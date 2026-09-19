"""Files-only provider values, owned by one live agent/DB/physical session.

Canonical rows are safe before publication. Only staging and the normal gateway's
active SQL/admission join can bind a row; copying IDs or text grants nothing.
No file read, custody refresh, durable payload, or restart reconstruction lives here.
"""
from __future__ import annotations


from dataclasses import dataclass, field
from typing import Any
import weakref


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        if any(type(k) is not str for k in value):
            raise TypeError('Files JSON keys must be text')
        return ('object', tuple((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, list):
        return ('array', tuple(_freeze(v) for v in value))
    if value is None or type(value) in (str, bool, int, float):
        return ('scalar', value)
    raise TypeError('Files content must be JSON data')


def _thaw(value: Any) -> Any:
    tag, body = value
    if tag == 'object':
        return {k: _thaw(v) for k, v in body}
    if tag == 'array':
        return [_thaw(v) for v in body]
    return body


def _lock(agent):
    from agent.session_persistence import _persist_lock
    return _persist_lock(agent)


@dataclass(eq=False, repr=False)
class _FilesLiveEntry:
    owner: Any
    db: Any
    session_id: str
    turn_id: str
    admission_id: Any
    safe: str
    row: Any
    bound_content: str
    prepared: Any = field(repr=False)
    sealed: Any = field(default=None, repr=False)
    row_id: Any = None
    state: str = 'PREPARING'

    def __repr__(self):
        return '<Files live context: private>'

    def retire(self):
        self.prepared = self.sealed = self.row = None
        self.state = 'RETIRED'

    def valid(self, agent):
        row = self.row
        return (self.owner() is agent and self.db is getattr(agent, '_session_db', None)
                and self.session_id == getattr(agent, 'session_id', None)
                and self.state != 'RETIRED' and isinstance(row, dict)
                and row.get('role') == 'user' and row.get('content') == self.bound_content
                and row.get('api_content') is None and not row.get('_compressed_summary'))


def _entries(agent):
    entries = getattr(agent, '_files_live_entries', ())
    return entries if isinstance(entries, (list, tuple)) else ()


def retire_files_context(agent):
    with _lock(agent):
        for entry in _entries(agent):
            if entry.owner() is agent:
                entry.retire()
        agent._files_live_entries = []


def prune_files_context(agent, messages, *, finish=False):
    with _lock(agent):
        kept = []
        for entry in _entries(agent):
            if (entry.valid(agent) and any(row is entry.row for row in messages)
                    and (not finish or entry.sealed is not None)):
                if finish:
                    entry.state = 'RETAINED'
                kept.append(entry)
            elif entry.owner() is agent:
                entry.retire()
        agent._files_live_entries = kept


def stage_files_context(agent, transcript, row, content):
    with _lock(agent):
        # A shallow-copied agent must never adopt its donor's table.
        entries: list[_FilesLiveEntry] = list(_entries(agent))
        for old in entries:
            if old.owner() is not agent:
                entries = []
                break
        agent._files_live_entries = entries
        entry = _FilesLiveEntry(weakref.ref(agent), getattr(agent, '_session_db', None),
            agent.session_id, agent._current_turn_id, transcript.admission_id,
            str(transcript), row, str(transcript), _freeze(content))
        entries.append(entry)
        agent._files_safe_results = True
        transcript.bind(row)


def files_entry(agent, row):
    with _lock(agent):
        return next((e for e in _entries(agent) if e.row is row and e.valid(agent)), None)


def require_current_files_context(agent, messages):
    from agent.session_persistence import FilesUserTranscript
    transcript = getattr(agent, '_persist_user_message_override', None)
    if not isinstance(transcript, FilesUserTranscript):
        return None
    entry = files_entry(agent, transcript.message)
    if entry is None or not any(row is entry.row for row in messages):
        raise RuntimeError('Files context unavailable: current input binding was replaced; reattach files')
    return entry


def seal_files_context(agent, messages, prefetch, plugin_context):
    from agent.turn_context import compose_user_api_content
    with _lock(agent):
        entry = require_current_files_context(agent, messages)
        if entry is not None and entry.sealed is None:
            payload = _thaw(entry.prepared)
            composed = compose_user_api_content(payload, prefetch, plugin_context)
            entry.sealed = _freeze(payload if composed is None else composed)
            entry.prepared = None
            entry.state = 'SEALED'


def merge_files_gateway_notes(agent, row, notes):
    from agent.turn_context import append_notes_to_multimodal_content
    with _lock(agent):
        entry = files_entry(agent, row)
        if entry is None or entry.sealed is not None:
            return False
        payload = _thaw(entry.prepared)
        if not isinstance(payload, list):
            return False
        append_notes_to_multimodal_content(payload, notes)
        entry.prepared = _freeze(payload)
        return True


def files_provider_content(agent, row, *, preparing=False):
    with _lock(agent):
        entry = files_entry(agent, row)
        if entry is None:
            return False, None
        frozen = entry.sealed if entry.sealed is not None else entry.prepared if preparing else None
        if frozen is None:
            raise RuntimeError('Files context unavailable: provider content is not sealed')
        return True, _thaw(frozen)


def files_estimate_view(agent, messages):
    """Ephemeral pricing copy only; compression still receives safe canonical rows."""
    view = []
    expanded = False
    for row in messages:
        found, content = files_provider_content(agent, row, preparing=True)
        view.append({**row, 'content': content} if found else row)
        expanded |= found
    return view if expanded else messages


def capture_files_row_ids(agent):
    with _lock(agent):
        for entry in _entries(agent):
            if entry.valid(agent) and entry.row_id is None:
                row_id = entry.row.get('_row_id')
                if type(row_id) is int and entry.row.get('_db_persisted'):
                    entry.row_id = row_id


def safe_files_result(agent, result, *, force=False, trusted_carry=False):
    """Independent canonical snapshot, never an alias to retained safe history.

    Unknown alternate result implementations cannot prove a transcript mapping:
    refuse their messages rather than exporting an untyped provider request.
    """
    if not force and getattr(agent, '_files_safe_results', False) is not True:
        return result
    from agent.session_persistence import FilesUserTranscript
    transcript = getattr(agent, '_persist_user_message_override', None)
    failed = {'failed': True, 'completed': False, 'messages': [],
              'error': 'Files result unavailable', 'failure_reason': 'files_result_unavailable'}
    if not isinstance(result, dict):
        return failed
    canonical = getattr(agent, '_session_messages', ()) or ()
    rows = result.get('messages', [])
    try:
        # This is a serialization integrity check, NEVER reattachment authority.
        # Alternate runners may return clones, but not a raw provider transcript.
        safe_rows = [_freeze(row) for row in canonical]
        if (not isinstance(rows, list) or (not trusted_carry and
                any(_freeze(row) not in safe_rows for row in rows))):
            return failed
        out = _thaw(_freeze(result))
    except TypeError:
        return failed
    if isinstance(transcript, FilesUserTranscript):
        out.pop('current_turn_user_idx', None)
        out.pop('turn_id', None)
        for index, row in enumerate(rows):
            if row is transcript.message:
                out['current_turn_user_idx'] = index
                out['turn_id'] = getattr(agent, '_current_turn_id', '')
                break
        else:
            # Idempotent projection of the complete canonical snapshot: carry the
            # exact bound row's coordinate through the copy, not a text reanchor.
            if [_freeze(row) for row in rows] == safe_rows:
                for index, row in enumerate(canonical):
                    if row is transcript.message:
                        out['current_turn_user_idx'] = index
                        out['turn_id'] = getattr(agent, '_current_turn_id', '')
                        break
    return out


class FilesReplayBindings:
    """One normal gateway load; provisional bindings never authorize unselected rows."""
    def __init__(self, agent, history, session_id):
        self.agent = agent
        self.sources = []
        self.targets = []
        self.live_allowed = []
        if session_id != getattr(agent, 'session_id', None):
            return
        with _lock(agent):
            entries = [e for e in _entries(agent) if e.valid(agent) and e.sealed is not None]
            self.live_allowed = [e for e in entries if e.row_id is None]
            if not entries:
                return
            try:
                active = agent._session_db.get_messages(session_id)
            except Exception:
                return
            from agent.memory_manager import sanitize_context
            by_id = {r['id']: r for r in active}
            for entry in entries:
                if entry.row_id is None or not entry.admission_id:
                    continue
                stored = by_id.get(entry.row_id)
                if not (stored and stored.get('session_id') == session_id
                        and stored.get('role') == 'user'
                        and stored.get('platform_message_id') == entry.admission_id
                        and stored.get('content') == entry.safe and stored.get('api_content') is None):
                    continue
                # Both SQL and source must have unique admission identity: cloned
                # compaction rows with copied platform IDs are not a new authority.
                if sum(r.get('platform_message_id') == entry.admission_id for r in active) != 1:
                    continue
                self.live_allowed.append(entry)
                matches = [r for r in history if r.get('message_id', r.get('platform_message_id')) == entry.admission_id]
                if len(matches) != 1:
                    continue
                source = matches[0]
                if (source.get('role') == 'user'
                        and source.get('_row_id', entry.row_id) == entry.row_id
                        and source.get('content') == sanitize_context(entry.safe).strip()
                        and source.get('api_content') is None and not source.get('_compressed_summary')):
                    self.sources.append((source, entry))

    def transformed(self, source, target, *, unchanged):
        if unchanged:
            for original, entry in self.sources:
                if original is source:
                    self.targets.append((entry, target, target.get('content')))

    def commit(self, selected):
        with _lock(self.agent):
            for entry in _entries(self.agent):
                if entry.owner() is self.agent and entry not in self.live_allowed:
                    entry.retire()
            for entry, target, content in self.targets:
                if (entry.valid(self.agent) and any(row is target for row in selected)
                        and target.get('content') == content and target.get('api_content') is None):
                    entry.row, entry.bound_content = target, content
            prune_files_context(self.agent, selected)
        self.sources.clear()
        self.targets.clear()
        self.live_allowed.clear()
