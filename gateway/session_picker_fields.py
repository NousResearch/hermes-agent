"""What a session picker shows for each ``session.list`` row (``picker: true``), and the folder filter (#150091).

Location (``cwd``, ``workspace``) comes from the conversation's first row: the row ``workspace`` filters
by (``list_sessions_rich`` filters root rows, as ``hermes -c`` does), and the row a workspace move
rewrites with every other one. The rest comes from its current segment (a local receipt's entry, else
the compression tip), as ``list_sessions_rich`` projects a compressed chain onto its tip: the first
prompt there (``preview``), its branch, pin, newest activity and newest reply. The stored feed passes the
columns ``list_sessions_rich`` already read; the live feed reads them here. Either way a page is one
store checkout in a worker thread: batched lookups and one indexed probe per segment for its last reply.
"""
from __future__ import annotations

import json

# The segment's newest replies, newest first, on the (session_id, id) index: an assistant row with text,
# live or archived by an in-place compaction, never a failed turn's notice or hidden scaffolding. Each is
# projected as a display read projects it (``project_compaction_message_for_display``): a pure handoff the
# compressor wrote as an assistant row shows nothing, so the reply before it stands; a carrier shows the
# reply it wraps.
_LAST_REPLIES_SQL = (
    "SELECT content, _compressed_summary FROM messages INDEXED BY idx_messages_session_id WHERE session_id = ? AND role = 'assistant'"
    " AND (active = 1 OR compacted = 1) AND content IS NOT NULL AND content <> ''"
    " AND COALESCE(display_kind, '') NOT IN ('hidden', 'failed_turn'){visible} ORDER BY id DESC LIMIT 8")


def workspace_members(db, session_ids, workspace):
    """The ids among *session_ids* whose row is in *workspace*: the clause ``list_sessions_rich``
    filters the stored feed by (``hermes -c``'s rule: the same repo root, or at/under the folder)."""
    from hermes_state_sessions import _workspace_key_clause
    clause, params = _workspace_key_clause(workspace)
    with db._read_ctx() as conn:
        rows = conn.execute(f"SELECT s.id FROM sessions s WHERE s.id IN (SELECT value FROM json_each(?)) AND {clause}",
                            (json.dumps(list(session_ids)), *params)).fetchall()
    return {row[0] for row in rows}


def _last_reply(db, conn, segment):
    from agent.compaction_display import project_compaction_message_for_display
    from agent.message_content import flatten_message_text
    from hermes_state_common import _shape_preview
    from hermes_state_messages import DISPLAY_VISIBLE_SQL
    for content, summary in conn.execute(_LAST_REPLIES_SQL.format(visible=DISPLAY_VISIBLE_SQL), (segment,)):
        shown = project_compaction_message_for_display(
            {'role': 'assistant', 'content': db._decode_content(content), '_compressed_summary': bool(summary)})
        text = _shape_preview(flatten_message_text(shown['content'])) if shown is not None else ''
        if text:
            return text
    return None


def _ids(values):
    return json.dumps(sorted(set(values)))


def fill_picker_fields(db, items):
    """Set the picker fields on each ``(row, root, segment, listed)``: *listed* is the segment's
    ``list_sessions_rich`` row (its ``preview`` / ``last_active`` / branch / pin), None to read them."""
    from hermes_state_common import _shape_preview, _sql_preview_raw, _sql_session_last_active
    from hermes_state_sessions import workspace_key
    unlisted = [segment for _, _, segment, listed in items if listed is None]
    with db._read_ctx() as conn:
        roots = {row['id']: dict(row) for row in conn.execute(
            "SELECT id, cwd, git_repo_root FROM sessions WHERE id IN (SELECT value FROM json_each(?))",
            (_ids(root for _, root, _, _ in items),)).fetchall()}
        segments = {row['id']: dict(row) for row in conn.execute(
            f"SELECT s.id, s.git_branch, s.pinned, {_sql_preview_raw('s.id')}, {_sql_session_last_active('s')} "
            "AS last_active FROM sessions s WHERE s.id IN (SELECT value FROM json_each(?))",
            (_ids(unlisted),)).fetchall()} if unlisted else {}
        replies = {segment: _last_reply(db, conn, segment) for segment in {item[2] for item in items}}
    for row, root, segment, listed in items:
        if listed is None:
            listed = segments.get(segment)
            if listed is None:
                continue  # deleted since it was listed: the row keeps what the feed read
            listed = {**listed, 'preview': _shape_preview(listed['_preview_raw'])}
        first = roots.get(root) or {}
        row.update(preview=listed.get('preview') or None, cwd=first.get('cwd') or None,
                   workspace=workspace_key(first), git_branch=listed.get('git_branch') or None,
                   last_active_at=listed.get('_effective_last_active') or listed.get('last_active'),
                   pinned=bool(listed.get('pinned')), last_reply=replies[segment])


def current_segments(authority, session_ids):
    """``{logical id: current physical segment}`` in one store checkout: a local receipt's entry (extracted in
    SQL, as ``session_live_listing._store_facts`` does), else the compression tip
    (``session_viewer_close._physical_target``'s rule)."""
    from hermes_state_local import POLICY_PREFIX
    db = authority.db
    with db._read_ctx() as conn:
        rows = conn.execute(
            "SELECT s.id, s.end_reason, (SELECT json_extract(m.value, '$.entry.session_id') FROM state_meta m "
            "WHERE m.key = ? || s.id AND json_valid(m.value)) AS entry FROM sessions s "
            "WHERE s.id IN (SELECT value FROM json_each(?))", (POLICY_PREFIX, _ids(session_ids))).fetchall()
        # Only a row compression ended has a tip past itself: walked on this one connection.
        return {row['id']: row['entry'] or (db._compression_tip_on(conn, row['id'])
                                            if row['end_reason'] == 'compression' else row['id']) for row in rows}


def live_picker_page(authority, sessions, limit, workspace=None, picker=False):
    """The live feed's page: *sessions* (newest first) in *workspace*, cut to *limit*, with the picker
    fields when asked."""
    if workspace is not None:
        members = workspace_members(authority.db, [row['session_id'] for row in sessions], workspace)
        sessions = [row for row in sessions if row['session_id'] in members]
    page = sessions[:limit]
    if picker and page:
        segments = current_segments(authority, [row['session_id'] for row in page])
        fill_picker_fields(authority.db, [(row, row['session_id'], segments.get(row['session_id'], row['session_id']),
                                           None) for row in page])
    return page
