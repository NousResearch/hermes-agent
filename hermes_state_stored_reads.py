"""The store reads behind ``session.transcript``, the ``session.list`` picker fields and a delegated child's
lookup by ``subagent_id``, kept in this one module: the display set a resume and a transcript page read,
the compression lineage a delegated child displays, the state check a poller's cursor holds (it tells an
append from every other change), the compaction handoffs a page shows as rows (and those a display row
before the page already holds), one keyset page of a
segment, and the session facts a picker shows. Bound onto SessionDB through ``SessionMessagesMixin``. Each
method's docstring names its storage-interface port.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional

from hermes_state_common import _placeholders

# The rows that can hold a compaction handoff (``ContextCompressor.classify_summary_content``'s prefixes,
# the merged delimiter, or the stored flag) for each shown row: the row itself, or an earlier copy of its
# message (same ``message_uid``: a later compaction carried, re-wrapped or unwrapped the carrier, so the
# display dedupe shows one copy and the older handoff would be lost). ``compaction_summary`` confirms each.
# Read in a fixed number of indexed lookups per call, never one per shown row: the shown rows by primary
# key, then the copies of their uids, live ones on idx_messages_session_uid, archived or removed ones on
# idx_messages_session_active, and (a legacy nullable column read before an open healed it) NULL ones;
# ``active`` holds nothing else. Each copy is paired with the shown rows after it in Python.
_HANDOFF_ROW_SQL = (" g.role IN ('user', 'assistant') AND (g._compressed_summary = 1"
                    " OR instr(g.content, '[CONTEXT COMPACTION') > 0 OR instr(g.content, '[CONTEXT SUMMARY]') > 0)")
_SHOWN_ROWS_SQL = (
    "SELECT g.id AS _shown_id, g.message_uid AS _shown_uid, g.session_id IN (SELECT value FROM json_each(:segments))"
    " AND" + _HANDOFF_ROW_SQL + " AS _holds FROM messages g WHERE g.id IN (SELECT value FROM json_each(:shown))")
# The full rows of the shown rows that can hold a handoff themselves; every other shown row is read for its
# id, uid and flag only, so an ordinary page never loads its bodies (images, tool results) a second time.
_HOLDER_ROWS_SQL = "SELECT g.* FROM messages g WHERE g.id IN (SELECT value FROM json_each(:ids))"
_HANDOFF_COPIES_SQL = " UNION ALL ".join(
    f"SELECT g.* FROM messages g{index} WHERE g.session_id IN (SELECT value FROM json_each(:segments)) AND {active}"
    " AND g.message_uid IN (SELECT value FROM json_each(:uids)) AND g.id < :below AND" + _HANDOFF_ROW_SQL
    for index, active in ((' INDEXED BY idx_messages_session_uid', 'g.active = 1'),
                          (' INDEXED BY idx_messages_session_active', 'g.active = 0'), ('', 'g.active IS NULL')))

# The handoff rows (as ``transcript_handoff_rows`` finds them) that a shown row of :sid before slot :before holds:
# driven from the matching handoff copies (by uid on idx_messages_session_uid; a legacy uid-less one by its
# content), each joined to the rows that hold it, kept when one is its slot's chosen display row.
_HANDOFF_SHOWN_BEFORE_SQL = (
    "SELECT DISTINCT g.* FROM messages g JOIN messages w ON w.session_id = g.session_id AND (w.id = g.id OR ("
    "g.message_uid IS NOT NULL AND g.message_uid <> '' AND w.message_uid = g.message_uid AND w.id > g.id))"
    " WHERE g.session_id = :sid AND g.role IN ('user', 'assistant') AND ({match})"
    " AND (g._compressed_summary = 1 OR instr(g.content, '[CONTEXT COMPACTION') > 0"
    " OR instr(g.content, '[CONTEXT SUMMARY]') > 0)"
    " AND w.display_order < :before AND w.id = (SELECT candidate.id FROM messages AS candidate"
    " WHERE candidate.session_id = :sid AND candidate.display_order = w.display_order"
    " AND (candidate.active = 1 OR candidate.compacted = 1){visible}{later}"
    " ORDER BY candidate.active DESC, candidate.id DESC LIMIT 1) ORDER BY g.id")

_REMOVED_IDENTITIES_SQL = (
    "SELECT session_id, id, role, timestamp, tool_call_id, tool_calls, tool_name, display_kind, display_metadata, "
    "message_uid, absorbed_message_uids, display_identity, active, compacted, "
    "CASE WHEN display_identity IS NULL THEN content END AS content "
    "FROM messages WHERE session_id IN ({ids}) AND active = 0 AND compacted = 0 ORDER BY id")

# The current display slot of the message whose first durable row is :durable: the newest shown copy of
# its identity (the live one first; a carried tail's copy when the original is removed), else the shown
# summary carrier that wraps it (same ``message_uid``; a carrier's content, so its identity, is new), else
# that row itself while it is shown. Each read uses its own index: the row by primary key, its identity's
# shown copies on idx_messages_display_identity, its live carrier on idx_messages_session_uid.
_CURSOR_ROW_SQL = ("SELECT display_identity, display_order, active, compacted, message_uid FROM messages"
                   " WHERE id = :durable AND session_id = :sid")
_CURSOR_COPY_SQL = (
    "SELECT display_order FROM messages INDEXED BY idx_messages_display_identity WHERE session_id = :sid"
    " AND display_identity = :identity AND (active = 1 OR compacted = 1) AND display_order IS NOT NULL"
    " ORDER BY active DESC, id DESC LIMIT 1")
_CURSOR_CARRIER_SQL = (
    "SELECT display_order FROM messages INDEXED BY idx_messages_session_uid WHERE session_id = :sid"
    " AND message_uid = :uid AND active = 1 AND _compressed_summary = 1 AND display_order IS NOT NULL"
    " ORDER BY id DESC LIMIT 1")

# The segment's newest replies, newest first, on the (session_id, id) index: an assistant row with text,
# live or archived by an in-place compaction, never a failed turn's notice or hidden scaffolding.
_LAST_REPLIES_SQL = (
    "SELECT content, _compressed_summary FROM messages INDEXED BY idx_messages_session_id WHERE session_id = ? AND role = 'assistant'"
    " AND (active = 1 OR compacted = 1) AND content IS NOT NULL AND content <> ''"
    " AND COALESCE(display_kind, '') NOT IN ('hidden', 'failed_turn'){visible} ORDER BY id DESC LIMIT 8")

# The read connection each thread's ``transcript_read_snapshot`` pinned, per SessionDB.
_SNAPSHOTS = threading.local()


def _pins() -> dict[int, Any]:
    pins = getattr(_SNAPSHOTS, 'pins', None)
    if pins is None:
        pins = _SNAPSHOTS.pins = {}
    return pins


def _ids(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(values)))


class _ReadCheck:
    """``transcript_read_snapshot``'s ``consistent()``; ``pinned``: one read transaction (else validated)."""

    def __init__(self, check, *, pinned: bool):
        self._check, self.pinned = check, pinned

    def __call__(self) -> bool:
        return bool(self._check())


class SessionStoredReadsMixin:
    """Stored-session reads for clients that read without attaching (``session.transcript``, the picker)."""

    # -- one read generation -------------------------------------------------------------------------

    def _pinned_read_conn(self) -> Any:
        """The connection this thread's open ``transcript_read_snapshot`` holds, else None (``_read_ctx``
        then checks one out as usual)."""
        return _pins().get(id(self))

    def _write_generation(self) -> Any:
        """What any committed or pending write moves: the writer connection (a reopen is a new one), its
        ``total_changes`` (this process's writes, all on it) and its ``PRAGMA data_version`` (every
        other connection's commits). A closed store has none: an object equal to nothing."""
        with self._lock:
            conn = self._conn
            if conn is None:
                return object()
            return conn, conn.total_changes, conn.execute('PRAGMA main.data_version').fetchone()[0]

    @contextmanager
    def transcript_read_snapshot(self) -> Iterator[_ReadCheck]:
        """Every read this thread makes through this SessionDB inside the block sees ONE generation: one
        read transaction on one pooled connection, which ``_read_ctx`` hands to each nested read. Yields
        ``consistent()``: false when the transaction ended early (a transient I/O error rolled it back),
        so the caller reads again. Without a pooled reader (not WAL, a read-only store, or the pool's
        ceiling: reads use the writer connection under the write lock) nothing is pinned, since a write
        from inside would wait on its own lock, and no lock is held across the block; each read is its
        own, and ``consistent()`` is false when any write landed between the block's start and the call
        (``_write_generation``), so the reads saw one generation exactly when it is true. ``.pinned``
        tells the two apart. Re-entrant.

        Storage interface: port as ``read_snapshot``."""
        pins = _pins()
        if id(self) in pins:
            conn = pins[id(self)]
            yield _ReadCheck(lambda: conn.in_transaction, pinned=True)
            return
        with self._read_ctx(pooled_only=True) as conn:
            if conn is None:
                start = self._write_generation()
                yield _ReadCheck(lambda: self._write_generation() == start, pinned=False)
                return
            began = not conn.in_transaction
            if began:
                conn.execute('BEGIN')
            pins[id(self)] = conn
            try:
                yield _ReadCheck(lambda: conn.in_transaction, pinned=True)
            finally:
                pins.pop(id(self), None)
                if began and conn.in_transaction:
                    conn.execute('ROLLBACK')

    # -- the display set -----------------------------------------------------------------------------

    def _with_removed_identities(self, rows, session_ids: list[str], conn=None) -> list[Any]:
        """*rows* (ordered by id) merged with the identities of *session_ids*' removed rows (``active=0,
        compacted=0``), which ``_dedupe_display_generations(drop_removed=True)`` needs only to hide a copy,
        never to show: no body (a legacy row without a stored display identity keeps its content for the
        key). *conn*: read on the caller's transaction, else on a read of its own (``self`` is a SessionDB
        or a worker's ``CompressionSnapshot``).

        Storage interface: port as ``display_removed_identities``."""
        sql = _REMOVED_IDENTITIES_SQL.format(ids=_placeholders(session_ids))
        removed = conn.execute(sql, tuple(session_ids)).fetchall() if conn is not None else self._read_all(sql, tuple(session_ids))
        return sorted([*rows, *removed], key=lambda row: row["id"]) if removed else list(rows)

    def _display_set(self, sql: str, session_ids: list[str]) -> list[Any]:
        """*sql*'s rows (bound to *session_ids*) with the removed rows' identities, in ONE read transaction:
        a compaction committing between the two reads marks a row removed after it was read live, so the
        removed read would hide the message while the live read never saw its new copy.

        Storage interface: port as ``display_set``."""
        def read(conn):
            began = not conn.in_transaction  # inside a transcript_read_snapshot: its transaction
            if began:
                conn.execute("BEGIN")
            try:
                return self._with_removed_identities(conn.execute(sql, tuple(session_ids)).fetchall(), session_ids, conn)
            finally:
                if began and conn.in_transaction:
                    conn.execute("ROLLBACK")
        return self._read_retrying_ioerr(read)

    # -- session.transcript --------------------------------------------------------------------------

    def transcript_lineage_ids(self, session_id: str) -> list[str]:
        """The compression lineage a delegated child's transcript displays: ``_resume_lineage_ids``, or the
        session alone while its row carries no ``_delegate_from``. A queued child's row is reserved with
        no ``model_config`` (``_reserve_child_transcript``); its delegation marker stays in memory until its
        first turn writes it, and a delegator that rotated through compression meanwhile would read as
        the child's own earlier segment. Every continuation of a child copies the marker.

        Storage interface: port as ``delegated_child_lineage_ids``."""
        row = self.get_session(session_id)
        cfg = row.get("model_config") if row else None
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except ValueError:
                cfg = None
        if not isinstance(cfg, dict) or not cfg.get("_delegate_from"):
            return [session_id]
        return self._resume_lineage_ids(session_id)

    def transcript_cursor_state(self, session_id: str, through_id: int,
                                lineage_ids: list[str] | None = None) -> tuple[int, int, int]:
        """``(live rows, inactive rows, transcript_revision)`` over *lineage_ids* (by default
        ``transcript_lineage_ids``), every segment a full read shows. Live rows: the session's own with
        id <= *through_id* (only it is appended to) and every live row of its earlier segments. An append
        to the session moves none of them. A compaction or rewind (anywhere, including rows appended after
        *through_id*) moves the inactive count; deleting a held row, or any live row of an earlier segment,
        moves the first; an in-place rewrite of any displayed row (an earlier segment's reaction included)
        the last. While all three hold, the session's rows after *through_id* are all that changed.

        Storage interface: port as ``transcript_cursor_state``."""
        ids = list(lineage_ids) if lineage_ids else self.transcript_lineage_ids(session_id)
        marks = _placeholders(ids)
        row = self._read_one(
            f"SELECT (SELECT COUNT(*) FROM messages WHERE session_id IN ({marks}) AND active = 1"
            " AND (session_id <> ? OR id <= ?)),"
            f" (SELECT COUNT(*) FROM messages WHERE session_id IN ({marks}) AND active = 0),"
            f" (SELECT COALESCE(SUM(transcript_revision), 0) FROM sessions WHERE id IN ({marks}))",
            (*ids, session_id, int(through_id), *ids, *ids))
        return int(row[0]), int(row[1]), int(row[2])

    def transcript_display_index_complete(self, segment: str) -> bool:
        """Whether every displayed row of *segment* has its display slot (``_ensure_display_order``'s check,
        read only, on the caller's ``transcript_read_snapshot`` when one is open): a rewrite of a row's content
        or ``display_kind`` clears its slot until the next backfill, and a page on the index skips that row.
        A store without the display columns has no index: False.

        Storage interface: port as ``display_index_complete``."""
        from hermes_state_messages import _DISPLAY_INDEX_MISSING_SQL
        with self._read_ctx() as conn:
            columns = set(self._message_column_names(conn))
        if not {"display_order", "display_identity"} <= columns:
            return False
        return self._read_one(_DISPLAY_INDEX_MISSING_SQL, (segment,)) is None

    def transcript_handoff_rows(self, shown_ids: list[int], segment_ids: list[str]) -> dict[int, list[dict[str, Any]]]:
        """``{shown row id: [candidate handoff message, ...]}`` in id order, for display rows *shown_ids*
        of the lineage *segment_ids*: the shown row when it can hold a handoff, and every earlier stored
        copy of its message that can (``_HANDOFF_ROW_SQL``). Decoded as ``_row_to_message_dict`` does,
        each with its own ``id``; the caller confirms a handoff with ``compaction_summary``.

        Storage interface: port as ``transcript_handoff_rows``."""
        if not shown_ids:
            return {}
        segments = json.dumps(list(segment_ids))
        with self._read_ctx() as conn:
            shown = conn.execute(_SHOWN_ROWS_SQL, {'segments': segments, 'shown': json.dumps(sorted(set(shown_ids)))}).fetchall()
            uids = {row['_shown_uid'] for row in shown if row['_shown_uid']}
            copies = conn.execute(_HANDOFF_COPIES_SQL, {
                'segments': segments, 'uids': json.dumps(sorted(uids)),
                'below': max(row['_shown_id'] for row in shown)}).fetchall() if uids else []
            holder_ids = sorted(int(row['_shown_id']) for row in shown if row['_holds'])
            holders = {int(row['id']): row for row in conn.execute(
                _HOLDER_ROWS_SQL, {'ids': json.dumps(holder_ids)}).fetchall()} if holder_ids else {}
        by_uid: dict[str, list[Any]] = {}
        for row in sorted(copies, key=lambda row: row['id']):
            by_uid.setdefault(row['message_uid'], []).append(row)
        found: dict[int, list[dict[str, Any]]] = {}
        for row in sorted(shown, key=lambda row: row['_shown_id']):
            held = [copy for copy in by_uid.get(row['_shown_uid'], ()) if copy['id'] < row['_shown_id']] if row['_shown_uid'] else []
            holder = holders.get(int(row['_shown_id'])) if row['_holds'] else None
            held += [holder] if holder is not None else []
            for copy in held:
                message = self._row_to_message_dict(copy, warn_context='session.transcript', summary_flag=True)
                found.setdefault(int(row['_shown_id']), []).append(message)
        return found

    def transcript_handoffs_shown_before(self, segment: str, before: int, handoffs: Iterable[tuple[Any, Any]]) -> list[dict[str, Any]]:
        """The stored copies of *handoffs* (``(message_uid or None, content)``) that a row of *segment*'s
        display set at a ``display_order`` below *before* holds (``transcript_handoff_rows``'s rule): the row
        itself, or an earlier copy of its message. A row is in the display set when it is its slot's
        chosen row (``_display_rows_from_conn``'s). Decoded as ``transcript_handoff_rows``; the caller
        confirms each with ``compaction_summary``. A paged read shows a handoff once, at its first row.

        Storage interface: port as ``transcript_handoffs_shown_before``."""
        from hermes_state_compaction_display import not_a_later_tool_copy_sql
        from hermes_state_messages import DISPLAY_VISIBLE_SQL
        handoffs = list(handoffs)
        uids = sorted({uid for uid, _ in handoffs if uid})
        # Structured (content-block) carriers are matched by their stored encoding, as strings are.
        contents = sorted({self._encode_content(content) for uid, content in handoffs
                           if not uid and isinstance(content, (str, list, dict))})
        match = [*(['g.message_uid IN (SELECT value FROM json_each(:uids))'] if uids else []),
                 *(["(g.message_uid IS NULL OR g.message_uid = '') AND g.content IN (SELECT value FROM json_each(:contents))"]
                   if contents else [])]
        if not match:
            return []
        sql = _HANDOFF_SHOWN_BEFORE_SQL.format(match=' OR '.join(f'({part})' for part in match), visible=DISPLAY_VISIBLE_SQL,
                                               later=not_a_later_tool_copy_sql('candidate'))
        with self._read_ctx() as conn:
            rows = conn.execute(sql, {'sid': segment, 'before': int(before), 'uids': json.dumps(uids),
                                      'contents': json.dumps(contents)}).fetchall()
        return [self._row_to_message_dict(row, warn_context='session.transcript', summary_flag=True) for row in rows]

    def transcript_segment_page(self, segment: str, *, latest: bool, after: Optional[tuple[int, int]],
                                count: int, inclusive: bool = False) -> tuple[list[Any], dict[int, int], list[Any]]:
        """``(rows, first durable ids, context row)`` of one keyset page of *segment*'s display index, in one
        read transaction (the cursor's slot, the page and its durable ids see one generation). *after*:
        ``(position, durable)`` of the cursor's row; the page starts at that message's current slot
        (``_CURSOR_*_SQL``), else at *position* when it was rewound or deleted, and holds *count* slots past
        it in reading order (from it when *inclusive*). *rows* are ``_display_rows_from_conn`` rows in
        display order; the context row is the display row before the page (empty when there is none).

        Storage interface: port as ``transcript_segment_page``."""
        def read(conn):
            began = not conn.in_transaction  # inside a transcript_read_snapshot: its transaction
            if began:
                conn.execute('BEGIN')
            try:
                position = None
                if after is not None:
                    found = self._transcript_cursor_slot(conn, segment, after[1])
                    position = after[0] if found is None else found
                    if inclusive:
                        position += 1 if latest else -1
                raw = self._display_rows_from_conn(conn, segment, limit=count, latest=latest, after=position)
                firsts = self._transcript_durable_firsts(conn, segment, raw)
                before = self._display_rows_from_conn(conn, segment, limit=1, latest=True,
                                                      after=min(row['display_order'] for row in raw)) if raw else []
                return raw, firsts, before
            finally:
                if began and conn.in_transaction:
                    conn.execute('ROLLBACK')
        return self._read_retrying_ioerr(read)

    @staticmethod
    def _transcript_cursor_slot(conn, segment: str, durable: int) -> Optional[int]:
        """The slot ``_CURSOR_*_SQL`` find for *durable*, or None (the message was rewound or deleted)."""
        row = conn.execute(_CURSOR_ROW_SQL, {'sid': segment, 'durable': durable}).fetchone()
        if row is None:
            return None
        if row['display_identity'] is not None:
            copy = conn.execute(_CURSOR_COPY_SQL, {'sid': segment, 'identity': row['display_identity']}).fetchone()
            if copy is not None:
                return copy[0]
        if row['message_uid']:
            carrier = conn.execute(_CURSOR_CARRIER_SQL, {'sid': segment, 'uid': row['message_uid']}).fetchone()
            if carrier is not None:
                return carrier[0]
        shown = (row['active'] == 1 or row['compacted'] == 1) and row['display_order'] is not None
        return row['display_order'] if shown else None

    @staticmethod
    def _transcript_durable_firsts(conn, segment: str, raw) -> dict[int, int]:
        """``{row id: first durable id}`` for the page rows a compaction re-wrote: a carried copy (its removed
        original has the same identity) and a summary carrier (the first row with its ``message_uid``; only a
        carrier folds by uid, so a pruned tool result keeps its own identity, as the display dedupe keeps it).
        A copy and a carrier keep the original's timestamp (``_insert_message_rows`` stamps it on the live
        dict for exactly this), so both reads stay on the (session_id, active, timestamp) index."""
        identities = list({row['display_identity'] for row in raw if row['display_identity'] is not None})
        stamps = list({row['timestamp'] for row in raw if row['timestamp'] is not None})
        by_identity = dict(conn.execute(
            "SELECT display_identity, MIN(id) FROM messages INDEXED BY idx_messages_session_active WHERE "
            f"session_id = ? AND active = 0 AND timestamp IN ({_placeholders(stamps)}) AND compacted = 0 "
            f"AND display_identity IN ({_placeholders(identities)}) GROUP BY display_identity",
            (segment, *stamps, *identities)).fetchall()) if identities and stamps else {}
        carriers = [row for row in raw if row['_compressed_summary'] and row['message_uid'] and row['timestamp'] is not None]
        uids, carrier_stamps = list({row['message_uid'] for row in carriers}), list({row['timestamp'] for row in carriers})
        by_uid = dict(conn.execute(
            "SELECT message_uid, MIN(id) FROM messages INDEXED BY idx_messages_session_active WHERE session_id = ?"
            f" AND active = 0 AND timestamp IN ({_placeholders(carrier_stamps)}) AND message_uid IN ({_placeholders(uids)})"
            " GROUP BY message_uid", (segment, *carrier_stamps, *uids)).fetchall()) if carriers else {}
        return {row['id']: min(row['display_order'], by_identity.get(row['display_identity'], row['id']),
                               by_uid.get(row['message_uid'], row['id']) if row['_compressed_summary'] else row['id'])
                for row in raw}

    def transcript_tool_calls_before(self, segment_ids: list[str], before_id: int, limit: int = 16) -> list[tuple[Any, Any]]:
        """``(tool_calls, tool_call_uids)`` JSON of the *limit* newest assistant rows with tool calls among
        *segment_ids* below row id *before_id*, nearest first: where a page's opening tool results find
        the calls that made them.

        Storage interface: port as ``transcript_tool_calls_before``."""
        with self._read_ctx() as conn:
            return [tuple(row) for row in conn.execute(
                "SELECT tool_calls, tool_call_uids FROM messages WHERE session_id IN (SELECT value FROM json_each(?))"
                " AND role = 'assistant' AND tool_calls IS NOT NULL AND id < ? ORDER BY id DESC LIMIT ?",
                (json.dumps(list(segment_ids)), int(before_id), int(limit))).fetchall()]

    # -- session.list picker fields ------------------------------------------------------------------

    def sessions_in_workspace(self, session_ids: Iterable[str], workspace: str) -> set[str]:
        """The ids among *session_ids* whose row is in *workspace*: the clause ``list_sessions_rich``
        filters the stored feed by (``hermes -c``'s rule: the same repo root, or at/under the folder).

        Storage interface: port as ``sessions_in_workspace``."""
        from hermes_state_sessions import _workspace_key_clause
        clause, params = _workspace_key_clause(workspace)
        with self._read_ctx() as conn:
            rows = conn.execute(f"SELECT s.id FROM sessions s WHERE s.id IN (SELECT value FROM json_each(?)) AND {clause}",
                                (json.dumps(list(session_ids)), *params)).fetchall()
        return {row[0] for row in rows}

    def picker_session_facts(self, root_ids: Iterable[str], unlisted_ids: Iterable[str],
                             segment_ids: Iterable[str]) -> tuple[dict, dict, dict]:
        """``(roots, unlisted, replies)`` in one store checkout. *roots*: ``{id: {id, cwd, git_repo_root}}``
        of *root_ids*. *unlisted*: ``{id: {id, git_branch, pinned, _preview_raw, last_active}}`` of
        *unlisted_ids* (segments ``list_sessions_rich`` did not list). *replies*: ``{segment: [(content,
        is summary carrier), ...]}``, each segment's newest assistant rows with text, newest first (at most
        8; live or archived by an in-place compaction, never hidden or a failed turn's notice), content
        decoded.

        Storage interface: port as ``picker_session_facts``."""
        from hermes_state_common import _sql_preview_raw, _sql_session_last_active
        from hermes_state_messages import DISPLAY_VISIBLE_SQL
        unlisted_ids = list(unlisted_ids)
        with self._read_ctx() as conn:
            roots = {row['id']: dict(row) for row in conn.execute(
                "SELECT id, cwd, git_repo_root FROM sessions WHERE id IN (SELECT value FROM json_each(?))",
                (_ids(root_ids),)).fetchall()}
            unlisted = {row['id']: dict(row) for row in conn.execute(
                f"SELECT s.id, s.git_branch, s.pinned, {_sql_preview_raw('s.id')}, {_sql_session_last_active('s')} "
                "AS last_active FROM sessions s WHERE s.id IN (SELECT value FROM json_each(?))",
                (_ids(unlisted_ids),)).fetchall()} if unlisted_ids else {}
            sql = _LAST_REPLIES_SQL.format(visible=DISPLAY_VISIBLE_SQL)
            replies = {segment: [(self._decode_content(content), bool(summary))
                                 for content, summary in conn.execute(sql, (segment,))]
                       for segment in set(segment_ids)}
        return roots, unlisted, replies

    def session_current_segments(self, session_ids: Iterable[str]) -> dict[str, str]:
        """``{logical id: current physical segment}`` in one store checkout: a local receipt's entry (extracted
        in SQL, as ``session_live_listing._store_facts`` does), else the compression tip
        (``session_viewer_close._physical_target``'s rule). Ids with no session row are absent.

        Storage interface: port as ``current_segments``."""
        from hermes_state_local import POLICY_PREFIX
        with self._read_ctx() as conn:
            rows = conn.execute(
                "SELECT s.id, s.end_reason, (SELECT json_extract(m.value, '$.entry.session_id') FROM state_meta m "
                "WHERE m.key = ? || s.id AND json_valid(m.value)) AS entry FROM sessions s "
                "WHERE s.id IN (SELECT value FROM json_each(?))", (POLICY_PREFIX, _ids(session_ids))).fetchall()
            # Only a row compression ended has a tip past itself: walked on this one connection.
            return {row['id']: row['entry'] or (self._compression_tip_on(conn, row['id'])
                                                if row['end_reason'] == 'compression' else row['id']) for row in rows}

    # -- delegated children --------------------------------------------------------------------------

    def subagent_session_ids(self, subagent_id: str, marker_key: str) -> list[str]:
        """Ids of the ``source='subagent'`` sessions whose ``model_config`` carries *subagent_id* under
        *marker_key*, newest ``started_at`` first (ties: id descending). The caller proves ownership.

        Storage interface: port as ``subagent_session_ids``."""
        from hermes_state_common import _sql_json_extract
        rows = self._read_all(
            "SELECT id FROM sessions WHERE source = 'subagent' AND "
            f"{_sql_json_extract('model_config', '$.' + marker_key)} = ? ORDER BY started_at DESC, id DESC",
            (subagent_id,))
        return [row["id"] for row in rows]
