"""The store reads behind ``session.transcript``, the ``session.list`` picker fields and a delegated child's
lookup by ``subagent_id``: the lineage display set with its durable keys, one keyset page of a segment, the
compaction handoffs a page shows as rows (and those a display row before the page already holds), the state
check a poller's cursor holds (it tells an append from every other change), the session facts a picker shows
and the segments a picker reads."""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from session_rules.turns import flatten_message_text, project_compaction_message_for_display

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema_ddl import _sql_json_extract
from hermes_state_common import _shape_preview
from persistence.sqlite.conversation._store.compaction_display import not_a_later_tool_copy_sql
from persistence.sqlite.conversation._store.compression import _compression_tip_on
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence.sqlite.conversation._store.messages import (
    _decode_content, _dedupe_display_generations, _display_rows_from_conn, _encode_content, _fetch_conversation_rows,
    _message_column_names, _resume_lineage_ids, _row_to_message_dict, _rows_to_conversation,
)
from persistence.sqlite.conversation._store.messages_sql import _DISPLAY_INDEX_MISSING_SQL, DISPLAY_VISIBLE_SQL
from persistence.sqlite.conversation._store.sessions import _workspace_key_clause, get_session
from persistence.sqlite.conversation._store.sql_common import _placeholders, _sql_preview_raw, _sql_session_last_active

# The rows that can hold a compaction handoff (``ContextCompressor.classify_summary_content``'s prefixes,
# the merged delimiter, or the stored flag) for each shown row: the row itself, or an earlier copy of its
# message (same ``message_uid``: a later compaction carried, re-wrapped or unwrapped the carrier, so the
# display dedupe shows one copy and the older handoff would be lost). The caller confirms each.
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


def _ids(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(values)))


def _in_one_read(holder: StoreHolder, read):
    """``read(conn)`` in one read transaction (inside a ``transcript_read_snapshot``: its transaction)."""
    def run(conn):
        began = not conn.in_transaction
        if began:
            conn.execute("BEGIN")
        try:
            return read(conn)
        finally:
            if began and conn.in_transaction:
                conn.execute("ROLLBACK")
    return holder._read_retrying_ioerr(run)


# -- session.transcript ------------------------------------------------------------------------------

def display_generations(holder: StoreHolder, session_id: str, lineage_ids: list[str]) -> list[tuple[dict[str, Any], int]]:
    """``[(replay dict with its row id, first durable id), ...]`` of the lineage's display set: the one display
    dedupe over every segment, keyed by each message's first durable row, a removed original's included."""
    first_ids: dict[int, int] = {}
    rows = _dedupe_display_generations(holder, _fetch_conversation_rows(
        holder, lineage_ids, "", with_session_id=False, with_compacted=True), drop_removed=True, first_ids=first_ids)
    messages = _rows_to_conversation(holder, rows, session_id=session_id, include_ancestors=True,
                                     repair_alternation=False, include_row_ids=True)
    return [(message, first_ids.get(message["_row_id"], message["_row_id"])) for message in messages]


def transcript_lineage_ids(holder: StoreHolder, session_id: str) -> list[str]:
    """The compression lineage a delegated child's transcript displays: ``_resume_lineage_ids``, or the
    session alone while its row carries no ``_delegate_from``. A queued child's row is reserved with no
    ``model_config``; its delegation marker stays in memory until its first turn writes it, and a delegator
    that rotated through compression meanwhile would read as the child's own earlier segment. Every
    continuation of a child copies the marker."""
    row = get_session(holder, session_id)
    cfg = row.get("model_config") if row else None
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except ValueError:
            cfg = None
    if not isinstance(cfg, dict) or not cfg.get("_delegate_from"):
        return [session_id]
    return _resume_lineage_ids(holder, session_id)


def transcript_cursor_state(holder: StoreHolder, session_id: str, through_id: int, lineage_ids: list[str]) -> tuple[int, int, int]:
    """``(live rows, inactive rows, transcript_revision)`` over *lineage_ids*: the session's own live rows with
    id <= *through_id* (only it is appended to) and every live row of its other segments; every inactive row;
    the summed revision."""
    ids = list(lineage_ids) or [session_id]
    marks = _placeholders(ids)
    row = holder._read_one(
        f"SELECT (SELECT COUNT(*) FROM messages WHERE session_id IN ({marks}) AND active = 1"
        " AND (session_id <> ? OR id <= ?)),"
        f" (SELECT COUNT(*) FROM messages WHERE session_id IN ({marks}) AND active = 0),"
        f" (SELECT COALESCE(SUM(transcript_revision), 0) FROM sessions WHERE id IN ({marks}))",
        (*ids, session_id, int(through_id), *ids, *ids))
    return int(row[0]), int(row[1]), int(row[2])


def transcript_display_index_complete(holder: StoreHolder, segment: str) -> bool:
    """Whether every displayed row of *segment* has its display slot (``_ensure_display_order``'s check, read
    only, on the caller's ``transcript_read_snapshot`` when one is open). A store without the display columns
    has no index: False."""
    with holder._read_ctx() as conn:
        columns = set(_message_column_names(holder, conn))
    if not {"display_order", "display_identity"} <= columns:
        return False
    return holder._read_one(_DISPLAY_INDEX_MISSING_SQL, (segment,)) is None


def transcript_handoff_rows(holder: StoreHolder, shown_ids: list[int], segment_ids: list[str]) -> dict[int, list[dict[str, Any]]]:
    """``{shown row id: [candidate handoff message, ...]}`` in id order, for display rows *shown_ids* of the
    lineage *segment_ids*: the shown row when it can hold a handoff, and every earlier stored copy of its
    message that can (``_HANDOFF_ROW_SQL``), decoded as ``_row_to_message_dict`` does."""
    if not shown_ids:
        return {}
    segments = json.dumps(list(segment_ids))
    with holder._read_ctx() as conn:
        shown = conn.execute(_SHOWN_ROWS_SQL, {"segments": segments, "shown": json.dumps(sorted(set(shown_ids)))}).fetchall()
        uids = {row["_shown_uid"] for row in shown if row["_shown_uid"]}
        copies = conn.execute(_HANDOFF_COPIES_SQL, {
            "segments": segments, "uids": json.dumps(sorted(uids)),
            "below": max(row["_shown_id"] for row in shown)}).fetchall() if uids else []
        holder_ids = sorted(int(row["_shown_id"]) for row in shown if row["_holds"])
        holders = {int(row["id"]): row for row in conn.execute(
            _HOLDER_ROWS_SQL, {"ids": json.dumps(holder_ids)}).fetchall()} if holder_ids else {}
    by_uid: dict[str, list[Any]] = {}
    for row in sorted(copies, key=lambda row: row["id"]):
        by_uid.setdefault(row["message_uid"], []).append(row)
    found: dict[int, list[dict[str, Any]]] = {}
    for row in sorted(shown, key=lambda row: row["_shown_id"]):
        held = [copy for copy in by_uid.get(row["_shown_uid"], ()) if copy["id"] < row["_shown_id"]] if row["_shown_uid"] else []
        own = holders.get(int(row["_shown_id"])) if row["_holds"] else None
        held += [own] if own is not None else []
        for copy in held:
            message = _row_to_message_dict(holder, copy, warn_context="session.transcript", summary_flag=True)
            found.setdefault(int(row["_shown_id"]), []).append(message)
    return found


def transcript_handoffs_shown_before(holder: StoreHolder, segment: str, before: int, handoffs: Iterable[tuple[Any, Any]]) -> list[dict[str, Any]]:
    """The stored copies of *handoffs* (``(message_uid or None, content)``) that a row of *segment*'s display set
    at a ``display_order`` below *before* holds (``transcript_handoff_rows``'s rule), decoded as it decodes them."""
    handoffs = list(handoffs)
    uids = sorted({uid for uid, _ in handoffs if uid})
    # Structured (content-block) carriers are matched by their stored encoding, as strings are.
    contents = sorted({_encode_content(content) for uid, content in handoffs
                       if not uid and isinstance(content, (str, list, dict))})
    match = [*(["g.message_uid IN (SELECT value FROM json_each(:uids))"] if uids else []),
             *(["(g.message_uid IS NULL OR g.message_uid = '') AND g.content IN (SELECT value FROM json_each(:contents))"]
               if contents else [])]
    if not match:
        return []
    sql = _HANDOFF_SHOWN_BEFORE_SQL.format(match=" OR ".join(f"({part})" for part in match), visible=DISPLAY_VISIBLE_SQL,
                                           later=not_a_later_tool_copy_sql("candidate"))
    with holder._read_ctx() as conn:
        rows = conn.execute(sql, {"sid": segment, "before": int(before), "uids": json.dumps(uids),
                                  "contents": json.dumps(contents)}).fetchall()
    return [_row_to_message_dict(holder, row, warn_context="session.transcript", summary_flag=True) for row in rows]


def transcript_segment_page(holder: StoreHolder, segment: str, *, latest: bool, after: Optional[tuple[int, int]],
                            count: int, inclusive: bool = False) -> tuple[list[Any], dict[int, int], list[Any]]:
    """``(rows, first durable ids, context row)`` of one keyset page of *segment*'s display index, in one read
    transaction (the cursor's slot, the page and its durable ids see one generation). *rows* are
    ``_display_rows_from_conn`` rows in display order; the context row is the display row before the page."""
    def read(conn):
        position = None
        if after is not None:
            found = _transcript_cursor_slot(conn, segment, after[1])
            position = after[0] if found is None else found
            if inclusive:
                position += 1 if latest else -1
        raw = _display_rows_from_conn(conn, segment, limit=count, latest=latest, after=position)
        firsts = _transcript_durable_firsts(conn, segment, raw)
        before = _display_rows_from_conn(conn, segment, limit=1, latest=True,
                                         after=min(row["display_order"] for row in raw)) if raw else []
        return raw, firsts, before
    return _in_one_read(holder, read)


def segment_page_messages(holder: StoreHolder, segment: str, *, latest: bool, after: Optional[tuple[int, int]],
                          count: int, inclusive: bool = False):
    """``(messages with their (display_order, durable) key, full, edge key, first slot)`` of one segment page:
    the rows get the conversation normalization a resume applies, with the display row before the page as
    context (dropped) so a prompt/reply pair split by the page edge is still one pair."""
    raw, firsts, before = transcript_segment_page(holder, segment, latest=latest, after=after, count=count, inclusive=inclusive)
    keys = {row["id"]: (row["display_order"], firsts[row["id"]]) for row in raw}
    context = {row["id"] for row in before}
    messages = [message for message in _rows_to_conversation(
        holder, [*before, *raw], session_id=segment, include_ancestors=False, repair_alternation=False, include_row_ids=True)
        if message["_row_id"] not in context]
    edge = None if not raw else keys[raw[0]["id"] if latest else raw[-1]["id"]]
    start = min((row["display_order"] for row in raw), default=None)
    return [(message, keys[message["_row_id"]]) for message in messages], len(raw) == count, edge, start


def _transcript_cursor_slot(conn, segment: str, durable: int) -> Optional[int]:
    """The slot ``_CURSOR_*_SQL`` find for *durable*, or None (the message was rewound or deleted)."""
    row = conn.execute(_CURSOR_ROW_SQL, {"sid": segment, "durable": durable}).fetchone()
    if row is None:
        return None
    if row["display_identity"] is not None:
        copy = conn.execute(_CURSOR_COPY_SQL, {"sid": segment, "identity": row["display_identity"]}).fetchone()
        if copy is not None:
            return copy[0]
    if row["message_uid"]:
        carrier = conn.execute(_CURSOR_CARRIER_SQL, {"sid": segment, "uid": row["message_uid"]}).fetchone()
        if carrier is not None:
            return carrier[0]
    shown = (row["active"] == 1 or row["compacted"] == 1) and row["display_order"] is not None
    return row["display_order"] if shown else None


def _transcript_durable_firsts(conn, segment: str, raw) -> dict[int, int]:
    """``{row id: first durable id}`` for the page rows a compaction re-wrote: a carried copy (its removed
    original has the same identity) and a summary carrier (the first row with its ``message_uid``; only a
    carrier folds by uid, so a pruned tool result keeps its own identity, as the display dedupe keeps it).
    A copy and a carrier keep the original's timestamp, so both reads stay on the (session_id, active,
    timestamp) index."""
    identities = list({row["display_identity"] for row in raw if row["display_identity"] is not None})
    stamps = list({row["timestamp"] for row in raw if row["timestamp"] is not None})
    by_identity = dict(conn.execute(
        "SELECT display_identity, MIN(id) FROM messages INDEXED BY idx_messages_session_active WHERE "
        f"session_id = ? AND active = 0 AND timestamp IN ({_placeholders(stamps)}) AND compacted = 0 "
        f"AND display_identity IN ({_placeholders(identities)}) GROUP BY display_identity",
        (segment, *stamps, *identities)).fetchall()) if identities and stamps else {}
    carriers = [row for row in raw if row["_compressed_summary"] and row["message_uid"] and row["timestamp"] is not None]
    uids, carrier_stamps = list({row["message_uid"] for row in carriers}), list({row["timestamp"] for row in carriers})
    by_uid = dict(conn.execute(
        "SELECT message_uid, MIN(id) FROM messages INDEXED BY idx_messages_session_active WHERE session_id = ?"
        f" AND active = 0 AND timestamp IN ({_placeholders(carrier_stamps)}) AND message_uid IN ({_placeholders(uids)})"
        " GROUP BY message_uid", (segment, *carrier_stamps, *uids)).fetchall()) if carriers else {}
    return {row["id"]: min(row["display_order"], by_identity.get(row["display_identity"], row["id"]),
                           by_uid.get(row["message_uid"], row["id"]) if row["_compressed_summary"] else row["id"])
            for row in raw}


def transcript_tool_calls_before(holder: StoreHolder, segment_ids: list[str], before_id: int, limit: int = 16) -> list[tuple[Any, Any]]:
    """``(tool_calls, tool_call_uids)`` JSON of the *limit* newest assistant rows with tool calls among
    *segment_ids* below row id *before_id*, nearest first."""
    with holder._read_ctx() as conn:
        return [tuple(row) for row in conn.execute(
            "SELECT tool_calls, tool_call_uids FROM messages WHERE session_id IN (SELECT value FROM json_each(?))"
            " AND role = 'assistant' AND tool_calls IS NOT NULL AND id < ? ORDER BY id DESC LIMIT ?",
            (json.dumps(list(segment_ids)), int(before_id), int(limit))).fetchall()]


# -- session.list picker fields ----------------------------------------------------------------------

def sessions_in_workspace(holder: StoreHolder, session_ids: Iterable[str], workspace: str) -> set[str]:
    """The ids among *session_ids* whose row is in *workspace*: the clause the listing filters by
    (``hermes -c``'s rule: the same repo root, or at/under the folder)."""
    clause, params = _workspace_key_clause(workspace)
    with holder._read_ctx() as conn:
        rows = conn.execute(f"SELECT s.id FROM sessions s WHERE s.id IN (SELECT value FROM json_each(?)) AND {clause}",
                            (json.dumps(list(session_ids)), *params)).fetchall()
    return {row[0] for row in rows}


def _last_reply(candidates) -> Optional[str]:
    """The newest of a segment's replies that shows text, projected as a display read projects it: a pure
    handoff the compressor wrote as an assistant row shows nothing, so the reply before it stands; a carrier
    shows the reply it wraps."""
    for content, summary in candidates:
        shown = project_compaction_message_for_display(
            {"role": "assistant", "content": content, "_compressed_summary": summary})
        text = _shape_preview(flatten_message_text(shown["content"])) if shown is not None else ""
        if text:
            return text
    return None


def picker_session_facts(holder: StoreHolder, root_ids: Iterable[str], unlisted_ids: Iterable[str],
                         segment_ids: Iterable[str]) -> tuple[dict, dict, dict]:
    """``(roots, unlisted, last replies)`` in one store checkout. *roots*: ``{id: {id, cwd, git_repo_root}}``.
    *unlisted*: ``{id: {id, git_branch, pinned, _preview_raw, last_active}}``. *last replies*: ``{segment: the
    last reply text or None}`` from each segment's newest assistant rows with text (at most 8)."""
    unlisted_ids = list(unlisted_ids)
    with holder._read_ctx() as conn:
        roots = {row["id"]: dict(row) for row in conn.execute(
            "SELECT id, cwd, git_repo_root FROM sessions WHERE id IN (SELECT value FROM json_each(?))",
            (_ids(root_ids),)).fetchall()}
        unlisted = {row["id"]: dict(row) for row in conn.execute(
            f"SELECT s.id, s.git_branch, s.pinned, {_sql_preview_raw('s.id')}, {_sql_session_last_active('s')} "
            "AS last_active FROM sessions s WHERE s.id IN (SELECT value FROM json_each(?))",
            (_ids(unlisted_ids),)).fetchall()} if unlisted_ids else {}
        sql = _LAST_REPLIES_SQL.format(visible=DISPLAY_VISIBLE_SQL)
        replies = {segment: [(_decode_content(content), bool(summary)) for content, summary in conn.execute(sql, (segment,))]
                   for segment in set(segment_ids)}
    return roots, unlisted, {segment: _last_reply(candidates) for segment, candidates in replies.items()}


def session_current_segments(holder: StoreHolder, session_ids: Iterable[str]) -> dict[str, str]:
    """``{logical id: current physical segment}`` in one store checkout: a local receipt's entry (extracted in
    SQL), else the compression tip of a row compression ended. Ids with no session row are absent."""
    with holder._read_ctx() as conn:
        rows = conn.execute(
            "SELECT s.id, s.end_reason, (SELECT json_extract(m.value, '$.entry.session_id') FROM state_meta m "
            "WHERE m.key = ? || s.id AND json_valid(m.value)) AS entry FROM sessions s "
            "WHERE s.id IN (SELECT value FROM json_each(?))", (POLICY_PREFIX, _ids(session_ids))).fetchall()
        # Only a row compression ended has a tip past itself: walked on this one connection.
        return {row["id"]: row["entry"] or (_compression_tip_on(holder, conn, row["id"])
                                            if row["end_reason"] == "compression" else row["id"]) for row in rows}


# -- delegated children ------------------------------------------------------------------------------

def subagent_session_ids(holder: StoreHolder, subagent_id: str, marker_key: str) -> list[str]:
    """Ids of the ``source='subagent'`` sessions whose ``model_config`` carries *subagent_id* under
    *marker_key*, newest ``started_at`` first (ties: id descending). The caller proves ownership."""
    rows = holder._read_all(
        "SELECT id FROM sessions WHERE source = 'subagent' AND "
        f"{_sql_json_extract('model_config', '$.' + marker_key)} = ? ORDER BY started_at DESC, id DESC",
        (subagent_id,))
    return [row["id"] for row in rows]
