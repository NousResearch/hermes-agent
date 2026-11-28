"""One-time repair of chats a stored safety refusal still locks.

A provider safety filter judges the whole thread, so a refused turn left in the model context refuses every later message. ``retain_refused_turn_for_display`` takes a new refusal out of the model context as it happens; a refusal stored by an older build is still in the transcript, and that chat stays locked. Those builds closed the refused turn with an assistant ``failed_turn`` boundary carrying ``display_metadata.error_surface.code == "content_policy_blocked"``, a shape current builds never write: their closers skip a refused turn and retain it instead. That boundary is the marker this repair keys on.

The repair runs as a data migration when a writable store opens, before any surface loads the transcript into a turn, and applies the same retention a new refusal gets: each refused turn leaves the model context (``active = 0``, ``rewind_count`` bumped as ``/undo`` does) and its user row stays on screen as the greyed ``content_refused`` notice when the user typed it (an internal row, such as a background-task notice, leaves the context without being shown). It only touches an interactive chat whose newest settled turns are all refused (the refusals form the transcript's tail). It leaves alone:

- a session nobody continues as a chat: a delegated subagent run, cron, batch, kanban and tool sessions (their one "turn" is the whole run, so retaining it would hollow out the record), and a chat never answered before a refused span that ran tools (an ACP task run refused at its end);
- a refusal followed by a real answer (the chat was not locked, and later turns were built on it);
- a chat whose refused text was already folded into a compaction summary (an archived refused boundary in the session, or one in a compression ancestor): the summary still carries it, so dropping rows would not unlock the chat, and rewriting a summary is not safe;
- a boundary without the error-card metadata (no ``error_surface``): nothing in the row says it was a refusal.

Candidates are found without the write lock: one index probe per chat session reads its newest settled assistant row (``idx_messages_session_id``), so a large store is never heap-scanned. The write lock is taken only per repaired chat, which re-reads the transcript inside that transaction. A session with a live turn lease or compression lock is deferred to the next open. The store records completion in ``state_meta`` so a finished store pays one read per open. Completion is one-shot: an older process still running after it is set can write the old shape again; current builds never do, so that window is upgrade-only.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import Optional

from persistence.conversation.messages import (
    CONTENT_REFUSED_DISPLAY_KIND, CONTENT_REFUSED_METADATA_KEY, CONTENT_REFUSED_NOTICE,
)
from persistence.sqlite._schema_ddl import _ephemeral_child_sql, _sql_json_extract

logger = logging.getLogger("hermes_state")

LEGACY_REFUSAL_REPAIR_DONE = "legacy_refusal_repair"
_SCAN_CURSOR = "legacy_refusal_repair_session_cursor"
_SCAN_COMPLETE = "legacy_refusal_repair_scanned"
_PENDING = "legacy_refusal_repair_pending"
_DISCOVER_BATCH = 250  # one cold batch stays near the budget
_SCAN_BUDGET_S = 1.0
_CODE = "content_policy_blocked"
# Sessions nobody continues as a chat. A delegate child spawned under a gateway turn inherits the
# gateway's source, so the ``_delegate_from`` marker and the ephemeral-child shape catch those too.
_NON_CHAT_SOURCES = ("subagent", "cron", "batch", "kanban", "curator", "tool")


def _discover_sql() -> str:
    sources = ", ".join(f"'{source}'" for source in _NON_CHAT_SOURCES)
    # The newest settled assistant row: a failure boundary or a real answer, the row _locked_tail
    # keys on. It is the boundary's display_metadata when that row is a failure boundary, else NULL.
    return f"""
SELECT s.id, (SELECT CASE WHEN m.display_kind = 'failed_turn' THEN COALESCE(m.display_metadata, '') END
              FROM messages m INDEXED BY idx_messages_session_id
              WHERE m.session_id = s.id AND m.active = 1 AND m.role = 'assistant'
                AND (m.display_kind = 'failed_turn'
                     OR (COALESCE(m.display_kind, '') = '' AND COALESCE(m.tool_calls, '') IN ('', '[]')))
              ORDER BY m.id DESC LIMIT 1) AS boundary
FROM sessions s
WHERE s.id > ? AND s.source NOT IN ({sources})
  AND {_sql_json_extract('s.model_config', '$._delegate_from')} IS NULL AND NOT {_ephemeral_child_sql('s')}
ORDER BY s.id LIMIT ?"""


_DISCOVER_SQL = _discover_sql()


def _meta(conn, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row[0]


def _set_meta(conn, key: str, value: str) -> None:
    conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def _is_refusal_boundary(row) -> bool:
    from persistence.sqlite.conversation._store.messages import _decode_display_metadata
    if row["role"] != "assistant" or row["display_kind"] != "failed_turn":
        return False
    surface = (_decode_display_metadata(row["display_metadata"]) or {}).get("error_surface")
    return isinstance(surface, dict) and surface.get("code") == _CODE


def _is_final_reply(row) -> bool:
    return row["role"] == "assistant" and not row["display_kind"] and row["tool_calls"] in (None, "", "[]")


def _continued_parent_sql() -> str:
    from persistence.sqlite.conversation._store.sql_common import _non_continuation_child_sql
    return ("SELECT p.id FROM sessions c JOIN sessions p ON p.id = c.parent_session_id "
            "WHERE c.id = ? AND p.end_reason = 'compression'\n" + _non_continuation_child_sql("c.", "p.id"))


def _compaction_holds_refusal(conn, session_id: str) -> bool:
    """True when a compaction summary may restate the refusal: an inactive refused boundary in this
    session (in-place compaction archives it, or carries it forward and retires the original) or any
    refused boundary in a compression ancestor (rotation). An ``/undo`` also leaves an inactive one;
    skipping that chat too is the safe side."""
    rows = conn.execute(
        "SELECT role, display_kind, display_metadata FROM messages WHERE session_id = ? AND active = 0 "
        "AND display_kind = 'failed_turn'", (session_id,)).fetchall()
    if any(_is_refusal_boundary(r) for r in rows):
        return True
    continued_parent_sql = _continued_parent_sql()
    current, seen = session_id, {session_id}
    while True:
        # Only a compression continuation inherits its parent's summary; a fork (branch, delegate,
        # reset) of a parent that later rotated does not.
        row = conn.execute(continued_parent_sql, (current,)).fetchone()
        parent = row[0] if row else None
        if not parent or parent in seen:
            return False
        rows = conn.execute(
            "SELECT role, display_kind, display_metadata FROM messages WHERE session_id = ? "
            "AND display_kind = 'failed_turn'", (parent,)).fetchall()
        if any(_is_refusal_boundary(r) for r in rows):
            return True
        seen.add(parent)
        current = parent


def _locked_tail(rows) -> Optional[list[tuple[Optional[int], list[int]]]]:
    """``[(shown_user_row_id, turn_row_ids), ...]`` for the refused turns that lock the chat, or None.

    The tail after the last real answer splits into turns at each ``failed_turn`` boundary. The chat is
    locked only when every turn from the first refused one through the newest boundary is refused and
    each starts at a user row that is not a compaction carrier. A turn head the user typed (the
    user-originated test the retention of a new refusal uses) is the row shown as refused; an internal
    head (a background-task or process notice, a hidden row) gets ``None``: it leaves the context unshown."""
    from session_rules.turns import is_compaction_summary_message, user_originated_turn_view
    from persistence.sqlite.conversation._store.messages import _decode_content, _decode_display_metadata
    # The newest settled turn (a real answer or a failure boundary) must be a refusal.
    last_boundary = max((i for i, r in enumerate(rows) if r["display_kind"] == "failed_turn" or _is_final_reply(r)),
                        default=-1)
    if last_boundary < 0 or not _is_refusal_boundary(rows[last_boundary]):
        return None
    start = 1 + max((i for i, r in enumerate(rows[:last_boundary]) if _is_final_reply(r)), default=-1)
    if not start and any(r["role"] == "tool" for r in rows[:last_boundary]):
        # Never answered and the refused span ran tools: a task run, not a chat. Dropping it empties
        # the record. A chat refused at its first plain message is still repaired (Bot Chat has no /new).
        return None
    turns, current = [], []
    for row in rows[start:last_boundary + 1]:
        current.append(row)
        if row["display_kind"] == "failed_turn":
            turns.append(current)
            current = []
    first = next((i for i, turn in enumerate(turns) if _is_refusal_boundary(turn[-1])), None)
    locked = turns[first:]
    if any(not _is_refusal_boundary(turn[-1]) for turn in locked):
        return None  # another failure sits between the refusals: not the shape this repair proves
    plan = []
    for turn in locked:
        head = turn[0]
        if head["role"] != "user" or head["_compressed_summary"] or is_compaction_summary_message(
                {"role": "user", "content": _decode_content(head["content"])}):
            return None
        view = {"role": "user", "content": _decode_content(head["content"]), "display_kind": head["display_kind"],
                "display_metadata": _decode_display_metadata(head["display_metadata"])}
        shown = int(head["id"]) if user_originated_turn_view(view) is not None else None
        plan.append((shown, [int(r["id"]) for r in turn]))
    return plan


def _repair_session(holder, conn, session_id: str, notice: str) -> str:
    """``"repaired"`` / ``"skipped"`` / ``"deferred"`` (live lease or compression lock) for one session,
    inside the caller's write transaction."""
    from persistence.sqlite._state.errors import SessionCompressionInProgressError, SessionTurnLeaseLostError
    from persistence.sqlite.conversation._store.messages import (
        _active_transcript_counts, _check_transcript_write_guards, _decode_display_metadata, _encode_display_metadata,
    )
    row = conn.execute("SELECT end_reason FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None or row[0] == "compression" or _compaction_holds_refusal(conn, session_id):
        return "skipped"
    rows = conn.execute(
        "SELECT id, role, content, tool_calls, display_kind, display_metadata, _compressed_summary FROM messages "
        "WHERE session_id = ? AND active = 1 AND role NOT IN ('session_meta', 'system') ORDER BY id",
        (session_id,)).fetchall()
    plan = _locked_tail(rows)
    if not plan:
        return "skipped"
    try:
        _check_transcript_write_guards(
            holder, conn, session_id, None, reject_active_turn_lease=True, reject_active_compression_lock=True)
    except (SessionTurnLeaseLostError, SessionCompressionInProgressError):
        return "deferred"
    for user_row_id, turn_ids in plan:
        conn.execute(f"UPDATE messages SET active = 0 WHERE id IN ({','.join('?' for _ in turn_ids)})", turn_ids)
        if user_row_id is None:
            continue
        meta = _decode_display_metadata(conn.execute(
            "SELECT display_metadata FROM messages WHERE id = ?", (user_row_id,)).fetchone()[0]) or {}
        meta[CONTENT_REFUSED_METADATA_KEY] = {"notice": notice}
        conn.execute("UPDATE messages SET compacted = 1, display_kind = ?, display_metadata = ? WHERE id = ?",
                     (CONTENT_REFUSED_DISPLAY_KIND, _encode_display_metadata(meta), user_row_id))
    message_count, tool_call_count = _active_transcript_counts(conn, session_id)
    # rewind_count as the retention of a new refusal (rewind_in_transaction) bumps it: a live view's /undo
    # checks its warm history against the durable rows and refuses instead of acting on the stale ones.
    conn.execute("UPDATE sessions SET message_count = ?, tool_call_count = ?, "
                 "rewind_count = COALESCE(rewind_count, 0) + 1 WHERE id = ?",
                 (message_count, tool_call_count, session_id))
    return "repaired"


def _discover(conn, cursor: str, deadline: float) -> tuple[list[str], str, bool]:
    """Candidate chat ids after ``cursor`` (by ``sessions.id``), read outside any write transaction;
    returns ``(found, new_cursor, reached_the_end)``. Stops at ``deadline`` between batches."""
    found: list[str] = []
    while True:
        rows = conn.execute(_DISCOVER_SQL, (cursor, _DISCOVER_BATCH)).fetchall()
        found += [r[0] for r in rows if r[1] is not None and _CODE in r[1]]
        if len(rows) < _DISCOVER_BATCH:
            return found, cursor if not rows else rows[-1][0], True
        cursor = rows[-1][0]
        if time.monotonic() >= deadline:
            return found, cursor, False


@contextlib.contextmanager
def _write_txn(conn):
    """BEGIN IMMEDIATE ... COMMIT. A COMMIT that fails (a reader holding the file in DELETE journal mode)
    leaves the transaction open, so it is rolled back too: the open's writer connection must come back
    to its caller with no transaction, or every later write fails at BEGIN."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _pending(conn) -> list[str]:
    return json.loads(_meta(conn, _PENDING) or "[]")


def repair_legacy_refused_turns(holder, conn) -> dict[str, int]:
    """Advance the one-time repair on the writer connection ``conn`` (autocommit, no open transaction).

    Discovery reads the next batches of chat sessions with no write lock; each candidate is then
    repaired in its own short transaction (``_repair_session`` re-reads and re-checks it there, so a
    chat another process already repaired is skipped). The pass shares one ``_SCAN_BUDGET_S`` budget,
    with at least one repair per open; unreached and deferred chats stay queued (``_PENDING``). One
    last short transaction records the cursor and queue, or the done marker. Returns outcome counts."""
    counts = {"repaired": 0, "skipped": 0, "deferred": 0}
    if _meta(conn, LEGACY_REFUSAL_REPAIR_DONE) is not None:
        return counts  # a finished store pays this one read per open
    started = time.monotonic()
    cursor = _meta(conn, _SCAN_CURSOR) or ""
    complete = _meta(conn, _SCAN_COMPLETE) is not None
    found: list[str] = []
    if not complete:
        found, cursor, complete = _discover(conn, cursor, started + _SCAN_BUDGET_S)
    queue = list(dict.fromkeys([*_pending(conn), *found]))
    handled: set[str] = set()
    keep: list[str] = []
    for index, session_id in enumerate(queue):
        if index and time.monotonic() >= started + _SCAN_BUDGET_S:
            keep += queue[index:]
            break
        with _write_txn(conn):
            outcome = _repair_session(holder, conn, session_id, CONTENT_REFUSED_NOTICE)
        counts[outcome] += 1
        handled.add(session_id)
        if outcome == "deferred":
            keep.append(session_id)
    with _write_txn(conn):
        if _meta(conn, LEGACY_REFUSAL_REPAIR_DONE) is None:
            # Merge with what another process queued meanwhile; a deferred chat goes to the back.
            stored = [sid for sid in _pending(conn) if sid not in handled and sid not in keep]
            remaining = list(dict.fromkeys([*stored, *keep]))
            complete = complete or _meta(conn, _SCAN_COMPLETE) is not None
            if complete and not remaining:
                _set_meta(conn, LEGACY_REFUSAL_REPAIR_DONE, "1")
                conn.execute("DELETE FROM state_meta WHERE key IN (?, ?, ?)", (_SCAN_CURSOR, _SCAN_COMPLETE, _PENDING))
            else:
                # Another process may have read further: whoever read a range queued its finds.
                _set_meta(conn, _SCAN_CURSOR, max(cursor, _meta(conn, _SCAN_CURSOR) or ""))
                if complete:
                    _set_meta(conn, _SCAN_COMPLETE, "1")
                _set_meta(conn, _PENDING, json.dumps(remaining))
    if counts["repaired"] or counts["deferred"]:
        logger.info("Legacy safety-refusal repair: %(repaired)d chat(s) unlocked, %(skipped)d left as-is, "
                    "%(deferred)d deferred (busy)", counts)
    return counts


def advance_legacy_refusal_repair(holder, conn) -> None:
    """Open-time entry point: a repair that cannot run now (state.db busy, an unexpected row) must
    never fail the open; it is retried on the next one."""
    try:
        repair_legacy_refused_turns(holder, conn)
    except Exception:  # health: allow BLE001 -- a one-time repair must never fail opening state.db
        logger.warning("Legacy safety-refusal repair skipped this open; it retries on the next one",
                       exc_info=True)


__all__ = ["LEGACY_REFUSAL_REPAIR_DONE", "advance_legacy_refusal_repair", "repair_legacy_refused_turns"]
