"""One-time repair of chats locked by a safety refusal stored before #149708.

A provider safety filter judges the whole thread, so a refused turn left in the model context refuses
every later message. #149708 takes a NEW refusal out of the model context as it happens
(``SessionRewindMixin.retain_refused_turn_for_display``); a refusal stored by an older build is still
in the transcript, and that chat stays locked. Those builds closed the refused turn with an
assistant ``failed_turn`` boundary carrying ``display_metadata.error_surface.code ==
"content_policy_blocked"`` (``agent/conversation_loop.py::_close_durable_failed_turn``), a shape
current builds never write: their closers skip a refused turn and retain it instead. That boundary
is the marker this repair keys on.

The repair runs as a data migration when a writable ``SessionDB`` opens, before any surface builds a
turn from the transcript, and applies the same retention #149708 applies to a new refusal: each
refused turn leaves the model context (``active = 0``) and its user row stays on screen as the
greyed ``content_refused`` notice. It only touches a locked chat, meaning one whose newest settled
turns are all refused (the refusals form the transcript's tail). It leaves alone:

- a refusal followed by a real answer (the chat was not locked, and later turns were built on it);
- a chat whose refused text was already folded into a compaction summary (an archived refused
  boundary in the session, or one in a compression ancestor): the summary still carries it, so
  dropping rows would not unlock the chat, and rewriting a summary is not safe;
- a boundary from a build older than the error-card metadata (no ``error_surface``): nothing in the
  row says it was a refusal.

A session with a live turn lease or compression lock is deferred to the next open. The store records
completion in ``state_meta`` so a repaired store is never scanned again.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger("hermes_state")

LEGACY_REFUSAL_REPAIR_DONE = "legacy_refusal_repair"
_SCAN_CURSOR = "legacy_refusal_repair_rowid"
_PENDING = "legacy_refusal_repair_pending"
_SCAN_CHUNK = 20000
_SCAN_BUDGET_S = 1.0
_CODE = "content_policy_blocked"

_CANDIDATES_SQL = (
    "SELECT DISTINCT session_id FROM messages WHERE rowid > ? AND rowid <= ? AND role = 'assistant' "
    "AND display_kind = 'failed_turn' AND display_metadata LIKE '%content_policy_blocked%'")


def _meta(conn, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row[0]


def _set_meta(conn, key: str, value: str) -> None:
    conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def _is_refusal_boundary(db, row) -> bool:
    if row["role"] != "assistant" or row["display_kind"] != "failed_turn":
        return False
    surface = (db._decode_display_metadata(row["display_metadata"]) or {}).get("error_surface")
    return isinstance(surface, dict) and surface.get("code") == _CODE


def _is_final_reply(row) -> bool:
    return row["role"] == "assistant" and not row["display_kind"] and row["tool_calls"] in (None, "", "[]")


def _compaction_holds_refusal(db, conn, session_id: str) -> bool:
    """True when a compaction summary may restate the refusal: an archived refused boundary in this
    session (in-place compaction) or any refused boundary in a compression ancestor (rotation)."""
    rows = conn.execute(
        "SELECT role, display_kind, display_metadata FROM messages WHERE session_id = ? AND active = 0 "
        "AND compacted = 1 AND display_kind = 'failed_turn'", (session_id,)).fetchall()
    if any(_is_refusal_boundary(db, r) for r in rows):
        return True
    current, seen = session_id, {session_id}
    while True:
        row = conn.execute("SELECT parent_session_id FROM sessions WHERE id = ?", (current,)).fetchone()
        parent = row[0] if row else None
        if not parent or parent in seen:
            return False
        ended = conn.execute("SELECT end_reason FROM sessions WHERE id = ?", (parent,)).fetchone()
        if not ended or ended[0] != "compression":
            return False
        rows = conn.execute(
            "SELECT role, display_kind, display_metadata FROM messages WHERE session_id = ? "
            "AND display_kind = 'failed_turn'", (parent,)).fetchall()
        if any(_is_refusal_boundary(db, r) for r in rows):
            return True
        seen.add(parent)
        current = parent


def _locked_tail(db, rows) -> Optional[list[tuple[int, list[int]]]]:
    """``[(user_row_id, turn_row_ids), ...]`` for the refused turns that lock the chat, or None.

    The tail after the last real answer splits into turns at each ``failed_turn`` boundary. The chat is
    locked only when every turn from the first refused one through the newest boundary is refused and
    each starts at an ordinary user row (not a compaction carrier)."""
    from agent.context_compressor import is_compaction_summary_message
    # The newest settled turn (a real answer or a failure boundary) must be a refusal.
    last_boundary = max((i for i, r in enumerate(rows) if r["display_kind"] == "failed_turn" or _is_final_reply(r)),
                        default=-1)
    if last_boundary < 0 or not _is_refusal_boundary(db, rows[last_boundary]):
        return None
    start = 1 + max((i for i, r in enumerate(rows[:last_boundary]) if _is_final_reply(r)), default=-1)
    turns, current = [], []
    for row in rows[start:last_boundary + 1]:
        current.append(row)
        if row["display_kind"] == "failed_turn":
            turns.append(current)
            current = []
    first = next((i for i, turn in enumerate(turns) if _is_refusal_boundary(db, turn[-1])), None)
    locked = turns[first:]
    if any(not _is_refusal_boundary(db, turn[-1]) for turn in locked):
        return None  # another failure sits between the refusals: not the shape this repair proves
    plan = []
    for turn in locked:
        head = turn[0]
        if head["role"] != "user" or head["_compressed_summary"] or is_compaction_summary_message(
                {"role": "user", "content": db._decode_content(head["content"])}):
            return None
        plan.append((int(head["id"]), [int(r["id"]) for r in turn]))
    return plan


def _repair_session(db, conn, session_id: str, notice: str) -> str:
    """``"repaired"`` / ``"skipped"`` / ``"deferred"`` (live lease or compression lock) for one session,
    inside the caller's write transaction."""
    from agent.turn_failure_copy import CONTENT_REFUSED_DISPLAY_KIND, CONTENT_REFUSED_METADATA_KEY
    from hermes_state_errors import SessionCompressionInProgressError, SessionTurnLeaseLostError
    row = conn.execute("SELECT end_reason FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None or row[0] == "compression" or _compaction_holds_refusal(db, conn, session_id):
        return "skipped"
    rows = conn.execute(
        "SELECT id, role, content, tool_calls, display_kind, display_metadata, _compressed_summary FROM messages "
        "WHERE session_id = ? AND active = 1 AND role NOT IN ('session_meta', 'system') ORDER BY id",
        (session_id,)).fetchall()
    plan = _locked_tail(db, rows)
    if not plan:
        return "skipped"
    try:
        db._check_transcript_write_guards(
            conn, session_id, None, reject_active_turn_lease=True, reject_active_compression_lock=True)
    except (SessionTurnLeaseLostError, SessionCompressionInProgressError):
        return "deferred"
    for user_row_id, turn_ids in plan:
        conn.execute(f"UPDATE messages SET active = 0 WHERE id IN ({','.join('?' for _ in turn_ids)})", turn_ids)
        meta = db._decode_display_metadata(conn.execute(
            "SELECT display_metadata FROM messages WHERE id = ?", (user_row_id,)).fetchone()[0]) or {}
        meta[CONTENT_REFUSED_METADATA_KEY] = {"notice": notice}
        conn.execute("UPDATE messages SET compacted = 1, display_kind = ?, display_metadata = ? WHERE id = ?",
                     (CONTENT_REFUSED_DISPLAY_KIND, db._encode_display_metadata(meta), user_row_id))
    message_count, tool_call_count = db._active_transcript_counts(conn, session_id)
    conn.execute("UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                 (message_count, tool_call_count, session_id))
    return "repaired"


def _scan(conn, deadline: float) -> tuple[list[str], bool]:
    """Candidate session ids from the next rowid slices, and whether the scan reached the end."""
    done_through = int(_meta(conn, _SCAN_CURSOR) or 0)
    high = conn.execute("SELECT MAX(rowid) FROM messages").fetchone()[0] or 0
    found: list[str] = []
    while done_through < high:
        found += [r[0] for r in conn.execute(_CANDIDATES_SQL, (done_through, done_through + _SCAN_CHUNK))]
        done_through += _SCAN_CHUNK
        if done_through < high and time.monotonic() >= deadline:
            _set_meta(conn, _SCAN_CURSOR, str(done_through))
            return found, False
    _set_meta(conn, _SCAN_CURSOR, str(done_through))
    return found, True


def repair_legacy_refused_turns(db, conn) -> dict[str, int]:
    """Advance the one-time repair on the writer connection ``conn`` (autocommit, no open transaction).
    Bounded per open (``_SCAN_BUDGET_S`` of scanning); the rest continues on later opens. Returns the
    per-outcome session counts of this pass."""
    from agent.turn_failure_copy import CONTENT_REFUSED_NOTICE
    counts = {"repaired": 0, "skipped": 0, "deferred": 0}
    if _meta(conn, LEGACY_REFUSAL_REPAIR_DONE) is not None:
        return counts  # read before BEGIN IMMEDIATE: a repaired store never takes the write lock here
    conn.execute("BEGIN IMMEDIATE")
    try:
        if _meta(conn, LEGACY_REFUSAL_REPAIR_DONE) is not None:
            conn.execute("COMMIT")
            return counts
        pending = json.loads(_meta(conn, _PENDING) or "[]")
        found, scan_complete = _scan(conn, time.monotonic() + _SCAN_BUDGET_S)
        deferred = []
        for session_id in dict.fromkeys([*pending, *found]):
            outcome = _repair_session(db, conn, session_id, CONTENT_REFUSED_NOTICE)
            counts[outcome] += 1
            if outcome == "deferred":
                deferred.append(session_id)
        _set_meta(conn, _PENDING, json.dumps(deferred))
        if scan_complete and not deferred:
            _set_meta(conn, LEGACY_REFUSAL_REPAIR_DONE, "1")
            conn.execute("DELETE FROM state_meta WHERE key IN (?, ?)", (_SCAN_CURSOR, _PENDING))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    if counts["repaired"] or counts["deferred"]:
        logger.info("Legacy safety-refusal repair: %(repaired)d chat(s) unlocked, %(skipped)d left as-is, "
                    "%(deferred)d deferred (busy)", counts)
    return counts


def advance_legacy_refusal_repair(db, conn) -> None:
    """Open-time entry point: a repair that cannot run now (state.db busy, an unexpected row) must
    never fail the open; it is retried on the next one."""
    try:
        repair_legacy_refused_turns(db, conn)
    except Exception:  # health: allow BLE001 -- a one-time repair must never fail opening state.db
        logger.warning("Legacy safety-refusal repair skipped this open; it retries on the next one",
                       exc_info=True)


__all__ = ["LEGACY_REFUSAL_REPAIR_DONE", "advance_legacy_refusal_repair", "repair_legacy_refused_turns"]
