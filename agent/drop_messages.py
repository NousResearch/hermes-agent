"""drop_messages — remove trailing turns from a live session history.

Public API
----------
drop_messages(session, n=None) -> DropResult

The function mutates *session._session_messages* in place and synchronises
the SQLite session store (via *session._session_db*) so the removal
survives process restart.

Design goals
------------
- No original message content in DropResult or logs (privacy/safety).
- Atomic-ish: state.db is updated first; on failure the in-memory list is
  left unchanged so the caller is never left with memory/DB out of sync.
- Orphan tool-call correction: if dropping messages leaves an
  ``assistant(tool_calls=[...])`` with no matching ``tool`` results, that
  assistant turn is included in the drop automatically.
- Refusal counter reset: any session-state retry counters whose names are
  listed in ``_REFUSAL_COUNTER_ATTRS`` are zeroed when the drop succeeds.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Marker stamped on each message dict once it has been written to the
# SQLite session store (mirrors the constant in run_agent.py).  We
# duplicate the string here to avoid a circular import.
_DB_PERSISTED_MARKER = "_db_persisted"

# Session-state attributes that track consecutive refusal / retry counts.
# These are reset to 0 when drop_messages succeeds so a poisoned-session
# recovery does not keep the counter at the poisoned value.  The list is
# intentionally conservative: only well-known counters whose continued
# elevation would cause incorrect retry behaviour after the drop.
_REFUSAL_COUNTER_ATTRS = (
    "_empty_content_retries",
    "_thinking_prefill_retries",
    "_dropped_toolcall_retries",
    "_content_filter_refusal_count",
)


@dataclass(frozen=True)
class DroppedMessageInfo:
    """Metadata about one removed message — no original content."""

    role: str
    length: int  # byte-length of the JSON-serialised content field
    sha256: str  # hex digest of that serialisation


@dataclass
class DropResult:
    """Outcome of a :func:`drop_messages` call."""

    dropped: int
    """Number of messages actually removed from the in-memory history."""

    messages_info: List[DroppedMessageInfo] = field(default_factory=list)
    """Metadata (role, length, sha256) for each removed message."""

    db_deactivated: int = 0
    """Number of rows soft-deleted in state.db (0 when no DB is attached)."""

    error: Optional[str] = None
    """Human-readable reason when the call was a no-op or partially failed."""

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _content_bytes(msg: Dict[str, Any]) -> bytes:
    """Return a stable byte representation of a message's content field."""
    import json

    content = msg.get("content")
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True).encode(
            "utf-8", errors="surrogatepass"
        )
    except (TypeError, ValueError):
        return repr(content).encode("utf-8", errors="surrogatepass")


def _message_info(msg: Dict[str, Any]) -> DroppedMessageInfo:
    """Build a :class:`DroppedMessageInfo` without retaining original text."""
    raw = _content_bytes(msg)
    return DroppedMessageInfo(
        role=str(msg.get("role") or ""),
        length=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _find_last_tool_result_start(messages: List[Dict[str, Any]]) -> Optional[int]:
    """Return the index of the last ``role='tool'`` message, or None."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "tool":
            return i
    return None


def _get_tool_call_ids(msg: Dict[str, Any]) -> set:
    """Extract the set of tool-call IDs from an assistant message dict."""
    tool_calls = msg.get("tool_calls") or []
    ids: set = set()
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            cid = tc.get("id") or tc.get("tool_call_id")
            if cid:
                ids.add(cid)
    return ids


def _correct_orphan_tool_calls(
    messages: List[Dict[str, Any]], drop_start: int
) -> int:
    """Extend *drop_start* backward to avoid orphaned assistant tool calls.

    After dropping ``messages[drop_start:]``, any ``assistant(tool_calls)``
    whose results are entirely in the dropped tail would become orphaned —
    the provider would receive a tool-call with no matching result, which is
    invalid.  Walk backward from *drop_start* to find the nearest such
    assistant turn and include it (and everything after it up to the
    original *drop_start*) in the drop.

    Returns the adjusted *drop_start* (unchanged when no orphan would arise).
    """
    tail = messages[drop_start:]

    # Collect tool_call_ids whose results live entirely in the dropped tail.
    dropped_result_ids: set = set()
    for m in tail:
        if m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid:
                dropped_result_ids.add(cid)

    if not dropped_result_ids:
        return drop_start

    # Walk backward through the surviving prefix to find the assistant that
    # owns the dropped results.
    for i in range(drop_start - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "assistant":
            continue
        call_ids = _get_tool_call_ids(msg)
        if not call_ids:
            continue
        # If this assistant issued any call whose result is in the dropped
        # tail, include the whole assistant turn in the drop.
        if call_ids & dropped_result_ids:
            return i

    return drop_start


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def drop_messages(
    session: Any,
    n: Optional[int] = None,
) -> DropResult:
    """Remove trailing messages from *session* and synchronise state.db.

    Parameters
    ----------
    session:
        Any object that exposes:
          - ``._session_messages``  — the mutable in-memory history list
          - ``.session_id``         — str session identifier (may be None/empty)
          - ``._session_db``        — Optional[SessionDB]
    n:
        Number of messages to remove from the tail.  When *None* (default),
        removes from the last ``role='tool'`` message to the end (inclusive).
        When there is no tool result in the history, returns a no-op
        DropResult with an explanatory ``error``.

    Returns
    -------
    DropResult
        Always returns a DropResult; never raises.  Check ``.ok`` /
        ``.error`` for outcome.  Original message content is never stored
        in the result or in log output.
    """
    messages = getattr(session, "_session_messages", None)
    if not isinstance(messages, list):
        return DropResult(
            dropped=0, error="session has no _session_messages attribute"
        )

    if not messages:
        return DropResult(dropped=0, error="history is empty — nothing to drop")

    # ── Determine initial drop_start ─────────────────────────────────────
    if n is None:
        idx = _find_last_tool_result_start(messages)
        if idx is None:
            return DropResult(
                dropped=0,
                error="no tool result in history — nothing to drop",
            )
        drop_start = idx
    else:
        if n <= 0:
            return DropResult(dropped=0, error=f"n must be positive, got {n!r}")
        drop_start = max(0, len(messages) - n)

    # ── Orphan tool-call correction ────────────────────────────────────────
    drop_start = _correct_orphan_tool_calls(messages, drop_start)

    tail = messages[drop_start:]
    if not tail:
        return DropResult(dropped=0, error="drop range is empty after boundary check")

    # ── Collect metadata before removal (no content retained) ─────────────
    info_list = [_message_info(m) for m in tail]
    n_persisted = sum(1 for m in tail if m.get(_DB_PERSISTED_MARKER))

    # ── Synchronise state.db first (rollback on failure keeps memory clean) ─
    db_deactivated = 0
    session_id: str = getattr(session, "session_id", None) or ""
    session_db = getattr(session, "_session_db", None)

    if n_persisted > 0 and session_db is not None and session_id:
        try:
            db_deactivated = session_db.deactivate_last_n_active_messages(
                session_id, n_persisted
            )
        except Exception as exc:
            logger.warning(
                "drop_messages: state.db deactivation failed for session %s — "
                "in-memory history is NOT modified (err=%s)",
                session_id,
                exc,
            )
            return DropResult(
                dropped=0,
                error=f"state.db write failed: {exc}",
            )

    # ── Mutate in-memory history ──────────────────────────────────────────
    del messages[drop_start:]

    # ── Reset refusal / retry counters ─────────────────────────────────────
    for attr in _REFUSAL_COUNTER_ATTRS:
        if hasattr(session, attr):
            try:
                setattr(session, attr, 0)
            except Exception:
                pass  # read-only or property — best effort

    logger.info(
        "drop_messages: session=%s dropped=%d db_deactivated=%d roles=%s",
        session_id or "(no-id)",
        len(info_list),
        db_deactivated,
        [i.role for i in info_list],
    )

    return DropResult(
        dropped=len(info_list),
        messages_info=info_list,
        db_deactivated=db_deactivated,
    )


__all__ = ["drop_messages", "DropResult", "DroppedMessageInfo"]
