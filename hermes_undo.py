"""Shared half-turn undo/redo core for Hermes sessions.

Concurrency precondition: callers must serialize undo()/redo()/
on_user_message_appended() per session_id. The in-memory ``_states`` dict is
unlocked and undo() does a non-atomic read (get_messages) then write
(rewind_to_message), so two concurrent same-session undos could push spurious
stack entries. This is safe today because every surface serializes per session:
the CLI is a single-thread REPL, the gateway rejects undo/redo while an agent is
running and runs the core synchronously on one event loop, and the TUI rejects
undo while a turn is in flight. Any future surface that drives the core off the
event-loop thread (or adds an ``await`` inside it) must add a per-session lock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hermes_state import SessionDB


@dataclass(frozen=True)
class UndoOp:
    n: int
    rewound_ids: List[int]


@dataclass
class UndoRedoState:
    undo_stack: List[UndoOp] = field(default_factory=list)
    redo_stack: List[UndoOp] = field(default_factory=list)


_states: Dict[str, UndoRedoState] = {}
_session_db: Optional[SessionDB] = None


def _get_db() -> SessionDB:
    global _session_db
    if _session_db is None:
        _session_db = SessionDB()
    return _session_db


def get_state(session_id: str) -> UndoRedoState:
    return _states.setdefault(session_id, UndoRedoState())


def clear_state(session_id: Optional[str] = None) -> None:
    """Test/helper reset for the in-memory undo/redo holder."""
    if session_id is None:
        _states.clear()
    else:
        _states.pop(session_id, None)


def _party(role: Any) -> str:
    if role == "user":
        return "user"
    if role in {"assistant", "tool"}:
        return "assistant"
    return "other"


def compute_half_turn_target(
    active_messages: List[Dict[str, Any]], n: int
) -> Optional[int]:
    """Return the inclusive id target for undoing ``n`` half-turns.

    Consecutive rows with the same party form one half-turn. ``other`` rows
    participate in boundary detection but are never counted or returned.
    """
    if n < 1:
        n = 1

    group_starts: List[int] = []
    current_party: Optional[str] = None
    current_start: Optional[int] = None

    for msg in active_messages:
        msg_party = _party(msg.get("role"))
        if current_party is None or msg_party != current_party:
            if current_party != "other" and current_start is not None:
                group_starts.append(current_start)
            current_party = msg_party
            current_start = msg.get("id")

    if current_party != "other" and current_start is not None:
        group_starts.append(current_start)

    if not group_starts:
        return None

    target_index = max(0, len(group_starts) - n)
    return group_starts[target_index]


def _new_tail(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not messages:
        return None
    return max(messages, key=lambda m: m["id"])


def _tail_before_target(
    active_messages: List[Dict[str, Any]], target_id: int
) -> Optional[Dict[str, Any]]:
    before = [m for m in active_messages if m["id"] < target_id]
    return _new_tail(before)


def _prefill_from_tail(tail: Optional[Dict[str, Any]]) -> Optional[str]:
    if tail and tail.get("role") == "user" and isinstance(tail.get("content"), str):
        return tail["content"]
    return None


def undo(session_id: str, n: int) -> Dict[str, Any]:
    db = _get_db()
    state = get_state(session_id)
    msgs = db.get_messages(session_id, include_inactive=False)
    target_id = compute_half_turn_target(msgs, n)
    if target_id is None:
        return {
            "rewound_ids": [],
            "prefill_text": None,
            "message": "nothing to undo",
        }

    tail = _tail_before_target(msgs, target_id)
    if (
        tail is not None
        and tail.get("role") == "user"
        and not isinstance(tail.get("content"), str)
    ):
        target_id = tail["id"]

    result = db.rewind_to_message(
        session_id, target_id, require_user_role=False
    )
    rewound_ids = list(result.get("rewound_ids", []))
    if not rewound_ids:
        # Nothing was actually deactivated (degenerate/raced rewind). Don't push
        # an empty op or clobber a pending redo — match the None-path contract of
        # touching neither stack.
        return {
            "rewound_ids": [],
            "prefill_text": None,
            "message": "nothing to undo",
        }
    state.undo_stack.append(UndoOp(n=n, rewound_ids=rewound_ids))
    state.redo_stack.clear()

    active_after = db.get_messages(session_id, include_inactive=False)
    return {
        "rewound_ids": rewound_ids,
        "prefill_text": _prefill_from_tail(_new_tail(active_after)),
    }


def redo(session_id: str, m: int) -> Dict[str, Any]:
    db = _get_db()
    state = get_state(session_id)
    if m <= 0:
        return {
            "reactivated_count": 0,
            "new_tail_id": None,
            "prefill_text": None,
            "message": "nothing to redo",
        }

    k = min(m, len(state.undo_stack))
    if k == 0:
        session = db.get_session(session_id) or {}
        message = "nothing to redo"
        if (session.get("rewind_count") or 0) > 0:
            message = "nothing to redo (redo history doesn't survive a restart)"
        return {
            "reactivated_count": 0,
            "new_tail_id": None,
            "prefill_text": None,
            "message": message,
        }

    reactivated_total = 0
    for _ in range(k):
        op = state.undo_stack.pop()
        reactivated = db.restore_ids(session_id, op.rewound_ids)
        if reactivated == 0 and op.rewound_ids:
            # NONE of this op's rows could be restored — the transcript was
            # rewritten out from under the stack (/compress, /retry, and any
            # other replace_messages flow hard-delete + renumber rows). The redo
            # branch is meaningless now, so discard the whole stack rather than
            # raising: redo across a transcript rewrite is impossible, same as
            # redo after a restart.
            state.undo_stack.clear()
            state.redo_stack.clear()
            return {
                "reactivated_count": 0,
                "new_tail_id": None,
                "prefill_text": None,
                "message": "nothing to redo (transcript changed since undo)",
            }
        if reactivated != len(op.rewound_ids):
            # PARTIAL restore — some rows came back, some didn't. This is a
            # genuine desync (not a clean transcript rewrite), so fail loud to
            # surface latent corruption rather than silently half-redoing.
            raise RuntimeError(
                "redo invariant violated: restored "
                f"{reactivated} of {len(op.rewound_ids)} rewound rows"
            )
        reactivated_total += reactivated
        state.redo_stack.append(op)

    # Counter asymmetry is intentional: rewind_count increments per low-level
    # rewind_to_message call; redo_count increments once per /redo command.
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET redo_count = COALESCE(redo_count, 0) + 1 "
            "WHERE id = ?",
            (session_id,),
        )
    )

    active_after = db.get_messages(session_id, include_inactive=False)
    tail = _new_tail(active_after)
    return {
        "reactivated_count": reactivated_total,
        "new_tail_id": tail["id"] if tail else None,
        "prefill_text": None,
    }


def on_user_message_appended(session_id: str) -> None:
    """Invalidate the redo branch when a new user message is appended.

    Mirrors a text editor: once you type after undoing, the previously-undone
    content can no longer be redone. ``redo()`` consumes ``undo_stack`` (the
    pending-redoable ops), so that is the stack that must be cleared here;
    ``redo_stack`` (the already-redone history) is cleared too so a fresh
    message starts from a clean slate.
    """
    state = get_state(session_id)
    state.undo_stack.clear()
    state.redo_stack.clear()
