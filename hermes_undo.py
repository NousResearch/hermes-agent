"""Shared redo stack for Hermes ``/undo`` → ``/redo``.

``/undo`` already exists on every surface (CLI, gateway, TUI). It soft-deletes
the rewound rows — they stay in ``state.db`` with ``active=0`` for audit — which
means the rewind is physically reversible. This module is the small amount of
bookkeeping needed to actually offer that reversal to the user: it remembers
*which rows* each undo deactivated, so ``/redo`` can flip exactly those rows
back.

Design notes:

* **No second rewind engine.** The undo path stays owned by the existing
  callers (``SessionDB.rewind_to_message`` and its CAS guards). This module
  never rewinds anything; it only records the ids that a rewind reported and
  replays them via ``SessionDB.restore_ids``. Recording is done by calling
  :func:`record_undo` after a rewind commits.
* **In-memory, by design.** The redo branch is a UI affordance, not durable
  state — the same contract a text editor offers. It does not survive a
  restart, and :func:`redo` detects that case (a session with a non-zero
  ``rewind_count`` but an empty stack) so the user is told *why* there is
  nothing to redo instead of a bare "nothing to redo".
* **New input invalidates the redo branch**, again mirroring an editor: once
  you type after undoing, the undone content is no longer reachable. See
  :func:`on_user_message_appended`.

Concurrency precondition: callers must serialize ``record_undo``/``redo``/
``on_user_message_appended`` per ``session_id``. The in-memory ``_states``
mapping is unlocked. This holds today because every surface serializes per
session — the CLI is a single-threaded REPL, the gateway refuses undo/redo
while an agent is running and drives this module synchronously on one event
loop, and the TUI refuses undo while a turn is in flight. Any future surface
that drives this module off the event-loop thread must add a per-session lock.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from hermes_state import SessionDB

logger = logging.getLogger(__name__)


def _is_transient_db_error(exc: BaseException) -> bool:
    """True for a retryable failure of a redo write (a SQLite lock/busy).

    A non-transient failure (schema error, real bug) returns False so a partial
    redo is reported as non-retryable rather than telling the user to run
    ``/redo`` again into an identical re-failure.
    """
    if isinstance(exc, sqlite3.OperationalError):
        msg = str(exc).lower()
        return "locked" in msg or "busy" in msg
    return False


@dataclass(frozen=True)
class UndoOp:
    """One recorded undo: the rows it deactivated, newest operation last."""

    n: int
    rewound_ids: List[int]
    #: Rows the rewind INSERTED (a carrier rewind's compaction-handoff replacement); retired on redo.
    inserted_ids: List[int] = field(default_factory=list)


@dataclass
class UndoRedoState:
    #: Undo operations available to replay. ``redo()`` pops from here.
    undo_stack: List[UndoOp] = field(default_factory=list)
    #: Operations already replayed, kept so a later undo/redo cycle is coherent.
    redo_stack: List[UndoOp] = field(default_factory=list)


# Bound the in-memory holder so a long-running gateway touching many distinct
# sessions does not grow _states without limit. The state is purely an
# in-memory redo branch that already does not survive a restart, so evicting
# the least-recently-used session is graceful: a later /redo on an evicted
# session behaves exactly like /redo after a restart, which redo() handles.
_STATE_CAP = 2048
_states: "OrderedDict[str, UndoRedoState]" = OrderedDict()
_session_db: Optional[SessionDB] = None


def _get_db() -> SessionDB:
    global _session_db
    if _session_db is None:
        _session_db = SessionDB()
    return _session_db


def get_state(session_id: str) -> UndoRedoState:
    """Return (creating if needed) the redo bookkeeping for *session_id*."""
    state = _states.get(session_id)
    if state is None:
        state = UndoRedoState()
        _states[session_id] = state
        while len(_states) > _STATE_CAP:
            _states.popitem(last=False)
    else:
        _states.move_to_end(session_id)
    return state


def clear_state(session_id: Optional[str] = None) -> None:
    """Drop the redo branch for *session_id* (or all sessions when None)."""
    if session_id is None:
        _states.clear()
    else:
        _states.pop(session_id, None)


def record_undo(
    session_id: str, n: int, rewound_ids: List[int], inserted_ids: Optional[List[int]] = None,
) -> None:
    """Bank a committed rewind so ``/redo`` can replay it.

    *rewound_ids* are the rows the rewind actually deactivated — the
    ``rewound_ids`` field of a ``SessionDB.rewind_to_message`` result. Call this
    only after the rewind has committed.

    A rewind that deactivated nothing is not recorded: banking an empty
    operation would let ``/redo`` report success for a no-op. Recording a real
    undo clears the redo history, because the branch that history described is
    no longer the branch the user is on.
    """
    ids = [int(i) for i in (rewound_ids or [])]
    if not ids:
        return
    state = get_state(session_id)
    inserted = [int(i) for i in (inserted_ids or []) if i is not None]
    state.undo_stack.append(UndoOp(n=n, rewound_ids=ids, inserted_ids=inserted))
    state.redo_stack.clear()


def has_redoable(session_id: str) -> bool:
    """True when at least one undo operation is available to replay."""
    return bool(_states.get(session_id, UndoRedoState()).undo_stack)


def redo(session_id: str, m: int = 1) -> Dict[str, Any]:
    """Replay up to *m* recorded undo operations, newest first.

    Returns a dict with ``reactivated_count`` (rows flipped back to active),
    ``new_tail_id``, and — when only part of the request could be replayed —
    ``partial``/``partial_retryable``/``message``. ``reactivated_count == 0``
    with a ``message`` is the healthy "nothing to redo" outcome, not an error.
    """
    db = _get_db()
    state = get_state(session_id)
    if m <= 0:
        return {
            "reactivated_count": 0,
            "new_tail_id": None,
            "message": "nothing to redo",
        }

    k = min(m, len(state.undo_stack))
    if k == 0:
        # Distinguish "you have not undone anything" from "your redo branch was
        # lost". A session that has been rewound but has an empty stack means
        # the process restarted (or the LRU evicted it) — say so, because
        # otherwise the user sees "nothing to redo" right after a visible undo.
        message = "nothing to redo"
        try:
            session = db.get_session(session_id) or {}
            if (session.get("rewind_count") or 0) > 0:
                message = "nothing to redo (redo history doesn't survive a restart)"
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.debug("redo: session lookup failed for %s: %r", session_id, exc)
        return {
            "reactivated_count": 0,
            "new_tail_id": None,
            "message": message,
        }

    reactivated_total = 0
    ops_redone = 0
    transcript_changed = False
    transient_partial = False
    partial_hard_error = False

    for _ in range(k):
        op = state.undo_stack.pop()
        try:
            reactivated = db.restore_ids(session_id, op.rewound_ids, deactivate_ids=op.inserted_ids)
        except Exception as exc:
            # redo() does one write per operation. If an earlier operation
            # already committed, its rows are live in the DB, so this exception
            # must not unwind into the caller as "nothing changed" — that would
            # desync the screen and let a retry double-redo the earlier work.
            # Push the failed operation back (it did not commit, so it stays
            # recoverable) and report an honest partial. Classify here as well
            # as at the caller: a transient lock is worth retrying, a real bug
            # is not (retrying re-fails identically).
            state.undo_stack.append(op)
            if reactivated_total > 0:
                if _is_transient_db_error(exc):
                    transient_partial = True
                else:
                    partial_hard_error = True
                    logger.error(
                        "redo: op failed with a non-transient error after %s row(s) "
                        "already reactivated for %s (partial, not retryable): %r",
                        reactivated_total, session_id, exc, exc_info=True,
                    )
                break
            raise

        if reactivated == 0 and op.rewound_ids:
            # None of this operation's rows could be restored: the transcript
            # was rewritten out from under the stack (/compress and /retry
            # hard-delete and renumber rows). The redo branch is dead from here
            # on, so stop and discard the rest of the stack rather than raising
            # — redo across a transcript rewrite is impossible, same as redo
            # after a restart. Progress from earlier operations in this same
            # /redo is kept and reported, not thrown away.
            transcript_changed = True
            break

        if reactivated != len(op.rewound_ids):
            # A partial restore of a SINGLE operation: some rows came back,
            # some did not. That is a genuine desync rather than a clean
            # rewrite, so fail loud instead of silently half-redoing.
            raise RuntimeError(
                "redo invariant violated: restored "
                f"{reactivated} of {len(op.rewound_ids)} rewound rows"
            )

        reactivated_total += reactivated
        ops_redone += 1
        state.redo_stack.append(op)

    if transcript_changed:
        state.undo_stack.clear()
        if reactivated_total == 0:
            state.redo_stack.clear()
            return {
                "reactivated_count": 0,
                "new_tail_id": None,
                "message": "nothing to redo (transcript changed since undo)",
            }

    # Past this point the reactivation writes have COMMITTED. The trailing
    # counter bump and tail read are cosmetic and must fail soft: raising here
    # would unwind into the caller as "nothing changed" for a redo that did
    # happen, which invites a double-redo on retry.
    try:
        db.bump_redo_count(session_id)
    except Exception as exc:
        logger.warning(
            "redo: post-commit redo_count bump failed for %s "
            "(reactivation already committed): %r", session_id, exc,
        )

    try:
        active_after = db.get_messages(session_id, include_inactive=False)
        new_tail_id = max((m_["id"] for m_ in active_after), default=None)
    except Exception as exc:
        logger.warning(
            "redo: post-commit tail read failed for %s "
            "(reactivation already committed): %r", session_id, exc,
        )
        new_tail_id = None

    result: Dict[str, Any] = {
        "reactivated_count": reactivated_total,
        "new_tail_id": new_tail_id,
    }
    if transcript_changed:
        result["partial"] = True
        result["partial_retryable"] = False
        result["message"] = (
            f"redid {ops_redone} operation(s); the rest can't be redone "
            "(transcript changed since undo)"
        )
    elif transient_partial:
        result["partial"] = True
        result["partial_retryable"] = True
        result["message"] = (
            f"redid {ops_redone} operation(s); the rest hit a transient error — "
            "run /redo again to continue"
        )
    elif partial_hard_error:
        result["partial"] = True
        result["partial_retryable"] = False
        result["message"] = (
            f"redid {ops_redone} operation(s); the rest hit an error and was "
            "logged (not retryable)"
        )
    return result


def on_user_message_appended(session_id: str) -> None:
    """Invalidate the redo branch when a new user message is appended.

    Mirrors a text editor: once you type after undoing, the undone content can
    no longer be redone. ``redo()`` consumes ``undo_stack``, so that is the
    stack this must clear; ``redo_stack`` is cleared too so a fresh message
    starts from a clean slate.
    """
    state = get_state(session_id)
    state.undo_stack.clear()
    state.redo_stack.clear()


def record_rewind(session_id: str, session_db: Any, n: int, outcome: Any) -> None:
    """Bank a committed ``SessionDB.rewind_user_turn`` outcome (the shared rewind every surface uses).

    Only a plain ``/undo`` should call this: ``/retry`` and rollback replace the turn rather than
    leaving a branch to return to. Best-effort: bookkeeping must never fail the undo that committed.
    """
    global _session_db
    try:
        if session_db is not None:
            _session_db = session_db
        replacement = getattr(outcome, "replacement_id", None)
        record_undo(session_id, n, list(getattr(outcome, "rewound_ids", None) or []),
                    [replacement] if isinstance(replacement, int) else None)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("redo bookkeeping skipped for %s: %r", session_id, exc)
