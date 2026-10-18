"""Exit-time reconciliation for kanban workers.

WHY THIS EXISTS
---------------
A kanban worker must end with ``kanban_complete`` or ``kanban_block``. The
in-loop guard (``agent.kanban_stop``) nudges a worker that tries to finish with
plain text, and it works — but it only runs on the conversation loop's *normal*
exit. The loop has ~27 early ``return`` paths that bypass it entirely:

  * thinking-budget exhaustion (all output tokens spent on reasoning)
  * output truncation after N continuation attempts
  * truncated tool-call arguments
  * iteration-budget exhaustion (``break`` out of the loop)
  * compression / output-cap / stream-drop failures

Measured on one fleet (32,749 runs): 4,582 runs (32.3% of all execution
failures) ended this way, and only ~1,900 runs ever logged a nudge, so the
in-loop guard reaches a small minority of them. Those runs then cost a full
retry each (median 5 retries, max 286, to reach completion).

THE FIX
-------
Reconcile at the *process* boundary instead of trying to hook every return
path. The one-shot exit funnel (``hermes_cli/quiet_single_query.py::
exit_single_query``) is where every worker turn converges before ``sys.exit``.
Calling this there covers all ~27 bypass paths with a single call site.

This does NOT complete the task on the worker's behalf — silently marking work
done would be worse than the bug. It records an explicit
``protocol_violation`` event with the true reason (truncation, budget, etc.), so:

  * the board shows *why* the run did not land (today it shows nothing), and
  * the retry worker receives corrective context instead of a bare failure.

Safety: never raises. A guard that crashes would swallow the worker's real exit
code and turn a diagnosable failure into an opaque one.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TERMINAL_TOOLS = frozenset({"kanban_complete", "kanban_block"})

# Action names returned to the caller (logged by cli.py).
SKIPPED = "skipped"
RECORDED = "recorded"
NOT_RUNNING = "not_running"
ERROR = "error"


def _enabled() -> bool:
    """True when this process is a dispatcher-spawned kanban worker.

    ``KANBAN_EXIT_RECONCILE=0`` is an explicit kill switch (matches the
    ``HERMES_KANBAN_STOP_NUDGE`` convention in ``agent.kanban_stop``).
    """
    flag = (os.environ.get("KANBAN_EXIT_RECONCILE") or "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip())


def _tool_name(tc: Any) -> str:
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return str(fn.get("name") or "")
        return str(tc.get("name") or "")
    fn = getattr(tc, "function", None)
    if fn is not None:
        return str(getattr(fn, "name", "") or "")
    return str(getattr(tc, "name", "") or "")


def session_called_terminal_tool(result: Optional[dict]) -> bool:
    """True if the conversation already invoked kanban_complete / kanban_block."""
    if not isinstance(result, dict):
        return False
    for msg in result.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if _tool_name(tc) in _TERMINAL_TOOLS:
                    return True
        elif msg.get("role") == "tool":
            if str(msg.get("name") or "") in _TERMINAL_TOOLS:
                return True
    return False


def _exit_reason(completed: bool, result: Optional[dict]) -> str:
    """Human-meaningful reason, so the board says WHY rather than 'violation'."""
    if isinstance(result, dict):
        err = result.get("error")
        if err:
            return str(err)[:300]
        if result.get("partial"):
            return ("run ended partial (truncation / budget / compression) "
                    "without a terminal kanban call")
    return ("run ended without a terminal kanban call"
            if not completed else "run reported completed but no terminal kanban call")


@contextlib.contextmanager
def _board_connection():
    """Open the board DB on either runtime layout.

    Upstream exposes ``hermes_cli.kanban_db_connect.connect_closing``; older
    builds (and the fleet fork) expose ``kanban_db.connect``.
    """
    try:
        from hermes_cli import kanban_db_connect as kbc
    except ImportError:
        kbc = None
    if kbc is not None and hasattr(kbc, "connect_closing"):
        with kbc.connect_closing() as conn:
            yield conn
        return
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        yield conn
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def reconcile_worker_exit(
    *,
    completed: bool,
    result: Optional[dict] = None,
    task_id: Optional[str] = None,
) -> str:
    """Record a protocol_violation if this worker is still running with no terminal call.

    Returns one of: ``skipped``, ``recorded``, ``not_running``, ``error``.
    Never raises.
    """
    try:
        if not _enabled():
            return SKIPPED
        if session_called_terminal_tool(result):
            return SKIPPED

        tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip()
        if not tid:
            return SKIPPED

        from hermes_cli import kanban_db as kb

        with _board_connection() as conn:  # resolves HERMES_KANBAN_DB / board itself
            row = conn.execute(
                "SELECT status FROM tasks WHERE id = ?", (tid,)
            ).fetchone()
            if row is None:
                return SKIPPED
            status = row["status"] if hasattr(row, "keys") else row[0]
            if status != "running":
                # kanban_complete/kanban_block (or a reclaim) already moved it.
                return NOT_RUNNING

            reason = _exit_reason(completed, result)
            payload = {
                "exit_reconcile": True,
                "session_called_terminal": False,
                "completed_flag": bool(completed),
                "reason": reason,
            }
            kb._append_event(conn, tid, "protocol_violation", payload)
            with contextlib.suppress(Exception):
                conn.commit()
            logger.info(
                "kanban exit reconcile: recorded protocol_violation task=%s completed=%s "
                "reason=%s", tid, completed, reason[:120],
            )
            return RECORDED
    except Exception:  # never raise: a guard crash would mask the real exit code
        logger.debug("kanban exit reconcile failed", exc_info=True)
        return ERROR


__all__ = ["reconcile_worker_exit", "session_called_terminal_tool"]
