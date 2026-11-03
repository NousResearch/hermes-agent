"""Turn a Kanban worker's guardrail halt into a terminal card action.

A tool-loop guardrail halt (``agent/tool_guardrails.py``) ends the turn with a normal
final response and ``turn_exit_reason == "guardrail_halt"``. That is the right shape
for an INTERACTIVE session — the model explains itself and asks the human how to
proceed — but it is the wrong shape for a HEADLESS Kanban worker: nobody is there
to answer, so the worker exits rc=0 with no terminal Kanban call. The dispatcher
books that as a protocol violation and re-spawns the card from scratch, which lands
in the same guardrail again. After the configured violations the card is given up.

Observed 2026-11-03 00:00-00:09 UTC on card ``t_d8d6c78f``: three runs, each ending
``[kanban-worker-exit] rc=0`` after ``idempotent_no_progress_block`` on
``search_files``, each booking a protocol violation, ending in ``gave_up``.

This module closes the gap: when a Kanban worker's turn ends in a guardrail halt,
the worker records the halt on the card with ``kanban_block`` so the run reaches a
TERMINAL state. The card is then a human-input problem (a loop the model cannot
break on its own), which is exactly what it is.

Deliberately NOT done here:

* The card is NOT retried. A guardrail halt means the same call repeated and
  returned the same result; re-running the same turn reproduces it.
* The guardrail's own advice ("send ``continue``") is not actionable here — there is
  no user in the loop. The block reason says what actually happened instead.
* Non-worker sessions are untouched: a human at a terminal still gets the normal
  "how would you like to proceed?" reply.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The turn-exit reason ``agent/turn_tool_round.py`` sets when the tool guardrail
# halts the turn.
GUARDRAIL_HALT_EXIT_REASON = "guardrail_halt"

# Block kind: the model hit a loop it cannot break by itself. That is a capability
# problem for THIS attempt, not a transient one — a transient block would be
# respawned by the dispatcher and would hit the same guardrail again.
BLOCK_KIND = "capability"


def _worker_task_id() -> Optional[str]:
    """This process's own Kanban task id, if it is a dispatcher-spawned worker."""
    return (os.environ.get("HERMES_KANBAN_TASK") or "").strip() or None


def is_guardrail_halt(result: Any) -> bool:
    """True when a settled turn result ended in a tool-guardrail halt."""
    if not isinstance(result, dict):
        return False
    return str(result.get("turn_exit_reason") or "") == GUARDRAIL_HALT_EXIT_REASON


def _guardrail_detail(result: dict) -> str:
    """A one-line description of which guardrail fired."""
    guardrail = result.get("guardrail")
    if isinstance(guardrail, dict):
        code = guardrail.get("code")
        tool = guardrail.get("tool_name")
        count = guardrail.get("count")
        if code:
            bits = [str(code)]
            if tool:
                bits.append(f"tool={tool}")
            if count:
                bits.append(f"count={count}")
            return " (".join(bits[:1]) + (", " + ", ".join(bits[1:]) + ")" if len(bits) > 1 else ")")
    return ""


def block_reason(result: Any) -> str:
    """The human-readable reason recorded on the card."""
    detail = _guardrail_detail(result) if isinstance(result, dict) else ""
    base = (
        "The worker's tool guardrail halted the turn"
        + (f" ({detail})" if detail else "")
        + ": a read-only tool call returned the same result repeatedly, so the model "
        "re-entered the same call instead of making progress."
    )
    return (
        base
        + " This is a headless worker, so there is nobody to send `continue` to; the turn "
        "ended without a terminal card action. The card needs a human to either narrow the "
        "investigation (a specific file, log window, or command) or to approve a different "
        "tool path. Retrying unchanged reproduces the same halt."
    )


def block_guardrail_halt(result: Any, *, emit=None) -> bool:
    """Record a guardrail halt as a terminal ``kanban_block`` on this worker's card.

    Returns True when the card was blocked. Returns False — without raising — when
    this is not a Kanban worker, the turn did not end in a guardrail halt, or the
    block could not be recorded (the dispatcher will then book the usual protocol
    violation, which is the correct fallback).
    """
    task_id = _worker_task_id()
    if not task_id:
        return False
    if not is_guardrail_halt(result):
        return False

    reason = block_reason(result)
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        conn = kbc.connect(board=None)
    except Exception:
        # The board could not be opened at all. The dispatcher's protocol-violation
        # path is the correct fallback, so this must not raise — but the traceback
        # is what tells an operator WHY the card was not blocked.
        logger.warning(
            "Guardrail halt on kanban task %s: could not open the board to block it",
            task_id, exc_info=True,
        )
        return False

    try:
        run_id = _worker_run_id()
        ok = kb.block_task(
            conn, task_id, reason=reason, kind=BLOCK_KIND, expected_run_id=run_id)
        if not ok:
            logger.warning(
                "Guardrail halt on kanban task %s: block_task did not land (unknown id or "
                "the task is not in running/ready).", task_id,
            )
            return False
        if emit is not None:
            emit(f"[kanban] guardrail halt recorded as a block on {task_id}: {reason}")
        logger.info("Guardrail halt recorded as a block on kanban task %s", task_id)
        return True
    except Exception:
        # The board write failed. Same reasoning as above: the protocol-violation
        # fallback is correct, but the traceback is the only evidence of why.
        logger.warning(
            "Guardrail halt on kanban task %s: block failed", task_id, exc_info=True)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            logger.warning(
                "Closing the kanban board connection after a guardrail block failed",
                exc_info=True)


def _worker_run_id() -> Optional[int]:
    """This worker's dispatcher run id, when the dispatcher pinned one."""
    raw = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


__all__ = [
    "BLOCK_KIND",
    "GUARDRAIL_HALT_EXIT_REASON",
    "block_guardrail_halt",
    "block_reason",
    "is_guardrail_halt",
]
