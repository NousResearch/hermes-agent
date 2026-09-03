"""Record an abandoned kanban turn on the board instead of exiting silently.

``run_conversation`` has bail-out paths that ``return`` early without reaching
``finalize_turn`` — currently the "response stayed truncated after every
continuation attempt" path in :mod:`agent.conversation_loop`. A dispatcher-
spawned worker that leaves through one of them abandons its card: the process
exits cleanly, the turn is never finalized, and no terminal board tool is
called. The dispatcher's orphan reconciler later finds the pid gone and writes
``crashed: pid N not alive``.

That label describes the corpse, not the cause. It reads as "the process was
killed", so every investigation starts by looking for a killer that does not
exist — memory pressure, an external dependency, someone's cleanup script.
Meanwhile the real reason (the model blew past its output cap) is recorded
nowhere on the card, and the operator sees a task that looks crashed but whose
work may be complete.

This module converts that silent abandonment into a ``blocked`` row carrying
the actual reason. Blocking (rather than completing) is deliberate: the turn
did not finish, so the card genuinely needs a human or a retry — we are only
making the failure legible, not claiming success.

Policy lives in :mod:`agent.kanban_stop`, which is side-effect free. This
module is the side-effecting counterpart and is kept separate so that contract
holds.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

from agent.kanban_stop import session_called_kanban_terminal

logger = logging.getLogger(__name__)

# Truncation is retryable in principle — a shorter answer may fit — so the
# recurrence breaker should be allowed to see it repeat and route the card to
# triage rather than us pre-judging it as a hard capability limit.
_DEFAULT_KIND = "transient"

_MAX_REASON_CHARS = 2000


def block_abandoned_kanban_task(
    messages: Iterable[dict] | None,
    reason: str,
    *,
    kind: str = _DEFAULT_KIND,
    task_id: Optional[str] = None,
) -> bool:
    """Mark this worker's kanban task ``blocked`` with ``reason``.

    Returns True only when a block was actually written. Returns False —
    without raising — in every other case:

    * not a dispatcher-spawned kanban worker (``HERMES_KANBAN_TASK`` unset);
    * the session already called ``kanban_complete`` / ``kanban_block``, so the
      card has a terminal state the worker chose and we must not overwrite it;
    * the board write failed.

    Never raises. It is called on a path that is already giving up; a failure
    to annotate the card must not turn a bad turn into a crash.
    """
    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not tid:
        return False

    if session_called_kanban_terminal(messages):
        # The worker closed its own card before this bail-out. Its verdict wins.
        return False

    text = (reason or "").strip() or "Turn abandoned without a terminal board call."
    if len(text) > _MAX_REASON_CHARS:
        text = text[: _MAX_REASON_CHARS - 1] + "…"

    try:
        from hermes_cli import kanban_db as kb

        with kb.connect_closing() as conn:
            ok = kb.block_task(conn, tid, reason=text, kind=kind)
    except Exception:
        logger.warning(
            "could not record abandoned kanban turn for task=%s — the card will "
            "fall to the orphan reconciler and show as 'pid not alive'",
            tid,
            exc_info=True,
        )
        return False

    if ok:
        logger.info("recorded abandoned kanban turn as blocked task=%s kind=%s", tid, kind)
    else:
        # Not running/ready any more — another writer already moved it.
        logger.info("abandoned kanban turn not recorded (task=%s not blockable)", tid)
    return bool(ok)


__all__ = ["block_abandoned_kanban_task"]
