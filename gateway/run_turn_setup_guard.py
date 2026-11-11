"""The pre-agent setup guard (#138358): session resolve, turn prepare and the ``agent:start`` hooks
run before the inactivity watchdog exists, so a hang there held the turn's slot (and the session's
durable FIFO behind it) forever. The guard bounds that phase and lets /stop or /new end it.

It runs on the turn's own task, so ContextVars set during prepare stay visible to the agent run.
Ending setup cancels that task and converts the cancellation at the guard, so the turn returns
normally: its finally blocks release the slot, lease and marker, and a FIFO drain awaiting it
settles the admission and moves on.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, Optional

from agent.i18n import t

logger = logging.getLogger("gateway.run")

_TIMEOUT = "timeout"
_STOPPED = "stopped"


class PreagentSetupStopped(Exception):
    """/stop or /new ended the turn while it was still in setup; their own reply already told the user."""


class PreagentSetupTimeout(Exception):
    """Setup outlived ``HERMES_AGENT_TIMEOUT``. Not a ``TimeoutError``: setup's own timeouts (the
    turn lease's ``TurnLeaseTimeoutError``) must still reach the caller that answers them."""


@asynccontextmanager
async def preagent_setup_guard(runner, session_key: str):
    """Bound setup by ``HERMES_AGENT_TIMEOUT`` (``agent.gateway_timeout``; ``<= 0`` = unbounded) and
    publish a stop handle on the turn slot. Raises ``PreagentSetupTimeout`` past the deadline and
    ``PreagentSetupStopped`` after ``cancel_preagent_setup``."""
    from gateway.run import _dump_wedged_turn_stacks, _float_env
    delay = _float_env("HERMES_AGENT_TIMEOUT", 1800)
    task = asyncio.current_task()
    turn = runner._session_state(session_key).turn
    reason = None

    def _end(why: str) -> bool:
        nonlocal reason
        if reason is not None or task is None or task.done():
            return False
        reason = why
        task.cancel()
        return True

    def _fire() -> None:
        try:
            frames = task.get_stack(limit=25) if task is not None else []
            stack = "".join(traceback.format_stack(frames[-1])) if frames else ""
            logger.error(
                "Pre-agent setup timed out for %s after %.0fs (HERMES_AGENT_TIMEOUT / "
                "agent.gateway_timeout); cancelling the wedged turn:\n%s",
                session_key, delay, stack or "<no asyncio stack>",
            )
            _dump_wedged_turn_stacks(session_key)
        except Exception:
            logger.error("Pre-agent setup timed out for %s after %.0fs; stack dump failed",
                         session_key, delay, exc_info=True)
        _end(_TIMEOUT)

    handle = asyncio.get_running_loop().call_later(delay, _fire) if delay > 0 else None
    def stop() -> bool:
        return _end(_STOPPED)

    turn.setup_stop = stop
    try:
        try:
            yield
        except asyncio.CancelledError:
            if reason is None or (task is not None and task.cancelling() > 1):
                raise  # a real cancellation (shutdown, the drain's) is never swallowed
        if reason is not None:
            # Also when setup swallowed the cancel and finished: the turn was ended either way, and
            # the pending cancel request must not outlive it on this (possibly the drain's) task.
            if task is not None:
                task.uncancel()
            if reason == _TIMEOUT:
                raise PreagentSetupTimeout from None
            raise PreagentSetupStopped from None
    finally:
        if handle is not None:
            handle.cancel()
        if turn.setup_stop is stop:
            turn.setup_stop = None


def cancel_preagent_setup(runner, session_key: str) -> bool:
    """End the session's turn if it is still in setup (before any agent ran). True when it was."""
    state = runner._peek_session_state(session_key)
    stop = state.turn.setup_stop if state is not None else None
    if stop is None or not stop():
        return False
    logger.info("Cancelled pre-agent setup for session %s", session_key)
    return True


async def run_preagent_setup(runner, event, source, quick_key: str, run_generation: int) -> tuple[Optional[tuple], Any]:
    """Session resolve, turn prepare and ``agent:start`` under the guard. ``(setup, None)`` to run
    the agent, setup = ``(source, session_entry, session_key, prepared, env_tokens, hook_ctx)``;
    ``(None, reply)`` when the turn ends here. A deadline ends it with a visible resend notice, so
    a person's message is never dropped silently; a /stop or /new answers for itself."""
    try:
        async with preagent_setup_guard(runner, quick_key):
            resolved = await runner._hmwa_resolve_session(event, source)
            if resolved is None:
                return None, None
            source, session_entry, session_key = resolved
            prepared, env_tokens = await runner._hmwa_prepare_turn(
                event, source, session_entry, session_key, quick_key, run_generation,
            )
            if not isinstance(prepared, runner._PreparedTurn):
                return None, prepared
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "chat_id": source.chat_id or "",
                "thread_id": str(source.thread_id) if getattr(source, "thread_id", None) else "",
                "chat_type": getattr(source, "chat_type", "") or "",
                "session_id": session_entry.session_id,
                "message": prepared.message_text[:500],
            }
            await runner.hooks.emit("agent:start", hook_ctx)
    except (PreagentSetupTimeout, PreagentSetupStopped) as ended:
        runner._clear_session_env([])  # prepare may have bound them before the hang
        return None, (t("gateway.busy.setup_timeout") if isinstance(ended, PreagentSetupTimeout) else None)
    return (source, session_entry, session_key, prepared, env_tokens, hook_ctx), None


def pending_setup_busy_notice(runner, session_key: str) -> Optional[str]:
    """Visible busy ack for a follow-up queued behind a turn still in setup. Fresh setup stays
    silent (the historical race-avoidance); past ``session_stall_timeout`` (0 disables) the user
    gets the same class of notice as a stalled running turn, debounced to one per 30 s."""
    import os
    import time
    from gateway.run import _float_env
    if os.environ.get("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true").lower() != "true":
        return None
    timeout = _float_env("HERMES_SESSION_STALL_TIMEOUT", 300)
    state = runner._peek_session_state(session_key)
    started = float(getattr(getattr(state, "turn", None), "started_ts", 0.0) or 0.0)
    if timeout <= 0 or state is None or not started:
        return None
    now = time.time()
    age = now - started
    if age < timeout or now - float(state.turn.busy_ack_ts or 0.0) < 30:
        return None
    state.turn.busy_ack_ts = now
    return t("gateway.busy.setup_waiting", minutes=max(1, int(age // 60)))
