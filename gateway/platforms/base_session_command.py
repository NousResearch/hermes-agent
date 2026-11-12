"""``/stop``, ``/new`` and ``/reset`` reaching a busy session (``BasePlatformAdapter``).

The command bypasses the active-session guard, so it runs inline under a command guard of its own.
The adapter cancels the session's task only on the runner's verdict: its stop/reset handlers run
past admission and slash access and mark the event (``MessageEvent.accept_interrupt``). A refused
sender in a shared session must not abort another user's turn or drop their held input (#136759).
An adapter with no authorization check registered has no verdict to wait for and always cancels.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

# Command guard -> the guard it stands in for, while that command is in flight. Overlapping commands
# chain to the owner's guard, so a refusal never hands the session to another command's guard.
_command_guard_owners: dict[asyncio.Event, asyncio.Event | None] = {}


async def dispatch_active_session_command(adapter, event, session_key: str, cmd: str) -> None:
    """Reply, then cancel + drain if accepted; refused, the owner's turn and held input stay."""
    logger.debug("[%s] Command '/%s' bypassing active-session guard for %s", adapter.name, cmd, session_key)
    current_guard = adapter._active_sessions.get(session_key)
    current_guard = _command_guard_owners.get(current_guard, current_guard)
    command_guard = adapter._active_sessions[session_key] = asyncio.Event()
    _command_guard_owners[command_guard] = current_guard
    if (held := adapter._text_debounce_store().pop(session_key, None)) is not None:
        held.cancel_timer()
    cancelled = False
    try:
        await adapter._dispatch_inline_reply(event, log_cmd=cmd)
        if adapter._authorization_check is None or getattr(event, "_interrupt_accepted", False) is True:
            adapter._discard_text_debounce(session_key)  # adapter-held media too (Telegram, #81370)
            await adapter.cancel_session_processing(session_key, release_guard=False, discard_pending=False)
            cancelled = True
    finally:
        _command_guard_owners.pop(command_guard, None)
        if not cancelled:
            await _restore_after_refused_command(adapter, session_key, command_guard, current_guard, held)
    if cancelled:
        await adapter._drain_pending_after_session_command(session_key, command_guard)


async def _restore_after_refused_command(adapter, session_key, command_guard, current_guard, held) -> None:
    """Hand the session back as the command found it: a still-running owner gets its guard back and
    drains the held text itself; an owner that finished during the reply left nobody to drain it,
    so it runs now instead of waiting in the pending slot for the next stale-lock heal to drop it."""
    if held is not None:
        adapter._text_debounce_store()[session_key] = held
    if adapter._active_sessions.get(session_key) is not command_guard:
        await adapter._flush_text_debounce_now(session_key)  # a newer owner took the session; it drains
        return
    if (current_guard is not None and session_key in adapter._session_tasks
            and not adapter._session_task_is_stale(session_key)):
        adapter._active_sessions[session_key] = current_guard
        await adapter._flush_text_debounce_now(session_key)
        return
    await adapter._drain_pending_after_session_command(session_key, command_guard)
