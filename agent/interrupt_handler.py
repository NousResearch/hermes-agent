"""Interrupt handling helpers extracted from AIAgent for modularity."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def interrupt(runner, message: str = None) -> None:
    """
    Request the agent to interrupt its current tool-calling loop.

    Call this from another thread (e.g., input handler, message receiver)
    to gracefully stop the agent and process a new message.

    Also signals long-running tool executions (e.g. terminal commands)
    to terminate early, so the agent can respond immediately.

    Args:
        message: Optional new message that triggered the interrupt.
                 If provided, the agent will include this in its response context.

    Example (CLI):
        # In a separate input thread:
        if user_typed_something:
            agent.interrupt(user_input)

    Example (Messaging):
        # When new message arrives for active session:
        if session_has_running_agent:
            running_agent.interrupt(new_message.text)
    """
    from tools.interrupt import set_interrupt as _set_interrupt

    runner._interrupt_requested = True
    runner._interrupt_message = message
    # Signal all tools to abort any in-flight operations immediately.
    # Scope the interrupt to this agent's execution thread so other
    # agents running in the same process (gateway) are not affected.
    if runner._execution_thread_id is not None:
        _set_interrupt(True, runner._execution_thread_id)
        runner._interrupt_thread_signal_pending = False
    else:
        # The interrupt arrived before run_conversation() finished
        # binding the agent to its execution thread. Defer the tool-level
        # interrupt signal until startup completes instead of targeting
        # the caller thread by mistake.
        runner._interrupt_thread_signal_pending = True
    # Fan out to concurrent-tool worker threads.  Those workers run tools
    # on their own tids (ThreadPoolExecutor workers), so `is_interrupted()`
    # inside a tool only sees an interrupt when their specific tid is in
    # the `_interrupted_threads` set.  Without this propagation, an
    # already-running concurrent tool (e.g. a terminal command hung on
    # network I/O) never notices the interrupt and has to run to its own
    # timeout.  See `_run_tool` for the matching entry/exit bookkeeping.
    # `getattr` fallback covers test stubs that build AIAgent via
    # object.__new__ and skip __init__.
    _tracker = getattr(runner, "_tool_worker_threads", None)
    _tracker_lock = getattr(runner, "_tool_worker_threads_lock", None)
    if _tracker is not None and _tracker_lock is not None:
        with _tracker_lock:
            _worker_tids = list(_tracker)
        for _wtid in _worker_tids:
            try:
                _set_interrupt(True, _wtid)
            except Exception:
                pass
    # Propagate interrupt to any running child agents (subagent delegation)
    with runner._active_children_lock:
        children_copy = list(runner._active_children)
    for child in children_copy:
        try:
            child.interrupt(message)
        except Exception as e:
            logger.debug("Failed to propagate interrupt to child agent: %s", e)
    if not runner.quiet_mode:
        print("\n⚡ Interrupt requested" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))


def clear_interrupt(runner) -> None:
    """Clear any pending interrupt request and the per-thread tool interrupt signal."""
    from tools.interrupt import set_interrupt as _set_interrupt

    runner._interrupt_requested = False
    runner._interrupt_message = None
    runner._interrupt_thread_signal_pending = False
    if runner._execution_thread_id is not None:
        _set_interrupt(False, runner._execution_thread_id)
    # Also clear any concurrent-tool worker thread bits.  Tracked
    # workers normally clear their own bit on exit, but an explicit
    # clear here guarantees no stale interrupt can survive a turn
    # boundary and fire on a subsequent, unrelated tool call that
    # happens to get scheduled onto the same recycled worker tid.
    # `getattr` fallback covers test stubs that build AIAgent via
    # object.__new__ and skip __init__.
    _tracker = getattr(runner, "_tool_worker_threads", None)
    _tracker_lock = getattr(runner, "_tool_worker_threads_lock", None)
    if _tracker is not None and _tracker_lock is not None:
        with _tracker_lock:
            _worker_tids = list(_tracker)
        for _wtid in _worker_tids:
            try:
                _set_interrupt(False, _wtid)
            except Exception:
                pass
    # A hard interrupt supersedes any pending /steer — the steer was
    # meant for the agent's next tool-call iteration, which will no
    # longer happen. Drop it instead of surprising the user with a
    # late injection on the post-interrupt turn.
    _steer_lock = getattr(runner, "_pending_steer_lock", None)
    if _steer_lock is not None:
        with _steer_lock:
            runner._pending_steer = None


def is_interrupted(runner) -> bool:
    """Check if an interrupt has been requested."""
    return runner._interrupt_requested


def interruptible_api_call(runner, api_kwargs: dict):
    """
    Run the API call in a background thread so the main conversation loop
    can detect interrupts without waiting for the full HTTP round-trip.
    """
    from agent.chat_completion_helpers import interruptible_api_call as _impl
    return _impl(runner, api_kwargs)
