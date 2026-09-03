"""Bounded synchronous execution of Relay awaitables.

The Relay adapters (``agent.relay_tools``, ``agent.relay_llm``) drive async
Relay operations from synchronous call sites via ``asyncio.run``. A bare
``asyncio.run(value)`` has no deadline: an awaitable that never resolves parks
the conversation-turn thread in the event-loop selector forever with no log
output (#79568 — observed in production with a wedged ``skill_view``; the
session's ``ended_at``/``end_reason`` stay NULL and the client spins on
"ruminating…" indefinitely). The concurrent tool path is already bounded by
``HERMES_CONCURRENT_TOOL_TIMEOUT_S``; this module gives the synchronous funnel
an equivalent backstop.

Two layers, because one is not enough:

1. **Cooperative ceiling** — ``asyncio.wait_for(value, ceiling)``. Cleanly
   cancels a wedge that is parked at an ``await`` (the observed production
   class): ``finally`` blocks run, the loop closes, nothing leaks. The timer
   can only fire while the loop is idle, so a tool that *legitimately* blocks
   the loop synchronously past the ceiling (terminal foreground commands run
   up to 600s; an approval prompt adds up to 300s) is NOT interrupted
   mid-execution — the overdue timer fires at the next await point, and the
   adapters' post-dispatch recovery (``execute()``'s ``except BaseException``
   fallback) returns the already-computed tool result.
2. **Hard abandon** — the awaitable runs on a daemon worker thread, and the
   calling thread waits at most ``ceiling * _HARD_ABANDON_MULTIPLIER``. A
   wedge that swallows ``CancelledError`` or blocks the loop synchronously
   forever defeats layer 1 (``wait_for`` joins the cancellation, so it hangs
   exactly like the bare ``asyncio.run``). When the hard deadline expires the
   worker thread is abandoned — never joined — mirroring the concurrent
   executor's ``shutdown(wait=False)`` policy for hung workers, and
   ``TimeoutError`` is raised so the turn can continue.

The same abandon policy applies to ``KeyboardInterrupt``: Ctrl-C interrupts
the calling thread's wait promptly, but the worker (and the tool it is
running) is abandoned rather than cancelled and may complete detached —
matching how the concurrent executor treats its workers on interrupt.

The worker is wrapped with ``tools.thread_context.propagate_context_to_thread``
so the turn's ContextVars and the thread-local approval/sudo callbacks reach
the tool callback despite the thread shift (same audited mechanism the
concurrent executor uses).

``HERMES_TOOL_EXECUTION_CEILING_S`` tunes the cooperative ceiling (default
420s, matching ``HERMES_CONCURRENT_TOOL_TIMEOUT_S``); any value <= 0 disables
both layers. This is an internal operator knob like its concurrent sibling —
not user-facing configuration.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TOOL_EXECUTION_CEILING_S = 420.0

# The hard-abandon deadline is a multiple of the cooperative ceiling because
# layer 1's timer is starved while a tool blocks the loop synchronously: a
# legitimate slow tool must be allowed to finish before the runner gives up on
# the thread and its result is lost. Stock worst case is a 300s approval
# prompt followed by a 600s foreground terminal command = 900s; 3x the 420s
# default gives 1260s of headroom above that.
_HARD_ABANDON_MULTIPLIER = 3.0

# Floor for the managed-LLM twin: a single codex call may legitimately run up
# to HERMES_CODEX_HARD_TIMEOUT_SECONDS (1500s) inside the synchronous provider
# callback, which would outlive the tool path's 1260s hard deadline.
LLM_HARD_DEADLINE_FLOOR_S = 1800.0


def tool_execution_ceiling_seconds() -> float | None:
    """Cooperative upper bound for one synchronous Relay execution, or None.

    Reads ``HERMES_TOOL_EXECUTION_CEILING_S``. Empty/unset -> the 420s
    default; non-numeric -> warn and use the default; any value that is not
    strictly positive (including negatives and NaN) -> None (disabled),
    matching ``_resolve_concurrent_tool_timeout``'s contract.
    """
    raw = os.getenv("HERMES_TOOL_EXECUTION_CEILING_S", "").strip()
    if not raw:
        return _DEFAULT_TOOL_EXECUTION_CEILING_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "invalid HERMES_TOOL_EXECUTION_CEILING_S=%r; using %.0fs",
            raw,
            _DEFAULT_TOOL_EXECUTION_CEILING_S,
        )
        return _DEFAULT_TOOL_EXECUTION_CEILING_S
    if not value > 0:
        return None
    return value


def run_awaitable(
    value: Any,
    *,
    on_loop_error: str,
    hard_deadline_floor: float = 0.0,
) -> Any:
    """Run *value* to completion off any event loop, bounded by the ceiling.

    Non-awaitables pass through unchanged. Raises ``RuntimeError`` with
    *on_loop_error* when called from an event-loop thread (unchanged adapter
    behavior). ``hard_deadline_floor`` raises the layer-2 abandon deadline for
    call sites whose synchronous callbacks legitimately run longer than the
    tool path's stock worst case (the managed-LLM twin).
    """
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(on_loop_error)

    ceiling = tool_execution_ceiling_seconds()
    if ceiling is None:
        return asyncio.run(value)
    hard_deadline = max(ceiling * _HARD_ABANDON_MULTIPLIER, hard_deadline_floor)
    return _run_bounded(value, ceiling, hard_deadline)


def _run_bounded(value: Any, ceiling: float, hard_deadline: float) -> Any:
    # Lazy import: thread_context lazily imports tools.terminal_tool, which is
    # heavy; keep this module a cheap leaf until a bounded run actually happens.
    from tools.thread_context import propagate_context_to_thread

    outcome: dict[str, Any] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            outcome["result"] = asyncio.run(asyncio.wait_for(value, timeout=ceiling))
        except BaseException as exc:  # re-raised on the calling thread below
            outcome["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(
        target=propagate_context_to_thread(_worker),
        name="hermes-relay-bounded-await",
        daemon=True,
    )
    thread.start()
    if not done.wait(hard_deadline):
        # Layer 1 failed: the awaitable swallows CancelledError or blocks the
        # loop synchronously, so wait_for is parked joining a cancellation
        # that will never finish. Abandon the daemon thread — joining a hung
        # worker is exactly the forever-wedge this module exists to prevent.
        logger.warning(
            "Relay awaitable ignored the %.0fs execution ceiling; "
            "abandoning its worker thread after %.0fs",
            ceiling,
            hard_deadline,
        )
        raise TimeoutError(
            f"execution exceeded the {ceiling:.0f}s ceiling and did not honor "
            f"cancellation within {hard_deadline:.0f}s; worker thread abandoned"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]
