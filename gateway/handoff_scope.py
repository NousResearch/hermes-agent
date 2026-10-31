"""A permanently warm single worker for the handoff watcher's scope walk.

``loop.run_in_executor`` calls ``executor.submit()`` on the EVENT LOOP thread, and every pool that has
to GROW at submit time starts a thread there. ``Thread.start()`` returns only once the new thread is
running, so on a host under memory-commit pressure it holds the loop for as long as that thread takes
to bootstrap — measured at over 30 s on 2026-10-31 01:00, past the three missed liveness probes, so
the shutdown watchdog exited the gateway with code 75 and killed every mid-flight Kanban worker.

A pool whose worker is ALREADY parked turns ``submit()`` into a queue put. That is the whole point of
this module: one worker, never idle-timed-out, so a watcher tick can never start a thread on the loop.
The only submit that starts a thread is :func:`warm`, which boot calls from an executor thread.

Sharing one worker serializes the ticks of both watchers (``_handoff_watcher`` and
``_loop_wakeup_watcher``). They only await the walk, so contention costs latency, never loop time.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import threading
from typing import Any, Callable

# A worker that cannot start is a real fault (OS thread limit); bound the boot wait so it surfaces as
# a slow first tick instead of hanging the warm-up.
_WARM_TIMEOUT_S = 30.0

_pool: concurrent.futures.Executor | None = None
_pool_lock = threading.Lock()


def _scope_pool() -> concurrent.futures.Executor:
    """The process-wide worker every watcher tick shares; created on first use, never replaced.

    ``ThreadPoolExecutor`` (not the turn pool's ``_UnboundedThreadExecutor``): its one worker blocks
    on the work queue until ``shutdown()``, which is exactly the parked worker ``submit()`` needs.
    """
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="hermes-handoff-scope")
        return _pool


def warm() -> bool:
    """Start the pool's worker now, so no later tick pays ``Thread.start()``.

    Call this OFF the loop (the boot warm-up runs it in an executor): the call itself starts the
    thread, which is the cost this module exists to keep off the loop thread.
    """
    future = _scope_pool().submit(lambda: True)
    done, _pending = concurrent.futures.wait([future], timeout=_WARM_TIMEOUT_S)
    return bool(done) and future.result()


async def resolve_off_loop(func: Callable[..., Any], /, *args: Any) -> Any:
    """Run blocking ``func(*args)`` on the warm worker while the loop keeps serving probes.

    The executor hop carries the caller's context, like ``_run_in_executor_with_context``, so a
    contextvar a caller bound before the hop still reads the same on the worker.
    """
    ctx = contextvars.copy_context()

    def _run() -> Any:
        return ctx.run(func, *args)

    return await asyncio.get_running_loop().run_in_executor(_scope_pool(), _run)
