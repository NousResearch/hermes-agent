"""Cold-start init deadline and abandoned-task guards for ``TelegramAdapter``.

Extracted verbatim from ``plugins/platforms/telegram/adapter.py`` as part of
the god-file decomposition campaign. All helpers are module-level functions
(no adapter state); ``adapter.py`` re-imports the ones its staying methods
still call so behavior is unchanged.
"""

import asyncio
import faulthandler
import logging
import threading

logger = logging.getLogger("plugins.platforms.telegram.adapter")


def _consume_abandoned_task(task: asyncio.Task) -> None:
    """Observe a detached task's terminal exception to avoid noisy loop logs."""
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("Abandoned Telegram init task failed after timeout", exc_info=True)

# Grace period after the wall-clock deadline fires: if the event loop still
# hasn't processed the expiry callback by then, the loop thread itself is
# blocked in a synchronous call — the exact state in which every asyncio-based
# timeout (including this helper's own expiry hand-off) goes silent, so the
# gateway hangs at "attempt 1/8" with no further output (#63309).
_LOOP_BLOCKED_DUMP_GRACE = 5.0

def _dump_loop_blocked_diagnostics(timeout: float, grace: float) -> None:
    """Emit diagnostics from the deadline timer thread when the loop is stuck.

    Runs OFF the event loop, so it works precisely when the loop cannot. The
    faulthandler dump names the frame the loop thread is blocked in — the one
    piece of information #63309-class hangs otherwise never surface.
    """
    logger.warning(
        "[Telegram] init deadline (%.0fs) expired but the event loop has not "
        "processed the expiry after a further %.0fs — the loop thread appears "
        "BLOCKED in a synchronous call, which is why no timeout fires (#63309). "
        "Dumping all thread stacks to stderr to identify the blocking frame.",
        timeout,
        grace,
    )
    try:
        faulthandler.dump_traceback(all_threads=True)
    except Exception:
        logger.debug("faulthandler traceback dump failed", exc_info=True)

async def _await_with_thread_deadline(awaitable, timeout: float, *, on_abandon=None):
    """Await with a wall-clock deadline that does not depend on loop timers.

    ``asyncio.wait_for`` schedules its timeout on the event loop and then waits
    for cancellation to propagate.  PTB/httpcore initialization can sit inside
    cancellation-shielded anyio scopes, so a timed-out initialize() may never
    hand control back to the retry ladder under some supervisors.  This helper
    lets a daemon ``threading.Timer`` wake the loop and, on timeout, abandons
    the shielded task instead of awaiting cancellation completion.

    ``on_abandon`` (optional) is a zero-arg callable returning an awaitable that
    is scheduled as a detached best-effort cleanup when the task is abandoned on
    timeout.  The abandoned initialize() may leave a half-built httpx client /
    connection pool open (it never completed and we do not await its
    cancellation), so the caller uses this to shut that state down and avoid
    leaking a pool per retry attempt.  Cleanup runs detached and its own errors
    are swallowed, so it can never re-block the retry ladder.
    """
    task = asyncio.ensure_future(awaitable)
    loop = asyncio.get_running_loop()
    deadline = loop.create_future()
    # Set the moment the loop actually runs the expiry callback (or the helper
    # exits normally). threading.Event so the watchdog thread can read it
    # without touching asyncio state from off-loop.
    loop_processed_expiry = threading.Event()

    def _mark_expired() -> None:
        loop_processed_expiry.set()
        if not deadline.done():
            deadline.set_result(None)

    def _expire_from_thread() -> None:
        loop.call_soon_threadsafe(_mark_expired)

    def _watchdog_check() -> None:
        # The deadline fired _LOOP_BLOCKED_DUMP_GRACE ago but the loop never
        # ran _mark_expired: the loop thread is stuck in a synchronous call.
        # Diagnose from this thread — the loop can't.
        if not loop_processed_expiry.is_set():
            _dump_loop_blocked_diagnostics(timeout, _LOOP_BLOCKED_DUMP_GRACE)

    timer = threading.Timer(max(timeout, 0.0), _expire_from_thread)
    timer.daemon = True
    timer.start()
    watchdog = threading.Timer(
        max(timeout, 0.0) + _LOOP_BLOCKED_DUMP_GRACE, _watchdog_check
    )
    watchdog.daemon = True
    watchdog.start()
    try:
        done, _ = await asyncio.wait(
            {task, deadline},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            if not deadline.done():
                deadline.cancel()
            return await task

        task.cancel()
        task.add_done_callback(_consume_abandoned_task)
        if on_abandon is not None:
            # Detached best-effort cleanup: close the half-built app's httpx
            # client/pool so an abandoned attempt can't leak sockets across the
            # retry ladder. Detached + exception-observed so it never re-blocks
            # or re-hangs the ladder we are trying to advance.
            cleanup = asyncio.ensure_future(_run_abandon_cleanup(on_abandon))
            cleanup.add_done_callback(_consume_abandoned_task)
        raise asyncio.TimeoutError()
    finally:
        timer.cancel()
        watchdog.cancel()
        # cancel() cannot stop a Timer whose callback is already running;
        # setting the event closes that race so a completed await can never
        # be misreported as a blocked loop.
        loop_processed_expiry.set()

async def _first_completed(*futures: "asyncio.Future") -> None:
    """Return when the first of ``futures`` completes.

    Used by the strict cold-start readiness gate to wait on "progress OR
    polling error", whichever fires first (#67498). Does not cancel the
    losers — the caller owns their lifecycle.
    """
    await asyncio.wait(set(futures), return_when=asyncio.FIRST_COMPLETED)

async def _run_abandon_cleanup(on_abandon) -> None:
    """Run the abandonment cleanup coroutine, swallowing any failure.

    Wrapped so a cleanup that itself hangs or raises cannot surface as an
    unhandled task error or block anything — it is fully fire-and-forget.
    """
    try:
        result = on_abandon()
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            await result
    except Exception:
        logger.debug("Abandoned Telegram init cleanup failed", exc_info=True)

async def _shutdown_abandoned_app(app) -> None:
    """Release a half-built PTB app's httpx transports after init was abandoned.

    ``Application.shutdown()`` / ``Bot.shutdown()`` are gated on the app's
    ``_initialized`` / ``_requests_initialized`` flags, which a wedged
    ``initialize()`` (the case this whole path exists for) may never have set —
    so calling only ``app.shutdown()`` no-ops and leaks the connection pool it
    was meant to close.  ``HTTPXRequest`` builds its ``httpx.AsyncClient``
    eagerly in its constructor and its ``shutdown()`` gates only on
    ``client.is_closed``, so closing the request transports directly releases
    the pool regardless of PTB init state.  We try the clean path first, then
    fall back to the transports.  All best-effort and swallowed.
    """
    if app is None:
        return
    try:
        await app.shutdown()
    except Exception:
        logger.debug("Abandoned Telegram app.shutdown() failed", exc_info=True)
    # Directly close the underlying request transports (bypasses PTB's
    # init-gated shutdown so the eagerly-built httpx pool is released even when
    # the abandoned initialize() never flipped _initialized).
    bot = getattr(app, "bot", None)
    requests = getattr(bot, "_request", None) if bot is not None else None
    if not requests:
        return
    for request in requests:
        shutdown = getattr(request, "shutdown", None)
        if shutdown is None:
            continue
        try:
            result = shutdown()
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result
        except Exception:
            logger.debug("Abandoned Telegram request shutdown failed", exc_info=True)
