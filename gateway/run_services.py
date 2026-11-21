"""Background-service lifecycle owner for the gateway runner.

Extracted from ``gateway/run.py`` so the runner delegates plugin
background-service startup/shutdown here instead of re-absorbing lifecycle
logic into the monolith (the ``run.py`` decomposition's no-regrowth
boundary).

Lifecycle contract — the one authoritative statement
====================================================

* ``service_factory(config_dict, gateway_runner)`` returns an object
  exposing ``async start()`` and ``async stop()``.
* ``start()`` reports success by **completing**: a coroutine that returns
  without raising — including the common untyped ``return`` / ``return
  None`` shape — counts as started. Only an explicit ``return False`` or a
  raised exception counts as failed. (Requiring a truthy return would
  misclassify conforming ``-> None`` services as failed and leak them
  without ever calling ``stop()``.)
* Start is **transactional**: once a service instance exists, every
  non-committed start path (``False``, raise, timeout) still owns the
  instance's cleanup — the runtime calls ``stop()`` on it, bounded, so a
  partial start never leaks a task or socket. After a timeout the cleanup
  first fences on the detached generation's actual settlement
  (:data:`_SERVICE_DETACH_GRACE_S`) so ``stop()`` never runs concurrently
  with a cancel-swallowing ``start()`` that is still executing; a generation
  still running after the grace keeps ownership of its partial allocations
  (logged, not raced).
* Both ``start()`` and ``stop()`` are **bounded**
  (:data:`_SERVICE_START_TIMEOUT_S` / :data:`_SERVICE_STOP_TIMEOUT_S`): a
  third-party service cannot wedge gateway startup or shutdown
  indefinitely. On deadline the coroutine is cancelled and detached — the
  same shape as adapter teardown (``run_adapters._wait_or_detach``),
  because ``asyncio.wait_for`` would wait forever for a cancelled child
  that swallows ``CancelledError``.
* One bad service never prevents later services from starting, nor the
  gateway shutdown from settling.
"""

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Bounds for third-party service lifecycle calls. Generous enough for a
# service that opens sockets / hydrates caches on start, tight enough that a
# wedged plugin cannot stall gateway startup or hold shutdown past systemd's
# TimeoutStopSec (which would end in SIGKILL and skip atexit cleanup).
_SERVICE_START_TIMEOUT_S = 30.0
_SERVICE_STOP_TIMEOUT_S = 15.0
# After a start() deadline the task is cancelled and detached; a cancel-
# swallowing start() may still be executing. Cleanup first waits up to this
# grace bound for the detached generation to actually settle, so teardown
# never runs concurrently with an in-flight setup on the same instance.
_SERVICE_DETACH_GRACE_S = 5.0


def _consume_detached(task: "asyncio.Task") -> None:
    """Swallow the eventual outcome of a detached (timed-out) service task
    so it never surfaces as an unhandled-exception warning."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("Detached background-service task error: %r", exc)


async def _await_bounded(
    awaitable: Any, timeout: float, label: str, phase: str
) -> tuple[bool, Any, "asyncio.Task"]:
    """Await *awaitable* for up to *timeout* seconds.

    Returns ``(completed, result, task)``. On deadline the task is cancelled
    and detached (not awaited — a cancel-swallowing service must not block
    the caller) and ``(False, None, task)`` is returned; the task is handed
    back so cleanup can fence on its actual settlement. A raising awaitable
    re-raises here so the caller's error path owns the cleanup decision.
    """
    task = asyncio.ensure_future(awaitable)
    done: set = set()
    try:
        done, _pending = await asyncio.wait({task}, timeout=timeout)
    finally:
        if task not in done:  # timed out, or our own cancellation
            task.cancel()
            task.add_done_callback(_consume_detached)
    if task not in done:
        logger.warning(
            "Background service '%s' %s timed out after %.0fs; task detached",
            label, phase, timeout,
        )
        return False, None, task
    return True, task.result(), task


async def _cleanup_failed_start(
    svc_name: str, svc: Any, pending: Optional["asyncio.Task"] = None
) -> None:
    """Bounded ``stop()`` on an instance whose start did not commit, so a
    partial start (task spawned, socket opened, then False/raise/timeout)
    never leaks.

    ``pending`` is the detached start generation after a deadline: a cancel-
    swallowing ``start()`` may still be executing, and running ``stop()``
    concurrently with the in-flight setup would tear down one object mid-
    allocation (anything allocated after ``stop()`` ran leaks with no owner,
    since ``stop()`` will not run twice). The fence: wait up to
    :data:`_SERVICE_DETACH_GRACE_S` for the generation to actually settle,
    and only then call ``stop()``. A generation that is STILL running after
    the grace keeps ownership of its partial allocations — cleanup is skipped
    with a warning rather than raced.
    """
    if pending is not None and not pending.done():
        done, _ = await asyncio.wait({pending}, timeout=_SERVICE_DETACH_GRACE_S)
        if pending not in done:
            logger.warning(
                "Background service '%s' detached start() is still running "
                "%.0fs after its deadline; skipping stop() so teardown cannot "
                "race the in-flight setup — the detached task keeps ownership "
                "of its partial allocations",
                svc_name, _SERVICE_DETACH_GRACE_S,
            )
            return
    stop = getattr(svc, "stop", None)
    if stop is None:
        return
    try:
        await _await_bounded(stop(), _SERVICE_STOP_TIMEOUT_S, svc_name, "post-failure stop()")
    except Exception as e:
        logger.warning(
            "Background service '%s' post-failure cleanup error: %s", svc_name, e,
        )


async def _start_one(runner: Any, svc_name: str, svc: Any) -> None:
    """Transactional start of one service instance (see module docstring)."""
    try:
        completed, result, task = await _await_bounded(
            svc.start(), _SERVICE_START_TIMEOUT_S, svc_name, "start()",
        )
    except Exception as e:
        logger.error(
            "Background service '%s' startup error: %s", svc_name, e, exc_info=True,
        )
        await _cleanup_failed_start(svc_name, svc)
        return
    if completed and result is not False:
        # Committed: completion (any non-False result, None included) is the
        # success signal — see the lifecycle contract above.
        runner.services[svc_name] = svc
        logger.info("Background service '%s' started", svc_name)
        return
    if completed:
        logger.warning(
            "Background service '%s' reported start failure (returned False)",
            svc_name,
        )
        await _cleanup_failed_start(svc_name, svc)
        return
    # Deadline: hand the detached generation to cleanup so it can fence on
    # settlement before touching the instance.
    await _cleanup_failed_start(svc_name, svc, pending=task)


async def start_plugin_background_services(runner: Any) -> None:
    """Instantiate and start each enabled background service.

    Iterates ``runner.config.services`` (loaded from ``services:`` in
    config.yaml) and looks each name up in the profile-scoped
    ``service_registry``. Misses are logged and skipped — they're either a
    typo, a disabled plugin, or a service we removed but config still
    references. A failing/hanging service is contained per entry and never
    aborts the remaining services or gateway startup.
    """
    from gateway.service_registry import service_registry

    services_cfg = getattr(runner.config, "services", None) or {}
    if not services_cfg:
        return

    for svc_name, svc_config in services_cfg.items():
        if not isinstance(svc_config, dict) or not svc_config.get("enabled", False):
            continue

        if not service_registry.is_registered(svc_name):
            logger.warning(
                "Background service '%s' is enabled in config.yaml but no "
                "plugin registered it. Is the plugin installed and enabled?",
                svc_name,
            )
            continue

        svc = service_registry.create_service(svc_name, svc_config, runner)
        if svc is None:
            continue

        await _start_one(runner, svc_name, svc)


async def stop_plugin_background_services(runner: Any) -> None:
    """Stop every running background service and clear the runner's dict.

    Called from shutdown BEFORE adapter teardown — services produce events
    that deliver through the adapters, so producers stop first. A raising or
    hanging ``stop()`` is bounded and logged and does not block the
    remaining services or the rest of shutdown.
    """
    services = getattr(runner, "services", None)
    if not services:
        return
    # Settlement is unconditional: an outer cancellation (gateway shutdown
    # task cancelled while awaiting one slow stop()) must not skip the
    # remaining services or the ownership clear. Each service's bounded wait
    # keeps its own deadline; a caller-side CancelledError is absorbed per
    # service (and the current task un-cancelled so later awaits still run),
    # the loop finishes stopping the rest, the dict is cleared in a finally,
    # and the cancellation is re-raised afterwards.
    cancelled: Optional[BaseException] = None
    try:
        for svc_name, svc in list(services.items()):
            try:
                completed, _result, _task = await _await_bounded(
                    svc.stop(), _SERVICE_STOP_TIMEOUT_S, svc_name, "stop()",
                )
                if completed:
                    logger.info("Background service '%s' stopped", svc_name)
            except asyncio.CancelledError as e:
                cancelled = e
                task = asyncio.current_task()
                if task is not None:
                    task.uncancel()
                logger.warning(
                    "Shutdown cancelled while stopping background service "
                    "'%s'; finishing the remaining stops before re-raising",
                    svc_name,
                )
            except Exception as e:
                logger.error("Background service '%s' stop error: %s", svc_name, e)
    finally:
        services.clear()
    if cancelled is not None:
        raise cancelled
