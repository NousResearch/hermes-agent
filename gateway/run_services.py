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
  partial start never leaks a task or socket.
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
) -> tuple[bool, Any]:
    """Await *awaitable* for up to *timeout* seconds.

    Returns ``(completed, result)``. On deadline the task is cancelled and
    detached (not awaited — a cancel-swallowing service must not block the
    caller) and ``(False, None)`` is returned. A raising awaitable re-raises
    here so the caller's error path owns the cleanup decision.
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
        return False, None
    return True, task.result()


async def _cleanup_failed_start(svc_name: str, svc: Any) -> None:
    """Bounded ``stop()`` on an instance whose start did not commit, so a
    partial start (task spawned, socket opened, then False/raise/timeout)
    never leaks."""
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
        completed, result = await _await_bounded(
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
    for svc_name, svc in list(services.items()):
        try:
            completed, _result = await _await_bounded(
                svc.stop(), _SERVICE_STOP_TIMEOUT_S, svc_name, "stop()",
            )
            if completed:
                logger.info("Background service '%s' stopped", svc_name)
        except Exception as e:
            logger.error("Background service '%s' stop error: %s", svc_name, e)
    services.clear()
