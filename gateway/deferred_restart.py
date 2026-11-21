"""Constrained acceptance channel for the runner's existing supervised restart."""
from __future__ import annotations

from concurrent.futures import Future
import logging
import math
import os

SUPPORTED_SUPERVISORS = frozenset({"systemd", "launchd", "external"})
HANDLER_TIMEOUT = 5.0
REQUEST_TIMEOUT = 8.0
logger = logging.getLogger(__name__)


def valid_drain_timeout(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def deferred_restart_verb(runner, loop):
    """Socket worker handler; a timed-out handoff may still execute exactly once."""
    def handle() -> dict:
        from gateway.control_socket import _detect_supervisor
        from gateway.restart import is_gateway_supervisor_process

        pid = os.getpid()
        if (os.name != "posix" or (supervisor := _detect_supervisor()) not in SUPPORTED_SUPERVISORS
                or not is_gateway_supervisor_process()):
            return {"accepted": False, "reason": "unsupported_supervisor", "pid": pid}
        drain = runner._restart_drain_timeout
        metadata = {"pid": pid, "supervisor": supervisor}
        if not valid_drain_timeout(drain):
            return {"accepted": False, "reason": "unsupported_drain_timeout", **metadata}
        metadata["drain_timeout"] = drain
        result = Future()

        def request() -> None:
            try:
                # stop() sets its task before yielding; signal handlers also stamp intent
                # before scheduling stop(). Check here, even after the worker's deadline.
                stopping = (runner._stop_task is not None
                            or getattr(runner, "_stop_requested_by_signal", False)
                            or getattr(runner, "_signal_initiated_shutdown", False)
                            or (runner._draining and not runner._restart_requested))
                accepted = False if stopping else runner.request_restart(detached=False, via_service=True)
                if type(accepted) is not bool:
                    raise TypeError("restart acceptance must be boolean")
                ack = {"accepted": accepted, **metadata}
                if not accepted:
                    ack["reason"] = "already_stopping"
                result.set_result(ack)
            except Exception as exc:
                logger.exception("Deferred restart callback failed")
                result.set_exception(exc)

        loop.call_soon_threadsafe(request)
        return result.result(timeout=HANDLER_TIMEOUT)

    return handle
