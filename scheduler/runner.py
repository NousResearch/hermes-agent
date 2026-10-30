"""Run untrusted scheduled handlers without letting them wedge the turn.

The contract matches the hot-path hook dispatcher (``hermes_cli/plugins_dispatch.py``): a daemon
worker plus ``done.wait(timeout)``, the worker is **abandoned without joining** on timeout, the
same callback is suppressed for a window afterwards, and abandoned workers are capped. Async
handlers are cancelled (``asyncio.wait_for``) rather than abandoned.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("hermes.scheduler")

DEFAULT_TIMEOUT_S = 0.05
DEFAULT_SUPPRESSION_S = 60.0
DEFAULT_MAX_ABANDONED = 3


@dataclass(frozen=True)
class CallbackOutcome:
    status: str  # "ok" | "error" | "timeout" | "skipped"
    reason: str = ""
    duration_ms: int = 0
    error: str | None = None
    value: Any = None


class BoundedCallbackRunner:
    """Per-engine runner; one instance per scheduler engine."""

    def __init__(
        self, *, timeout_s: float = DEFAULT_TIMEOUT_S, suppression_s: float = DEFAULT_SUPPRESSION_S,
        max_abandoned: int = DEFAULT_MAX_ABANDONED,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.suppression_s = float(suppression_s)
        self.max_abandoned = int(max_abandoned)
        self._lock = threading.Lock()
        self._running: dict[tuple, object] = {}
        self._abandoned: dict[tuple, set] = {}
        self._suppressed_until: dict[tuple, float] = {}

    def run(
        self, *, task_key: tuple[str, str], session_id: str | None, seq: Any,
        fn: Callable[[], Any], timeout_s: float | None = None,
    ) -> CallbackOutcome:
        suppression_key = (task_key, session_id)
        gate_key = (*suppression_key, seq)
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        token = object()
        with self._lock:
            until = self._suppressed_until.get(suppression_key)
            if gate_key in self._running:
                return CallbackOutcome("skipped", reason="still_running")
            if until is not None and until > time.monotonic():
                return CallbackOutcome("skipped", reason="suppressed")
            abandoned = self._abandoned.get(suppression_key)
            if abandoned and len(abandoned) >= self.max_abandoned:
                return CallbackOutcome("skipped", reason="abandoned_cap")
            self._suppressed_until.pop(suppression_key, None)
            self._running[gate_key] = token

        started = time.monotonic()
        done = threading.Event()
        outcome: dict[str, Any] = {}
        failure: dict[str, BaseException] = {}
        context = contextvars.copy_context()

        def _release() -> None:
            with self._lock:
                if self._running.get(gate_key) is token:
                    self._running.pop(gate_key, None)
                    workers = self._abandoned.get(suppression_key)
                    if workers is not None:
                        workers.discard(gate_key)
                        if not workers:
                            self._abandoned.pop(suppression_key, None)

        def _runner() -> None:
            try:
                result = context.run(fn)
                if inspect.isawaitable(result):
                    result = asyncio.run(self._await_bounded(result, timeout))
                outcome["value"] = result
            except TimeoutError as exc:
                failure["timeout"] = exc
            except BaseException as exc:  # untrusted handler, reported as an outcome
                failure["exc"] = exc
            finally:
                _release()
                done.set()

        thread = threading.Thread(
            target=_runner, name=f"hermes-sched-{task_key[1]}"[:40], daemon=True)
        try:
            thread.start()
        except RuntimeError as exc:
            _release()
            return CallbackOutcome("skipped", reason="no_thread", error=str(exc))
        # Async handlers are cancelled inside the worker; give cancellation a short grace.
        wait_s = timeout + 0.5 if inspect.iscoroutinefunction(fn) else timeout
        if not done.wait(timeout=wait_s):  # never join: the handler may be wedged
            with self._lock:
                self._suppressed_until[suppression_key] = time.monotonic() + self.suppression_s
                if self._running.get(gate_key) is token:
                    self._abandoned.setdefault(suppression_key, set()).add(gate_key)
            logger.warning("scheduler task %s timed out after %.0fms; worker abandoned",
                           task_key[1], timeout * 1000)
            return CallbackOutcome("timeout", reason="timeout", duration_ms=self._elapsed(started))
        if "timeout" in failure:
            logger.warning("scheduler task %s timed out after %.0fms and was cancelled",
                           task_key[1], timeout * 1000)
            return CallbackOutcome("timeout", reason="cancelled", duration_ms=self._elapsed(started))
        if "exc" in failure:
            exc = failure["exc"]
            logger.warning("scheduler task %s raised: %s", task_key[1], exc)
            return CallbackOutcome("error", reason=type(exc).__name__, error=str(exc)[:2000],
                                   duration_ms=self._elapsed(started))
        return CallbackOutcome("ok", duration_ms=self._elapsed(started), value=outcome.get("value"))

    @staticmethod
    async def _await_bounded(awaitable: Any, timeout: float) -> Any:
        return await asyncio.wait_for(awaitable, timeout)

    @staticmethod
    def _elapsed(started: float) -> int:
        return int((time.monotonic() - started) * 1000)
