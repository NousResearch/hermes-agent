"""Presentation lifetimes for canonical cron executions; never an execution gate.

The gateway owns presenters and immutable admission identities. The invocation still
owns its decision poll, cancellation checks, and original tool invocation.
"""

from __future__ import annotations

import contextvars
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class ApprovalUnavailable(RuntimeError):
    """This presentation no longer represents a live worker invocation."""


@dataclass(frozen=True)
class ApprovalRequest:
    profile_home: str
    job_id: str
    execution_id: str
    session_id: str
    external_session_id: str
    call_id: str
    invocation_id: str
    deadline: float


class ApprovalLifetime:
    """Latched cancellation with the invocation's original monotonic deadline."""

    def __init__(self, deadline: float, live: Callable[[], bool], *, clock=time.monotonic):
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError("Invalid approval deadline")
        self.deadline = deadline
        self.ended = threading.Event()
        self.reason = None
        self._live = live
        self._clock = clock
        self._lock = threading.RLock()

    def close(self, reason: str) -> None:
        with self._lock:
            if self.reason is None:
                self.reason = reason
                self.ended.set()

    def check(self) -> None:
        with self._lock:
            if self._clock() >= self.deadline:
                self.close("expired")
            elif not self._live():
                self.close("disconnected")
            if self.reason is not None:
                raise ApprovalUnavailable(self.reason)

    @property
    def remaining_seconds(self) -> float:
        self.check()
        return max(0.0, self.deadline - self._clock())

    def request(self, operation: Callable):
        self.check()
        result = operation()
        self.check()
        return result


_registrations: dict[tuple[str, str], PresenterRegistration] = {}
_lock = threading.RLock()


class PresenterRegistration:
    """One profile/plugin-owned presenter generation, disposed on plugin unload."""

    def __init__(self, home: str, plugin_id: str, callback: Callable):
        if not callable(callback):
            raise TypeError("Approval presenter must be callable")
        self.home = str(Path(home).resolve())
        self.plugin_id = plugin_id
        self.callback = callback
        self.context = contextvars.copy_context()
        self.closed = threading.Event()
        with _lock:
            key = (self.home, plugin_id)
            previous = _registrations.get(key)
            if previous is not None:
                previous.dispose()
            _registrations[key] = self

    def dispose(self) -> None:
        with _lock:
            self.closed.set()
            key = (self.home, self.plugin_id)
            if _registrations.get(key) is self:
                _registrations.pop(key)


def presenters_for(home: str) -> dict[str, PresenterRegistration]:
    home = str(Path(home).resolve())
    with _lock:
        return {name: item for (profile, name), item in _registrations.items()
                if profile == home and not item.closed.is_set()}
