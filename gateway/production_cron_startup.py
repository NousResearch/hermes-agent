"""Fail-closed activation gate for the production in-process cron scheduler.

The production gateway must not advertise readiness while cron startup is an
unobserved future side effect, and it must not execute a due job before the
gateway has published its final systemd READY receipt.  This module supplies
that mechanical ordering boundary without deciding which jobs should run.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class ProductionCronStartupError(RuntimeError):
    """The exact production cron thread did not reach its required state."""


class ProductionCronActivationGate:
    """Start the exact built-in cron provider parked, then activate it once.

    ``prepare()`` starts a thread, exercises the real ticker-heartbeat store,
    and proves two parked in-process heartbeats before it returns.  The wrapped
    provider is not entered until ``activate()`` is called.  Production calls
    ``activate()`` only after systemd READY has been published.  ``stop()``
    wakes a parked thread so failed startup cannot leak a daemon forever.
    """

    _PARK_POLL_SECONDS = 0.025

    def __init__(
        self,
        *,
        provider: Any,
        stop_event: threading.Event,
        adapters: Any,
        loop: Any,
        interval: int = 60,
    ) -> None:
        from cron.scheduler_provider import InProcessCronScheduler

        if type(provider) is not InProcessCronScheduler:
            raise ProductionCronStartupError(
                "production cron requires the exact built-in provider"
            )
        if not isinstance(stop_event, threading.Event):
            raise TypeError("stop_event must be threading.Event")
        if type(interval) is not int or interval <= 0:
            raise ValueError("interval must be a positive integer")

        self.provider = provider
        self.stop_event = stop_event
        self.adapters = adapters
        self.loop = loop
        self.interval = interval

        self._condition = threading.Condition()
        self._activation_requested = threading.Event()
        self._activated = threading.Event()
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None
        self._parked_heartbeats = 0
        self._failure: BaseException | None = None

    @property
    def thread(self) -> threading.Thread | None:
        return self._thread

    @property
    def parked_heartbeats(self) -> int:
        with self._condition:
            return self._parked_heartbeats

    @property
    def activated(self) -> bool:
        return self._activated.is_set()

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def _record_parked_heartbeat(self) -> None:
        with self._condition:
            self._parked_heartbeats += 1
            self._condition.notify_all()

    def _record_failure(self, exc: BaseException) -> None:
        with self._condition:
            self._failure = exc
            self._condition.notify_all()

    def _run(self) -> None:
        try:
            # Prove that the exact cron store/heartbeat path is writable from
            # this real thread before READY.  This is liveness metadata only;
            # no due-job scan or task execution occurs while the gate is
            # parked.  The built-in provider records the same heartbeat again
            # when activation releases it.
            from cron.jobs import record_ticker_heartbeat

            record_ticker_heartbeat()
            while (
                not self.stop_event.is_set()
                and not self._activation_requested.is_set()
            ):
                self._record_parked_heartbeat()
                self._activation_requested.wait(self._PARK_POLL_SECONDS)

            if self.stop_event.is_set():
                return

            self._activated.set()
            with self._condition:
                self._condition.notify_all()
            self.provider.start(
                self.stop_event,
                adapters=self.adapters,
                loop=self.loop,
                interval=self.interval,
            )
        except BaseException as exc:
            self._record_failure(exc)
        finally:
            self._finished.set()
            with self._condition:
                self._condition.notify_all()

    def _raise_if_failed_unlocked(self) -> None:
        if self._failure is not None:
            raise ProductionCronStartupError(
                "production cron thread failed"
            ) from self._failure

    def prepare(self, *, timeout: float = 5.0) -> None:
        """Start and prove a live parked thread without entering the provider."""

        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        with self._condition:
            if self._thread is not None:
                raise ProductionCronStartupError(
                    "production cron gate was already prepared"
                )
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="production-cron-gated",
            )
            self._thread.start()
            deadline = time.monotonic() + float(timeout)
            while self._parked_heartbeats < 2:
                self._raise_if_failed_unlocked()
                if self._finished.is_set():
                    raise ProductionCronStartupError(
                        "production cron thread exited before activation"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProductionCronStartupError(
                        "production cron thread did not park"
                    )
                self._condition.wait(timeout=remaining)

            if not self._thread.is_alive():
                raise ProductionCronStartupError(
                    "production cron thread is not alive while parked"
                )

    def activate(self, *, timeout: float = 5.0) -> None:
        """Release the parked thread exactly once after READY publication."""

        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        with self._condition:
            if self._thread is None or self._parked_heartbeats < 2:
                raise ProductionCronStartupError(
                    "production cron gate was not prepared"
                )
            if self._activation_requested.is_set():
                raise ProductionCronStartupError(
                    "production cron gate was already activated"
                )
            if self.stop_event.is_set():
                raise ProductionCronStartupError(
                    "production cron stop was requested before activation"
                )
            self._activation_requested.set()
            deadline = time.monotonic() + float(timeout)
            while not self._activated.is_set():
                self._raise_if_failed_unlocked()
                if self._finished.is_set():
                    raise ProductionCronStartupError(
                        "production cron thread exited during activation"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProductionCronStartupError(
                        "production cron activation timed out"
                    )
                self._condition.wait(timeout=remaining)

            self._raise_if_failed_unlocked()
            if self._thread is None or not self._thread.is_alive():
                raise ProductionCronStartupError(
                    "production cron thread exited during activation"
                )

    def stop(self) -> None:
        """Stop the provider or wake a still-parked startup thread."""

        self.stop_event.set()
        self._activation_requested.set()
        try:
            self.provider.stop()
        finally:
            with self._condition:
                self._condition.notify_all()


__all__ = [
    "ProductionCronActivationGate",
    "ProductionCronStartupError",
]
