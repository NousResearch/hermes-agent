"""Per-provider request admission: ``max_in_flight`` and ``requests_per_minute``.

Every physical provider request takes a permit here before it is sent (see
``relay_llm`` and ``interruptible_streaming_api_call``); a stream keeps its permit
until it is exhausted, closed or abandoned. Budgets are process-local and keyed by
(Hermes home, provider id): profiles multiplexed into one process never share or
throttle each other's budget, matching how each profile reads its own config.

Waiting is interruptible, and a waiter that gives up never holds a permit. When
the main loop and auxiliary tasks both wait on one provider, admission alternates
between the two queues so neither can starve the other.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

# How often a blocked waiter re-checks its interrupt flag.
_POLL_SECONDS = 0.1
_ASYNC_POLL_SECONDS = 0.02
# x-ratelimit-*-requests describes a one-minute window; a larger reset is not that window.
_HEADER_WINDOW_CAP_SECONDS = 60.0

MAIN, AUXILIARY = "main", "auxiliary"

# Gate keys whose permit the current execution context already holds. A provider call
# nested inside another call to the same provider (a Codex-shim auxiliary client opening
# a Relay stream inside the outer callback, a stream worker thread) is the same physical
# request: taking a second permit would deadlock at ``max_in_flight: 1``.
_held_gates: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "hermes_llm_admission_held", default=frozenset()
)
_gates: dict[str, "_Gate"] = {}
_gates_lock = threading.Lock()


@dataclass(frozen=True)
class _Limits:
    max_in_flight: int | None = None
    requests_per_minute: float | None = None
    # ``requests_per_minute`` is set (a number or ``auto``): also honor x-ratelimit headers.
    pace_from_headers: bool = False

    @property
    def active(self) -> bool:
        return bool(self.max_in_flight or self.requests_per_minute or self.pace_from_headers)


def _provider_identity(provider: Any) -> str:
    raw = str(provider or "").strip().lower()
    if not raw or raw.startswith("custom:"):
        return raw
    from hermes_cli.providers import normalize_provider

    return normalize_provider(raw)


def _positive(value: Any, cast: Callable[[Any], Any]) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = cast(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and math.isfinite(number) else None


def _configured_limits(provider: Any, identity: str) -> _Limits | None:
    """``providers.<id>.max_in_flight`` / ``requests_per_minute`` for the active profile."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:
        return None
    providers = config.get("providers") if isinstance(config, dict) else None
    if not isinstance(providers, dict):
        return None
    raw = str(provider or "").strip().lower()
    candidates = [raw, identity]
    if identity.startswith("custom:"):
        candidates.insert(0, identity.partition(":")[2])
    entry = next((providers[key] for key in candidates if isinstance(providers.get(key), dict)), None)
    if entry is None:
        return None
    rpm_raw = entry.get("requests_per_minute")
    auto = isinstance(rpm_raw, str) and rpm_raw.strip().lower() == "auto"
    rpm = None if auto else _positive(rpm_raw, float)
    limits = _Limits(
        max_in_flight=_positive(entry.get("max_in_flight"), int),
        requests_per_minute=rpm,
        pace_from_headers=auto or rpm is not None,
    )
    return limits if limits.active else None


def _profile_key() -> str:
    from hermes_constants import get_hermes_home

    return str(get_hermes_home())


class _Waiter:
    __slots__ = ("role",)

    def __init__(self, role: str) -> None:
        self.role = role


class _Gate:
    """One (profile, provider) budget: an in-flight cap plus a start-time pacer."""

    def __init__(self, key: str, limits: _Limits) -> None:
        self.key = key
        self.limits = limits
        self._cond = threading.Condition()
        self._in_flight = 0
        self._queues: dict[str, deque[_Waiter]] = {MAIN: deque(), AUXILIARY: deque()}
        self._last_served = AUXILIARY  # on a tie the main loop goes first
        self._next_start = 0.0  # monotonic: earliest time the next request may start
        self._header_interval = 0.0
        self._header_until = 0.0

    def reconfigure(self, limits: _Limits) -> None:
        with self._cond:
            if limits != self.limits:
                self.limits = limits
                self._cond.notify_all()

    def _head(self) -> _Waiter | None:
        main, aux = self._queues[MAIN], self._queues[AUXILIARY]
        if main and aux:
            return (aux if self._last_served == MAIN else main)[0]
        queue = main or aux
        return queue[0] if queue else None

    def _interval(self, now: float) -> float:
        rpm = self.limits.requests_per_minute
        interval = 60.0 / rpm if rpm else 0.0
        if self.limits.pace_from_headers and now < self._header_until:
            interval = max(interval, self._header_interval)
        return interval

    def _try_admit(self, waiter: _Waiter) -> float | None:
        """Admit ``waiter`` (returns None) or return how long it should wait. Lock held."""
        if self._head() is not waiter:
            return _POLL_SECONDS
        cap = self.limits.max_in_flight
        if cap and self._in_flight >= cap:
            return _POLL_SECONDS
        now = time.monotonic()
        delay = self._next_start - now
        if delay > 0:
            return delay
        self._queues[waiter.role].popleft()
        self._in_flight += 1
        self._last_served = waiter.role
        self._next_start = now + self._interval(now)
        self._cond.notify_all()
        return None

    def _abandon(self, waiter: _Waiter) -> None:
        with self._cond:
            with contextlib.suppress(ValueError):
                self._queues[waiter.role].remove(waiter)
            self._cond.notify_all()

    def acquire(self, role: str, cancelled: Callable[[], bool] | None) -> None:
        waiter = _Waiter(role)
        with self._cond:
            self._queues[role].append(waiter)
            try:
                while True:
                    if cancelled is not None and cancelled():
                        raise InterruptedError("Provider admission wait interrupted")
                    wait = self._try_admit(waiter)
                    if wait is None:
                        return
                    self._cond.wait(timeout=min(wait, _POLL_SECONDS))
            except BaseException:
                with contextlib.suppress(ValueError):
                    self._queues[role].remove(waiter)
                self._cond.notify_all()
                raise

    async def acquire_async(self, role: str, cancelled: Callable[[], bool] | None) -> None:
        waiter = _Waiter(role)
        with self._cond:
            self._queues[role].append(waiter)
        try:
            while True:
                if cancelled is not None and cancelled():
                    raise InterruptedError("Provider admission wait interrupted")
                with self._cond:
                    wait = self._try_admit(waiter)
                if wait is None:
                    return
                await asyncio.sleep(min(wait, _ASYNC_POLL_SECONDS))
        except BaseException:
            self._abandon(waiter)
            raise

    def release(self) -> None:
        with self._cond:
            self._in_flight = max(0, self._in_flight - 1)
            self._cond.notify_all()

    def note_request_window(self, limit: int, remaining: int, reset_seconds: float) -> None:
        """Spread the provider-reported remaining requests over the rest of its window."""
        if limit <= 0 or reset_seconds <= 0:
            return
        now = time.monotonic()
        window = min(reset_seconds, _HEADER_WINDOW_CAP_SECONDS)
        with self._cond:
            self._header_until = now + window
            if remaining <= 0:
                self._header_interval = window
                self._next_start = max(self._next_start, now + window)
            else:
                self._header_interval = window / remaining
            self._cond.notify_all()


def _gate_for(provider: Any) -> _Gate | None:
    identity = _provider_identity(provider)
    if not identity:
        return None
    limits = _configured_limits(provider, identity)
    key = f"{_profile_key()}\0{identity}"
    with _gates_lock:
        gate = _gates.get(key)
        if limits is None:
            # Unset later: keep the gate object for permits still in flight, stop gating new work.
            if gate is not None:
                gate.reconfigure(_Limits())
            return None
        if gate is None:
            gate = _gates[key] = _Gate(key, limits)
            logger.info("Provider %s: admission limits %s", identity, limits)
        else:
            gate.reconfigure(limits)
        return gate


def provider_has_limit(provider: Any) -> bool:
    return _gate_for(provider) is not None


def _role(role: str | None) -> str:
    return AUXILIARY if role == AUXILIARY else MAIN


class ProviderPermit:
    """One held admission (or a no-op when unlimited or re-entrant); release is idempotent."""

    __slots__ = ("key", "_gate", "_released", "_lock")

    def __init__(self, key: str = "", gate: _Gate | None = None) -> None:
        self.key, self._gate = key, gate
        self._released = False
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def active(self) -> Iterator[None]:
        """Mark the enclosed code as running under this permit (nested calls re-enter)."""
        if not self.key:
            yield
            return
        token = _held_gates.set(_held_gates.get() | {self.key})
        try:
            yield
        finally:
            _held_gates.reset(token)

    def release(self) -> None:
        if self._gate is None:
            return
        with self._lock:
            if self._released:
                return
            self._released = True
        self._gate.release()


def acquire_provider_slot(
    provider: Any, *, role: str | None = None, cancelled: Callable[[], bool] | None = None,
) -> ProviderPermit:
    gate = _gate_for(provider)
    if gate is None:
        return ProviderPermit()
    if gate.key in _held_gates.get():
        return ProviderPermit(gate.key)
    gate.acquire(_role(role), cancelled)
    return ProviderPermit(gate.key, gate)


async def acquire_provider_slot_async(
    provider: Any, *, role: str | None = None, cancelled: Callable[[], bool] | None = None,
) -> ProviderPermit:
    gate = _gate_for(provider)
    if gate is None:
        return ProviderPermit()
    if gate.key in _held_gates.get():
        return ProviderPermit(gate.key)
    await gate.acquire_async(_role(role), cancelled)
    return ProviderPermit(gate.key, gate)


@contextlib.contextmanager
def provider_slot(
    provider: Any, *, role: str | None = None, cancelled: Callable[[], bool] | None = None,
) -> Iterator[None]:
    permit = acquire_provider_slot(provider, role=role, cancelled=cancelled)
    try:
        with permit.active():
            yield
    finally:
        permit.release()


@contextlib.asynccontextmanager
async def provider_slot_async(
    provider: Any, *, role: str | None = None, cancelled: Callable[[], bool] | None = None,
):
    permit = await acquire_provider_slot_async(provider, role=role, cancelled=cancelled)
    try:
        with permit.active():
            yield
    finally:
        permit.release()


def note_rate_limit_state(provider: Any, state: Any) -> None:
    """Seed pacing from a parsed ``RateLimitState`` (``agent.rate_limit_tracker``)."""
    gate = _gate_for(provider)
    if gate is None or not gate.limits.pace_from_headers:
        return
    bucket = getattr(state, "requests_min", None)
    if bucket is None:
        return
    gate.note_request_window(int(bucket.limit), int(bucket.remaining), float(bucket.remaining_seconds_now))


def _reset_provider_limiters() -> None:
    """Drop cached gates (test helper)."""
    with _gates_lock:
        _gates.clear()
