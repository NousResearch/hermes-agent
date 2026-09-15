"""Process-wide provider request concurrency limits."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
from dataclasses import dataclass, field
from typing import Iterator


_active_provider_slots: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "active_llm_provider_slots", default=frozenset()
)
_provider_limiters: dict[str, threading.BoundedSemaphore] = {}
_provider_limiters_lock = threading.Lock()


def _provider_identity(provider: str) -> str:
    raw = str(provider or "").strip().lower()
    if raw.startswith("custom:"):
        return raw
    from hermes_cli.providers import normalize_provider

    return normalize_provider(raw)


def _configured_max_in_flight(provider: str) -> int | None:
    """Read ``providers.<id>.max_in_flight`` as a positive integer."""
    identity = _provider_identity(provider)
    if not identity:
        return None
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:
        return None
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    if not isinstance(providers, dict):
        return None

    raw = str(provider or "").strip().lower()
    candidates = [raw, identity]
    if identity.startswith("custom:"):
        candidates.insert(0, identity.partition(":")[2])
    entry = next(
        (providers[key] for key in candidates if key in providers and isinstance(providers[key], dict)),
        None,
    )
    if entry is None:
        return None
    value = entry.get("max_in_flight")
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


def _provider_limiter(provider: str) -> tuple[str, threading.BoundedSemaphore | None]:
    """Return the provider's process-lifetime limiter, initialized on first use."""
    identity = _provider_identity(provider)
    if not identity:
        return identity, None
    with _provider_limiters_lock:
        if identity in _provider_limiters:
            return identity, _provider_limiters[identity]
        limit = _configured_max_in_flight(provider)
        if limit is None:
            return identity, None
        limiter = _provider_limiters[identity] = threading.BoundedSemaphore(limit)
        return identity, limiter


def provider_has_limit(provider: str) -> bool:
    return _provider_limiter(provider)[1] is not None


@dataclass
class ProviderPermit:
    """One idempotently releasable provider slot."""

    identity: str
    limiter: threading.BoundedSemaphore | None = None
    _released: bool = False
    _release_lock: threading.Lock = field(default_factory=threading.Lock)

    @contextlib.contextmanager
    def active(self) -> Iterator[None]:
        """Mark only the physical provider callback as re-entrant."""
        if self.limiter is None:
            yield
            return
        token = _active_provider_slots.set(_active_provider_slots.get() | {self.identity})
        try:
            yield
        finally:
            _active_provider_slots.reset(token)

    def release(self) -> None:
        if self.limiter is None:
            return
        with self._release_lock:
            if self._released:
                return
            self._released = True
            self.limiter.release()


def acquire_provider_slot(provider: str, *, cancelled=None) -> ProviderPermit:
    identity, limiter = _provider_limiter(provider)
    if limiter is None or identity in _active_provider_slots.get():
        return ProviderPermit(identity)
    while not limiter.acquire(timeout=0.1):
        if cancelled is not None and cancelled():
            raise InterruptedError("Provider concurrency wait interrupted")
    if cancelled is not None and cancelled():
        limiter.release()
        raise InterruptedError("Provider concurrency wait interrupted")
    return ProviderPermit(identity, limiter)


async def acquire_provider_slot_async(provider: str, *, cancelled=None) -> ProviderPermit:
    identity, limiter = _provider_limiter(provider)
    if limiter is None or identity in _active_provider_slots.get():
        return ProviderPermit(identity)
    # A non-blocking acquire loop stays cancellation-safe: there is no background
    # worker that can acquire after its cancelled coroutine has stopped owning it.
    while not limiter.acquire(blocking=False):
        if cancelled is not None and cancelled():
            raise InterruptedError("Provider concurrency wait interrupted")
        await asyncio.sleep(0.01)
    if cancelled is not None and cancelled():
        limiter.release()
        raise InterruptedError("Provider concurrency wait interrupted")
    return ProviderPermit(identity, limiter)


@contextlib.contextmanager
def provider_slot(provider: str, *, cancelled=None) -> Iterator[None]:
    permit = acquire_provider_slot(provider, cancelled=cancelled)
    try:
        with permit.active():
            yield
    finally:
        permit.release()


@contextlib.asynccontextmanager
async def provider_slot_async(provider: str, *, cancelled=None):
    permit = await acquire_provider_slot_async(provider, cancelled=cancelled)
    try:
        with permit.active():
            yield
    finally:
        permit.release()


def _reset_provider_limiters() -> None:
    """Drop cached provider limiters (test helper)."""
    with _provider_limiters_lock:
        _provider_limiters.clear()
