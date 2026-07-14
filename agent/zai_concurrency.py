"""Process-local concurrency gate for Z.AI / GLM model calls.

Z.AI Coding Plan can return HTTP 429 / code 1305 under concurrent load.
Subagent and Mixture-of-Agents fan-out can otherwise keep several long-lived
requests and their retries in flight at once. This module applies one strict
process-local bound to calls that resolve to the Z.AI endpoint.

The gate deliberately does not use the model name alone: GLM models can be
served by local or third-party endpoints that should not be throttled. A call
is gated only when its resolved provider or base URL identifies Z.AI.

Scope and limitations
---------------------
- The semaphore is process-local. Separate Hermes processes that share one API
  key each have their own limit, so fleet operators may need a lower
  per-process value.
- Non-Z.AI providers and a disabled gate pass through unchanged.
- Saturated calls queue instead of bypassing the cap. Interactive waits poll
  for an interrupt. A positive acquire timeout raises locally without sending
  a request; the default timeout of zero waits until a slot or an interrupt.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Optional

from utils import env_float, env_int

# Field reports show the Coding Plan endpoint remains stable at two concurrent
# calls while four can still trigger 1305 overload responses.
# 0 disables the gate entirely.
_ZAI_MAX_CONCURRENT = max(0, env_int("HERMES_ZAI_MAX_CONCURRENT", 2))

# 0 waits until a slot becomes available. A positive value raises
# ZaiConcurrencyTimeout after the configured number of seconds.
_ZAI_ACQUIRE_TIMEOUT = max(0.0, env_float("HERMES_ZAI_ACQUIRE_TIMEOUT_S", 0.0))
_ZAI_ACQUIRE_POLL_INTERVAL = 0.1

_ZAI_HOST_MARKERS = ("api.z.ai", "bigmodel.cn")


class ZaiConcurrencyTimeout(TimeoutError):
    """Raised when a configured Z.AI slot wait expires."""


class _ZaiConcurrencyGate:
    """Lazy, thread-safe holder for the process-wide Z.AI semaphore."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sem: Optional[threading.BoundedSemaphore] = None
        self._configured_max = _ZAI_MAX_CONCURRENT

    def _semaphore(self) -> Optional[threading.BoundedSemaphore]:
        if self._configured_max <= 0:
            return None
        if self._sem is None:
            with self._lock:
                if self._sem is None:
                    self._sem = threading.BoundedSemaphore(self._configured_max)
        return self._sem

    def is_zai(self, *, provider: Any, model: Any, base_url: Any) -> bool:
        """Return whether the resolved destination is Z.AI / Zhipu."""
        host = str(base_url or "").lower()
        if any(marker in host for marker in _ZAI_HOST_MARKERS):
            return True
        provider_name = str(provider or "").lower()
        return provider_name in {"zai", "zhipu", "glm", "z-ai", "z.ai"}


_gate = _ZaiConcurrencyGate()


def is_zai_request(*, provider: Any, model: Any, base_url: Any) -> bool:
    """Return whether a model call targets Z.AI / Zhipu."""
    return _gate.is_zai(provider=provider, model=model, base_url=base_url)


def acquire_zai_slot(
    *,
    provider: Any,
    model: Any,
    base_url: Any,
    interrupt_check: Optional[Callable[[], bool]] = None,
) -> "_SlotHandle":
    """Return a context manager that acquires a Z.AI slot on entry.

    Creating a handle without entering it never consumes capacity. Non-Z.AI
    requests and a disabled gate receive a no-op context manager.
    """
    if not is_zai_request(provider=provider, model=model, base_url=base_url):
        return _NoopHandle()
    sem = _gate._semaphore()
    if sem is None:
        return _NoopHandle()
    return _SlotHandle(
        sem,
        timeout=_ZAI_ACQUIRE_TIMEOUT,
        interrupt_check=interrupt_check,
    )


class _SlotHandle:
    """Acquire one semaphore slot on entry and release it on exit."""

    __slots__ = ("_acquired", "_entered", "_interrupt_check", "_sem", "_timeout")

    def __init__(
        self,
        sem: threading.BoundedSemaphore,
        *,
        timeout: float,
        interrupt_check: Optional[Callable[[], bool]],
    ) -> None:
        self._sem = sem
        self._timeout = timeout
        self._interrupt_check = interrupt_check
        self._entered = False
        self._acquired = False

    def __enter__(self) -> "_SlotHandle":
        if self._entered:
            raise RuntimeError("Z.AI concurrency slot handle cannot be re-entered")
        self._entered = True

        deadline = time.monotonic() + self._timeout if self._timeout > 0 else None
        while True:
            if self._interrupt_check is not None and self._interrupt_check():
                raise InterruptedError(
                    "Agent interrupted while waiting for a Z.AI concurrency slot"
                )

            wait_for = _ZAI_ACQUIRE_POLL_INTERVAL
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ZaiConcurrencyTimeout(
                        f"Timed out after {self._timeout:g}s waiting for a "
                        "Z.AI concurrency slot"
                    )
                wait_for = min(wait_for, remaining)

            if self._sem.acquire(timeout=wait_for):
                self._acquired = True
                if self._interrupt_check is not None and self._interrupt_check():
                    self._acquired = False
                    self._sem.release()
                    raise InterruptedError(
                        "Agent interrupted while waiting for a Z.AI concurrency slot"
                    )
                return self

    def __exit__(self, *exc: Any) -> None:
        if self._acquired:
            self._acquired = False
            self._sem.release()


class _NoopHandle:
    """Pass-through context manager for non-Z.AI or disabled calls."""

    __slots__ = ()

    def __enter__(self) -> "_NoopHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def configured_max_concurrent() -> int:
    """Return the configured Z.AI in-flight cap; zero means disabled."""
    return _ZAI_MAX_CONCURRENT


def _reset_for_tests(max_concurrent: int, timeout: float) -> None:
    """Rebuild the process-local gate with explicit settings."""
    global _ZAI_MAX_CONCURRENT, _ZAI_ACQUIRE_TIMEOUT, _gate
    _ZAI_MAX_CONCURRENT = max(0, max_concurrent)
    _ZAI_ACQUIRE_TIMEOUT = max(0.0, timeout)
    _gate = _ZaiConcurrencyGate()
