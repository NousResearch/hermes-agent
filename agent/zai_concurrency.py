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
- User-facing controls live under ``providers.zai`` in the active profile's
  ``config.yaml``. No non-secret environment variables are consulted.
- Non-Z.AI providers and a disabled gate pass through unchanged.
- Saturated calls queue instead of bypassing the cap. Interactive waits poll
  for an interrupt. A positive acquire timeout raises locally without sending
  a request; the default timeout of zero waits until a slot or an interrupt.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from math import isfinite
from typing import Any, Optional
from urllib.parse import urlparse

# Field reports show the Coding Plan endpoint remains stable at two concurrent
# calls while four can still trigger 1305 overload responses.
_DEFAULT_ZAI_MAX_CONCURRENT = 2
_DEFAULT_ZAI_ACQUIRE_TIMEOUT = 0.0
_ZAI_PROVIDER_CONFIG_KEY = "zai"


def _non_negative_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, float) and not value.is_integer():
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _non_negative_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if isfinite(parsed) and parsed >= 0 else default


def _read_configured_limits() -> tuple[int, float]:
    """Read the active profile's Z.AI concurrency settings from config.yaml."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:
        config = {}

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(_ZAI_PROVIDER_CONFIG_KEY, {})
        if isinstance(providers, dict)
        else {}
    )
    if not isinstance(provider_config, dict):
        provider_config = {}

    max_concurrent = _non_negative_int(
        provider_config.get("max_concurrent"),
        _DEFAULT_ZAI_MAX_CONCURRENT,
    )
    acquire_timeout = _non_negative_float(
        provider_config.get("acquire_timeout_seconds"),
        _DEFAULT_ZAI_ACQUIRE_TIMEOUT,
    )
    return max_concurrent, acquire_timeout


# Loaded lazily with this module, after profile selection has set HERMES_HOME.
# 0 disables the gate entirely.
_ZAI_MAX_CONCURRENT, _ZAI_ACQUIRE_TIMEOUT = _read_configured_limits()

# 0 waits until a slot becomes available. A positive value raises
# ZaiConcurrencyTimeout after the configured number of seconds.
_ZAI_ACQUIRE_POLL_INTERVAL = 0.1

# Deliberately asymmetric: Zhipu serves API traffic across bigmodel.cn, so the
# whole registrable domain is gated, while on z.ai only the api. subtree is —
# gating chat.z.ai-style non-API properties would throttle traffic that never
# touches the API concurrency quota.
_ZAI_HOST_SUFFIXES = ("api.z.ai", "bigmodel.cn")


def _resolved_hostname(base_url: Any) -> str:
    raw = str(base_url or "").strip()
    if not raw:
        return ""
    # An opaque ``https:host/path`` form (no ``//``) hides the host in the
    # path; peel the scheme so the host still parses. Only web schemes are
    # peeled — a bare ``host:port`` must not be mistaken for one.
    raw = re.sub(r"^(?:https?|wss?):(?!//)", "", raw, flags=re.IGNORECASE)
    # A leading scheme (``https://…``) or protocol-relative prefix (``//…``)
    # lets urlparse read the netloc directly. Anything else is a bare
    # ``host[:port]/path`` — even when a later path or query segment contains
    # ``://`` — and needs a ``//`` prefix so the host is parsed as netloc
    # rather than path.
    if raw.startswith("//") or re.match(r"^[a-z][a-z0-9+.-]*://", raw, re.IGNORECASE):
        parsed = urlparse(raw)
    else:
        parsed = urlparse(f"//{raw}")
    return str(parsed.hostname or "").rstrip(".").lower()


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
        """Return whether the resolved destination is Z.AI / Zhipu.

        Host-first: a resolvable base-URL host is authoritative, so the gate
        follows the *destination*, not the provider label. This keeps the
        module's promise that GLM models served by a local or third-party
        endpoint are not throttled — a ``zai``/``glm`` provider whose
        ``GLM_BASE_URL`` is overridden to a non-Z.AI host is correctly left
        ungated. The provider name is consulted only when no host can be parsed
        (empty or malformed base URL), where the built-in Z.AI provider is the
        intended target.
        """
        host = _resolved_hostname(base_url)
        if host:
            return any(
                host == suffix or host.endswith(f".{suffix}")
                for suffix in _ZAI_HOST_SUFFIXES
            )
        provider_name = str(provider or "").strip().lower()
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


def configured_acquire_timeout() -> float:
    """Return the configured Z.AI slot-acquire timeout in seconds."""
    return _ZAI_ACQUIRE_TIMEOUT


def _reset_for_tests(max_concurrent: int, timeout: float) -> None:
    """Rebuild the process-local gate with explicit settings."""
    global _ZAI_MAX_CONCURRENT, _ZAI_ACQUIRE_TIMEOUT, _gate
    _ZAI_MAX_CONCURRENT = max(0, max_concurrent)
    _ZAI_ACQUIRE_TIMEOUT = max(0.0, timeout)
    _gate = _ZaiConcurrencyGate()
