"""Process-wide per-provider LLM request gate (#109889).

Several agents / subagents / profiles sharing one provider account with a hard
concurrency limit (e.g. an external API that allows a single in-flight request
per key) otherwise collide into 429 storms: every collision retries, and the
retries multiply the load. This gate makes that contention local and explicit —
requests queue in-process instead of being rejected upstream:

    providers:
      my-endpoint:
        max_in_flight: 1     # 0 / unset = unlimited (default: no gate at all)

One gate per provider id per process, shared by every path that issues a
provider request (main agent loop, auxiliary/relay clients, streaming), so all
models on the same provider/account draw on one budget. Sync and async callers
share it too: a thread driving the main loop and an event loop driving an
auxiliary call cannot both be "the one request" the provider allows.

Re-entrant per execution context (thread / asyncio task): a provider request
nested inside another request to the same provider — an aux call made while a
main-path stream is in flight — runs without taking a second permit. Blocking on
a permit its own caller holds would deadlock at ``max_in_flight: 1``.

Unconfigured providers take the fast path: the gate is never built and the block
is skipped, so default behavior is unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

_GATES_LOCK = threading.Lock()
_GATES: Dict[str, "_ProviderGate"] = {}
# Providers whose malformed limit was already reported; warn once per process.
_BAD_LIMIT_WARNED: set = set()
# Providers whose configured cap was already announced.
_ANNOUNCED: set = set()

# Providers whose permit the CURRENT execution context already holds, as
# ``{provider: depth}``. Copied on write so a context copied into a worker thread
# cannot corrupt the parent's counts.
_HELD: "contextvars.ContextVar[Optional[Dict[str, int]]]" = contextvars.ContextVar(
    "hermes_llm_gate_held", default=None)


class _Permit:
    """One held provider permit (or a re-entrant no-op when ``_depth`` > 0)."""

    __slots__ = ("_provider", "_gate", "_depth")

    def __init__(self, provider: str, gate: Optional["_ProviderGate"], depth: int) -> None:
        self._provider = provider
        self._gate = gate
        self._depth = depth

    def release(self) -> None:
        """Release the permit. Safe to call exactly once per acquire, from any thread."""
        _exit_context(self._provider, self._depth)
        if self._gate is not None:
            self._gate.release()


class _ProviderGate:
    """Counting gate allowing at most ``limit`` concurrent holders.

    Sync holders wait on a Condition; async holders wait on an :class:`asyncio.Event`
    their own loop drives — never blocking the loop and never parking a worker
    thread per waiter. Permits are only ever taken by a waiter that has already
    woken up, so a cancelled async waiter can leave without stranding a permit.
    """

    def __init__(self, provider: str, limit: int) -> None:
        self.provider = provider
        self.limit = limit
        self._cond = threading.Condition()
        self._in_flight = 0
        self._async_waiters: List[Tuple[asyncio.Event, Any]] = []

    # ── acquiring ───────────────────────────────────────────────────────
    def acquire(self) -> None:
        """Block until a permit is free (sync callers)."""
        with self._cond:
            if self._in_flight >= self.limit:
                logger.debug("%s: queued for provider concurrency permit (%d in flight, limit %d)",
                             self.provider, self._in_flight, self.limit)
            while self._in_flight >= self.limit:
                self._cond.wait()
            self._in_flight += 1

    async def acquire_async(self) -> None:
        """Await a permit without blocking the event loop."""
        loop = asyncio.get_running_loop()
        while True:
            event = asyncio.Event()
            with self._cond:
                if self._in_flight < self.limit:
                    self._in_flight += 1
                    return
                self._async_waiters.append((event, loop))
            try:
                await event.wait()
            finally:
                # Cancelled (or woken) while queued: never took a permit, so just
                # stop being a waiter. The waker may have consumed our entry already.
                with self._cond:
                    self._async_waiters = [w for w in self._async_waiters if w[0] is not event]

    def release(self) -> None:
        with self._cond:
            if self._in_flight <= 0:
                raise ValueError(
                    f"LLM concurrency permit released without a matching acquire (provider {self.provider!r})")
            self._in_flight -= 1
            self._wake_locked()
            self._cond.notify()

    def set_limit(self, limit: int) -> None:
        """Apply a reconfigured limit in place — never rebuilt, so in-flight permits survive."""
        with self._cond:
            self.limit = limit
            self._wake_locked()
            self._cond.notify_all()

    def _wake_locked(self) -> None:
        """Wake every waiter; they re-check the counter themselves under the lock.

        Broadcast rather than a strict FIFO hand-off: the herd is tiny (a handful of
        agents) and a woken-but-cancelled waiter must not be able to strand a permit.
        """
        waiters, self._async_waiters = self._async_waiters, []
        for event, loop in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:  # waiter's loop is already closed (shutdown)
                pass


def provider_max_in_flight(provider: Any) -> Optional[int]:
    """``providers.<id>.max_in_flight`` for ``provider``, or None when unlimited/unset.

    Malformed values (non-numeric, <= 0) warn once and mean unlimited, matching the
    "unconfigured" path.
    """
    provider_id = str(provider or "").strip()
    if not provider_id:
        return None
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.config_providers import find_provider_entry

        config = load_config_readonly()
        providers = config.get("providers") if isinstance(config, dict) else None
        _stored, entry = find_provider_entry(providers, provider_id)
    except Exception as exc:  # config must never break a request
        logger.debug("providers.%s.max_in_flight lookup failed: %s", provider_id, exc)
        return None
    raw = (entry or {}).get("max_in_flight")
    if raw is None or raw == "":
        return None
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0:
        with _GATES_LOCK:
            first = provider_id not in _BAD_LIMIT_WARNED
            _BAD_LIMIT_WARNED.add(provider_id)
        if first:
            logger.warning("providers.%s.max_in_flight=%r is not a positive integer — ignoring "
                           "(provider concurrency stays unlimited)", provider_id, raw)
        return None
    return limit


def _gate_for(provider: Any) -> Optional["_ProviderGate"]:
    """Cached gate for ``provider``, or None when no limit is configured."""
    limit = provider_max_in_flight(provider)
    if limit is None:
        return None
    provider_id = str(provider).strip()
    with _GATES_LOCK:
        gate = _GATES.get(provider_id)
        if gate is None:
            gate = _GATES[provider_id] = _ProviderGate(provider_id, limit)
        elif gate.limit != limit:
            gate.set_limit(limit)
        if provider_id not in _ANNOUNCED:
            _ANNOUNCED.add(provider_id)
            logger.info("Provider %s: capping concurrent LLM requests at %d "
                        "(providers.%s.max_in_flight)", provider_id, limit, provider_id)
        return gate


def _enter_context(provider: str) -> int:
    """Record that this context holds ``provider``; returns the previous depth."""
    counts = dict(_HELD.get() or {})
    depth = counts.get(provider, 0)
    counts[provider] = depth + 1
    _HELD.set(counts)
    return depth


def _exit_context(provider: str, depth: int) -> None:
    counts = dict(_HELD.get() or {})
    if depth:
        counts[provider] = depth
    else:
        counts.pop(provider, None)
    _HELD.set(counts)


def acquire_provider_permit(provider: Any) -> Optional[_Permit]:
    """Take a provider permit (blocking), or return None when unlimited.

    Callers that hand the work to another thread or return a live stream must call
    ``permit.release()`` themselves — see :func:`release_permit_when_stream_ends`.
    """
    gate = _gate_for(provider)
    if gate is None:
        return None
    depth = _enter_context(gate.provider)
    if not depth:
        gate.acquire()
    return _Permit(gate.provider, gate if not depth else None, depth)


async def async_acquire_provider_permit(provider: Any) -> Optional[_Permit]:
    """Async twin of :func:`acquire_provider_permit`; cancellable while queued."""
    gate = _gate_for(provider)
    if gate is None:
        return None
    depth = _enter_context(gate.provider)
    if not depth:
        try:
            await gate.acquire_async()
        except BaseException:
            _exit_context(gate.provider, depth)
            raise
    return _Permit(gate.provider, gate if not depth else None, depth)


@contextlib.contextmanager
def provider_request_slot(provider: Any) -> Iterator[None]:
    """Hold one provider permit for the duration of the block (no-op when unlimited).

    Use for anything that can be bracketed synchronously: non-streaming requests and
    whole streaming calls that return only once the stream is finished.
    """
    permit = acquire_provider_permit(provider)
    try:
        yield
    finally:
        if permit is not None:
            permit.release()


@contextlib.asynccontextmanager
async def async_provider_request_slot(provider: Any) -> Any:
    """Async twin of :func:`provider_request_slot`."""
    permit = await async_acquire_provider_permit(provider)
    try:
        yield
    finally:
        if permit is not None:
            permit.release()


def release_permit_when_stream_ends(stream: Any, permit: Optional[_Permit]) -> Iterator[Any]:
    """Iterate ``stream``, releasing ``permit`` when it ends or its consumer gives up.

    The streaming counterpart of the context managers: the permit lives exactly as
    long as the provider stream does, so a queued request cannot start while the
    provider is still sending this one.
    """
    if permit is None:
        yield from stream
        return
    try:
        yield from stream
    finally:
        permit.release()


def reset_provider_gates() -> None:
    """Drop cached gates and context bookkeeping (test helper)."""
    with _GATES_LOCK:
        _GATES.clear()
        _ANNOUNCED.clear()
        _BAD_LIMIT_WARNED.clear()
    _HELD.set(None)
