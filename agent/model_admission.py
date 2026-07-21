"""Process-wide admission control for outbound model requests.

The controller deliberately knows nothing about a particular model SDK.  A
caller acquires a permit immediately before starting a request and completes
it when the response (including a streaming response) is finished.  Keeping
that boundary here lets every transport share the same global and per-target
limits without holding a worker thread for async callers.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Generic, TypeVar
from urllib.parse import urlsplit, urlunsplit

__all__ = [
    "ModelAdmissionCancelled",
    "ModelAdmissionPermit",
    "ModelAdmissionRegistry",
    "ModelAdmissionSettings",
    "ModelAdmissionTimeout",
    "ModelTarget",
    "normalize_model_target",
]


class ModelAdmissionTimeout(TimeoutError):
    """Raised when a request cannot enter before its queue deadline."""


class ModelAdmissionCancelled(RuntimeError):
    """Raised when a queued synchronous request is cancelled."""


@dataclass(frozen=True, slots=True)
class ModelTarget:
    """A credential-free key for one provider endpoint and model."""

    provider: str
    base_url: str
    model: str


def _normalize_base_url(base_url: str | None) -> str:
    raw = "" if base_url is None else str(base_url).strip()
    if not raw:
        return ""

    # urlsplit only recognizes a network location when a scheme or ``//`` is
    # present.  The latter also lets us safely strip credentials from the
    # scheme-less endpoint forms accepted by some custom providers.
    has_scheme = "://" in raw
    try:
        parsed = urlsplit(raw if has_scheme else f"//{raw}")
        scheme = parsed.scheme.casefold() if has_scheme else ""
        host = (parsed.hostname or "").casefold().rstrip(".")
    except ValueError:
        # Admission must not make an already-invalid transport URL leak its
        # credentials through a parser exception.  A short digest keeps bad
        # endpoints isolated from each other while remaining safe to snapshot.
        digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
        return f"invalid-url:{digest[:16]}"

    if host:
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = parsed.port
        except ValueError:
            # An invalid port must not make diagnostics echo the original URL,
            # which may contain credentials.  The transport will report the
            # malformed endpoint separately when it attempts the request.
            port = None
        if port is not None and not (
            (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
        ):
            host = f"{host}:{port}"

    path = parsed.path.rstrip("/")
    if scheme:
        return urlunsplit((scheme, host, path, "", ""))
    if host:
        return f"{host}{path}"

    # Non-network schemes (for example a local test adapter) have no hostname.
    # urlunsplit still removes any query and fragment from their key.
    return urlunsplit((parsed.scheme.casefold(), "", path, "", ""))


def normalize_model_target(
    provider: str | None,
    base_url: str | None,
    model: str | None,
) -> ModelTarget:
    """Return a stable target key that never retains URL credentials."""

    return ModelTarget(
        provider=("" if provider is None else str(provider)).strip().casefold(),
        base_url=_normalize_base_url(base_url),
        model=("" if model is None else str(model)).strip().casefold(),
    )


def _snapshot_base_url(base_url: str) -> str:
    """Render only an endpoint origin; route paths may contain credentials."""

    if not base_url:
        return ""
    if base_url.startswith("invalid-url:"):
        return "invalid-url"
    has_scheme = "://" in base_url
    try:
        parsed = urlsplit(base_url if has_scheme else f"//{base_url}")
        host = parsed.netloc if has_scheme else (parsed.netloc or parsed.hostname or "")
    except ValueError:
        return "opaque-endpoint"
    if has_scheme and host:
        return f"{parsed.scheme.casefold()}://{host}"
    if host:
        return host
    return "opaque-endpoint"


def _snapshot_route_id(base_url: str) -> str:
    if not base_url:
        return ""
    return hashlib.sha256(base_url.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ModelAdmissionSettings:
    """Limits and recovery policy for :class:`ModelAdmissionRegistry`."""

    enabled: bool = True
    max_in_flight: int = 64
    per_target: int = 8
    min_per_target: int = 1
    queue_timeout_seconds: float = 120.0
    additive_successes: int = 16
    retry_after_max_seconds: float = 600.0
    idle_state_ttl_seconds: float = 3600.0
    max_target_states: int = 1024
    wait_poll_seconds: float = 0.1

    def __post_init__(self) -> None:
        positive_ints = {
            "max_in_flight": self.max_in_flight,
            "per_target": self.per_target,
            "min_per_target": self.min_per_target,
            "additive_successes": self.additive_successes,
            "max_target_states": self.max_target_states,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_per_target > self.per_target:
            raise ValueError("min_per_target cannot exceed per_target")

        non_negative = {
            "queue_timeout_seconds": self.queue_timeout_seconds,
            "retry_after_max_seconds": self.retry_after_max_seconds,
            "idle_state_ttl_seconds": self.idle_state_ttl_seconds,
        }
        for name, value in non_negative.items():
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not math.isfinite(float(self.wait_poll_seconds))
            or self.wait_poll_seconds <= 0
        ):
            raise ValueError("wait_poll_seconds must be finite and positive")


@dataclass(slots=True)
class _TargetState:
    limit: int
    last_used: float
    in_flight: int = 0
    queued: int = 0
    notified: int = 0
    congestion_epoch: int = 0
    successes_toward_increase: int = 0
    blocked_until: float = 0.0
    rate_limit_events: int = 0
    rate_limit_streak: int = 0


@dataclass(eq=False, slots=True)
class _Waiter:
    target: ModelTarget
    state: _TargetState
    deadline: float | None
    sync_event: threading.Event
    loop: asyncio.AbstractEventLoop | None = None
    event: asyncio.Event | None = None
    active: bool = True
    notified: bool = False
    first_check: bool = True
    terminal_error: BaseException | None = None


def _iter_error_chain(error: BaseException) -> Iterator[BaseException]:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending and len(seen) < 12:
        current = pending.pop(0)
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        yield current
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                pending.append(linked)


def _status_code(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _is_rate_limit_error(error: BaseException) -> bool:
    for current in _iter_error_chain(error):
        if _status_code(_safe_attr(current, "status_code")) == 429:
            return True
        response = _safe_attr(current, "response")
        if _status_code(_safe_attr(response, "status_code")) == 429:
            return True
        class_name = type(current).__name__.casefold().replace("_", "")
        if "ratelimit" in class_name or class_name == "toomanyrequestserror":
            return True
    return False


def _header_value(headers: Any, name: str) -> Any:
    if headers is None:
        return None
    getter = _safe_attr(headers, "get")
    if callable(getter):
        for candidate in (name, name.casefold(), name.upper()):
            try:
                value = getter(candidate)
            except Exception:
                continue
            if value is not None:
                return value
    items = _safe_attr(headers, "items")
    if callable(items):
        try:
            for key, value in items():
                if str(key).casefold() == name.casefold():
                    return value
        except Exception:
            pass
    return None


def _retry_after_value(error: BaseException) -> Any:
    for current in _iter_error_chain(error):
        direct = _safe_attr(current, "retry_after")
        if direct is not None:
            return direct
        response = _safe_attr(current, "response")
        value = _header_value(_safe_attr(response, "headers"), "Retry-After")
        if value is not None:
            return value
        value = _header_value(_safe_attr(current, "headers"), "Retry-After")
        if value is not None:
            return value
    return None


def _parse_retry_after(value: Any, wall_now: float, maximum: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        delay = float(value)
    else:
        rendered = str(value).strip()
        try:
            delay = float(rendered)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(rendered)
            except (TypeError, ValueError, OverflowError):
                return None
            if parsed is None:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            try:
                delay = parsed.timestamp() - wall_now
            except (OverflowError, OSError, ValueError):
                return None
    if not math.isfinite(delay):
        return None
    return min(max(delay, 0.0), maximum)


class ModelAdmissionRegistry:
    """Thread-safe global and per-model request admission registry."""

    def __init__(
        self,
        settings: ModelAdmissionSettings | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings or ModelAdmissionSettings()
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._condition = threading.Condition(threading.RLock())
        self._states: dict[ModelTarget, _TargetState] = {}
        self._waiters: deque[_Waiter] = deque()
        self._notified_waiters: set[_Waiter] = set()
        self._in_flight = 0
        self._notified = 0

    def acquire(
        self,
        provider: str | None,
        base_url: str | None,
        model: str | None,
        *,
        timeout: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ModelAdmissionPermit:
        """Synchronously wait for and return a request permit."""

        if not self.settings.enabled:
            return ModelAdmissionPermit.noop()
        target = normalize_model_target(provider, base_url, model)
        deadline = self._deadline(timeout)

        with self._condition:
            waiter = self._enqueue_locked(target, deadline=deadline)
        try:
            while True:
                if cancel_check is not None and cancel_check():
                    error = ModelAdmissionCancelled("Model admission was cancelled")
                    with self._condition:
                        self._remove_waiter_locked(waiter, terminal_error=error)
                        self._dispatch_waiters_locked()
                    raise error
                with self._condition:
                    now = self._monotonic()
                    if not waiter.active:
                        raise waiter.terminal_error or ModelAdmissionCancelled(
                            "Model admission waiter is no longer active"
                        )
                    if not waiter.first_check and self._deadline_expired(waiter, now):
                        error = ModelAdmissionTimeout("Model admission timed out")
                        self._remove_waiter_locked(waiter, terminal_error=error)
                        self._dispatch_waiters_locked(now)
                        raise error
                    if not waiter.notified and self._has_reservable_capacity_locked(
                        waiter.state, now
                    ):
                        self._dispatch_waiters_locked(now)
                    if waiter.notified and self._can_admit_locked(waiter, now):
                        return self._admit_locked(waiter, now)
                    if waiter.notified:
                        self._revoke_notification_locked(waiter)
                        self._dispatch_waiters_locked(now)
                    waiter.first_check = False
                    remaining = None if deadline is None else deadline - now
                    if remaining is not None and remaining <= 0:
                        error = ModelAdmissionTimeout("Model admission timed out")
                        self._remove_waiter_locked(waiter, terminal_error=error)
                        self._dispatch_waiters_locked(now)
                        raise error
                    waiter.sync_event.clear()
                    wait_for = self.settings.wait_poll_seconds
                    if remaining is not None:
                        wait_for = min(wait_for, remaining)
                waiter.sync_event.wait(timeout=wait_for)
        except BaseException:
            with self._condition:
                self._remove_waiter_locked(waiter)
                self._dispatch_waiters_locked()
            raise

    async def acquire_async(
        self,
        provider: str | None,
        base_url: str | None,
        model: str | None,
        *,
        timeout: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ModelAdmissionPermit:
        """Asynchronously wait without occupying a worker thread."""

        if not self.settings.enabled:
            return ModelAdmissionPermit.noop()
        target = normalize_model_target(provider, base_url, model)
        deadline = self._deadline(timeout)
        loop = asyncio.get_running_loop()
        event = asyncio.Event()

        with self._condition:
            waiter = self._enqueue_locked(
                target,
                deadline=deadline,
                loop=loop,
                event=event,
            )
        try:
            while True:
                if cancel_check is not None and cancel_check():
                    error = ModelAdmissionCancelled("Model admission was cancelled")
                    with self._condition:
                        self._remove_waiter_locked(waiter, terminal_error=error)
                        self._dispatch_waiters_locked()
                    raise error
                with self._condition:
                    now = self._monotonic()
                    if not waiter.active:
                        raise waiter.terminal_error or ModelAdmissionCancelled(
                            "Model admission waiter is no longer active"
                        )
                    if not waiter.first_check and self._deadline_expired(waiter, now):
                        error = ModelAdmissionTimeout("Model admission timed out")
                        self._remove_waiter_locked(waiter, terminal_error=error)
                        self._dispatch_waiters_locked(now)
                        raise error
                    if not waiter.notified and self._has_reservable_capacity_locked(
                        waiter.state, now
                    ):
                        self._dispatch_waiters_locked(now)
                    if waiter.notified and self._can_admit_locked(waiter, now):
                        return self._admit_locked(waiter, now)
                    if waiter.notified:
                        self._revoke_notification_locked(waiter)
                        self._dispatch_waiters_locked(now)
                    waiter.first_check = False
                    remaining = None if deadline is None else deadline - now
                    if remaining is not None and remaining <= 0:
                        error = ModelAdmissionTimeout("Model admission timed out")
                        self._remove_waiter_locked(waiter, terminal_error=error)
                        self._dispatch_waiters_locked(now)
                        raise error
                    event.clear()
                    wait_for = self.settings.wait_poll_seconds
                    if remaining is not None:
                        wait_for = min(wait_for, remaining)
                try:
                    await asyncio.wait_for(event.wait(), timeout=wait_for)
                except TimeoutError:
                    pass
        except BaseException:
            with self._condition:
                self._remove_waiter_locked(waiter)
                self._dispatch_waiters_locked()
            raise

    def snapshot(self) -> dict[str, Any]:
        """Return bounded, credential-free diagnostics for observability."""

        with self._condition:
            now = self._monotonic()
            self._prune_stale_waiters_locked(now, scan_all=True)
            self._dispatch_waiters_locked(now)
            self._cleanup_idle_locked(now)
            targets = [
                {
                    "provider": target.provider,
                    "base_url": _snapshot_base_url(target.base_url),
                    "route_id": _snapshot_route_id(target.base_url),
                    "model": target.model,
                    "limit": state.limit,
                    "in_flight": state.in_flight,
                    "queued": state.queued,
                    "congestion_epoch": state.congestion_epoch,
                    "successes_toward_increase": state.successes_toward_increase,
                    "blocked_for_seconds": max(0.0, state.blocked_until - now),
                    "rate_limit_events": state.rate_limit_events,
                }
                for target, state in sorted(
                    self._states.items(),
                    key=lambda item: (
                        item[0].provider,
                        item[0].base_url,
                        item[0].model,
                    ),
                )
            ]
            return {
                "enabled": self.settings.enabled,
                "global": {
                    "in_flight": self._in_flight,
                    "max_in_flight": self.settings.max_in_flight,
                    "queued": len(self._waiters),
                },
                "targets": targets,
            }

    def _deadline(self, timeout: float | None) -> float | None:
        duration = self.settings.queue_timeout_seconds if timeout is None else timeout
        duration = float(duration)
        if math.isnan(duration):
            raise ValueError("timeout must not be NaN")
        if math.isinf(duration):
            if duration > 0:
                return None
            duration = 0.0
        return self._monotonic() + max(0.0, duration)

    def _enqueue_locked(
        self,
        target: ModelTarget,
        *,
        deadline: float | None,
        loop: asyncio.AbstractEventLoop | None = None,
        event: asyncio.Event | None = None,
    ) -> _Waiter:
        now = self._monotonic()
        self._cleanup_idle_locked(now)
        state = self._states.get(target)
        if state is None:
            state = _TargetState(limit=self.settings.per_target, last_used=now)
            self._states[target] = state
        state.queued += 1
        state.last_used = now
        waiter = _Waiter(
            target=target,
            state=state,
            deadline=deadline,
            sync_event=threading.Event(),
            loop=loop,
            event=event,
        )
        self._waiters.append(waiter)
        # Enqueuing behind a blocked/saturated target (or a full global pool)
        # cannot make any waiter newly eligible.  Avoid rescanning the whole
        # queue on every such enqueue; the next capacity/state transition or
        # the waiter's bounded poll will dispatch eligible work.
        if self._has_reservable_capacity_locked(state, now):
            self._dispatch_waiters_locked(now)
        return waiter

    def _has_reservable_capacity_locked(
        self,
        state: _TargetState,
        now: float,
    ) -> bool:
        return (
            self._in_flight + self._notified < self.settings.max_in_flight
            and now >= state.blocked_until
            and state.in_flight + state.notified < state.limit
        )

    def _can_admit_locked(self, waiter: _Waiter, now: float) -> bool:
        return (
            waiter.active
            and waiter.notified
            and self._in_flight < self.settings.max_in_flight
            and waiter.state.in_flight < waiter.state.limit
            and now >= waiter.state.blocked_until
        )

    def _admit_locked(self, waiter: _Waiter, now: float) -> ModelAdmissionPermit:
        self._remove_waiter_locked(waiter)
        waiter.state.in_flight += 1
        waiter.state.last_used = now
        self._in_flight += 1
        self._dispatch_waiters_locked(now)
        return ModelAdmissionPermit(
            registry=self,
            target=waiter.target,
            congestion_epoch=waiter.state.congestion_epoch,
        )

    def _remove_waiter_locked(
        self,
        waiter: _Waiter,
        *,
        terminal_error: BaseException | None = None,
        wake: bool = False,
    ) -> None:
        if not waiter.active:
            return
        try:
            self._waiters.remove(waiter)
        except ValueError:
            waiter.active = False
            return
        waiter.active = False
        if waiter.notified:
            self._revoke_notification_locked(waiter)
        waiter.state.queued = max(0, waiter.state.queued - 1)
        waiter.state.last_used = self._monotonic()
        if terminal_error is not None:
            waiter.terminal_error = terminal_error
        if wake:
            self._signal_waiter_locked(waiter)

    def _revoke_notification_locked(self, waiter: _Waiter) -> None:
        if not waiter.notified:
            return
        waiter.notified = False
        waiter.state.notified = max(0, waiter.state.notified - 1)
        self._notified = max(0, self._notified - 1)
        self._notified_waiters.discard(waiter)

    @staticmethod
    def _deadline_expired(waiter: _Waiter, now: float) -> bool:
        return waiter.deadline is not None and now >= waiter.deadline

    @staticmethod
    def _loop_closed(waiter: _Waiter) -> bool:
        if waiter.loop is None:
            return False
        try:
            return waiter.loop.is_closed()
        except Exception:
            return True

    def _drop_stale_waiter_locked(self, waiter: _Waiter, now: float) -> bool:
        if not waiter.active:
            return True
        if self._loop_closed(waiter):
            self._remove_waiter_locked(waiter)
            return True
        if not waiter.first_check and self._deadline_expired(waiter, now):
            self._remove_waiter_locked(
                waiter,
                terminal_error=ModelAdmissionTimeout("Model admission timed out"),
                wake=True,
            )
            return True
        return False

    def _prune_stale_waiters_locked(self, now: float, *, scan_all: bool) -> None:
        candidates = self._waiters if scan_all else self._notified_waiters
        for waiter in tuple(candidates):
            self._drop_stale_waiter_locked(waiter, now)

    def _signal_waiter_locked(self, waiter: _Waiter) -> bool:
        if waiter.loop is None or waiter.event is None:
            waiter.sync_event.set()
            return True
        if self._loop_closed(waiter):
            return False
        try:
            waiter.loop.call_soon_threadsafe(waiter.event.set)
            return True
        except RuntimeError:
            return False

    def _dispatch_waiters_locked(self, now: float | None = None) -> None:
        now = self._monotonic() if now is None else now
        # Already-signalled waiters are at most max_in_flight, so this stale
        # check stays bounded even when the queue contains thousands of tasks.
        self._prune_stale_waiters_locked(now, scan_all=False)
        available = self.settings.max_in_flight - self._in_flight - self._notified
        if available <= 0:
            return

        for waiter in tuple(self._waiters):
            if available <= 0:
                break
            if waiter.notified or self._drop_stale_waiter_locked(waiter, now):
                continue
            state = waiter.state
            if now < state.blocked_until:
                continue
            if state.in_flight + state.notified >= state.limit:
                continue

            waiter.notified = True
            state.notified += 1
            self._notified += 1
            self._notified_waiters.add(waiter)
            if self._signal_waiter_locked(waiter):
                available -= 1
            else:
                self._remove_waiter_locked(waiter)

    def _revoke_target_notifications_locked(self, state: _TargetState) -> None:
        for waiter in tuple(self._notified_waiters):
            if waiter.state is state:
                self._revoke_notification_locked(waiter)

    def _complete(
        self,
        target: ModelTarget,
        congestion_epoch: int,
        *,
        succeeded: bool,
        error: BaseException | None,
    ) -> None:
        try:
            is_rate_limit = error is not None and _is_rate_limit_error(error)
            retry_after = (
                _parse_retry_after(
                    _retry_after_value(error),
                    self._wall_clock(),
                    self.settings.retry_after_max_seconds,
                )
                if is_rate_limit and error is not None
                else None
            )
        except Exception:
            # A non-standard SDK exception must never prevent capacity release.
            is_rate_limit = False
            retry_after = None

        with self._condition:
            now = self._monotonic()
            state = self._states.get(target)
            self._in_flight = max(0, self._in_flight - 1)
            if state is None:
                self._dispatch_waiters_locked(now)
                return
            state.in_flight = max(0, state.in_flight - 1)
            state.last_used = now

            if is_rate_limit:
                state.rate_limit_events += 1
                new_congestion_epoch = congestion_epoch == state.congestion_epoch
                if new_congestion_epoch:
                    state.rate_limit_streak += 1
                    state.limit = max(
                        self.settings.min_per_target,
                        state.limit // 2,
                    )
                    state.congestion_epoch += 1
                    state.successes_toward_increase = 0
                delay = retry_after
                if delay is None and new_congestion_epoch:
                    delay = min(
                        self.settings.retry_after_max_seconds,
                        float(2 ** min(state.rate_limit_streak - 1, 6)),
                    )
                if delay is not None:
                    state.blocked_until = max(state.blocked_until, now + delay)
                self._revoke_target_notifications_locked(state)
            elif succeeded and congestion_epoch == state.congestion_epoch:
                state.rate_limit_streak = 0
                if state.limit < self.settings.per_target:
                    state.successes_toward_increase += 1
                    if (
                        state.successes_toward_increase
                        >= self.settings.additive_successes
                    ):
                        state.limit += 1
                        state.successes_toward_increase = 0

            self._cleanup_idle_locked(now)
            self._dispatch_waiters_locked(now)

    def _cleanup_idle_locked(self, now: float) -> None:
        idle = [
            (target, state)
            for target, state in self._states.items()
            if state.in_flight == 0
            and state.queued == 0
            and state.notified == 0
            and now >= state.blocked_until
        ]
        ttl = self.settings.idle_state_ttl_seconds
        for target, state in idle:
            if now - state.last_used >= ttl:
                self._states.pop(target, None)

        overflow = len(self._states) - self.settings.max_target_states
        if overflow <= 0:
            return
        remaining_idle = sorted(
            (
                (target, state)
                for target, state in self._states.items()
                if state.in_flight == 0
                and state.queued == 0
                and state.notified == 0
                and now >= state.blocked_until
            ),
            key=lambda item: item[1].last_used,
        )
        for target, _state in remaining_idle[:overflow]:
            self._states.pop(target, None)


_T = TypeVar("_T")


class ModelAdmissionPermit:
    """An idempotent permit whose lifetime covers the complete response."""

    def __init__(
        self,
        *,
        registry: ModelAdmissionRegistry | None,
        target: ModelTarget | None,
        congestion_epoch: int,
    ) -> None:
        self._registry = registry
        self._target = target
        self._congestion_epoch = congestion_epoch
        self._finish_lock = threading.Lock()
        self._finished = False
        self._stream_owned = False

    @classmethod
    def noop(cls) -> ModelAdmissionPermit:
        return cls(registry=None, target=None, congestion_epoch=0)

    @property
    def is_noop(self) -> bool:
        return self._registry is None

    @property
    def is_finished(self) -> bool:
        with self._finish_lock:
            return self._finished

    def succeed(self) -> None:
        self._finish(succeeded=True, error=None)

    def fail(self, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        self._finish(succeeded=False, error=error)

    def release(self) -> None:
        """Release without treating an early cancellation as a success."""

        self._finish(succeeded=False, error=None)

    def _finish(
        self,
        *,
        succeeded: bool,
        error: BaseException | None,
        defer_if_stream_owned: bool = False,
    ) -> None:
        with self._finish_lock:
            if self._finished:
                return
            if defer_if_stream_owned and self._stream_owned:
                return
            self._finished = True
        if self._registry is not None and self._target is not None:
            self._registry._complete(
                self._target,
                self._congestion_epoch,
                succeeded=succeeded,
                error=error,
            )

    def _claim_stream_ownership(self) -> None:
        with self._finish_lock:
            if self._finished:
                raise RuntimeError("Cannot wrap a stream with a finished permit")
            if self._stream_owned:
                raise RuntimeError("A permit can own only one stream")
            self._stream_owned = True

    def wrap_stream(self, stream: Any) -> _PermitStream[_T]:
        self._claim_stream_ownership()
        try:
            return _PermitStream(stream, self)
        except BaseException as error:
            self.fail(error)
            raise

    # A descriptive alias is useful at integration sites that handle both
    # sync and async SDKs in the same function.
    wrap_sync_stream = wrap_stream

    def wrap_async_stream(self, stream: Any) -> _PermitAsyncStream[_T]:
        self._claim_stream_ownership()
        try:
            return _PermitAsyncStream(stream, self)
        except BaseException as error:
            self.fail(error)
            raise

    def __enter__(self) -> ModelAdmissionPermit:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        if exc is None:
            self._finish(
                succeeded=True,
                error=None,
                defer_if_stream_owned=True,
            )
        else:
            self.fail(exc)
        return False

    async def __aenter__(self) -> ModelAdmissionPermit:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        if exc is None:
            self._finish(
                succeeded=True,
                error=None,
                defer_if_stream_owned=True,
            )
        else:
            self.fail(exc)
        return False

    def __del__(self) -> None:
        try:
            self.release()
        except BaseException:
            pass


class _PermitStream(Generic[_T]):
    """Iterator proxy that releases its permit at the real stream boundary."""

    def __init__(self, stream: Any, permit: ModelAdmissionPermit) -> None:
        self._stream = stream
        self._entered_stream: Any = None
        self._context_entered = False
        self._exhausted = False
        self._pending_error: BaseException | None = None
        self._permit = permit
        try:
            self._iterator: Iterator[_T] | None = iter(stream)
        except TypeError:
            # Some SDKs return a context manager whose entered value is the
            # iterable stream.  Defer iterator construction until __enter__.
            if not callable(_safe_attr(stream, "__enter__")):
                raise
            self._iterator = None

    def __iter__(self) -> _PermitStream[_T]:
        return self

    def _active_stream(self) -> Any:
        return (
            self._entered_stream if self._entered_stream is not None else self._stream
        )

    def _record_error(self, error: BaseException) -> None:
        if self._context_entered:
            if self._pending_error is None:
                self._pending_error = error
        else:
            self._permit.fail(error)

    def __next__(self) -> _T:
        try:
            if self._iterator is None:
                self._iterator = iter(self._stream)
            return next(self._iterator)
        except StopIteration:
            self._exhausted = True
            if not self._context_entered:
                self._permit.succeed()
            raise
        except BaseException as error:
            self._record_error(error)
            raise

    def close(self) -> None:
        try:
            target = self._active_stream()
            close = _safe_attr(target, "close")
            if callable(close):
                close()
        except BaseException as error:
            self._record_error(error)
            raise
        else:
            # A manager can still fail in __exit__ (including with a 429), so
            # only the manager exit owns completion once context entry began.
            if not self._context_entered:
                self._permit.release()

    def __enter__(self) -> _PermitStream[_T]:
        self._context_entered = True
        enter = _safe_attr(self._stream, "__enter__")
        try:
            entered = enter() if callable(enter) else self._stream
            self._entered_stream = entered
            if callable(enter):
                self._iterator = iter(entered)
            return self
        except BaseException as error:
            self._permit.fail(error)
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        suppressed = False
        try:
            exit_method = _safe_attr(self._stream, "__exit__")
            if callable(exit_method):
                suppressed = bool(exit_method(exc_type, exc, traceback))
            else:
                target = self._active_stream()
                close = _safe_attr(target, "close")
                if callable(close):
                    close()
        except BaseException as close_error:
            self._pending_error = None
            self._permit.fail(close_error)
            raise
        if exc is not None:
            self._permit.fail(exc)
        elif self._pending_error is not None:
            self._permit.fail(self._pending_error)
        elif self._exhausted:
            self._permit.succeed()
        else:
            self._permit.release()
        self._pending_error = None
        return suppressed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._active_stream(), name)

    def __del__(self) -> None:
        # This is a safety net for consumers that abandon a stream without
        # closing it.  Explicit close/context management remains deterministic.
        try:
            self._permit.release()
        except BaseException:
            pass


class _PermitAsyncStream(Generic[_T]):
    """Async iterator proxy with the same lifetime rules as `_PermitStream`."""

    def __init__(
        self,
        stream: Any,
        permit: ModelAdmissionPermit,
    ) -> None:
        self._stream = stream
        self._entered_stream: Any = None
        self._context_entered = False
        self._exhausted = False
        self._pending_error: BaseException | None = None
        self._permit = permit
        iterator_factory = _safe_attr(stream, "__aiter__")
        if callable(iterator_factory):
            self._iterator: AsyncIterator[_T] | None = iterator_factory()
        elif callable(_safe_attr(stream, "__aenter__")):
            # As with sync SDKs, an async stream manager may only expose the
            # iterator returned from its context entry.
            self._iterator = None
        else:
            raise TypeError("stream is not an async iterable or context manager")

    def __aiter__(self) -> _PermitAsyncStream[_T]:
        return self

    def _active_stream(self) -> Any:
        return (
            self._entered_stream if self._entered_stream is not None else self._stream
        )

    def _record_error(self, error: BaseException) -> None:
        if self._context_entered:
            if self._pending_error is None:
                self._pending_error = error
        else:
            self._permit.fail(error)

    async def __anext__(self) -> _T:
        try:
            if self._iterator is None:
                raise TypeError("async stream context must be entered before iteration")
            return await self._iterator.__anext__()
        except StopAsyncIteration:
            self._exhausted = True
            if not self._context_entered:
                self._permit.succeed()
            raise
        except BaseException as error:
            self._record_error(error)
            raise

    async def aclose(self) -> None:
        try:
            target = self._active_stream()
            aclose = _safe_attr(target, "aclose")
            close = _safe_attr(target, "close")
            if callable(aclose):
                await aclose()
            elif callable(close):
                result = close()
                if hasattr(result, "__await__"):
                    await result
        except BaseException as error:
            self._record_error(error)
            raise
        else:
            # As in the sync wrapper, __aexit__ owns the final outcome after
            # context entry because it can still surface a rate-limit error.
            if not self._context_entered:
                self._permit.release()

    async def __aenter__(self) -> _PermitAsyncStream[_T]:
        self._context_entered = True
        enter = _safe_attr(self._stream, "__aenter__")
        try:
            entered = await enter() if callable(enter) else self._stream
            self._entered_stream = entered
            if callable(enter):
                self._iterator = entered.__aiter__()
            return self
        except BaseException as error:
            self._permit.fail(error)
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        suppressed = False
        try:
            exit_method = _safe_attr(self._stream, "__aexit__")
            if callable(exit_method):
                suppressed = bool(await exit_method(exc_type, exc, traceback))
            else:
                target = self._active_stream()
                aclose = _safe_attr(target, "aclose")
                close = _safe_attr(target, "close")
                if callable(aclose):
                    await aclose()
                elif callable(close):
                    result = close()
                    if hasattr(result, "__await__"):
                        await result
        except BaseException as close_error:
            self._pending_error = None
            self._permit.fail(close_error)
            raise
        if exc is not None:
            self._permit.fail(exc)
        elif self._pending_error is not None:
            self._permit.fail(self._pending_error)
        elif self._exhausted:
            self._permit.succeed()
        else:
            self._permit.release()
        self._pending_error = None
        return suppressed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._active_stream(), name)

    def __del__(self) -> None:
        try:
            self._permit.release()
        except BaseException:
            pass
