"""Non-blocking status probes for local llama.cpp runtimes."""

from __future__ import annotations

from functools import lru_cache
import math
import ssl
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Optional
from urllib.parse import urlparse

import httpx

from agent.model_metadata import _auth_headers, _localhost_to_ipv4, _normalize_base_url, is_local_endpoint
from agent.ssl_verify import resolve_httpx_verify


class LocalModelState(str, Enum):
    """Authoritative residency states reported by a local runtime."""

    LOADED = "loaded"
    LOADING = "loading"
    SLEEPING = "sleeping"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LocalModelRoute:
    """A probeable local model route. Credentials are never shown in repr."""

    provider: str
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    extra_headers: tuple[tuple[str, str], ...] = field(default_factory=tuple, repr=False)
    ssl_ca_cert: str = field(default="", repr=False)
    ssl_verify: Optional[bool] = field(default=None, repr=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.provider, self.base_url, self.model)


@dataclass(frozen=True)
class LocalModelProbeResult:
    state: LocalModelState
    supported: Optional[bool]
    checked_at: float


@dataclass(frozen=True)
class LocalModelDisplay:
    bar: str
    label: str
    style: str

    @property
    def text(self) -> str:
        return f"{self.bar} {self.label}"


def canonicalize_route_url(value: object) -> str:
    """Return a stable URL identity for cross-client route comparisons."""

    try:
        return str(httpx.URL(str(value or ""))).rstrip("/")
    except Exception:
        return ""


def _format_remaining(seconds: int) -> str:
    minutes, seconds = divmod(max(0, seconds), 60)
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def build_local_model_display(
    result: LocalModelProbeResult,
    *,
    idle_timeout_seconds: Optional[float],
    last_activity_at: Optional[float],
    now: float,
    turn_live: bool,
) -> LocalModelDisplay:
    """Build a binary residency bar plus an explicitly estimated countdown."""

    if result.state is LocalModelState.SLEEPING:
        return LocalModelDisplay("[░░░░░░░░░░]", "off", "sleeping")
    if result.state is LocalModelState.LOADING:
        return LocalModelDisplay("[▒▒▒▒▒▒▒▒▒▒]", "load", "loading")
    if result.state is not LocalModelState.LOADED:
        return LocalModelDisplay("[░░░░░░░░░░]", "?", "unknown")

    label = "busy" if turn_live else "on"
    if not turn_live and idle_timeout_seconds and idle_timeout_seconds > 0 and last_activity_at is not None:
        remaining = math.ceil(idle_timeout_seconds - max(0.0, now - last_activity_at))
        if remaining > 0:
            label = f"~{_format_remaining(remaining)}"
    return LocalModelDisplay("[██████████]", label, "loaded")


def build_local_model_route(
    *,
    provider: str,
    base_url: str,
    model: str,
    api_key: str = "",
    extra_headers: Optional[Mapping[str, str]] = None,
    ssl_ca_cert: str = "",
    ssl_verify: Optional[bool] = None,
) -> Optional[LocalModelRoute]:
    """Return a probe route only for local HTTP(S) endpoints."""

    normalized = canonicalize_route_url(_normalize_base_url(base_url or ""))
    try:
        parsed = urlparse(normalized)
    except Exception:
        return None
    # Rewriting localhost avoids slow IPv6 fallback for plain HTTP, but doing
    # so for HTTPS breaks certificates whose SAN contains localhost, not the IP.
    if parsed.scheme == "http":
        normalized = _localhost_to_ipv4(normalized)
        parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not is_local_endpoint(normalized):
        return None
    return LocalModelRoute(
        provider=str(provider or ""),
        base_url=normalized,
        model=str(model or ""),
        api_key=str(api_key or ""),
        extra_headers=tuple(
            sorted(
                (str(name), str(value))
                for name, value in (extra_headers or {}).items()
                if value is not None
            )
        ),
        ssl_ca_cert=str(ssl_ca_cert or ""),
        ssl_verify=ssl_verify if isinstance(ssl_verify, bool) else None,
    )


def _props_urls(base_url: str) -> tuple[str, ...]:
    server_url = base_url[:-3] if base_url.endswith("/v1") else base_url
    # Current llama.cpp serves props at the server root. Keep /v1/props as a
    # compatibility fallback, but avoid generating a 404 on every poll.
    return (f"{server_url}/props", f"{server_url}/v1/props")


@lru_cache(maxsize=32)
def _resolve_probe_verify(
    base_url: str,
    ca_bundle: str,
    ssl_verify: Optional[bool],
) -> bool | ssl.SSLContext:
    """Resolve and cache the active route's TLS policy."""

    return resolve_httpx_verify(
        ca_bundle=ca_bundle or None,
        ssl_verify=ssl_verify,
        base_url=base_url,
    )


def probe_llamacpp_status(
    route: LocalModelRoute,
    *,
    timeout: float = 1.0,
) -> LocalModelProbeResult:
    """Probe llama.cpp's read-only props endpoint without waking the model."""

    checked_at = time.monotonic()
    try:
        headers = _auth_headers(route.api_key)
        headers.update(dict(route.extra_headers))
        with httpx.Client(
            timeout=timeout,
            headers=headers,
            trust_env=False,
            verify=_resolve_probe_verify(
                route.base_url,
                route.ssl_ca_cert,
                route.ssl_verify,
            ),
        ) as client:
            for url in _props_urls(route.base_url):
                response = client.get(url)
                if response.status_code == 404:
                    continue
                if response.status_code == 503:
                    return LocalModelProbeResult(
                        state=LocalModelState.LOADING,
                        supported=None,
                        checked_at=checked_at,
                    )
                if response.status_code != 200:
                    return LocalModelProbeResult(
                        state=LocalModelState.UNKNOWN,
                        supported=None,
                        checked_at=checked_at,
                    )
                try:
                    payload = response.json()
                except Exception:
                    return LocalModelProbeResult(
                        state=LocalModelState.UNKNOWN,
                        supported=None,
                        checked_at=checked_at,
                    )
                if not isinstance(payload, dict) or "default_generation_settings" not in payload:
                    return LocalModelProbeResult(
                        state=LocalModelState.UNKNOWN,
                        supported=False,
                        checked_at=checked_at,
                    )
                is_sleeping = payload.get("is_sleeping")
                if not isinstance(is_sleeping, bool):
                    return LocalModelProbeResult(
                        state=LocalModelState.UNKNOWN,
                        supported=True,
                        checked_at=checked_at,
                    )
                return LocalModelProbeResult(
                    state=LocalModelState.SLEEPING if is_sleeping else LocalModelState.LOADED,
                    supported=True,
                    checked_at=checked_at,
                )
    except Exception:
        return LocalModelProbeResult(
            state=LocalModelState.UNKNOWN,
            supported=None,
            checked_at=checked_at,
        )

    return LocalModelProbeResult(
        state=LocalModelState.UNKNOWN,
        supported=False,
        checked_at=checked_at,
    )


class LocalModelStatusMonitor:
    """Single-flight background probe cache for the active local route."""

    def __init__(
        self,
        *,
        probe: Callable[[LocalModelRoute], LocalModelProbeResult] = probe_llamacpp_status,
        poll_interval: float = 5.0,
        max_backoff: float = 30.0,
        unsupported_interval: float = 300.0,
        auto_poll: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._probe = probe
        self._poll_interval = max(0.1, float(poll_interval))
        self._max_backoff = max(self._poll_interval, float(max_backoff))
        self._unsupported_interval = max(self._poll_interval, float(unsupported_interval))
        self._auto_poll = bool(auto_poll)
        self._failure_delay = self._poll_interval
        self._clock = clock
        self._lock = threading.Lock()
        self._route: Optional[LocalModelRoute] = None
        self._generation = 0
        self._closed = False
        self._probe_in_flight = False
        self._next_probe_at = 0.0
        self._snapshot = LocalModelProbeResult(
            state=LocalModelState.UNKNOWN,
            supported=None,
            checked_at=0.0,
        )

    def observe(self, route: Optional[LocalModelRoute]) -> Optional[LocalModelProbeResult]:
        """Return cached status and schedule a due probe without blocking."""

        worker: Optional[threading.Thread] = None
        with self._lock:
            if self._closed:
                return None
            if route != self._route:
                self._generation += 1
                self._route = route
                self._next_probe_at = 0.0
                self._failure_delay = self._poll_interval
                self._snapshot = LocalModelProbeResult(
                    state=LocalModelState.UNKNOWN,
                    supported=None,
                    checked_at=0.0,
                )
            if route is None:
                return None

            now = self._clock()
            if not self._probe_in_flight and now >= self._next_probe_at:
                self._probe_in_flight = True
                generation = self._generation
                worker = threading.Thread(
                    target=self._run_probe,
                    args=(route, generation),
                    daemon=True,
                    name="hermes-local-model-status",
                )
            snapshot = self._snapshot

        if worker is not None:
            worker.start()
        if snapshot.supported is False:
            return None
        return snapshot

    def _run_probe(self, route: LocalModelRoute, generation: int) -> None:
        try:
            result = self._probe(route)
        except Exception:
            result = LocalModelProbeResult(
                state=LocalModelState.UNKNOWN,
                supported=None,
                checked_at=self._clock(),
            )

        timer: Optional[threading.Timer] = None
        replacement_worker: Optional[threading.Thread] = None
        with self._lock:
            if generation != self._generation or route != self._route:
                self._probe_in_flight = False
                if not self._closed and self._route is not None:
                    self._probe_in_flight = True
                    replacement_worker = threading.Thread(
                        target=self._run_probe,
                        args=(self._route, self._generation),
                        daemon=True,
                        name="hermes-local-model-status",
                    )
            else:
                self._snapshot = result
                self._probe_in_flight = False
                if result.supported is False:
                    delay = self._unsupported_interval
                    self._failure_delay = self._poll_interval
                elif result.state is LocalModelState.UNKNOWN and result.supported is None:
                    delay = self._failure_delay
                    self._failure_delay = min(self._failure_delay * 2, self._max_backoff)
                else:
                    delay = self._poll_interval
                    self._failure_delay = self._poll_interval
                self._next_probe_at = self._clock() + delay
                if self._auto_poll and not self._closed:
                    timer = threading.Timer(
                        delay,
                        self._run_scheduled_probe,
                        args=(route, generation),
                    )
                    timer.daemon = True
                    timer.name = "hermes-local-model-status-timer"

        if replacement_worker is not None:
            replacement_worker.start()
        if timer is not None:
            timer.start()

    def _run_scheduled_probe(self, route: LocalModelRoute, generation: int) -> None:
        with self._lock:
            if (
                self._closed
                or generation != self._generation
                or route != self._route
                or self._probe_in_flight
            ):
                return
            self._probe_in_flight = True
        self._run_probe(route, generation)

    def close(self) -> None:
        """Invalidate in-flight work; daemon probes never delay process exit."""

        with self._lock:
            self._generation += 1
            self._closed = True
            self._route = None
            self._probe_in_flight = False
