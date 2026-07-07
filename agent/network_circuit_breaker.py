"""Host-level network circuit breaker utilities.

This module is intentionally below provider-specific retry logic. It only
opens on local host connectivity failures such as ephemeral port exhaustion
(``EADDRNOTAVAIL``) or broken source-route selection (``EHOSTUNREACH``).
Provider 4xx/5xx responses and ordinary request timeouts are left to the
existing model retry/fallback machinery.
"""

from __future__ import annotations

import errno
import logging
import os
import subprocess
import threading
import time
from collections import Counter
from typing import Mapping

logger = logging.getLogger(__name__)

_HOST_CONNECT_ERRNOS = {
    errno.EADDRNOTAVAIL,
    errno.EHOSTUNREACH,
    errno.ENETUNREACH,
}

_HOST_CONNECT_PHRASES = (
    "can't assign requested address",
    "cannot assign requested address",
    "no route to host",
    "network is unreachable",
)

_NETSTAT_STATES = {
    "CLOSED",
    "LISTEN",
    "SYN_SENT",
    "SYN_RECEIVED",
    "ESTABLISHED",
    "CLOSE_WAIT",
    "FIN_WAIT_1",
    "CLOSING",
    "LAST_ACK",
    "FIN_WAIT_2",
    "TIME_WAIT",
}


class NetworkCircuitOpen(ConnectionError):
    """Raised when host networking is in cooldown."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    cur: BaseException | None = exc
    depth = 0
    while cur is not None and depth < 16:
        ident = id(cur)
        if ident in seen:
            break
        seen.add(ident)
        depth += 1
        yield cur
        cur = cur.__cause__ or cur.__context__


def is_host_connectivity_error(exc: BaseException) -> bool:
    """True for local host/socket failures that retry storms worsen."""
    for item in _iter_exception_chain(exc):
        err_no = getattr(item, "errno", None)
        if err_no in _HOST_CONNECT_ERRNOS:
            return True
        if isinstance(item, OSError) and getattr(item, "args", None):
            first = item.args[0]
            if isinstance(first, int) and first in _HOST_CONNECT_ERRNOS:
                return True
        text = str(item).lower()
        if any(phrase in text for phrase in _HOST_CONNECT_PHRASES):
            return True
    return False


class NetworkCircuitBreaker:
    """Small thread-safe failure counter with a cooldown window."""

    def __init__(self, *, threshold: int | None = None, cooldown_seconds: float | None = None):
        self.threshold = max(1, threshold if threshold is not None else _env_int("HERMES_NETWORK_CIRCUIT_THRESHOLD", 3))
        self.cooldown_seconds = max(1.0, cooldown_seconds if cooldown_seconds is not None else _env_float("HERMES_NETWORK_CIRCUIT_COOLDOWN_SECONDS", 120.0))
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_until = 0.0
        self._reason = ""

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_until = 0.0
            self._reason = ""

    def before_request(self, surface: str = "network") -> None:
        now = time.time()
        with self._lock:
            if self._opened_until > now:
                remaining = int(self._opened_until - now) + 1
                reason = self._reason or "host network circuit open"
                raise NetworkCircuitOpen(
                    f"Host network circuit open for {surface}; retry after {remaining}s ({reason})"
                )
            if self._opened_until and self._opened_until <= now:
                self._opened_until = 0.0
                self._reason = ""
                self._failures = 0

    def record_success(self, surface: str = "network") -> None:
        with self._lock:
            self._failures = 0

    def record_failure(self, exc: BaseException, *, surface: str = "network") -> bool:
        if not is_host_connectivity_error(exc):
            return False
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold:
                self._opened_until = time.time() + self.cooldown_seconds
                self._reason = f"{type(exc).__name__}: {exc}"[:240]
                logger.warning(
                    "Host network circuit opened for %s after %d connectivity failures: %s",
                    surface,
                    self._failures,
                    self._reason,
                )
                return True
        return False

    def force_open(self, reason: str, *, cooldown_seconds: float | None = None) -> None:
        cooldown = max(1.0, cooldown_seconds if cooldown_seconds is not None else self.cooldown_seconds)
        with self._lock:
            self._opened_until = time.time() + cooldown
            self._reason = str(reason or "forced open")[:240]
            self._failures = max(self._failures, self.threshold)
            logger.warning("Host network circuit forced open for %.0fs: %s", cooldown, self._reason)


_GLOBAL_BREAKER = NetworkCircuitBreaker()


def get_global_network_breaker() -> NetworkCircuitBreaker:
    return _GLOBAL_BREAKER


def parse_netstat_state_counts(output: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for line in str(output or "").splitlines():
        for token in reversed(line.split()):
            state = token.strip()
            if state in _NETSTAT_STATES:
                counts[state] += 1
                break
    return dict(counts)


def socket_pressure_is_high(
    counts: Mapping[str, int],
    *,
    port_range_size: int,
    threshold_ratio: float = 0.70,
) -> bool:
    if port_range_size <= 0:
        return False
    pressure = (
        int(counts.get("TIME_WAIT", 0))
        + int(counts.get("FIN_WAIT_1", 0))
        + int(counts.get("LAST_ACK", 0))
    )
    return pressure >= int(port_range_size * threshold_ratio)


def current_socket_state_counts() -> dict[str, int]:
    try:
        output = subprocess.check_output(
            ["netstat", "-anv", "-f", "inet"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return {}
    return parse_netstat_state_counts(output)
