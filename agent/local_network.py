"""Machine-level connectivity gating for provider failover.

A transport failure is evidence about ONE route, never about the provider. When the
machine itself has no working network (resolver dead, no route, link down) every
*external* provider is equally unreachable, so switching to another external route
cannot recover the turn: it strands the session on a fallback model (full prompt-cache
re-read, different price and capabilities) and reports the user's own outage as a
provider fault.

The gate therefore answers a ROUTE question, not an ambient one:

* ``local_network_outage(error)`` — does this failure mean the machine cannot reach the
  outside world?
* ``route_needs_external_network(provider, base_url)`` — does the candidate route depend
  on that outside world?

Callers suppress only the combination (external route, while provably offline). A
loopback/LAN fallback — a local Ollama, LM Studio, or an on-prem ``provider: custom``
endpoint — stays eligible and can still keep the turn alive during a WAN outage; that is
the whole point of having one.

Verdicts are conservative in both directions. A false "offline" would suppress a
legitimate failover, so "outage" requires positive evidence: DNS-resolution/routing
wording in the exception chain (or a connect-class failure), then a probe that fails to
reach anything. ``connection refused`` (the host answered) and read timeouts (a connection
WAS established) can never be an outage; when the probe cannot run at all, only the hard
markers count.
"""

from __future__ import annotations

import asyncio
import errno
import ipaddress
import logging
import socket
import time
from typing import Any, Iterator, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# DNS-resolution and routing failures — the machine cannot reach the network at all.
# ``connection refused`` is deliberately absent: the host answered (RST), so the local
# network is fine and the failure belongs to the provider.
LOCAL_OUTAGE_MARKERS: tuple[str, ...] = (
    "temporary failure in name resolution",
    "name or service not known",
    "nodename nor servname provided, or not known",
    "getaddrinfo failed",
    "getaddrinfo enotfound",
    "eai_again",
    "no address associated with hostname",
    "network is unreachable",
    "network unreachable",
    "no route to host",
)

# Someone answered on the far side — never an outage, whatever else the text says.
_ANSWERED_MARKERS = ("connection refused", "econnrefused", "errno 111", "errno 61")

# Connect-class exception names (httpx + OpenAI SDK). These mean no socket was ever
# established; ``ReadTimeout``/``APITimeoutError`` are intentionally absent (see above).
_CONNECT_FAILURE_TYPES = frozenset({
    "ConnectError", "ConnectTimeout", "APIConnectionError", "APIConnectionTimeoutError",
})

# Provider-neutral EGRESS witnesses. A resolver answer is NOT one: a router/cache resolver
# can keep answering while the default route is dead, which is exactly the outage this gate
# exists for — so every witness here is a TCP connect to a bare IP (no DNS involved). Any
# completed handshake proves egress; a refusal counts too, because an RST still proves the
# packet left the machine. Port 53 is included because filtered networks that block 443 to
# these hosts commonly still allow DNS.
_PROBE_ENDPOINTS: tuple[tuple[str, int], ...] = (
    ("1.1.1.1", 443), ("8.8.8.8", 443), ("9.9.9.9", 443),
    ("223.5.5.5", 443), ("1.1.1.1", 53), ("8.8.8.8", 53),
)
_PROBE_TIMEOUT_SECONDS = 2.0
# Bound the whole sweep: the probe runs inside an error path, so it must never outlast the
# retry loop's patience. Six witnesses at the per-attempt timeout would be 12s; the deadline
# stops the sweep at five.
_PROBE_DEADLINE_SECONDS = 5.0
# A retry burst must not probe per attempt, but recovery has to be noticed promptly: the
# verdict is re-taken at most every 5s, so a restored link releases the hold at once.
_PROBE_CACHE_TTL_SECONDS = 5.0

# Waits between attempts while the machine is offline: short enough that a recovered
# hotspot is used immediately, long enough that a real outage does not spin.
_OUTAGE_WAIT_SCHEDULE = (5.0, 10.0, 20.0, 30.0)

# ``agent.network_outage_max_wait_seconds``: how long a turn holds the primary route
# waiting for connectivity before it gives up. 0 = fail fast instead of waiting.
_DEFAULT_MAX_WAIT_SECONDS = 120.0

# Provider slugs whose default endpoint is on this machine. An entry without ``base_url``
# is local for these and external for everything else.
_LOCAL_PROVIDER_SLUGS = frozenset({
    "ollama", "lmstudio", "lm-studio", "llamacpp", "llama-cpp", "vllm", "localai", "local",
})

# Suffixes that resolve on the local network, never through the WAN.
_LOCAL_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa", ".localdomain")

# (monotonic timestamp, verdict) of the last probe; verdict None = not probed yet.
_probe_cache: tuple[float, Optional[bool]] = (0.0, None)


def reset_connectivity_cache() -> None:
    """Drop the cached probe verdict (tests, and explicit re-checks)."""
    global _probe_cache
    _probe_cache = (0.0, None)


def probe_connectivity(
    *, timeout: float = _PROBE_TIMEOUT_SECONDS, cache_ttl: float = _PROBE_CACHE_TTL_SECONDS,
    deadline: float = _PROBE_DEADLINE_SECONDS, cache_only: bool = False,
) -> Optional[bool]:
    """Whether this machine can reach the outside world: ``True``/``False``/``None`` (unknown).

    Cached for ``cache_ttl`` seconds. ``cache_only`` returns the cached verdict and never
    performs I/O — the path for callers that must not block (see ``probe_connectivity_async``
    and the auxiliary ladder, which may be advanced on an event loop).
    """
    global _probe_cache
    now = time.monotonic()
    cached_at, cached = _probe_cache
    if cached is not None and now - cached_at < cache_ttl:
        return cached
    if cache_only:
        return None
    verdict = _probe_uncached(timeout=timeout, deadline=deadline)
    _probe_cache = (now, verdict)
    return verdict


async def probe_connectivity_async() -> Optional[bool]:
    """Off-loop refresh of the cached verdict (blocking socket work in a worker thread).

    The async auxiliary lane calls this before it classifies a transport error, so the
    sync ladder that follows can consume the verdict without touching a socket.
    """
    try:
        return await asyncio.to_thread(probe_connectivity)
    except Exception as exc:  # pragma: no cover - thread/loop shutdown races
        logger.debug("Off-loop connectivity probe failed: %s", exc)
        return None


def _probe_uncached(*, timeout: float, deadline: float) -> Optional[bool]:
    """One bounded sweep of egress witnesses; ``None`` when the probe itself is untrustworthy."""
    started = time.monotonic()
    for host, port in _PROBE_ENDPOINTS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError as exc:
            if isinstance(exc, ConnectionRefusedError) or getattr(exc, "errno", None) == errno.ECONNREFUSED:
                # Refused still proves the packet reached the far side: egress works.
                return True
            if time.monotonic() - started >= deadline:
                logger.debug(
                    "Connectivity probe deadline (%.1fs) reached after %s:%s: %s",
                    deadline, host, port, exc,
                )
                return False
            continue
        except Exception as exc:  # pragma: no cover - non-OSError probe faults
            logger.debug("Connectivity probe failed unexpectedly on %s:%s: %s", host, port, exc)
            return None
    return False


def _chain(error: Any) -> Iterator[BaseException]:
    """``error`` and its ``__cause__``/``__context__`` chain (each node once)."""
    seen: set[int] = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)


def failure_text(error: Any) -> str:
    """Lowercased text of the whole exception chain (SDKs wrap the OS error one level down)."""
    parts = []
    for node in _chain(error):
        parts.append(str(node))
        message = getattr(node, "message", None)
        if isinstance(message, str) and message:
            parts.append(message)
    return " ".join(parts).lower()


def has_local_outage_marker(error: Any) -> bool:
    """True when the failure text names a DNS-resolution or routing failure."""
    text = failure_text(error)
    return any(marker in text for marker in LOCAL_OUTAGE_MARKERS)


def is_connect_failure(error: Any) -> bool:
    """True when no socket was ever established (connect/route failure, not a stalled stream)."""
    return any(type(node).__name__ in _CONNECT_FAILURE_TYPES for node in _chain(error))


def local_network_outage(error: Any, *, allow_probe: bool = True) -> bool:
    """True when ``error`` most likely means this machine has no working network.

    ``allow_probe=False`` consumes only an already-cached verdict (no socket work at all) —
    used where blocking is unacceptable, e.g. the auxiliary ladder reachable from an event
    loop. With no cached verdict the hard markers still decide.
    """
    text = failure_text(error)
    if any(marker in text for marker in _ANSWERED_MARKERS):
        return False
    markers = any(marker in text for marker in LOCAL_OUTAGE_MARKERS)
    if not markers and not is_connect_failure(error):
        return False
    verdict = probe_connectivity(cache_only=not allow_probe)
    if verdict is None:
        # No trustworthy probe (or none allowed here): only explicit DNS/routing evidence counts.
        return markers
    return not verdict


def _is_local_host(host: str) -> bool:
    """True for a name/address that stays reachable while the WAN is down."""
    name = host.strip().strip("[]").lower()
    if not name:
        return False
    if name in {"localhost", "0.0.0.0"} or name.endswith(_LOCAL_HOST_SUFFIXES):
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        # A single-label name resolves on the local network, not through public DNS.
        return "." not in name
    return address.is_loopback or address.is_private or address.is_link_local


def route_needs_external_network(provider: str, base_url: str = "") -> bool:
    """False when the route stays reachable with the WAN down (loopback / on-prem endpoint).

    Decides whether an outage may suppress this candidate: a local route must remain
    eligible, because it is precisely the route that can still serve the turn.
    """
    host = (urlsplit(base_url).hostname or "") if base_url else ""
    if host:
        return not _is_local_host(host)
    return (provider or "").strip().lower() not in _LOCAL_PROVIDER_SLUGS


def network_outage_max_wait_seconds() -> float:
    """``agent.network_outage_max_wait_seconds`` (default 120s; 0 turns the wait off)."""
    try:
        from hermes_cli.config import load_config_readonly
        agent_cfg = (load_config_readonly() or {}).get("agent") or {}
        raw = agent_cfg.get("network_outage_max_wait_seconds", _DEFAULT_MAX_WAIT_SECONDS)
        return max(float(raw), 0.0)
    except Exception as exc:
        logger.debug("network_outage_max_wait_seconds fell back to default: %s", exc)
        return _DEFAULT_MAX_WAIT_SECONDS


def network_outage_wait_seconds(retry_count: int) -> float:
    """Wait before the next attempt while offline (5 → 10 → 20 → 30s, then flat)."""
    try:
        index = max(int(retry_count) - 1, 0)
    except (TypeError, ValueError):
        index = 0
    return _OUTAGE_WAIT_SCHEDULE[min(index, len(_OUTAGE_WAIT_SCHEDULE) - 1)]


def outage_wait_plan(waited: float, retry_count: int, *, cap: float) -> float:
    """Seconds this attempt may wait for connectivity, clamped to the remaining budget.

    One authority for both sides of the accounting: the caller charges exactly what this
    returns and sleeps exactly what this returns, so ``cap`` bounds the real wall-clock wait
    rather than a bookkeeping counter (0.0 = budget spent, fall back to normal backoff).
    """
    if cap <= 0:
        return 0.0
    remaining = max(cap - max(waited, 0.0), 0.0)
    return min(network_outage_wait_seconds(retry_count), remaining)
