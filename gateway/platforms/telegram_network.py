"""Telegram-specific network helpers.

Provides a hostname-preserving fallback transport for networks where
api.telegram.org resolves to an endpoint that is unreachable from the current
host. The transport keeps the logical request host and TLS SNI as
api.telegram.org while retrying the TCP connection against one or more fallback
IPv4 addresses.

It also owns construction of the underlying ``httpx.AsyncHTTPTransport`` used
for every Telegram API call — direct, proxied, or fallback-IP — so that the
keep-alive policy is consistent across paths.  Out of the box, ``httpx``
(and therefore PTB's ``HTTPXRequest``) uses ``keepalive_expiry=5.0``, which
means an idle connection is retired after 5 seconds.  On networks where the
TLS handshake to ``api.telegram.org`` is slow (for example, an HTTP CONNECT
proxy that tunnels traffic through a distant exit node, where upstream TLS
costs 5–10 seconds), every reply sent more than 5 seconds after the previous
one would trigger a fresh handshake — surfacing as the "typing indicator
appears 10+ seconds late, then sometimes snappy" symptom.  The same setups
also leak hundreds of CLOSE_WAIT sockets against the proxy port: the proxy
half-closes idle long-poll connections after Telegram's getUpdates timeout,
but httpx never notices because nothing is reading from the socket, so the
pool keeps growing until the process restarts.

``build_telegram_httpx_transport`` raises the keep-alive expiry to 10 minutes,
caps the per-origin pool, and enables TCP keepalive socket options so the
kernel detects half-open sockets at the transport layer.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from typing import Any, Iterable, Optional

import httpx

logger = logging.getLogger(__name__)

_TELEGRAM_API_HOST = "api.telegram.org"

# httpx.Limits / TCP keepalive defaults for Telegram traffic.
#
# Rationale (see module docstring): httpx's default keepalive_expiry of 5.0
# seconds is far too short when the transport-level TLS handshake costs
# multiple seconds, and the default pool sizing lets dead sockets accumulate.
_MAX_CONNECTIONS = 100
_MAX_KEEPALIVE_CONNECTIONS = 10
_KEEPALIVE_EXPIRY = 600.0  # seconds — 10 minutes
#
# Why these three numbers:
#   * max_connections=100 — the total pool size, including in-flight
#     requests.  PTB's default is 512; we keep it generous because the
#     send path can spike: a single user reply can spawn sendChatAction
#     refreshes (every 2 s while the agent thinks), the actual
#     sendMessage, optional editMessageText for streaming updates, plus
#     concurrent media uploads.  Setting this too low triggers
#     "Pool timeout: All connections in the connection pool are
#     occupied" under burst load.
#   * max_keepalive_connections=10 — the number of *idle* sockets we
#     hold open for reuse between requests.  This is the knob that
#     actually controls the resident TLS-tunnel footprint: dead
#     connections can't accumulate beyond this cap, but a transient
#     burst can still scale up to max_connections without backpressure.
#   * keepalive_expiry=600s — see below; long enough to span normal
#     "ask, walk away, come back" usage so the same TLS tunnel is
#     reused across idle gaps.

# TCP keepalive timings (seconds).  Linux exposes all three; macOS exposes
# TCP_KEEPALIVE (= idle); Windows is best-effort via SO_KEEPALIVE only.
# We feature-detect at runtime so the helper degrades gracefully.
_TCP_KEEPIDLE = 60
_TCP_KEEPINTVL = 30
_TCP_KEEPCNT = 3

# DNS-over-HTTPS providers used to discover Telegram API IPs that may differ
# from the (potentially unreachable) IP returned by the local system resolver.
_DOH_TIMEOUT = 4.0  # seconds — bounded so connect() isn't noticeably delayed

_DOH_PROVIDERS: list[dict] = [
    {
        "url": "https://dns.google/resolve",
        "params": {"name": _TELEGRAM_API_HOST, "type": "A"},
        "headers": {},
    },
    {
        "url": "https://cloudflare-dns.com/dns-query",
        "params": {"name": _TELEGRAM_API_HOST, "type": "A"},
        "headers": {"Accept": "application/dns-json"},
    },
]

# Last-resort IPs when DoH is also blocked.  These are stable Telegram Bot API
# endpoints in the 149.154.160.0/20 block (same seed used by OpenClaw).
_SEED_FALLBACK_IPS: list[str] = ["149.154.167.220"]


def _resolve_proxy_url(target_hosts=None) -> str | None:
    # Delegate to shared implementation (env vars + macOS system proxy detection)
    from gateway.platforms.base import resolve_proxy_url
    return resolve_proxy_url("TELEGRAM_PROXY", target_hosts=target_hosts)


def _telegram_socket_options() -> list[tuple[int, int, int]]:
    """Return TCP keepalive socket options for the current platform.

    The base SO_KEEPALIVE flag works everywhere; the per-connection timings
    (idle / interval / count) are Linux-specific, with a partial macOS
    equivalent.  Anything we can't detect is silently skipped — httpx accepts
    any iterable of ``(level, optname, value)`` triples.
    """
    options: list[tuple[int, int, int]] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]

    # Linux
    keepidle = getattr(socket, "TCP_KEEPIDLE", None)
    if keepidle is not None:
        options.append((socket.IPPROTO_TCP, keepidle, _TCP_KEEPIDLE))
    else:
        # macOS calls it TCP_KEEPALIVE; same semantics as Linux TCP_KEEPIDLE.
        keepalive_idle = getattr(socket, "TCP_KEEPALIVE", None)
        if keepalive_idle is not None:
            options.append((socket.IPPROTO_TCP, keepalive_idle, _TCP_KEEPIDLE))

    keepintvl = getattr(socket, "TCP_KEEPINTVL", None)
    if keepintvl is not None:
        options.append((socket.IPPROTO_TCP, keepintvl, _TCP_KEEPINTVL))

    keepcnt = getattr(socket, "TCP_KEEPCNT", None)
    if keepcnt is not None:
        options.append((socket.IPPROTO_TCP, keepcnt, _TCP_KEEPCNT))

    return options


def _telegram_httpx_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=_MAX_CONNECTIONS,
        max_keepalive_connections=_MAX_KEEPALIVE_CONNECTIONS,
        keepalive_expiry=_KEEPALIVE_EXPIRY,
    )


def build_telegram_httpx_transport(
    *,
    proxy_url: Optional[str] = None,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> httpx.AsyncHTTPTransport:
    """Construct an ``AsyncHTTPTransport`` tuned for Telegram traffic.

    Caller-supplied ``extra_kwargs`` win over our defaults so tests and power
    users can still override individual knobs (e.g. ``http2=True``,
    ``verify=False``).  The base policy is:

    * ``limits`` with a 10-minute keep-alive expiry and a bounded pool, so
      idle connections survive cross-turn gaps but a transient burst can't
      grow the pool unbounded.
    * TCP keepalive socket options so half-open sockets dropped silently by
      a CONNECT proxy are detected at the kernel layer (and the in-pool
      socket is retired) instead of triggering a fresh TLS handshake on
      the next request.
    * Optional ``proxy`` wiring — ``HTTPS_PROXY``-style env vars are picked
      up by callers via :func:`resolve_proxy_url`; only an explicit URL is
      threaded through here so ``httpx`` doesn't read the environment a
      second time and disagree with our ``NO_PROXY`` evaluation.
    """
    kwargs: dict[str, Any] = {
        "limits": _telegram_httpx_limits(),
        "socket_options": _telegram_socket_options(),
    }
    if proxy_url:
        kwargs["proxy"] = proxy_url
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return httpx.AsyncHTTPTransport(**kwargs)


class TelegramFallbackTransport(httpx.AsyncBaseTransport):
    """Retry Telegram Bot API requests via fallback IPs while preserving TLS/SNI.

    Requests continue to target https://api.telegram.org/... logically, but on
    connect failures the underlying TCP connection is retried against a known
    reachable IP. This is effectively the programmatic equivalent of
    ``curl --resolve api.telegram.org:443:<ip>``.
    """

    def __init__(self, fallback_ips: Iterable[str], **transport_kwargs):
        self._fallback_ips = list(dict.fromkeys(_normalize_fallback_ips(fallback_ips)))
        proxy_url = _resolve_proxy_url(target_hosts=[_TELEGRAM_API_HOST, *self._fallback_ips])
        # Caller may have already supplied ``proxy`` (or set it via env) — let
        # that win.  Same for ``limits`` / ``socket_options`` so power users
        # can still override the defaults.
        explicit_proxy = transport_kwargs.pop("proxy", None) or proxy_url
        self._primary = build_telegram_httpx_transport(
            proxy_url=explicit_proxy,
            extra_kwargs=transport_kwargs or None,
        )
        self._fallbacks = {
            ip: build_telegram_httpx_transport(
                proxy_url=explicit_proxy,
                extra_kwargs=transport_kwargs or None,
            )
            for ip in self._fallback_ips
        }
        self._sticky_ip: Optional[str] = None
        self._sticky_lock = asyncio.Lock()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != _TELEGRAM_API_HOST or not self._fallback_ips:
            return await self._primary.handle_async_request(request)

        sticky_ip = self._sticky_ip
        attempt_order: list[Optional[str]] = [sticky_ip] if sticky_ip else [None]
        if sticky_ip:
            attempt_order.append(None)  # retry primary DNS after sticky failure
        for ip in self._fallback_ips:
            if ip != sticky_ip:
                attempt_order.append(ip)

        last_error: Exception | None = None
        for ip in attempt_order:
            candidate = request if ip is None else _rewrite_request_for_ip(request, ip)
            transport = self._primary if ip is None else self._fallbacks[ip]
            try:
                response = await transport.handle_async_request(candidate)
                if ip is not None and self._sticky_ip != ip:
                    async with self._sticky_lock:
                        if self._sticky_ip != ip:
                            self._sticky_ip = ip
                            logger.warning(
                                "[Telegram] Primary api.telegram.org path unreachable; using sticky fallback IP %s",
                                ip,
                            )
                return response
            except Exception as exc:
                last_error = exc
                if not _is_retryable_connect_error(exc):
                    raise
                if ip is not None and ip == self._sticky_ip:
                    async with self._sticky_lock:
                        if self._sticky_ip == ip:
                            self._sticky_ip = None
                            logger.warning(
                                "[Telegram] Sticky fallback IP %s failed; resetting to primary DNS path",
                                ip,
                            )
                if ip is None:
                    logger.warning(
                        "[Telegram] Primary api.telegram.org connection failed (%s); trying fallback IPs %s",
                        exc,
                        ", ".join(self._fallback_ips),
                    )
                    continue
                logger.warning("[Telegram] Fallback IP %s failed: %s", ip, exc)
                continue

        if last_error is None:
            raise RuntimeError("All Telegram fallback IPs exhausted but no error was recorded")
        raise last_error

    async def aclose(self) -> None:
        await self._primary.aclose()
        for transport in self._fallbacks.values():
            await transport.aclose()


def _normalize_fallback_ips(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        raw = str(value).strip()
        if not raw:
            continue
        try:
            addr = ipaddress.ip_address(raw)
        except ValueError:
            logger.warning("Ignoring invalid Telegram fallback IP: %r", raw)
            continue
        if addr.version != 4:
            logger.warning("Ignoring non-IPv4 Telegram fallback IP: %s", raw)
            continue
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_unspecified:
            logger.warning("Ignoring private/internal Telegram fallback IP: %s", raw)
            continue
        normalized.append(str(addr))
    return normalized


def parse_fallback_ip_env(value: str | None) -> list[str]:
    if not value:
        return []
    parts = [part.strip() for part in value.split(",")]
    return _normalize_fallback_ips(parts)


def _resolve_system_dns() -> set[str]:
    """Return the IPv4 addresses that the OS resolver gives for api.telegram.org."""
    try:
        results = socket.getaddrinfo(_TELEGRAM_API_HOST, 443, socket.AF_INET)
        return {addr[4][0] for addr in results}
    except Exception:
        return set()


async def _query_doh_provider(
    client: httpx.AsyncClient, provider: dict
) -> list[str]:
    """Query one DoH provider and return A-record IPs."""
    try:
        resp = await client.get(
            provider["url"], params=provider["params"], headers=provider["headers"]
        )
        resp.raise_for_status()
        data = resp.json()
        ips: list[str] = []
        for answer in data.get("Answer", []):
            if answer.get("type") != 1:  # A record
                continue
            raw = answer.get("data", "").strip()
            try:
                ipaddress.ip_address(raw)
                ips.append(raw)
            except ValueError:
                continue
        return ips
    except Exception as exc:
        logger.debug("DoH query to %s failed: %s", provider["url"], exc)
        return []


async def discover_fallback_ips() -> list[str]:
    """Auto-discover Telegram API IPs via DNS-over-HTTPS.

    Resolves api.telegram.org through Google and Cloudflare DoH and returns all
    unique A records.  IPs that match the local system resolver are kept rather
    than excluded: in many networks the system-DNS IP is the most reliable path
    to api.telegram.org and a transient primary-path failure should be retried
    against the same address via the IP-rewrite path before the seed list is
    consulted (#14520).  Falls back to a hardcoded seed list only when DoH
    yields no usable answers.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(_DOH_TIMEOUT)) as client:
        doh_tasks = [_query_doh_provider(client, p) for p in _DOH_PROVIDERS]
        system_dns_task = asyncio.to_thread(_resolve_system_dns)
        results = await asyncio.gather(system_dns_task, *doh_tasks, return_exceptions=True)

    # results[0] = system DNS IPs (set), results[1:] = DoH IP lists
    system_ips: set[str] = results[0] if isinstance(results[0], set) else set()

    doh_ips: list[str] = []
    for r in results[1:]:
        if isinstance(r, list):
            doh_ips.extend(r)

    # Deduplicate preserving order
    seen: set[str] = set()
    candidates: list[str] = []
    for ip in doh_ips:
        if ip not in seen:
            seen.add(ip)
            candidates.append(ip)

    # Validate through existing normalization
    validated = _normalize_fallback_ips(candidates)

    if validated:
        logger.debug("Discovered Telegram fallback IPs via DoH: %s", ", ".join(validated))
        return validated

    logger.info(
        "DoH discovery yielded no usable IPs (system DNS: %s); using seed fallback IPs %s",
        ", ".join(system_ips) or "unknown",
        ", ".join(_SEED_FALLBACK_IPS),
    )
    return list(_SEED_FALLBACK_IPS)


def _rewrite_request_for_ip(request: httpx.Request, ip: str) -> httpx.Request:
    original_host = request.url.host or _TELEGRAM_API_HOST
    url = request.url.copy_with(host=ip)
    headers = request.headers.copy()
    headers["host"] = original_host
    extensions = dict(request.extensions)
    extensions["sni_hostname"] = original_host
    return httpx.Request(
        method=request.method,
        url=url,
        headers=headers,
        stream=request.stream,
        extensions=extensions,
    )


def _is_retryable_connect_error(exc: Exception) -> bool:
    return isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError))
