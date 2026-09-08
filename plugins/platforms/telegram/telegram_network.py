"""Telegram network helpers: a hostname-preserving fallback transport (Host + SNI stay
api.telegram.org while TCP retries known IPv4 literals) plus DoH-based IP discovery."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from typing import Iterable, Optional

import httpx

logger = logging.getLogger(__name__)

_TELEGRAM_API_HOST = "api.telegram.org"

# TCP keepalive so a half-open/CLOSE-WAIT long-poll errors out instead of blocking getUpdates forever
# (Windows leaves SO_KEEPALIVE off). Idle/interval knobs are best-effort per Python/OS combo.
# Windows does not enable SO_KEEPALIVE on new sockets by default, so a dead api.telegram.org peer can hang
# forever (#87057).
_TCP_KEEPALIVE_IDLE_S = 30
_TCP_KEEPALIVE_INTERVAL_S = 10
_TCP_KEEPALIVE_COUNT = 3


def tcp_keepalive_socket_options() -> list[tuple[int, int, int]]:
    """``setsockopt`` tuples for httpx ``socket_options``: always SO_KEEPALIVE, plus idle/interval/count
    when the interpreter exposes those option names."""
    options: list[tuple[int, int, int]] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    idle = getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None)
    for opt, value in ((idle, _TCP_KEEPALIVE_IDLE_S), (getattr(socket, "TCP_KEEPINTVL", None), _TCP_KEEPALIVE_INTERVAL_S),
                       (getattr(socket, "TCP_KEEPCNT", None), _TCP_KEEPALIVE_COUNT)):
        if opt is not None:
            options.append((socket.IPPROTO_TCP, opt, value))
    return options

# DNS-over-HTTPS providers: discover Telegram API IPs the (possibly unreachable) local resolver may not
# return. Bounded so connect() isn't delayed.
_DOH_TIMEOUT = 4.0
_DOH_PROVIDERS: list[dict] = [
    {"url": "https://dns.google/resolve", "params": {"name": _TELEGRAM_API_HOST, "type": "A"}, "headers": {}},
    {
        "url": "https://cloudflare-dns.com/dns-query", "params": {"name": _TELEGRAM_API_HOST, "type": "A"},
        "headers": {"Accept": "application/dns-json"}},
]
# Last-resort IPv4 Bot API endpoints (149.154.160.0/20). Used when DoH is blocked AND as
# first-try connect targets so a blackholed IPv6 AAAA for the hostname can't pin initialize().
SEED_FALLBACK_IPS: list[str] = ["149.154.166.110", "149.154.167.220"]
_UNSET = object()


def _resolve_proxy_url(target_hosts=None) -> str | None:
    from gateway.platforms.base import resolve_proxy_url  # env vars + macOS system proxy
    return resolve_proxy_url("TELEGRAM_PROXY", target_hosts=target_hosts)


class TelegramFallbackTransport(httpx.AsyncBaseTransport):
    """Reach the Bot API via known IPv4 literals first, dual-stack hostname last. Host + SNI stay on
    api.telegram.org (like ``curl --resolve``) so a blackholed IPv6 AAAA can't pin initialize()."""

    # Bound every pool: httpx's 100-connection default × (wedged endpoint + seed IPs) can outgrow the fd limit.
    # See #63311.
    _POOL_LIMITS = httpx.Limits(max_connections=8, max_keepalive_connections=4)

    def __init__(self, fallback_ips: Iterable[str], **transport_kwargs):
        self._fallback_ips = list(dict.fromkeys(_normalize_fallback_ips(fallback_ips)))
        proxy_url = _resolve_proxy_url(target_hosts=[_TELEGRAM_API_HOST, *self._fallback_ips])
        if proxy_url and "proxy" not in transport_kwargs:
            transport_kwargs["proxy"] = proxy_url
        transport_kwargs.setdefault("limits", self._POOL_LIMITS)
        transport_kwargs.setdefault("socket_options", tcp_keepalive_socket_options())
        self._transport_kwargs = transport_kwargs
        self._primary = httpx.AsyncHTTPTransport(**transport_kwargs)
        self._primary_lock = asyncio.Lock()
        self._primary_closed = False
        # Built on demand and discarded on failure — see _reset_fallback.
        self._fallbacks: dict[str, httpx.AsyncHTTPTransport] = {}
        self._fallback_lock = asyncio.Lock()
        # ``_UNSET`` / ``None`` / ``str`` = no sticky yet / sticky hostname / sticky IPv4.
        self._sticky_ip: object = _UNSET
        self._sticky_lock = asyncio.Lock()

    async def _get_fallback(self, ip: str) -> httpx.AsyncHTTPTransport:
        async with self._fallback_lock:
            transport = self._fallbacks.get(ip)
            if transport is None:
                transport = httpx.AsyncHTTPTransport(**self._transport_kwargs)
                self._fallbacks[ip] = transport
            return transport

    async def _reset_primary(self, transport: httpx.AsyncHTTPTransport) -> None:
        # Retryable primary failures leave half-closed sockets in the pool; replace the generation first.
        async with self._primary_lock:
            if self._primary_closed or transport is not self._primary:
                return
            self._primary = httpx.AsyncHTTPTransport(**self._transport_kwargs)
        try:
            await transport.aclose()
        except Exception as exc:
            logger.debug("[Telegram] Error closing primary transport: %s", exc)

    async def _reset_fallback(self, ip: str) -> None:
        """Discard a failed fallback pool: a peer-closed connect leaves a CLOSE_WAIT socket in it, and the
        poisoned pool would leak one fd per retry.

        Retaining the poisoned pool leaks one descriptor per retry until the process hits its file limit and
        can no longer accept connections or resolve DNS (#63311).
        """
        async with self._fallback_lock:
            transport = self._fallbacks.pop(ip, None)
        if transport is None:
            return
        try:
            await transport.aclose()
        except Exception as exc:  # closing a broken pool must never mask the real error
            logger.debug("[Telegram] Error closing fallback transport %s: %s", ip, exc)

    def _attempt_order(self) -> list[Optional[str]]:
        """Sticky path first, then IPv4 literals, dual-stack hostname last (a blackholed IPv6 path never
        errors — Happy Eyeballs waits on AAAA until the OS TCP timeout and can pin the loop)."""
        order: list[Optional[str]] = []
        if self._sticky_ip is not _UNSET:
            order.append(None if self._sticky_ip is None else str(self._sticky_ip))
        order.extend(ip for ip in self._fallback_ips if ip not in order)
        if None not in order:
            order.append(None)
        return order

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != _TELEGRAM_API_HOST or not self._fallback_ips:
            return await self._primary.handle_async_request(request)
        last_error: Exception | None = None
        for ip in self._attempt_order():
            candidate = request if ip is None else _rewrite_request_for_ip(request, ip)
            transport = self._primary if ip is None else await self._get_fallback(ip)
            try:
                response = await transport.handle_async_request(candidate)
                if self._sticky_ip is _UNSET or self._sticky_ip != ip:
                    async with self._sticky_lock:
                        if self._sticky_ip is _UNSET or self._sticky_ip != ip:
                            self._sticky_ip = ip
                            if ip is not None:
                                log = logger.warning if last_error is not None else logger.info
                                log("[Telegram] Using sticky IPv4 Telegram API path %s (dual-stack hostname tried last — #87015)", ip)
                return response
            except Exception as exc:
                last_error = exc
                if not _is_retryable_connect_error(exc):
                    raise
                if self._sticky_ip is not _UNSET and ip == self._sticky_ip:
                    async with self._sticky_lock:
                        if self._sticky_ip is not _UNSET and self._sticky_ip == ip:
                            self._sticky_ip = _UNSET
                            logger.warning(
                                "[Telegram] Sticky Telegram path %s failed; re-walking IPv4 literals before the hostname",
                                ip if ip is not None else "api.telegram.org")
                if ip is None:
                    await self._reset_primary(transport)
                    logger.warning("[Telegram] Dual-stack api.telegram.org path failed (%s)", exc)
                    continue
                logger.warning("[Telegram] IPv4 Telegram API IP %s failed: %s", ip, exc)
                await self._reset_fallback(ip)
                continue
        if last_error is None:
            raise RuntimeError("All Telegram fallback IPs exhausted but no error was recorded")
        raise last_error

    async def aclose(self) -> None:
        async with self._primary_lock:
            self._primary_closed = True
            primary = self._primary
        await primary.aclose()
        async with self._fallback_lock:
            transports = list(self._fallbacks.values())
            self._fallbacks.clear()
        for transport in transports:
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
        elif addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_unspecified:
            logger.warning("Ignoring private/internal Telegram fallback IP: %s", raw)
        else:
            normalized.append(str(addr))
    return normalized


def parse_fallback_ip_env(value: str | None) -> list[str]:
    return _normalize_fallback_ips(part.strip() for part in value.split(",")) if value else []


def _resolve_system_dns() -> set[str]:
    """Return the IPv4 addresses that the OS resolver gives for api.telegram.org."""
    try:
        results = socket.getaddrinfo(_TELEGRAM_API_HOST, 443, socket.AF_INET)
        return {addr[4][0] for addr in results}
    except Exception:
        return set()


async def _query_doh_provider(client: httpx.AsyncClient, provider: dict) -> list[str]:
    """Query one DoH provider and return A-record IPs."""
    try:
        resp = await client.get(provider["url"], params=provider["params"], headers=provider["headers"])
        resp.raise_for_status()
        data = resp.json()
        ips: list[str] = []
        for answer in data.get("Answer", []):
            if answer.get("type") != 1:  # A record
                continue
            raw = answer.get("data", "").strip()
            try:
                ipaddress.ip_address(raw)
            except ValueError:
                continue
            ips.append(raw)
        return ips
    except Exception as exc:
        logger.debug("DoH query to %s failed: %s", provider["url"], exc)
        return []


async def discover_fallback_ips() -> list[str]:
    """Resolve api.telegram.org via Google + Cloudflare DoH; unique A records, in order. IPs matching the
    system resolver are deliberately KEPT (often the most reliable path). Falls back to
    ``SEED_FALLBACK_IPS`` only when DoH yields nothing usable.

    IPs that match the local system resolver are kept rather than excluded: in many networks the system-DNS
    IP is the most reliable path to api.telegram.org and a transient primary-path failure should be retried
    against the same address via the IP-rewrite path before the seed list is consulted (#14520).
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(_DOH_TIMEOUT)) as client:
        system_dns_task = asyncio.ensure_future(asyncio.to_thread(_resolve_system_dns))
        results = await asyncio.gather(*[_query_doh_provider(client, p) for p in _DOH_PROVIDERS], return_exceptions=True)
    # The getaddrinfo leg has no timeout of its own and only feeds the log line below — bound it.
    # The system-resolver leg runs socket.getaddrinfo in a worker thread with no timeout of its own — a
    # wedged OS resolver (broken VPN/DNS) can sit for minutes. Its result only feeds the no-usable-answers
    # log line below, so it must never gate discovery: bound it and move on (#63309). The DoH legs are
    # already bounded by the client timeout above.
    system_ips: set[str] = set()
    try:
        system_result = await asyncio.wait_for(system_dns_task, timeout=_DOH_TIMEOUT)
        if isinstance(system_result, set):
            system_ips = system_result
    except Exception:
        logger.debug("System-DNS resolution for %s did not complete in time", _TELEGRAM_API_HOST)
    doh_ips = [ip for r in results if isinstance(r, list) for ip in r]
    validated = _normalize_fallback_ips(list(dict.fromkeys(doh_ips)))  # dedupe, keep order
    if validated:
        logger.debug("Discovered Telegram fallback IPs via DoH: %s", ", ".join(validated))
        return validated
    logger.info(
        "DoH discovery yielded no usable IPs (system DNS: %s); using seed fallback IPs %s",
        ", ".join(system_ips) or "unknown", ", ".join(SEED_FALLBACK_IPS))
    return list(SEED_FALLBACK_IPS)


def _rewrite_request_for_ip(request: httpx.Request, ip: str) -> httpx.Request:
    original_host = request.url.host or _TELEGRAM_API_HOST
    url = request.url.copy_with(host=ip)
    headers = request.headers.copy()
    headers["host"] = original_host
    extensions = dict(request.extensions)
    extensions["sni_hostname"] = original_host
    return httpx.Request(method=request.method, url=url, headers=headers, stream=request.stream, extensions=extensions)


def _is_retryable_connect_error(exc: Exception) -> bool:
    return isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError))
