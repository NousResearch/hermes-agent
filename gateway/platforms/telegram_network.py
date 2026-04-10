"""Telegram-specific network helpers.

Provides a hostname-preserving fallback transport for networks where
api.telegram.org resolves to an endpoint that is unreachable from the current
host. The transport keeps the logical request host and TLS SNI as
api.telegram.org while retrying the TCP connection against one or more fallback
IPv4 addresses.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import socket
from typing import Iterable, Optional

import httpx

logger = logging.getLogger(__name__)

_TELEGRAM_API_HOST = "api.telegram.org"

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


def _resolve_proxy_url() -> str | None:
    # Delegate to shared implementation (env vars + macOS system proxy detection)
    from gateway.platforms.base import resolve_proxy_url
    return resolve_proxy_url()


class TelegramFallbackTransport(httpx.AsyncBaseTransport):
    """Retry Telegram Bot API requests via fallback IPs while preserving TLS/SNI.

    Requests continue to target https://api.telegram.org/... logically, but on
    connect failures the underlying TCP connection is retried against a known
    reachable IP. This is effectively the programmatic equivalent of
    ``curl --resolve api.telegram.org:443:<ip>``.
    """

    def __init__(self, fallback_ips: Iterable[str], **transport_kwargs):
        self._fallback_ips = [ip for ip in dict.fromkeys(_normalize_fallback_ips(fallback_ips))]
        proxy_url = _resolve_proxy_url()
        if proxy_url and "proxy" not in transport_kwargs:
            transport_kwargs["proxy"] = proxy_url
        self._primary = httpx.AsyncHTTPTransport(**transport_kwargs)
        self._fallbacks = {
            ip: httpx.AsyncHTTPTransport(**transport_kwargs) for ip in self._fallback_ips
        }
        self._sticky_ip: Optional[str] = None
        self._sticky_lock = asyncio.Lock()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != _TELEGRAM_API_HOST or not self._fallback_ips:
            return await self._primary.handle_async_request(request)

        sticky_ip = self._sticky_ip
        attempt_order: list[Optional[str]] = [sticky_ip] if sticky_ip else [None]
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
                if ip is None:
                    logger.warning(
                        "[Telegram] Primary api.telegram.org connection failed (%s); trying fallback IPs %s",
                        exc,
                        ", ".join(self._fallback_ips),
                    )
                    continue
                logger.warning("[Telegram] Fallback IP %s failed: %s", ip, exc)
                continue

        assert last_error is not None
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

    Resolves api.telegram.org through Google and Cloudflare DoH, collects all
    unique IPs, and excludes the system-DNS-resolved IP (which is presumably
    unreachable on this network).  Falls back to a hardcoded seed list when DoH
    is also unavailable.
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

    # Deduplicate preserving order, exclude system-DNS IPs
    seen: set[str] = set()
    candidates: list[str] = []
    for ip in doh_ips:
        if ip not in seen and ip not in system_ips:
            seen.add(ip)
            candidates.append(ip)

    # Validate through existing normalization
    validated = _normalize_fallback_ips(candidates)

    if validated:
        logger.debug("Discovered Telegram fallback IPs via DoH: %s", ", ".join(validated))
        return validated

    logger.info(
        "DoH discovery yielded no new IPs (system DNS: %s); using seed fallback IPs %s",
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
