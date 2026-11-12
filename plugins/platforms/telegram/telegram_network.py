"""Telegram network helpers: a hostname-preserving fallback transport (Host + SNI stay
api.telegram.org while TCP retries known IPv4 literals) plus DoH-based IP discovery."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import socket
import ssl
import sys
import time
from typing import Iterable, Optional

import httpx

from agent.ssl_verify import platform_ssl_context
from plugins.platforms.telegram.telegram_pool import MAX_FALLBACK_IPS, PoolGeneration, fd_bounded_limits

logger = logging.getLogger(__name__)

_TELEGRAM_API_HOST = "api.telegram.org"


def _describe_transport_error(error: Exception) -> str:
    """Return a non-empty, secret-safe exception representation for diagnostics."""
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(repr(error), force=True)
    except Exception:
        return type(error).__name__

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
# A real A-record answer is well under 1 KiB; anything past this is a misbehaving endpoint, captive
# portal or proxy, so that provider is ignored rather than buffered (#55087).
_DOH_MAX_BODY_BYTES = 64 * 1024
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
# One request's whole fallback walk shares this CONNECT budget: half the adapter's 30s telegram-send
# deadline, so the walk ends with its own (safely retryable) ConnectTimeout before that deadline fires
# and masks it as an ambiguous, non-retryable TimeoutError (#136412).
_WALK_CONNECT_BUDGET_S = 15.0


def _env_number(key: str, default, cast):
    raw = os.getenv(key, "").strip()
    try:
        return cast(raw) if raw else default
    except (TypeError, ValueError):
        return default


def ptb_request_kwargs() -> dict:
    """PTB ``HTTPXRequest`` sizing and timeouts (``HERMES_TELEGRAM_HTTP_*`` overrides): PTB's 1s pool
    timeout trips "Pool timeout" on flaky networks. ``media_write_timeout`` is httpx's per-write
    stall budget, so 60s rides out congested-link buffer stalls."""
    return {
        "connection_pool_size": _env_number("HERMES_TELEGRAM_HTTP_POOL_SIZE", 512, int),
        "pool_timeout": _env_number("HERMES_TELEGRAM_HTTP_POOL_TIMEOUT", 8.0, float),
        "connect_timeout": _env_number("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", 10.0, float),
        "read_timeout": _env_number("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", 20.0, float),
        "write_timeout": _env_number("HERMES_TELEGRAM_HTTP_WRITE_TIMEOUT", 20.0, float),
        "media_write_timeout": 60.0,
    }


def getupdates_keepalive_connections() -> int:
    """Pooled sockets the getUpdates long-poll may reuse. Windows keeps 0: a long-poll socket the
    server closed sits in CLOSE_WAIT there and a reused one deadlocks polling (#87057). Elsewhere a
    fresh TCP+TLS connection per ~10 s poll parks every closed socket in client-side TIME_WAIT,
    which exhausts macOS's 16k ephemeral range after ~2 days and turns each connect blip on the
    path into a polling gap (#107880); one reused socket is enough for the single in-flight poll."""
    return 0 if sys.platform == "win32" else 1


def build_ptb_requests(request_cls, *, fallback_ips: list[str], proxy_url: Optional[str], verify=None) -> tuple:
    """The (general, getUpdates) ``request_cls`` (PTB ``HTTPXRequest``) pair the gateway adapter and
    the standalone sender both use. Each client is served by its own ``TelegramFallbackTransport``:
    the fallback-IP walk when ``fallback_ips`` and no proxy, else every request through
    ``proxy_url`` (or direct DNS); every client verifies with ``verify`` (default: the cached
    ``platform_ssl_context()``).

    The proxy decision made here is final: the transport gets ``proxy=proxy_url`` (``None`` =
    direct) so it never re-resolves HTTPS_PROXY against api.telegram.org, which sent a NO_PROXY'd
    custom ``base_url`` server through the proxy (#47188).

    CLOSE_WAIT fd leak (#31599, #18451): PTB builds its httpx client with only
    ``max_connections`` and httpx's default 5s keepalive; ``platform_httpx_limits()`` keepalive
    tuning is injected while keeping PTB's pool size, capped so every Telegram pool fits the process
    descriptor budget (#82678). getUpdates keeps ``getupdates_keepalive_connections()`` sockets (see
    there for the per-OS reason). httpx ignores client ``limits`` when a custom ``transport`` is
    supplied (#58790), so they go to the transport too."""
    from gateway.platforms._http_client_limits import platform_httpx_limits
    verify = platform_ssl_context() if verify is None else verify
    kwargs = ptb_request_kwargs()
    ips = _cap_fallback_ips(fallback_ips) if not proxy_url else []
    base = platform_httpx_limits()
    pool = updates = None
    if base is not None:
        pool = fd_bounded_limits(httpx.Limits(
            max_connections=kwargs["connection_pool_size"], max_keepalive_connections=base.max_keepalive_connections,
            keepalive_expiry=base.keepalive_expiry), paths=len(ips) + 1)
        updates = httpx.Limits(max_connections=pool.max_connections,
                               max_keepalive_connections=getupdates_keepalive_connections(),
                               keepalive_expiry=base.keepalive_expiry)
    transport_kwargs = {"socket_options": tcp_keepalive_socket_options(), "verify": verify,
                        "connect_budget": _WALK_CONNECT_BUDGET_S, "proxy": classic_proxy(proxy_url, verify)}
    extra = {"proxy": proxy_url} if proxy_url else {}

    def request(limits) -> object:
        limited = {"limits": limits} if limits is not None else {}
        transport = TelegramFallbackTransport(ips, **transport_kwargs, **limited)
        # A proxied or direct client keeps PTB's own ``proxy=`` shape; the transport serves it as
        # the ``all://`` mount, which replaces the proxy mount httpx builds from that argument.
        # ``trust_env=False``: the proxy was already resolved above (env, NO_PROXY,
        # ``gateway.trust_env``), so httpx must not add its own env-proxy mounts in front.
        route = {"transport": transport} if ips else {"mounts": {"all://": transport}, "trust_env": False}
        return request_cls(**kwargs, **extra, httpx_kwargs={"verify": verify, **route, **limited})

    return request(pool), request(updates)


def proxy_target_hosts(base_url: Optional[str], fallback_ips: Iterable[str] = ()) -> list[str]:
    """The hosts a Bot API request actually reaches, for the NO_PROXY check: a configured
    ``base_url`` server's ``host:port`` (a local ``telegram-bot-api`` on 127.0.0.1:8081 must
    match ``NO_PROXY=127.0.0.1``), else api.telegram.org and its fallback IPs."""
    if base_url:
        from urllib.parse import urlsplit
        parts = urlsplit(base_url if "//" in base_url else f"//{base_url}")
        try:
            port = parts.port or (80 if parts.scheme == "http" else 443)
        except ValueError:
            port = None
        host = parts.hostname or ""
        if host:
            host = f"[{host}]" if ":" in host else host
            return [f"{host}:{port}" if port else host]
    return [_TELEGRAM_API_HOST, *fallback_ips]


def telegram_verify(extra: Optional[dict], default):
    """The TLS context every Telegram client verifies with: ``default`` (the platform trust
    context; ``None`` lets ``build_ptb_requests`` pick it), or, with ``platforms.telegram.extra.tls_classic_key_exchange: true``, the same trust
    offering only the classic X25519 key share. That is for network paths that reset OpenSSL 3.5's
    larger post-quantum ClientHello, where every Bot API path then fails with an empty
    ``ConnectError`` (#147179). It keeps TLS 1.3, certificate and hostname checks. A value other
    than a boolean is refused with a warning and keeps the default."""
    value = (extra or {}).get("tls_classic_key_exchange", False)
    if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "on", "false", "0", "no", "off", ""}:
        value = value.strip().lower() in {"true", "1", "yes", "on"}
    if not isinstance(value, bool):
        logger.warning("[Telegram] Ignoring platforms.telegram.extra.tls_classic_key_exchange=%r: expected true or false",
                       value)
        return default
    if not value:
        return default
    logger.info("[Telegram] TLS: classic X25519 key exchange (tls_classic_key_exchange)")
    return _classic_key_exchange_context()


_CLASSIC_CONTEXTS: dict[int, ssl.SSLContext] = {}


def classic_proxy(proxy_url, verify):
    """``proxy_url`` for an httpx client verifying with ``verify``. An ``https://`` proxy has its own
    outer TLS handshake, which httpx runs with the proxy's ``ssl_context``, never the client's
    ``verify``; with the classic key exchange on, that outer hello would still be the large one the
    path resets, so the proxy leg gets the same classic context. Anything else passes unchanged."""
    if (isinstance(proxy_url, str) and proxy_url.strip().lower().startswith("https://")
            and any(verify is ctx for ctx in _CLASSIC_CONTEXTS.values())):
        return httpx.Proxy(proxy_url, ssl_context=verify)
    return proxy_url


def _classic_key_exchange_context() -> ssl.SSLContext:
    """A context with ``platform_ssl_context()``'s trust (platform store plus ``SSL_CERT_FILE``)
    offering only the classic X25519 share: a ~0.5 KB ClientHello instead of the ~1.5 KB
    post-quantum hybrid one. Cached per base context so clients keep one stable identity.
    truststore's injected context class keeps an inner context and does not forward
    ``set_ecdh_curve``, so the group is set on that inner context when there is one."""
    base = platform_ssl_context()
    ctx = _CLASSIC_CONTEXTS.get(id(base))
    if ctx is None:
        ctx = type(base)(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_default_certs()
        ca = os.environ.get("SSL_CERT_FILE", "").strip()
        if ca and os.path.isfile(ca):
            ctx.load_verify_locations(cafile=ca)
        getattr(ctx, "_ctx", ctx).set_ecdh_curve("X25519")
        _CLASSIC_CONTEXTS[id(base)] = ctx
    return ctx


def _resolve_proxy_url(target_hosts=None) -> str | None:
    from gateway.platforms.base import resolve_proxy_url  # env vars + macOS system proxy
    return resolve_proxy_url("TELEGRAM_PROXY", target_hosts=target_hosts)


class TelegramFallbackTransport(httpx.AsyncBaseTransport):
    """Reach the Bot API via known IPv4 literals first, dual-stack hostname last. Host + SNI stay on
    api.telegram.org (like ``curl --resolve``) so a blackholed IPv6 AAAA can't pin initialize().

    Every path is a :class:`PoolGeneration`: a failed path, and ``aclose()`` (PTB's
    ``HTTPXRequest.shutdown()`` drain, which reuses this object on the next ``initialize()``),
    swap in a fresh generation and retire the old one, which closes once the requests still on it
    finish, so neither in-flight requests nor their sockets are dropped (CLOSE_WAIT leak, #142971)."""

    # Bound every pool: httpx's 100-connection default × (wedged endpoint + seed IPs) can outgrow the fd limit.
    # See #63311.
    _POOL_LIMITS = httpx.Limits(max_connections=8, max_keepalive_connections=4)

    def __init__(self, fallback_ips: Iterable[str], *, connect_budget: Optional[float] = None, **transport_kwargs):
        # Seconds of CONNECT time one request's whole walk may spend; ``None`` leaves each path the
        # client's full per-request connect timeout.
        self._connect_budget = connect_budget
        self._fallback_ips = list(dict.fromkeys(_normalize_fallback_ips(fallback_ips)))
        if "proxy" not in transport_kwargs:
            proxy_url = _resolve_proxy_url(target_hosts=[_TELEGRAM_API_HOST, *self._fallback_ips])
            if proxy_url:
                transport_kwargs["proxy"] = classic_proxy(proxy_url, transport_kwargs.get("verify"))
        transport_kwargs.setdefault("limits", self._POOL_LIMITS)
        transport_kwargs.setdefault("socket_options", tcp_keepalive_socket_options())
        self._transport_kwargs = transport_kwargs
        # ``aclose()`` empties it; the next request then starts a fresh generation.
        self._primary: Optional[PoolGeneration] = PoolGeneration(**transport_kwargs)
        self._primary_lock = asyncio.Lock()
        # Built on demand and retired on failure — see _reset_fallback.
        self._fallbacks: dict[str, PoolGeneration] = {}
        self._fallback_lock = asyncio.Lock()
        # ``_UNSET`` / ``None`` / ``str`` = no sticky yet / sticky hostname / sticky IPv4.
        self._sticky_ip: object = _UNSET
        self._sticky_lock = asyncio.Lock()
        self._last_failure: tuple[str, str] | None = None

    async def _get_primary(self) -> PoolGeneration:
        async with self._primary_lock:
            if self._primary is None:
                self._primary = PoolGeneration(**self._transport_kwargs)
            return self._primary

    async def _get_fallback(self, ip: str) -> PoolGeneration:
        async with self._fallback_lock:
            transport = self._fallbacks.get(ip)
            if transport is None:
                transport = PoolGeneration(**self._transport_kwargs)
                self._fallbacks[ip] = transport
            return transport

    async def _reset_primary(self, transport: PoolGeneration) -> None:
        # Retryable primary failures leave half-closed sockets in the pool; replace the generation first.
        async with self._primary_lock:
            if transport is not self._primary:
                return
            self._primary = None
        await transport.retire()

    async def _reset_fallback(self, ip: str, transport: Optional[PoolGeneration] = None) -> None:
        """Retire a failed fallback pool: a peer-closed connect leaves a CLOSE_WAIT socket in it, and the
        poisoned pool would leak one fd per retry (#63311). Only the generation that failed is
        retired, never a fresh one a concurrent request already swapped in."""
        async with self._fallback_lock:
            current = self._fallbacks.get(ip)
            if current is None or (transport is not None and current is not transport):
                return
            del self._fallbacks[ip]
        await current.retire()

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

    def _connect_share(self, order: list, index: int, walk_started: float) -> float:
        """This path's slice of the remaining walk budget. The first (sticky / last-known-good) path
        keeps all but one average share, so at stock defaults (10s connect, 15s budget, 3 paths) it
        keeps its full 10s; later paths split what is left evenly."""
        remaining = self._connect_budget - (time.monotonic() - walk_started)
        paths_left = len(order) - index
        return remaining - remaining / paths_left if index == 0 and paths_left > 1 else remaining / paths_left

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != _TELEGRAM_API_HOST or not self._fallback_ips:
            return await (await self._get_primary()).handle_async_request(request)
        last_error: Exception | None = None
        order = self._attempt_order()
        walk_started = time.monotonic()
        for index, ip in enumerate(order):
            candidate = request
            if self._connect_budget is not None:
                share = self._connect_share(order, index, walk_started)
                if share <= 0 and last_error is not None:
                    break  # budget spent: surface the last ConnectTimeout before the send deadline masks it
                candidate = _request_with_connect_budget(request, share)
            if ip is not None:
                candidate = _rewrite_request_for_ip(candidate, ip)
            transport = await self._get_primary() if ip is None else await self._get_fallback(ip)
            try:
                response = await transport.handle_async_request(candidate)
                if self._last_failure is not None:
                    failed_path, failure = self._last_failure
                    self._last_failure = None
                    logger.info(
                        "[Telegram] Telegram API transport recovered via %s after %s failed: %s",
                        ip or _TELEGRAM_API_HOST,
                        failed_path,
                        failure,
                    )
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
                path = ip or _TELEGRAM_API_HOST
                failure = _describe_transport_error(exc)
                self._last_failure = (path, failure)
                if self._sticky_ip is not _UNSET and ip == self._sticky_ip:
                    async with self._sticky_lock:
                        if self._sticky_ip is not _UNSET and self._sticky_ip == ip:
                            self._sticky_ip = _UNSET
                            logger.warning(
                                "[Telegram] Sticky Telegram path %s failed; re-walking IPv4 literals before the hostname",
                                ip if ip is not None else "api.telegram.org")
                if ip is None:
                    await self._reset_primary(transport)
                    logger.warning("[Telegram] Dual-stack api.telegram.org path failed (%s)", failure)
                    continue
                logger.warning("[Telegram] IPv4 Telegram API IP %s failed: %s", ip, failure)
                await self._reset_fallback(ip, transport)
                continue
        if last_error is None:
            raise RuntimeError("All Telegram fallback IPs exhausted but no error was recorded")
        raise last_error

    async def aclose(self) -> None:
        """Retire every pool generation; the next request builds fresh ones, because PTB's
        ``HTTPXRequest.initialize()`` wraps its next client around this same transport after a
        ``shutdown()`` drain."""
        async with self._primary_lock:
            retired = [self._primary] if self._primary is not None else []
            self._primary = None
        async with self._fallback_lock:
            retired.extend(self._fallbacks.values())
            self._fallbacks.clear()
        # Every detached pool is retired before any close is awaited, and the closes run side by
        # side: one hung close (cut off by the adapter's bounded drain) must not leave the others
        # unretired, because nothing could reach them again once both maps are empty.
        for transport in retired:
            transport.mark_retired()
        await asyncio.gather(*(transport.retire() for transport in retired))


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


def _cap_fallback_ips(values: Iterable[str]) -> list[str]:
    """Normalized, deduped ``values`` with at most ``MAX_FALLBACK_IPS`` non-seed IPs (the FD cap).
    ``SEED_FALLBACK_IPS`` never count toward the cap and keep their place, so the tail
    ``discover_fallback_ips`` puts them in survives any number of DoH answers."""
    ips = list(dict.fromkeys(_normalize_fallback_ips(values)))
    dropped = [ip for ip in ips if ip not in SEED_FALLBACK_IPS][MAX_FALLBACK_IPS:]
    if dropped:
        logger.info("[Telegram] Using the first %d fallback IPs plus the seed IPs; ignoring %s",
                    MAX_FALLBACK_IPS, ", ".join(dropped))
    return [ip for ip in ips if ip not in dropped]


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
    """Query one DoH provider and return A-record IPs; a body over ``_DOH_MAX_BODY_BYTES`` counts as a
    failed provider. The raw (identity-requested) stream is counted, so a compressed body cannot
    inflate past the cap before it is measured."""
    try:
        headers = {**provider["headers"], "Accept-Encoding": "identity"}
        async with client.stream("GET", provider["url"], params=provider["params"], headers=headers) as resp:
            resp.raise_for_status()
            body = bytearray()
            async for chunk in resp.aiter_raw():
                body.extend(chunk)
                if len(body) > _DOH_MAX_BODY_BYTES:
                    raise ValueError(f"DoH response exceeds {_DOH_MAX_BODY_BYTES} bytes")
        data = json.loads(body)
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


async def discover_fallback_ips(verify=None) -> list[str]:
    """Resolve api.telegram.org via Google + Cloudflare DoH; unique A records, in order. IPs matching the
    system resolver are deliberately KEPT (often the most reliable path); ``SEED_FALLBACK_IPS`` always
    form the tail, so a one-record DoH answer cannot make one edge the fallback path's single point of
    failure (the adapter discovers once per connect, so a dead lone IP would strand polling).

    IPs that match the local system resolver are kept rather than excluded: in many networks the system-DNS
    IP is the most reliable path to api.telegram.org and a transient primary-path failure should be retried
    against the same address via the IP-rewrite path before the seed list is consulted (#14520).
    ``verify`` is the Telegram client's TLS context (``telegram_verify``), default the platform one.
    """
    verify = platform_ssl_context() if verify is None else verify
    async with httpx.AsyncClient(timeout=httpx.Timeout(_DOH_TIMEOUT), verify=verify) as client:
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
        fallback_ips = list(dict.fromkeys([*validated, *SEED_FALLBACK_IPS]))
        logger.debug("Discovered Telegram fallback IPs via DoH; effective fallback pool: %s", ", ".join(fallback_ips))
        return fallback_ips
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


def _request_with_connect_budget(request: httpx.Request, connect_budget: float) -> httpx.Request:
    """``request`` with its httpx ``timeout`` extension's connect phase narrowed to ``connect_budget``
    (the transport honours it per request). A tighter configured connect, or a request with no
    timeout dict, is returned unchanged."""
    timeout = request.extensions.get("timeout")
    if not isinstance(timeout, dict) or (timeout.get("connect") is not None and timeout["connect"] <= connect_budget):
        return request
    extensions = {**request.extensions, "timeout": {**timeout, "connect": max(connect_budget, 0.0)}}
    return httpx.Request(method=request.method, url=request.url, headers=request.headers, stream=request.stream,
                         extensions=extensions)


def _is_retryable_connect_error(exc: Exception) -> bool:
    return isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError))
