"""Broker server assembly: loopback-only binding, health endpoints, and the
aiohttp dependency gate.

This module must stay importable without aiohttp — the dependency is pulled
in lazily so ordinary Hermes imports never require the `oauth-broker` extra.
Startup refuses non-loopback binds before any dependency or socket work.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Dict, Optional

from agent.oauth_broker.account_slot import AccountSlot

logger = logging.getLogger(__name__)

DEFAULT_BROKER_PORT = 17880
DEFAULT_MAX_CONCURRENT_REQUESTS = 32

_CANONICAL_LOOPBACK_HOSTS = ("127.0.0.1", "::1")


class BrokerDependencyError(RuntimeError):
    """The broker's HTTP dependency is missing; carries install guidance."""


def validate_bind_host(host) -> str:
    """Accept only the canonical loopback literals; refuse everything else."""
    if not isinstance(host, str) or host not in _CANONICAL_LOOPBACK_HOSTS:
        raise ValueError(
            "oauth broker must bind a canonical loopback address "
            f"(one of {_CANONICAL_LOOPBACK_HOSTS}); refusing to start"
        )
    return host


def _require_aiohttp() -> None:
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        raise BrokerDependencyError(
            "The OAuth broker requires aiohttp, which is not installed. "
            "Install the pinned extra with: pip install 'hermes-agent[oauth-broker]' "
            "— nothing is installed automatically."
        ) from None


def create_server_app(
    *,
    slots: Dict[str, AccountSlot],
    local_key: str,
    upstream_origin: Optional[str] = None,
    request_body_limit: Optional[int] = None,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    client_session_factory: Optional[Callable] = None,
    allow_test_upstream: bool = False,
):
    """Build the full broker app: proxy routes plus /health endpoints."""
    _require_aiohttp()
    from aiohttp import web

    import agent.oauth_broker.proxy as proxy_mod

    app = proxy_mod.create_proxy_app(
        slots=slots,
        local_key=local_key,
        upstream_origin=(
            upstream_origin
            if upstream_origin is not None
            else proxy_mod.DEFAULT_UPSTREAM_ORIGIN
        ),
        request_body_limit=(
            request_body_limit
            if request_body_limit is not None
            else proxy_mod.DEFAULT_MAX_REQUEST_BYTES
        ),
        client_session_factory=client_session_factory,
        allow_test_upstream=allow_test_upstream,
    )

    key_bytes = local_key.encode("utf-8")
    slot_map = dict(slots)

    async def handle_health(request):
        # Liveness only — no account metadata on the unauthenticated route.
        return web.json_response({"status": "ok"})

    async def handle_health_detailed(request):
        if not proxy_mod._authorized(request, key_bytes):
            return proxy_mod._json_error(
                401, "unauthorized", "missing or invalid broker key"
            )
        accounts = []
        all_ready = True
        for alias in sorted(slot_map):
            status = slot_map[alias].status()
            account_ready = (
                status.present
                and status.healthy
                and not status.persistence_degraded
            )
            all_ready = all_ready and account_ready
            accounts.append(
                {
                    "alias": status.alias,
                    "present": status.present,
                    "healthy": account_ready,
                    "expires_at": status.expires_at,
                    "last_refresh_result": status.last_refresh_result,
                    "persistence_degraded": status.persistence_degraded,
                }
            )
        return web.json_response(
            {"status": "ok" if all_ready else "degraded", "accounts": accounts},
            status=200 if all_ready else 503,
        )

    app.router.add_get("/health", handle_health)
    app.router.add_get("/health/detailed", handle_health_detailed)

    if max_concurrent_requests and max_concurrent_requests > 0:
        semaphore = asyncio.Semaphore(max_concurrent_requests)

        @web.middleware
        async def _concurrency_guard(request, handler):
            if request.path in {"/health", "/health/detailed"}:
                return await handler(request)
            async with semaphore:
                return await handler(request)

        app.middlewares.append(_concurrency_guard)

    return app


def run_broker(
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_BROKER_PORT,
    slots: Dict[str, AccountSlot],
    local_key: str,
    upstream_origin: Optional[str] = None,
    request_body_limit: Optional[int] = None,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    client_session_factory: Optional[Callable] = None,
) -> None:
    """Run the broker until stopped. Fails closed before binding on any
    non-loopback host or missing dependency."""
    bind_host = validate_bind_host(host)
    app = create_server_app(
        slots=slots,
        local_key=local_key,
        upstream_origin=upstream_origin,
        request_body_limit=request_body_limit,
        max_concurrent_requests=max_concurrent_requests,
        client_session_factory=client_session_factory,
    )
    from aiohttp import web

    logger.info("oauth broker: starting on http://%s:%s", bind_host, port)
    # access_log=None: request lines carry query strings; our own proxy log
    # (request id, alias, status, duration) is the only per-request record.
    web.run_app(app, host=bind_host, port=port, print=None, access_log=None)


__all__ = [
    "DEFAULT_BROKER_PORT",
    "DEFAULT_MAX_CONCURRENT_REQUESTS",
    "BrokerDependencyError",
    "create_server_app",
    "run_broker",
    "validate_bind_host",
]
