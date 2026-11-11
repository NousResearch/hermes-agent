"""Telegram HTTP pool generations that close only once nothing still uses them.

Closing an httpcore pool while a request is mid-connect empties its connection list, and the
connection that finishes connecting afterwards serves its request and then sits in no list at all:
when Telegram's edge (or the proxy) later FINs it, the socket stays in CLOSE_WAIT until the
process exits. The Telegram transport retires a pool on every failed connect and on every
``HTTPXRequest.shutdown()`` drain, so busy gateways orphaned one socket per in-flight request on
each retirement. A :class:`PoolGeneration` counts the requests (and open response bodies) using
it; ``retire()`` closes it at once when idle, else when the last one finishes, with a grace bound
so a wedged long-poll cannot pin it forever.

A failed TLS handshake inside an HTTP proxy tunnel also leaves httpcore's tunnel connection
ACTIVE in the pool with no request owning it, holding a pool slot and the proxy socket forever;
:func:`_close_unowned_connections` reclaims those after a failed request.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# A lease ends when its PTB call ends: PTB reads every response through ``client.request``, which
# closes the body, and httpx closes it on cancellation too, so a stalled send is released by the
# adapter's own deadline. This bound only catches a body nobody ever closes; it sits far past any
# legitimate request (media uploads scale their budgets well beyond 300 s) so it never cuts one off.
RETIRE_GRACE_S = 3600.0
# Every Telegram pool together may use at most 1/_FD_SHARE of RLIMIT_NOFILE: two PTB clients, each
# with one pool per path (sticky IPv4 literals + dual-stack hostname).
_FD_SHARE = 4
_PTB_CLIENTS = 2
_MIN_POOL_CONNECTIONS = 4
MAX_FALLBACK_IPS = 4

_close_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.ensure_future(coro)
    _close_tasks.add(task)
    task.add_done_callback(_close_tasks.discard)


def _nofile_soft_limit() -> Optional[int]:
    try:
        import resource
    except ImportError:  # Windows: no RLIMIT_NOFILE (handles are not bounded the same way)
        return None
    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return soft if soft > 0 else None


def fd_bounded_limits(limits: httpx.Limits, paths: int) -> httpx.Limits:
    """``limits`` with ``max_connections`` capped so ``_PTB_CLIENTS × paths`` pools fit in their
    share of the process descriptor budget; the same object when no cap is needed."""
    soft = _nofile_soft_limit()
    if soft is None or limits.max_connections is None:
        return limits
    cap = max(_MIN_POOL_CONNECTIONS, soft // _FD_SHARE // (_PTB_CLIENTS * max(paths, 1)))
    if limits.max_connections <= cap:
        return limits
    logger.info("[Telegram] HTTP pool capped at %d connections per path (requested %d; RLIMIT_NOFILE %d, %d paths)",
                cap, limits.max_connections, soft, paths)
    keepalive = limits.max_keepalive_connections
    return httpx.Limits(max_connections=cap, max_keepalive_connections=cap if keepalive is None else min(keepalive, cap),
                        keepalive_expiry=limits.keepalive_expiry)


def _close_unowned_connections(transport) -> None:
    """Close pool connections that are neither idle, closed, nor serving a live request: the proxy
    tunnel httpcore leaves ACTIVE after a failed TLS-over-CONNECT handshake. A request's pool entry
    lives until its response body closes, so a connection no entry points at has no user left."""
    pool = getattr(transport, "_pool", None)
    requests = getattr(pool, "_requests", None)
    if requests is None:
        return
    owned = {id(entry.connection) for entry in requests if entry.connection is not None}
    for connection in list(getattr(pool, "connections", ())):
        if id(connection) not in owned and not connection.is_idle() and not connection.is_closed():
            logger.debug("[Telegram] Closing unowned pool connection %s", connection.info())
            _spawn(connection.aclose())


class _LeasedStream(httpx.AsyncByteStream):
    """A response body that returns its generation's lease when closed."""

    def __init__(self, stream, release) -> None:
        self._stream, self._release = stream, release

    async def __aiter__(self):
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        try:
            await self._stream.aclose()
        finally:
            self._release()


class PoolGeneration(httpx.AsyncBaseTransport):
    """One ``httpx.AsyncHTTPTransport`` plus a count of the requests still using it."""

    def __init__(self, **transport_kwargs) -> None:
        self._inner = httpx.AsyncHTTPTransport(**transport_kwargs)
        self._active = 0
        self._retired = False
        self._closed = False
        self._grace: Optional[asyncio.TimerHandle] = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._active += 1
        try:
            response = await self._inner.handle_async_request(request)
        except BaseException:
            _close_unowned_connections(self._inner)
            self._release()
            raise
        released = False

        def release_once() -> None:
            nonlocal released
            if not released:
                released = True
                self._release()

        response.stream = _LeasedStream(response.stream, release_once)
        return response

    def _release(self) -> None:
        self._active -= 1
        if self._retired and self._active <= 0:
            _spawn(self._close())

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._grace is not None:
            self._grace.cancel()  # the timer holds this generation; drop it once closed
            self._grace = None
        try:
            await self._inner.aclose()
        except (OSError, RuntimeError, httpx.HTTPError) as exc:  # a broken pool's close never masks the caller's error
            logger.debug("[Telegram] Error closing retired Telegram pool: %s", exc)

    async def retire(self) -> None:
        """Close now when idle; otherwise when the last request using it finishes, or after
        ``RETIRE_GRACE_S``."""
        self._retired = True
        if self._active <= 0:
            await self._close()
            return
        if self._grace is None:
            self._grace = asyncio.get_running_loop().call_later(RETIRE_GRACE_S, lambda: _spawn(self._close()))

    async def aclose(self) -> None:
        await self.retire()
