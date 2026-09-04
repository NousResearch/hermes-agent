"""HTTP server that forwards OpenAI-compatible requests to a configured upstream.

Listens on ``http://<host>:<port>/v1/<path>`` and forwards each request to
``<upstream-base-url>/<path>`` with the client's ``Authorization`` header
replaced by a freshly-resolved bearer from the configured adapter. The
response body is streamed through unchanged (SSE deltas preserved).

One narrow SSE compatibility shim applies after a *clean* upstream EOF:
when a ``text/event-stream`` response carries a terminal ``finish_reason``
or ``lastOne: true`` but omits the OpenAI ``data: [DONE]`` sentinel, the
proxy appends a single ``[DONE]`` frame. It never rewrites earlier frames,
never duplicates an upstream ``[DONE]``, and never synthesizes ``[DONE]``
after an error event or a mid-stream interrupt (see
:mod:`hermes_cli.proxy.sse_done`, issue #90848).

Otherwise the server does not mediate, log, or rewrite request/response
bodies — it is a credential-attaching forwarder.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Optional

try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential
from hermes_cli.proxy.sse_done import (
    DONE_SSE_FRAME,
    SseDoneTracker,
    content_type_is_sse,
)

logger = logging.getLogger(__name__)

# Stripped when forwarding upstream: ``host``/``content-length`` are recomputed by aiohttp,
# ``authorization`` is replaced with our bearer; everything else passes through.
_HOP_BY_HOP_HEADERS = frozenset({
    "host", "content-length", "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "authorization",
})
# aiohttp recomputes Content-Encoding/Content-Length on stream — let it.
_RESPONSE_DROP_HEADERS = _HOP_BY_HOP_HEADERS | {"content-encoding", "content-length"}

DEFAULT_PORT = 8645
DEFAULT_HOST = "127.0.0.1"
# Mirrors api_server's MAX_REQUEST_BYTES (10 MB); client_max_size bounds every read path,
# including chunked bodies.
MAX_REQUEST_BYTES = 10_000_000


def _require_aiohttp() -> None:
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp is required for `hermes proxy`. Run `hermes setup` to install it.")


def _json_error(status: int, message: str, code: str = "proxy_error") -> "web.Response":
    """OpenAI-style error JSON response."""
    body = {"error": {"message": message, "type": code, "code": code}}
    return web.json_response(body, status=status)


def _filter_headers(headers, drop: frozenset = _HOP_BY_HOP_HEADERS) -> dict:
    """Strip hop-by-hop (+ auth) headers; ``drop`` widens the set for upstream responses."""
    return {key: value for key, value in headers.items() if key.lower() not in drop}


async def _open_upstream(request: "web.Request", rel_path: str, body: bytes, cred: UpstreamCredential):
    """Send the request upstream with ``cred``; returns ``(session, response)`` or
    ``(error_response, None)``."""
    upstream_url = f"{cred.base_url.rstrip('/')}{rel_path}"
    if request.query_string:  # preserved verbatim
        upstream_url = f"{upstream_url}?{request.query_string}"
    fwd_headers = _filter_headers(request.headers)
    fwd_headers["Authorization"] = f"{cred.token_type} {cred.bearer}"
    logger.debug("proxy: forwarding %s %s -> %s (body=%d bytes)", request.method, rel_path, upstream_url, len(body))
    try:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=300))
    except Exception as exc:  # pragma: no cover - aiohttp setup issue
        return _json_error(500, f"proxy session init failed: {exc}"), None
    try:
        upstream_resp = await session.request(
            request.method, upstream_url, data=body if body else None, headers=fwd_headers, allow_redirects=False
        )
    except RuntimeError as exc:
        await session.close()
        return _json_error(500, str(exc)), None
    except aiohttp.ClientError as exc:
        await session.close()
        logger.warning("proxy: upstream connection failed: %s", exc)
        return _json_error(502, f"upstream connection failed: {exc}", code="upstream_unreachable"), None
    except asyncio.TimeoutError:
        await session.close()
        return _json_error(504, "upstream request timed out", code="upstream_timeout"), None
    except Exception:
        await session.close()
        raise
    return session, upstream_resp


async def _stream_back(request: "web.Request", session, upstream_resp) -> "web.StreamResponse":
    """Relay status + filtered headers, then the body chunk-by-chunk, appending a missing SSE
    ``[DONE]`` only after a clean EOF."""
    resp = web.StreamResponse(
        status=upstream_resp.status, headers=_filter_headers(upstream_resp.headers, _RESPONSE_DROP_HEADERS)
    )
    await resp.prepare(request)
    done_tracker: Optional[SseDoneTracker] = None
    if content_type_is_sse(upstream_resp.headers):
        done_tracker = SseDoneTracker()
    try:
        async for chunk in upstream_resp.content.iter_any():
            if chunk:
                if done_tracker is not None:
                    done_tracker.feed(chunk)
                await resp.write(chunk)
        if done_tracker is not None and done_tracker.should_append_done():
            try:
                await resp.write(DONE_SSE_FRAME)
            except Exception as exc:  # client hung up at EOF — harmless
                logger.debug("proxy: DONE append skipped: %s", exc)
    except (aiohttp.ClientError, asyncio.CancelledError, OSError) as exc:
        if done_tracker is not None:
            done_tracker.mark_interrupted()
        logger.warning("proxy: streaming interrupted: %s", exc)
    finally:
        upstream_resp.release()
        await session.close()
    await resp.write_eof()
    return resp


def create_app(adapter: UpstreamAdapter) -> "web.Application":
    """Build the aiohttp application bound to a specific upstream adapter.

    Every adapter method is synchronous and blocking (the Nous adapter takes the 15s cross-process
    ``_auth_store_lock()`` and may POST a token refresh; xAI rotates its key pool under a lock),
    so all three are run via ``asyncio.to_thread`` — a contended lock or refresh must never freeze
    the single loop and every other in-flight streaming completion.
    """
    _require_aiohttp()
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    # AppKey: forward-compat with aiohttp versions that strip bare-string keys.
    app[web.AppKey("adapter", UpstreamAdapter)] = adapter

    async def handle_health(request: "web.Request") -> "web.Response":
        # ``is_authenticated`` is documented as cheap (see UpstreamAdapter),
        # but both shipped adapters read auth state off disk, and the Nous one
        # does it under ``_auth_store_lock()`` — 15s cross-process. Offload it
        # so a healthcheck poll can never freeze the loop behind a lock held by
        # a concurrent ``hermes auth`` command.
        authenticated = await asyncio.to_thread(adapter.is_authenticated)
        return web.json_response(
            {
                "status": "ok",
                "upstream": adapter.display_name,
                "authenticated": authenticated,
            }
        )

    async def handle_proxy(request: "web.Request") -> "web.StreamResponse":
        rel_path = "/" + request.match_info.get("tail", "").lstrip("/")
        if rel_path not in adapter.allowed_paths:
            allowed = ", ".join(sorted(adapter.allowed_paths))
            return _json_error(
                404, f"Path /v1{rel_path} is not forwarded by this proxy. Allowed: {allowed}", code="path_not_allowed"
            )

        # ``UpstreamAdapter.get_credential`` is synchronous and hard-blocking:
        # the Nous adapter takes ``_auth_store_lock()`` (a cross-process lock
        # with a 15s timeout), reads auth.json, and may perform a token-refresh
        # POST, taking the lock a second time to persist a terminal error. Run
        # it on a worker thread so a refresh or a contended lock cannot freeze
        # every other in-flight streaming completion on this single loop.
        try:
            cred = await asyncio.to_thread(adapter.get_credential)
        except Exception as exc:
            logger.warning("proxy: credential resolution failed: %s", exc)
            return _json_error(401, str(exc), code="upstream_auth_failed")
        # Body read into memory once (chat/embeddings payloads are small); switch to streaming
        # if large multipart uploads ever need forwarding.
        body = await request.read()
        session, upstream_resp = await _open_upstream(request, rel_path, body, cred)
        if upstream_resp is None:
            return session
        if upstream_resp.status in {401, 429}:
            # Third and last blocking method on the adapter contract, and the
            # most expensive: the Nous adapter routes this straight into
            # ``_get_credential(force_refresh=True)``, so the refresh POST that
            # ``get_credential`` only performs near expiry is unconditional
            # here — under the same 15s cross-process ``_auth_store_lock()``.
            # The xAI adapter loads its key pool off disk and rotates it under
            # ``self._lock``. Offload it for the same reason as the two above.
            try:
                retry_cred = await asyncio.to_thread(
                    adapter.get_retry_credential,
                    failed_credential=cred,
                    status_code=upstream_resp.status,
                )
            except Exception as exc:
                logger.warning("proxy: retry credential resolution failed: %s", exc)
                retry_cred = None
            if retry_cred is not None:
                upstream_resp.release()
                await session.close()
                session, upstream_resp = await _open_upstream(request, rel_path, body, retry_cred)
                if upstream_resp is None:
                    return session_or_response
                session = session_or_response

        # Stream response back. Headers first, then chunked body.
        resp = web.StreamResponse(
            status=upstream_resp.status,
            headers=_filter_response_headers(upstream_resp.headers),
        )
        await resp.prepare(request)

        # Track SSE terminal markers so we can append a missing [DONE]
        # after clean EOF without rewriting any earlier frames.
        done_tracker: Optional[SseDoneTracker] = None
        if content_type_is_sse(upstream_resp.headers):
            done_tracker = SseDoneTracker()

        try:
            async for chunk in upstream_resp.content.iter_any():
                if chunk:
                    if done_tracker is not None:
                        done_tracker.feed(chunk)
                    await resp.write(chunk)
            if done_tracker is not None and done_tracker.should_append_done():
                try:
                    await resp.write(DONE_SSE_FRAME)
                except Exception as exc:  # client hung up at EOF — harmless
                    logger.debug("proxy: DONE append skipped: %s", exc)
        except (aiohttp.ClientError, asyncio.CancelledError, OSError) as exc:
            if done_tracker is not None:
                done_tracker.mark_interrupted()
            logger.warning("proxy: streaming interrupted: %s", exc)
        finally:
            upstream_resp.release()
            await session.close()

        await resp.write_eof()
        return resp

    # /health doesn't go through the upstream
    app.router.add_get("/health", handle_health)
    # Catch-all under /v1 — forwards if the path is allowed.
    app.router.add_route("*", "/v1/{tail:.*}", handle_proxy)

    app.router.add_get("/health", handle_health)  # never goes upstream
    app.router.add_route("*", "/v1/{tail:.*}", handle_proxy)  # forwards if the path is allowed
    return app


async def run_server(
    adapter: UpstreamAdapter,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    shutdown_event: Optional[asyncio.Event] = None,
) -> None:
    """Run the proxy in the current event loop until shutdown_event is set."""
    _require_aiohttp()
    app = create_app(adapter)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    logger.info("proxy: listening on http://%s:%d/v1 -> %s", host, port, adapter.display_name)
    stop_event = shutdown_event or asyncio.Event()
    if shutdown_event is None:  # we own the loop's lifetime → wire signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)  # windows-footgun: ok
            except NotImplementedError:
                pass  # Windows / restricted envs — Ctrl+C still raises KeyboardInterrupt
    try:
        await stop_event.wait()
    finally:
        logger.info("proxy: shutting down")
        await runner.cleanup()


__all__ = ["create_app", "run_server", "DEFAULT_HOST", "DEFAULT_PORT", "AIOHTTP_AVAILABLE"]
