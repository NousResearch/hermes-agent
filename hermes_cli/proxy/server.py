"""HTTP server that forwards OpenAI-compatible requests to a configured upstream.

Listens on ``http://<host>:<port>/v1/<path>`` and forwards each request to
``<upstream-base-url>/<path>`` with the client's ``Authorization`` header
replaced by a freshly-resolved bearer from the configured adapter. The
response is streamed back unmodified, preserving SSE.

The server is intentionally minimal: it does NOT mediate, log, transform,
or rewrite request/response bodies. It's a credential-attaching forwarder.

One exception: ``GET /v1/models`` responses get their catalog envelope
translated so both OpenAI-shaped clients (``{"data": [...]}``) and Codex
CLI (``{"models": [ModelInfo]}``) can consume the upstream catalog.
"""

from __future__ import annotations

import asyncio
import json
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

logger = logging.getLogger(__name__)

# Codex CLI decodes ``GET /v1/models`` into
# ``codex_protocol::openai_models::ModelsResponse { models: Vec<ModelInfo> }``
# and silently degrades (fallback metadata) when the requested model is
# absent. The upstream catalog is OpenRouter-shaped (``{"data": [...]}``),
# so we translate each entry into the ModelInfo shape Codex requires while
# keeping the original ``data`` array for OpenAI-shaped clients.
_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
)


def _to_codex_model_info(entry: dict) -> dict:
    """Map one OpenRouter catalog entry to Codex CLI's ModelInfo shape."""
    mid = entry.get("id") or entry.get("slug") or "unknown"
    ctx = entry.get("context_length")
    if isinstance(ctx, str):
        try:
            ctx = int(ctx)
        except ValueError:
            ctx = None
    reasoning = entry.get("reasoning") or {}
    efforts = reasoning.get("supported_efforts") or []
    if not isinstance(efforts, list):
        efforts = []
    efforts = [e for e in efforts if isinstance(e, str) and e in _REASONING_EFFORTS]
    supported_params = entry.get("supported_parameters") or []
    if not isinstance(supported_params, list):
        supported_params = []
    info: dict = {
        "slug": mid,
        "display_name": entry.get("name") or mid,
        "description": entry.get("description"),
        # Codex's legacy-compat deserializer hard-errors when an entry has
        # neither `base_instructions` nor `model_messages.instructions_template`.
        # Empty string is present-but-neutral: the real instructions travel
        # in the API request, not this catalog field.
        "base_instructions": "",
        "supported_reasoning_levels": [
            {"effort": e, "description": e} for e in efforts
        ],
        "shell_type": "default",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 0,
        "support_verbosity": False,
        "truncation_policy": {"mode": "tokens", "limit": ctx or 200_000},
        "supports_parallel_tool_calls": "parallel_tool_calls" in supported_params,
        "experimental_supported_tools": [],
    }
    if ctx is not None:
        info["context_window"] = ctx
    default_effort = reasoning.get("default_effort")
    if default_effort in _REASONING_EFFORTS:
        info["default_reasoning_level"] = default_effort
    return info


def _translate_models_payload(raw: bytes) -> bytes:
    """Return ``raw`` unchanged unless it is an OpenRouter-shaped catalog."""
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return raw
    data = payload.get("data")
    if not isinstance(data, list) or "models" in payload:
        return raw
    payload.setdefault("object", "list")
    payload["models"] = [
        _to_codex_model_info(e) for e in data if isinstance(e, dict)
    ]
    return json.dumps(payload).encode("utf-8")


def _normalize_responses_input(raw: bytes) -> bytes:
    """Interleave tool calls with their matching outputs; drop orphans.

    Some portal providers reject Responses API requests whose tool-call
    history groups calls separately from outputs — the shape Codex emits
    for parallel tool calls (``function_call, function_call, …,
    function_call_output, function_call_output, …``) — with a generic 400
    "Provider returned error". They require each call to be immediately
    followed by its matching output. Complete pairs are reordered to that
    canonical interleaved form; orphaned calls/outputs are dropped (an
    incomplete pair carries no usable tool result anyway).
    """
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return raw
    items = payload.get("input")
    if not isinstance(items, list):
        return raw
    call_types = ("function_call", "custom_tool_call")
    out_types = ("function_call_output", "custom_tool_call_output")
    outputs_by_call: dict = {}
    for it in items:
        if isinstance(it, dict) and it.get("type") in out_types:
            cid = it.get("call_id")
            if cid is not None:
                outputs_by_call.setdefault(cid, []).append(it)
    normalized: list = []
    emitted: set = set()
    changed = False
    for it in items:
        if not isinstance(it, dict):
            normalized.append(it)
            continue
        it_type = it.get("type")
        if it_type in call_types:
            outs = outputs_by_call.get(it.get("call_id"), [])
            if not outs:
                changed = True  # orphan call: drop (providers reject it)
                continue
            normalized.append(it)
            for out in outs:
                if id(out) not in emitted:
                    normalized.append(out)
                    emitted.add(id(out))
                    changed = True
        elif it_type in out_types:
            if id(it) not in emitted:
                changed = True  # orphan output: drop
                continue
        else:
            normalized.append(it)
    if not changed:
        return raw
    payload["input"] = normalized
    return json.dumps(payload).encode("utf-8")

# Headers we strip when forwarding to the upstream. ``host``/``content-length``
# are recomputed by aiohttp; ``authorization`` is replaced with our bearer.
# Everything else (content-type, accept, user-agent, x-* headers) passes through.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "authorization",  # we replace this one
    }
)

DEFAULT_PORT = 8645
DEFAULT_HOST = "127.0.0.1"
# Body cap for forwarded requests. Chat-completion payloads with long agent
# conversations can be large; mirror api_server's MAX_REQUEST_BYTES (10 MB).
# client_max_size bounds every read path, including chunked bodies.
MAX_REQUEST_BYTES = 10_000_000


def _json_error(status: int, message: str, code: str = "proxy_error") -> "web.Response":
    """Return an OpenAI-style error JSON response."""
    body = {"error": {"message": message, "type": code, "code": code}}
    return web.json_response(body, status=status)


def _filter_request_headers(headers: "aiohttp.typedefs.LooseHeaders") -> dict:
    """Strip hop-by-hop + auth headers from the inbound request."""
    out = {}
    for key, value in headers.items():
        if key.lower() in _HOP_BY_HOP_HEADERS:
            continue
        out[key] = value
    return out


def _filter_response_headers(headers) -> dict:
    """Strip hop-by-hop headers from the upstream response."""
    out = {}
    for key, value in headers.items():
        if key.lower() in _HOP_BY_HOP_HEADERS:
            continue
        # aiohttp recomputes Content-Encoding/Content-Length on stream — let it.
        if key.lower() in {"content-encoding", "content-length"}:
            continue
        out[key] = value
    return out


def create_app(adapter: UpstreamAdapter) -> "web.Application":
    """Build the aiohttp application bound to a specific upstream adapter."""
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError(
            "aiohttp is required for `hermes proxy`. Run `hermes setup` to install it."
        )

    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    # AppKey ensures forward-compat with future aiohttp versions that strip
    # bare-string keys.
    _adapter_key = web.AppKey("adapter", UpstreamAdapter)
    app[_adapter_key] = adapter

    async def handle_health(request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "status": "ok",
                "upstream": adapter.display_name,
                "authenticated": adapter.is_authenticated(),
            }
        )

    async def handle_proxy(request: "web.Request") -> "web.StreamResponse":
        # Extract the path *after* /v1
        rel_path = request.match_info.get("tail", "")
        rel_path = "/" + rel_path.lstrip("/")

        if rel_path not in adapter.allowed_paths:
            allowed = ", ".join(sorted(adapter.allowed_paths))
            return _json_error(
                404,
                f"Path /v1{rel_path} is not forwarded by this proxy. "
                f"Allowed: {allowed}",
                code="path_not_allowed",
            )

        try:
            cred = adapter.get_credential()
        except Exception as exc:
            logger.warning("proxy: credential resolution failed: %s", exc)
            return _json_error(401, str(exc), code="upstream_auth_failed")

        # Forward body verbatim. Read into memory once — request bodies for
        # chat/completions/embeddings are small (<1MB typically). If we ever
        # need to forward large multipart uploads we'll switch to streaming
        # the request body too.
        body = await request.read()

        # Some portal providers reject Responses requests whose tool-call
        # history groups calls separately from outputs (Codex's parallel-call
        # shape) with a generic 400. Interleave each call with its matching
        # output and drop orphans before forwarding.
        if rel_path == "/responses" and request.method.upper() == "POST" and body:
            body = _normalize_responses_input(body)

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=300)

        async def _send_upstream(active_cred: UpstreamCredential):
            upstream_url = f"{active_cred.base_url.rstrip('/')}{rel_path}"
            # Preserve query string verbatim.
            if request.query_string:
                upstream_url = f"{upstream_url}?{request.query_string}"

            fwd_headers = _filter_request_headers(request.headers)
            fwd_headers["Authorization"] = f"{active_cred.token_type} {active_cred.bearer}"

            logger.debug(
                "proxy: forwarding %s %s -> %s (body=%d bytes)",
                request.method, rel_path, upstream_url, len(body),
            )

            try:
                session = aiohttp.ClientSession(timeout=timeout)
            except Exception as exc:  # pragma: no cover - aiohttp setup issue
                raise RuntimeError(f"proxy session init failed: {exc}") from exc

            try:
                upstream_resp = await session.request(
                    request.method,
                    upstream_url,
                    data=body if body else None,
                    headers=fwd_headers,
                    allow_redirects=False,
                )
            except Exception:
                await session.close()
                raise
            return session, upstream_resp

        async def _open_upstream(active_cred: UpstreamCredential):
            try:
                return await _send_upstream(active_cred)
            except RuntimeError as exc:
                return _json_error(500, str(exc)), None
            except aiohttp.ClientError as exc:
                logger.warning("proxy: upstream connection failed: %s", exc)
                return (
                    _json_error(
                        502,
                        f"upstream connection failed: {exc}",
                        code="upstream_unreachable",
                    ),
                    None,
                )
            except asyncio.TimeoutError:
                return (
                    _json_error(
                        504,
                        "upstream request timed out",
                        code="upstream_timeout",
                    ),
                    None,
                )

        session_or_response, upstream_resp = await _open_upstream(cred)
        if upstream_resp is None:
            return session_or_response
        session = session_or_response

        if upstream_resp.status in {401, 429}:
            try:
                retry_cred = adapter.get_retry_credential(
                    failed_credential=cred,
                    status_code=upstream_resp.status,
                )
            except Exception as exc:
                logger.warning("proxy: retry credential resolution failed: %s", exc)
                retry_cred = None

            if retry_cred is not None:
                upstream_resp.release()
                await session.close()
                session_or_response, upstream_resp = await _open_upstream(retry_cred)
                if upstream_resp is None:
                    return session_or_response
                session = session_or_response

        # /models: translate the catalog envelope for Codex CLI while
        # preserving the original "data" shape for other OpenAI clients.
        if (
            rel_path == "/models"
            and request.method.upper() == "GET"
            and upstream_resp.status == 200
        ):
            raw = await upstream_resp.read()
            upstream_resp.release()
            await session.close()
            return web.Response(
                body=_translate_models_payload(raw),
                status=200,
                content_type="application/json",
            )

        # Stream response back. Headers first, then chunked body.
        resp = web.StreamResponse(
            status=upstream_resp.status,
            headers=_filter_response_headers(upstream_resp.headers),
        )
        await resp.prepare(request)

        try:
            async for chunk in upstream_resp.content.iter_any():
                if chunk:
                    await resp.write(chunk)
        except (aiohttp.ClientError, asyncio.CancelledError) as exc:
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

    return app


async def run_server(
    adapter: UpstreamAdapter,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    shutdown_event: Optional[asyncio.Event] = None,
) -> None:
    """Run the proxy in the current event loop until shutdown_event is set.

    If shutdown_event is None, runs until cancelled (Ctrl+C or SIGTERM).
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError(
            "aiohttp is required for `hermes proxy`. Run `hermes setup` to install it."
        )

    app = create_app(adapter)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    logger.info(
        "proxy: listening on http://%s:%d/v1 -> %s",
        host, port, adapter.display_name,
    )

    stop_event = shutdown_event or asyncio.Event()

    # Wire signal handlers when we own the loop's lifetime.
    if shutdown_event is None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)  # windows-footgun: ok
            except NotImplementedError:
                # Windows / restricted environments — Ctrl+C will still
                # raise KeyboardInterrupt and unwind us.
                pass

    try:
        await stop_event.wait()
    finally:
        logger.info("proxy: shutting down")
        await runner.cleanup()


__all__ = [
    "create_app",
    "run_server",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "AIOHTTP_AVAILABLE",
]
