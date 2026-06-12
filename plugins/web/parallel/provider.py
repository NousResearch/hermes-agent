"""Parallel.ai web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`.

Search runs on one of two transports, picked by credential:

- **No key →** the free hosted Search MCP at ``https://search.parallel.ai/mcp``
  (anonymous Streamable-HTTP JSON-RPC). This makes ``web_search`` work out of
  the box with zero setup, which is why ``parallel`` is the keyless default
  backend in :func:`tools.web_tools._get_backend`.
- **``PARALLEL_API_KEY`` →** the ``parallel`` SDK's v1 ``search`` / ``extract``
  REST endpoints (objective-tuned, mode-selectable, higher rate limits).

Extract mirrors search: keyed uses the async SDK (``AsyncParallel``) v1
``extract``; keyless uses the free MCP's ``web_fetch``. :meth:`extract` is
declared ``async def`` and the dispatcher in
:func:`tools.web_tools.web_extract_tool` detects coroutines via
:func:`inspect.iscoroutinefunction` and awaits.

Config keys this provider responds to::

    web:
      search_backend: "parallel"      # explicit per-capability
      extract_backend: "parallel"     # explicit per-capability
      backend: "parallel"             # shared fallback
      # Optional: search mode (default "advanced"; also "basic")
      # via the PARALLEL_SEARCH_MODE env var. REST path only.

Env vars::

    PARALLEL_API_KEY=...             # https://parallel.ai (optional — unlocks
                                     # the v1 REST Search API; without it,
                                     # search and extract use the free MCP)
    PARALLEL_SEARCH_MODE=advanced    # optional: basic|advanced (legacy
                                     # fast/one-shot map to basic, agentic to
                                     # advanced). REST path only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# Free hosted Search MCP — anonymous-friendly, used when no PARALLEL_API_KEY is
# configured. Docs: https://docs.parallel.ai/integrations/mcp/search-mcp
_MCP_SEARCH_URL = "https://search.parallel.ai/mcp"
_MCP_PROTOCOL_VERSION = "2025-06-18"
# Deliberately generic client identity. Project policy (see the telemetry PR
# policy in AGENTS.md) forbids third-party usage attribution without an
# explicit user opt-in, so neither clientInfo nor the User-Agent names
# hermes. MCP requires *a* clientInfo; a neutral one satisfies the spec
# without attributing traffic.
_MCP_CLIENT_NAME = "mcp-web-client"
_MCP_CLIENT_VERSION = "1.0.0"
_MCP_USER_AGENT = f"{_MCP_CLIENT_NAME}/{_MCP_CLIENT_VERSION}"
_MCP_TIMEOUT_SECONDS = 30.0

# Free-tier attribution. The hosted Search MCP is free to use; surfacing this
# on keyless results credits Parallel and matches the free-tier terms
# (https://parallel.ai/customer-terms).
_FREE_MCP_ATTRIBUTION = (
    "Search powered by the free Parallel Web Search MCP (https://parallel.ai)."
)


def _new_session_id() -> str:
    """Mint a fresh Parallel ``session_id`` for a single tool call.

    Per-call rather than process-global: one process serves many unrelated
    chats in the gateway/batch runners, and a shared id would pool their
    searches into one Parallel session. The prefix is deliberately generic
    (no hermes attribution — telemetry policy).
    """
    return f"{_MCP_CLIENT_NAME}-{uuid.uuid4().hex}"

# Module-level note: the canonical cache slots ``_parallel_client`` and
# ``_async_parallel_client`` live on :mod:`tools.web_tools` so tests that do
# ``tools.web_tools._parallel_client = None`` between cases see fresh state.
# The plugin reads/writes through that public module (see
# :func:`_get_sync_client` / :func:`_get_async_client`).


def _ensure_parallel_sdk_installed() -> None:
    """Trigger lazy install of the parallel SDK if it isn't present.

    Mirrors the lazy-deps pattern used by the legacy implementation.
    Swallows benign ImportError from the lazy_deps helper itself; if the
    SDK is genuinely missing the subsequent ``from parallel import ...``
    raises ImportError that the caller can handle.
    """
    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("search.parallel", prompt=False)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 — surface install hint as ImportError
        raise ImportError(str(exc))


def _get_sync_client() -> Any:
    """Lazy-load + cache the sync Parallel client.

    Cache lives on :mod:`tools.web_tools` (as ``_parallel_client``) so unit
    tests that reset that name between cases keep working.
    """
    import tools.web_tools as _wt

    cached = getattr(_wt, "_parallel_client", None)
    if cached is not None:
        return cached

    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        raise ValueError(
            "PARALLEL_API_KEY environment variable not set. "
            "Get your API key at https://parallel.ai"
        )

    _ensure_parallel_sdk_installed()
    from parallel import Parallel  # noqa: WPS433 — deliberately lazy

    client = Parallel(api_key=api_key)
    _wt._parallel_client = client
    return client


def _get_async_client() -> Any:
    """Lazy-load + cache the async Parallel client.

    Cache lives on :mod:`tools.web_tools` (as ``_async_parallel_client``).
    """
    import tools.web_tools as _wt

    cached = getattr(_wt, "_async_parallel_client", None)
    if cached is not None:
        return cached

    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        raise ValueError(
            "PARALLEL_API_KEY environment variable not set. "
            "Get your API key at https://parallel.ai"
        )

    _ensure_parallel_sdk_installed()
    from parallel import AsyncParallel  # noqa: WPS433 — deliberately lazy

    client = AsyncParallel(api_key=api_key)
    _wt._async_parallel_client = client
    return client


def _reset_clients_for_tests() -> None:
    """Drop both cached clients so tests can re-instantiate cleanly.

    Clears the canonical slots on :mod:`tools.web_tools` (where
    :func:`_get_sync_client` / :func:`_get_async_client` read/write them).
    """
    import tools.web_tools as _wt

    _wt._parallel_client = None
    _wt._async_parallel_client = None


# Backward-compatible aliases for the names that lived in tools.web_tools
# before the migration (matches existing tests + external callers).
_get_parallel_client = _get_sync_client
_get_async_parallel_client = _get_async_client


def _resolve_search_mode() -> str:
    """Return the validated v1 search mode (default "advanced").

    V1 collapses the three Beta modes into two. We accept the v1 values
    directly and map the legacy Beta values for back-compat with anyone who
    still sets ``PARALLEL_SEARCH_MODE=fast|one-shot|agentic``:

    - ``fast`` / ``one-shot`` → ``basic``  (lower latency)
    - ``agentic``             → ``advanced`` (higher quality, the v1 default)
    """
    mode = os.getenv("PARALLEL_SEARCH_MODE", "advanced").lower().strip()
    if mode == "basic" or mode in {"fast", "one-shot"}:
        return "basic"
    # advanced, legacy "agentic", and anything unrecognized → the v1 default.
    return "advanced"


# ---------------------------------------------------------------------------
# Free Search MCP transport (keyless path)
# ---------------------------------------------------------------------------
#
# A small hand-rolled Streamable-HTTP JSON-RPC client for the hosted Search
# MCP, rather than the full MCP-client subsystem: we only call two tools
# (``web_search`` / ``web_fetch``), so keeping it inline lets web_search and
# web_extract stay ordinary tools with the MCP endpoint as just their wire
# protocol.


def _mcp_headers(
    session_id: str | None,
    api_key: str | None,
    protocol_version: str | None = None,
) -> Dict[str, str]:
    """Headers for an MCP request.

    A Bearer token is attached only when we actually hold a key — the free
    endpoint is anonymous, and sending an empty/garbage token would make it
    401 instead of serving the anonymous tier. After ``initialize`` the
    Streamable-HTTP spec expects the negotiated ``MCP-Protocol-Version`` on
    every follow-up request, so we echo it once known.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": _MCP_USER_AGENT,
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    if protocol_version:
        headers["MCP-Protocol-Version"] = protocol_version
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _iter_mcp_messages(text: str):
    """Yield JSON-RPC message dicts from a plain-JSON or SSE response body.

    Handles ``application/json`` (a single object) and ``text/event-stream``
    (SSE: events separated by blank lines; an event's one-or-more ``data:``
    lines concatenate into a single JSON payload). Unparseable chunks and
    non-``data`` SSE fields (``event:``/``id:``/comments) are skipped.
    """
    def _emit(payload):
        # Streamable HTTP allows batching responses/notifications into a JSON
        # array — flatten so callers always see individual message dicts.
        if isinstance(payload, list):
            yield from payload
        elif payload is not None:
            yield payload

    body = (text or "").strip()
    if not body:
        return
    if body.startswith("{") or body.startswith("["):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return
        yield from _emit(parsed)
        return

    data_lines: List[str] = []

    def _flush():
        if not data_lines:
            return None
        try:
            return json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            return None

    for raw in body.split("\n"):
        line = raw.rstrip("\r")
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
        elif line.strip() == "":  # event boundary
            yield from _emit(_flush())
            data_lines = []
    yield from _emit(_flush())


def _mcp_response_envelope(text: str, request_id: str) -> Dict[str, Any]:
    """Select the JSON-RPC response for *request_id* from an MCP response body.

    Streamable-HTTP servers may emit progress/log notifications before the
    final result, so we scan the whole stream and return the result/error
    message whose ``id`` matches our request. Falls back to the last
    result/error-bearing message if no id matches; ``{}`` if none is present.
    """
    fallback: Dict[str, Any] = {}
    for msg in _iter_mcp_messages(text):
        if not isinstance(msg, dict) or not ("result" in msg or "error" in msg):
            continue
        if msg.get("id") == request_id:
            return msg
        fallback = msg
    return fallback


def _mcp_payload(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the tool result payload from a ``tools/call`` envelope.

    Prefers ``structuredContent`` (authoritative machine-readable form);
    otherwise scans text blocks for the first JSON-parseable one. Raises on a
    JSON-RPC error or a tool-level ``isError``.
    """
    if "error" in envelope:
        raise RuntimeError(f"Parallel MCP error: {str(envelope['error'])[:500]}")
    result = envelope.get("result") or {}
    if result.get("isError"):
        raise RuntimeError(f"Parallel MCP tool error: {str(result)[:500]}")

    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured

    for block in result.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text = str(block.get("text") or "")
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(
        f"Parallel MCP returned no parseable content: {str(result)[:500]}"
    )


def _mcp_call(
    tool_name: str, arguments: Dict[str, Any], api_key: str | None
) -> Dict[str, Any]:
    """Run the MCP handshake then a single ``tools/call`` and return its payload.

    initialize → (capture ``Mcp-Session-Id``) → notifications/initialized →
    tools/call ``tool_name``. Returns the parsed tool payload dict (see
    :func:`_mcp_payload`). A Bearer token is attached only when *api_key* is set.
    """
    with httpx.Client(timeout=_MCP_TIMEOUT_SECONDS) as client:
        # 1. initialize — capture the server-assigned MCP session id.
        init_id = str(uuid.uuid4())
        init = client.post(
            _MCP_SEARCH_URL,
            headers=_mcp_headers(None, api_key),
            json={
                "jsonrpc": "2.0",
                "id": init_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": _MCP_CLIENT_NAME,
                        "version": _MCP_CLIENT_VERSION,
                    },
                },
            },
        )
        init.raise_for_status()
        # Only echo a session id the server actually issued. Stateless
        # Streamable-HTTP servers may omit it; inventing one and sending it on
        # follow-up requests can get those requests rejected (the server never
        # created that session). When absent, the Mcp-Session-Id header is simply
        # omitted (see _mcp_headers). This is separate from the tool-arg
        # ``session_id`` below, which is a client-minted rate-limit/grouping id.
        mcp_session_id = init.headers.get("mcp-session-id")
        init_env = _mcp_response_envelope(init.text, init_id)
        # Echo the negotiated protocol version on every post-init request, per
        # the Streamable-HTTP spec (servers may enforce it).
        negotiated_version = (
            (init_env.get("result") or {}).get("protocolVersion")
            or _MCP_PROTOCOL_VERSION
        )

        # 2. notifications/initialized — required handshake ack.
        client.post(
            _MCP_SEARCH_URL,
            headers=_mcp_headers(mcp_session_id, api_key, negotiated_version),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

        # 3. tools/call.
        call_id = str(uuid.uuid4())
        call = client.post(
            _MCP_SEARCH_URL,
            headers=_mcp_headers(mcp_session_id, api_key, negotiated_version),
            json={
                "jsonrpc": "2.0",
                "id": call_id,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        call.raise_for_status()
        return _mcp_payload(_mcp_response_envelope(call.text, call_id))


def _mcp_web_search(query: str, limit: int, api_key: str | None) -> Dict[str, Any]:
    """Run a ``web_search`` tool call against the hosted Search MCP.

    Returns the standard provider search shape
    (``{"success": True, "data": {"web": [...]}}``). The MCP serves a fixed
    result count, so ``limit`` is applied client-side. The MCP requires
    ``objective`` (REST treats it as optional), so we mirror the query.
    """
    payload = _mcp_call(
        "web_search",
        {
            "objective": query,
            "search_queries": [query],
            "session_id": _new_session_id(),
        },
        api_key,
    )

    web_results: List[Dict[str, Any]] = []
    for i, result in enumerate((payload.get("results") or [])[: max(limit, 1)]):
        if not isinstance(result, dict):
            continue
        excerpts = result.get("excerpts") or []
        web_results.append(
            {
                "url": result.get("url") or "",
                "title": result.get("title") or "",
                "description": " ".join(excerpts) if excerpts else "",
                "position": i + 1,
            }
        )

    # Credit the free tier (anonymous path only — keyed search uses REST and
    # carries no attribution).
    return {
        "success": True,
        "data": {"web": web_results},
        "provider": "parallel",
        "attribution": _FREE_MCP_ATTRIBUTION,
    }


def _mcp_web_fetch(urls: List[str], api_key: str | None) -> List[Dict[str, Any]]:
    """Run a ``web_fetch`` tool call against the hosted Search MCP.

    Returns the per-URL extract shape that
    :func:`tools.web_tools.web_extract_tool` expects — exactly one row per input
    URL, in request order (including duplicates). We pass ``full_content=True``
    so the page body comes back as markdown (matching the keyed SDK path and
    what extract callers/summarizers expect), falling back to excerpts only when
    full content is absent. Any input the MCP didn't return is emitted as a
    per-URL error row.
    """
    payload = _mcp_call(
        "web_fetch",
        {"urls": list(urls), "full_content": True, "session_id": _new_session_id()},
        api_key,
    )

    # Index the response by URL, then emit one row per *input* URL in order so
    # duplicates and positional alignment with the request list are preserved.
    by_url: Dict[str, Dict[str, Any]] = {}
    for item in payload.get("results") or []:
        if isinstance(item, dict) and item.get("url"):
            by_url.setdefault(item["url"], item)

    results: List[Dict[str, Any]] = []
    for url in urls:
        item = by_url.get(url)
        if item is None:
            results.append(
                {
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": "extraction failed (no content returned)",
                    "metadata": {"sourceURL": url},
                }
            )
            continue
        title = item.get("title") or ""
        # Prefer the full page body; fall back to joined excerpts (mirrors the
        # keyed SDK extract path).
        content = item.get("full_content") or "\n\n".join(item.get("excerpts") or [])
        results.append(
            {
                "url": url,
                "title": title,
                "content": content,
                "raw_content": content,
                "metadata": {"sourceURL": url, "title": title},
            }
        )

    return results


class ParallelWebSearchProvider(WebSearchProvider):
    """Parallel.ai search + async extract provider."""

    @property
    def name(self) -> str:
        return "parallel"

    @property
    def display_name(self) -> str:
        return "Parallel"

    def is_available(self) -> bool:
        """Return True when ``PARALLEL_API_KEY`` is set.

        Deliberately key-based: this gates the registry's active-provider walk
        and the ``hermes tools`` picker (auto-selecting Parallel for a user who
        hasn't named it), so it must not claim availability on the keyless path.
        The keyless free-MCP path is reached independently via
        :func:`tools.web_tools._get_backend`'s ``parallel`` terminal default.
        """
        return bool(os.getenv("PARALLEL_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Parallel search (sync).

        With ``PARALLEL_API_KEY`` set, uses the v1 ``search`` REST endpoint with
        the configured mode (``PARALLEL_SEARCH_MODE`` env var, default
        "advanced"; limit requested via advanced_settings.max_results, capped at
        20). Without a key, falls back to the free hosted Search MCP so search
        still works with zero setup.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            api_key = os.getenv("PARALLEL_API_KEY", "").strip()
            if not api_key:
                logger.info(
                    "Parallel search (free MCP): '%s' (limit=%d)", query, limit
                )
                return _mcp_web_search(query, limit, api_key=None)

            mode = _resolve_search_mode()
            logger.info(
                "Parallel search (v1 REST): '%s' (mode=%s, limit=%d)",
                query, mode, limit,
            )
            # v1 Search API. Request the caller's limit via max_results (capped
            # at 20) so we don't rely on the API default — the slice below can
            # only trim, not ask for more.
            response = _get_sync_client().search(
                search_queries=[query],
                objective=query,
                mode=mode,
                session_id=_new_session_id(),
                advanced_settings={"max_results": min(max(limit, 1), 20)},
            )

            web_results = []
            for i, result in enumerate((response.results or [])[: max(limit, 1)]):
                excerpts = result.excerpts or []
                web_results.append(
                    {
                        "url": result.url or "",
                        "title": result.title or "",
                        "description": " ".join(excerpts) if excerpts else "",
                        "position": i + 1,
                    }
                )

            # Paid/REST path: no attribution and no "[Parallel]" label — the
            # branding is specifically for the free Search MCP tier.
            return {"success": True, "data": {"web": web_results}}
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except ImportError as exc:
            return {
                "success": False,
                "error": f"Parallel SDK not installed: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parallel search error: %s", exc)
            return {"success": False, "error": f"Parallel search failed: {exc}"}

    async def extract(
        self, urls: List[str], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs.

        With ``PARALLEL_API_KEY`` set, uses the async SDK's v1 ``extract`` for
        full page content. Without a key, falls back to the free hosted Search
        MCP's ``web_fetch`` tool so extraction works with zero setup, mirroring
        the keyless search path.

        Returns the legacy list-of-results shape that
        :func:`tools.web_tools.web_extract_tool` expects: one entry per
        successful URL plus one entry per failed URL with an ``error``
        field. Errors are not raised — they're returned as per-URL items.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            api_key = os.getenv("PARALLEL_API_KEY", "").strip()
            if not api_key:
                logger.info(
                    "Parallel extract (free MCP web_fetch): %d URL(s)", len(urls)
                )
                # _mcp_web_fetch is sync httpx; run off the event loop.
                return await asyncio.to_thread(_mcp_web_fetch, list(urls), None)

            logger.info("Parallel extract (v1 REST): %d URL(s)", len(urls))
            # v1 Extract API (client.extract, /v1/extract); full_content is set
            # via advanced_settings.
            response = await _get_async_client().extract(
                urls=urls,
                advanced_settings={"full_content": True},
                session_id=_new_session_id(),
            )

            results: List[Dict[str, Any]] = []
            for result in response.results or []:
                content = result.full_content or ""
                if not content:
                    content = "\n\n".join(result.excerpts or [])
                url = result.url or ""
                title = result.title or ""
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "content": content,
                        "raw_content": content,
                        "metadata": {"sourceURL": url, "title": title},
                    }
                )

            for error in response.errors or []:
                err_url = getattr(error, "url", "") or ""
                err_msg = (
                    getattr(error, "message", None)
                    or getattr(error, "content", None)
                    or getattr(error, "error_type", None)
                    or "extraction failed"
                )
                results.append(
                    {
                        "url": err_url,
                        "title": "",
                        "content": "",
                        "error": err_msg,
                        "metadata": {"sourceURL": err_url},
                    }
                )

            return results
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except ImportError as exc:
            return [
                {"url": u, "title": "", "content": "", "error": f"Parallel SDK not installed: {exc}"}
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parallel extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"Parallel extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Parallel",
            "badge": "free",
            "tag": (
                "Free web search + extraction via Parallel's hosted Search MCP "
                "— no key needed. Add PARALLEL_API_KEY for the v1 REST Search "
                "API (richer modes, higher limits)."
            ),
            "env_vars": [
                {
                    "key": "PARALLEL_API_KEY",
                    "prompt": "Parallel API key (optional — unlocks the v1 REST Search API)",
                    "url": "https://parallel.ai",
                },
            ],
        }
