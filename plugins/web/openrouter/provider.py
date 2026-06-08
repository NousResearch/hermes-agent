"""OpenRouter agentic web search + content fetch — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`.

Leverages OpenRouter server-side tools (``openrouter:web_search`` /
``openrouter:web_fetch``) to search and extract web content using your
existing OpenRouter credits — no separate Exa / Firecrawl / Parallel /
Tavily API key required.

How it works
------------
OpenRouter's ``openrouter:web_search`` and ``openrouter:web_fetch`` are
server-side tools available inside any chat-completions call.  This provider
makes a *lightweight* chat request with a cheap auxiliary model (Gemini 3
Flash Preview), passes the tool definition, and extracts the search/extract
results from the response ``annotations``.  The model is effectively a
pass-through — it adds ~$0.0003 in token cost on top of the $0.005/search
Exa/Parallel fee billed by OpenRouter.

Config keys this provider responds to::

    web:
      search_backend: "openrouter"   # explicit per-capability
      extract_backend: "openrouter"  # explicit per-capability
      backend: "openrouter"          # shared fallback

Auth: reads ``OPENROUTER_API_KEY`` from the environment (already present
when Hermes uses OpenRouter as its model provider).

Server-tool engines
-------------------
By default the provider uses Exa (returns real, direct URLs — no Google
redirects).  You can override with::

    export OPENROUTER_WEB_ENGINE=auto
    export OPENROUTER_WEB_ENGINE=native
    export OPENROUTER_WEB_ENGINE=parallel

Supported engines: ``auto``, ``native``, ``exa``, ``parallel``, ``firecrawl``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# ── OpenRouter API ────────────────────────────────────────────────────────

_OR_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# Cheap model that barely generates tokens — the search results come from
# OpenRouter's server-side tool execution, not from the model itself.
_DEFAULT_MODEL = "google/gemini-3-flash-preview"


def _get_api_key() -> Optional[str]:
    """Return the OpenRouter API key from the environment."""
    return os.getenv("OPENROUTER_API_KEY", "").strip() or None


def _get_search_engine() -> str:
    """Read the user's preferred search engine from env, or 'exa' (real URLs).

    ``auto`` / ``native`` engines (e.g. Google) return Google redirect URLs
    instead of direct result URLs.  ``exa`` returns real URLs and costs
    $0.005/search billed from OpenRouter credits.  Override with::

        export OPENROUTER_WEB_ENGINE=auto
        export OPENROUTER_WEB_ENGINE=parallel
    """
    return (os.getenv("OPENROUTER_WEB_ENGINE") or "exa").lower().strip()


def _build_tool_spec(tool_type: str, max_results: int = 5) -> dict:
    """Build an OpenRouter server-side tool definition.

    Parameters are passed alongside the ``type`` key — OpenRouter parses
    them from the same dict.  See
    https://openrouter.ai/docs/guides/features/plugins/web-search
    """
    spec: dict = {"type": tool_type}
    engine = _get_search_engine()
    params: dict[str, Any] = {"max_results": max_results}
    if engine != "auto":
        params["engine"] = engine
    spec["parameters"] = params
    return spec


def _call_or_tools(
    tool_type: str,
    query_or_url: str,
    max_results: int = 5,
) -> Optional[dict]:
    """Make a lightweight chat-completions call with an OpenRouter server-side tool.

    Returns the full response JSON on success, or ``None`` on failure.

    The model is a pass-through — it'll produce a short summary, but the
    actual search/extract results live in ``annotations[].url_citation``.
    We minimise token waste with ``max_tokens=50``.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning("OPENROUTER_API_KEY is not set")
        return None

    # Determine the user message text based on tool type
    if tool_type == "openrouter:web_search":
        user_msg = f"search: {query_or_url}"
    else:
        user_msg = f"fetch: {query_or_url}"

    body = {
        "model": _DEFAULT_MODEL,
        "messages": [{"role": "user", "content": user_msg}],
        "tools": [_build_tool_spec(tool_type, max_results)],
        "max_tokens": 50,
    }

    try:
        resp = httpx.post(
            _OR_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("OpenRouter API HTTP %s: %s", exc.response.status_code, exc)
        return None
    except httpx.RequestError as exc:
        logger.warning("OpenRouter API request error: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenRouter API call failed: %s", exc)
        return None

    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        logger.warning("OpenRouter API response not JSON: %s", exc)
        return None


def _extract_annotations(data: dict) -> List[dict]:
    """Extract ``url_citation`` entries from an OpenRouter chat response.

    Returns a list of ``{"url", "title", "content"}`` dicts, or empty list.
    """
    try:
        msg = data["choices"][0]["message"]
        annotations = msg.get("annotations") or []
    except (KeyError, IndexError, TypeError):
        return []

    results: List[dict] = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        if ann.get("type") != "url_citation":
            continue
        cit = ann.get("url_citation") or {}
        url = cit.get("url", "")
        title = cit.get("title", "")
        content = cit.get("content", "")
        if not url:
            continue
        results.append({"url": url, "title": title, "content": content or ""})

    return results


# ── Provider Class ────────────────────────────────────────────────────────


class OpenRouterWebSearchProvider(WebSearchProvider):
    """OpenRouter-backed search + extract provider using server-side tools.

    ``search()`` uses the ``openrouter:web_search`` tool internally.
    ``extract()`` uses the ``openrouter:web_fetch`` tool internally.

    Both methods work by making a lightweight chat-completions call to
    OpenRouter and extracting results from the response annotations.
    The cheap auxiliary model (Gemini 3 Flash Preview) acts as a
    pass-through — it's not doing the heavy lifting.
    """

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return "OpenRouter Agentic Web"

    def is_available(self) -> bool:
        """Return True when ``OPENROUTER_API_KEY`` is set."""
        return _get_api_key() is not None

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    # ── Search ───────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a web search via OpenRouter's ``openrouter:web_search``.

        Returns the standard Hermes response shape on success, or an error
        dict on failure.
        """
        if not query or not query.strip():
            return {"success": False, "error": "Empty search query"}

        if not _get_api_key():
            return {
                "success": False,
                "error": (
                    "OPENROUTER_API_KEY is not set. "
                    "Set it in ~/.hermes/.env or your environment."
                ),
            }

        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}
        except ImportError:
            pass

        logger.info(
            "OpenRouter search: '%s' (limit=%d, engine=%s)",
            query, limit, _get_search_engine(),
        )

        data = _call_or_tools("openrouter:web_search", query, limit)
        if data is None:
            return {
                "success": False,
                "error": "OpenRouter API call failed (see logs for details)",
            }

        annotations = _extract_annotations(data)

        # OpenRouter returns the search content as annotation url_citation
        # entries.  Map to the standard Hermes shape.
        web_results = [
            {
                "url": a["url"],
                "title": a["title"],
                "description": a["content"],
                "position": i + 1,
            }
            for i, a in enumerate(annotations[:limit])
        ]

        # Log cost info for transparency
        usage = data.get("usage") or {}
        cost = usage.get("cost", 0)
        logger.info(
            "OpenRouter search '%s': %d results, cost $%.6f",
            query, len(web_results), cost,
        )

        return {"success": True, "data": {"web": web_results}}

    # ── Extract ──────────────────────────────────────────────────────────

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Fetch page content via OpenRouter's ``openrouter:web_fetch``.

        Returns a list of result dicts shaped for the Hermes post-processing
        pipeline.  Each entry has the standard ``{url, title, content,
        raw_content, metadata}`` shape, or an ``error`` field on per-URL
        failure.

        Since ``openrouter:web_fetch`` processes one URL per call, we make
        sequential calls for multiple URLs.
        """
        if not urls:
            return []

        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]
        except ImportError:
            pass

        logger.info("OpenRouter extract: %d URL(s)", len(urls))

        results: List[Dict[str, Any]] = []
        for url in urls:
            if not url or not url.strip():
                results.append(
                    {"url": url or "", "title": "", "content": "", "error": "Empty URL"}
                )
                continue

            data = _call_or_tools("openrouter:web_fetch", url, max_results=1)
            if data is None:
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": "OpenRouter API call failed",
                    }
                )
                continue

            annotations = _extract_annotations(data)
            if annotations:
                ann = annotations[0]
                content = ann.get("content", "")
                results.append(
                    {
                        "url": url,
                        "title": ann.get("title", ""),
                        "content": content,
                        "raw_content": content,
                        "metadata": {
                            "sourceURL": url,
                            "title": ann.get("title", ""),
                        },
                    }
                )
            else:
                # No annotations returned — the model probably produced a
                # text response instead of fetching.  Fall back to the
                # model's content as a best-effort description.
                text = ""
                try:
                    text = (
                        (data.get("choices") or [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                except (IndexError, AttributeError):
                    pass

                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": text or "",
                        "raw_content": text or "",
                        "metadata": {"sourceURL": url},
                    }
                )

        return results

    # ── Setup Schema ─────────────────────────────────────────────────────

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenRouter Agentic Web",
            "badge": "paid",
            "tag": (
                "Search + fetch via OpenRouter server-side tools. "
                "Uses your existing OpenRouter credits — no separate API key."
            ),
            "env_vars": [
                {
                    "key": "OPENROUTER_API_KEY",
                    "prompt": "OpenRouter API key",
                    "url": "https://openrouter.ai/keys",
                },
            ],
        }