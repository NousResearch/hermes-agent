"""Custom OpenAI-compatible web search + extract — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Routes
both search and extract calls through an OpenAI-compatible
``/chat/completions`` endpoint that serves a model with built-in web
access (e.g. Perplexity Sonar, OpenAI gpt-4o with browsing).

Search-result extraction prefers structured ``search_results``, falls
back to a ``citations`` list (which may be strings or dicts), and as a
last resort returns the answer text itself as a single "result".

Config keys this provider responds to::

    web:
      search_backend: "custom"      # explicit per-capability
      extract_backend: "custom"     # explicit per-capability
      backend: "custom"             # shared fallback
      custom_base_url: "https://api.perplexity.ai"
      custom_model: "sonar"
      custom_api_key: "..."

Env vars (override config)::

    CUSTOM_SEARCH_API_KEY=...              # required
    CUSTOM_SEARCH_BASE_URL=https://...     # required
    CUSTOM_SEARCH_MODEL=sonar              # optional; default "sonar"
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


_DEFAULT_MODEL = "sonar"


def _load_web_config() -> Dict[str, Any]:
    """Return the ``web:`` section of config.yaml, or ``{}`` on any miss.

    Read at call time rather than import time so config edits and tests
    that patch ``hermes_cli.config.load_config`` see fresh values.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        web = cfg.get("web") if isinstance(cfg, dict) else None
        if isinstance(web, dict):
            return web
    except Exception as exc:  # noqa: BLE001 — config is best-effort
        logger.debug("Custom provider: could not load web config: %s", exc)
    return {}


def _resolve_api_key() -> str:
    """Return the API key from env or config.yaml; empty string when neither set."""
    env_key = os.getenv("CUSTOM_SEARCH_API_KEY", "").strip()
    if env_key:
        return env_key
    return (_load_web_config().get("custom_api_key") or "").strip()


def _resolve_base_url() -> str:
    """Return the base URL (env preferred, config fallback), trailing-slash stripped."""
    env_url = os.getenv("CUSTOM_SEARCH_BASE_URL", "").strip().rstrip("/")
    if env_url:
        return env_url
    return (_load_web_config().get("custom_base_url") or "").strip().rstrip("/")


def _resolve_model() -> str:
    """Return the model name; env > config > default."""
    env_model = os.getenv("CUSTOM_SEARCH_MODEL", "").strip()
    if env_model:
        return env_model
    cfg_model = (_load_web_config().get("custom_model") or "").strip()
    if cfg_model:
        return cfg_model
    return _DEFAULT_MODEL


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _chat_completion(prompt: str) -> Dict[str, Any]:
    """Call the custom backend via OpenAI-compatible chat completions.

    Raises ``ValueError`` when credentials/base_url are missing — callers
    convert to the typed ``{"success": False, "error": ...}`` envelope.
    """
    import httpx

    api_key = _resolve_api_key()
    if not api_key:
        raise ValueError(
            "Custom search backend requires an API key. "
            "Set CUSTOM_SEARCH_API_KEY or web.custom_api_key in config.yaml."
        )

    base_url = _resolve_base_url()
    if not base_url:
        raise ValueError(
            "Custom search backend requires a base URL. "
            "Set CUSTOM_SEARCH_BASE_URL or web.custom_base_url in config.yaml."
        )

    payload = {
        "model": _resolve_model(),
        "messages": [{"role": "user", "content": prompt}],
    }
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers=_build_headers(api_key),
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _extract_web_results(data: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    """Parse ``search_results`` → ``citations`` → answer text into the legacy shape."""
    web_results: List[Dict[str, Any]] = []

    search_results = data.get("search_results") or []
    if search_results:
        for i, sr in enumerate(search_results[:limit]):
            web_results.append({
                "title": sr.get("title", "") or "",
                "url": sr.get("url", "") or "",
                "description": sr.get("snippet", "") or sr.get("content", "") or "",
                "position": i + 1,
            })
        return web_results

    citations = data.get("citations") or []
    if citations:
        for i, c in enumerate(citations[:limit]):
            if isinstance(c, str):
                web_results.append({
                    "title": "",
                    "url": c,
                    "description": "",
                    "position": i + 1,
                })
            elif isinstance(c, dict):
                web_results.append({
                    "title": c.get("title", "") or "",
                    "url": c.get("url", "") or "",
                    "description": c.get("snippet", "") or c.get("content", "") or "",
                    "position": i + 1,
                })

    if web_results:
        return web_results

    # Last resort: surface the answer text itself as a single pseudo-result so
    # the caller still gets something useful out of a model that didn't
    # return structured citations.
    answer = ""
    choices = data.get("choices") or []
    if choices:
        answer = (choices[0].get("message") or {}).get("content", "") or ""
    if answer:
        web_results.append({
            "title": "Search Answer",
            "url": "",
            "description": answer[:2000],
            "position": 1,
        })

    return web_results


class CustomWebSearchProvider(WebSearchProvider):
    """OpenAI-compatible chat-completions backend with built-in web access."""

    @property
    def name(self) -> str:
        return "custom"

    @property
    def display_name(self) -> str:
        return "Custom (OpenAI-compatible)"

    def is_available(self) -> bool:
        """Return True when an API key is configured (env or config.yaml).

        Base URL absence is deliberately *not* checked here — explicit
        config-side selection should still surface a precise
        "set CUSTOM_SEARCH_BASE_URL" error rather than silently routing
        to a different backend.
        """
        return bool(_resolve_api_key())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            logger.info("Custom search: '%s' (limit: %d)", query, limit)
            data = _chat_completion(query)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Custom search error: %s", exc)
            return {"success": False, "error": f"Custom search failed: {exc}"}

        return {
            "success": True,
            "data": {"web": _extract_web_results(data, limit)},
        }

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for url in urls:
            try:
                prompt = (
                    f"Extract and summarise the main content from this URL: {url}\n"
                    "Return the full content in markdown format. "
                    "Include all key information, facts, and details."
                )
                logger.info("Custom extract: %s", url)
                data = _chat_completion(prompt)
                content = ""
                choices = data.get("choices") or []
                if choices:
                    content = (choices[0].get("message") or {}).get("content", "") or ""
                results.append({
                    "url": url,
                    "title": "",
                    "content": content,
                    "raw_content": content,
                    "metadata": {"sourceURL": url, "title": ""},
                })
            except Exception as exc:  # noqa: BLE001 — per-URL isolation
                logger.warning("Custom extract failed for %s: %s", url, exc)
                results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": str(exc),
                    "metadata": {"sourceURL": url},
                })
        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Custom (OpenAI-compatible)",
            "badge": "byo",
            "tag": "Any chat-completions endpoint with built-in search (e.g. Perplexity Sonar).",
            "env_vars": [
                {
                    "key": "CUSTOM_SEARCH_API_KEY",
                    "prompt": "API key for the custom search endpoint",
                },
                {
                    "key": "CUSTOM_SEARCH_BASE_URL",
                    "prompt": "Base URL (e.g. https://api.perplexity.ai)",
                },
                {
                    "key": "CUSTOM_SEARCH_MODEL",
                    "prompt": "Model name (default: sonar)",
                },
            ],
        }
