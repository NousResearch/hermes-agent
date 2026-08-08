"""Exa web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Uses the
official Exa SDK (``exa-py``) which is lazy-loaded via
:func:`tools.lazy_deps.ensure` so that cold-start CLI users don't pay the
SDK import cost when Exa isn't configured.

Config keys this provider responds to::

    web:
      search_backend: "exa"      # explicit per-capability
      extract_backend: "exa"     # explicit per-capability
      backend: "exa"             # shared fallback for both

Env var::

    EXA_API_KEY=...    # https://exa.ai (paid tier; free trial available)

The previous in-tree implementation lived at
``tools.web_tools._exa_search`` / ``_exa_extract``; this file is the
canonical replacement. Behavior is bit-for-bit identical aside from the
ABC method-name change.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# Module-level note: the canonical ``_exa_client`` cache slot lives on
# :mod:`tools.web_tools` so tests that do ``tools.web_tools._exa_client =
# None`` between cases see fresh state. The plugin reads/writes through
# that public module (see :func:`_get_exa_client`).


def _get_exa_client() -> Any:
    """Lazy-import and cache an Exa SDK client.

    Cache lives on :mod:`tools.web_tools` (as ``_exa_client``) so unit
    tests that reset that name between cases keep working. Raises
    ``ValueError`` when ``EXA_API_KEY`` is unset.
    """
    import tools.web_tools as _wt

    cached = getattr(_wt, "_exa_client", None)
    if cached is not None:
        return cached

    from agent.web_search_provider import get_provider_env

    api_key = get_provider_env("EXA_API_KEY")
    if not api_key:
        raise ValueError(
            "EXA_API_KEY environment variable not set. "
            "Get your API key at https://exa.ai"
        )

    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("search.exa", prompt=False)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 — lazy_deps surfaces install hints
        raise ImportError(str(exc))

    from exa_py import Exa  # noqa: WPS433 — deliberately lazy

    client = Exa(api_key=api_key)
    client.headers["x-exa-integration"] = "hermes-agent"
    _wt._exa_client = client
    return client


def _reset_client_for_tests() -> None:
    """Drop the cached Exa client so tests can re-instantiate cleanly."""
    import tools.web_tools as _wt

    _wt._exa_client = None


class ExaWebSearchProvider(WebSearchProvider):
    """Exa search + extract provider.

    Both methods are sync — Exa's SDK is sync-only. The web_extract_tool
    dispatcher wraps sync extracts via ``asyncio.to_thread`` when it
    needs to keep the event loop responsive.
    """

    @property
    def name(self) -> str:
        return "exa"

    @property
    def display_name(self) -> str:
        return "Exa"

    def is_available(self) -> bool:
        """Return True when Exa can service calls.

        Requires BOTH the ``EXA_API_KEY`` credential AND the ``exa_py``
        SDK to be importable. The SDK check is part of the WebSearchProvider
        ABC contract ("optional Python dep importable") — a key without the
        SDK would fail at request time, so it must not be reported as
        available (see post-B1 readiness audit). Cheap and offline:
        ``find_spec`` does not import the module.
        """
        from agent.web_search_provider import get_provider_env

        if not get_provider_env("EXA_API_KEY"):
            return False
        try:
            import importlib.util

            return importlib.util.find_spec("exa_py") is not None
        except Exception:  # noqa: BLE001 — a broken import probe means "not ready"
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5, search_depth: Optional[str] = None) -> Dict[str, Any]:
        """Execute an Exa search.

        Returns ``{"success": True, "data": {"web": [{...}, ...]}}`` on
        success, ``{"success": False, "error": str}`` on failure (incl.
        missing API key and SDK install errors).

        search_depth maps to Exa's ``type`` parameter:
            "fast" → "fast" (~450ms, optimized)
            "auto" / None → "auto" (~1s, balanced, default)
            "deep" → "deep" (4-15s, multi-step with reasoning)
            "deepest" → "deep-reasoning" (12-40s, maximum reasoning)
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            # Map search_depth to Exa type
            _type_map = {
                "fast": "fast",
                "deep": "deep",
                "deepest": "deep-reasoning",
            }
            exa_type = _type_map.get((search_depth or "").lower(), "auto")

            logger.info("Exa search: '%s' (limit=%d, type=%s)", query, limit, exa_type)
            response = _get_exa_client().search(
                query,
                num_results=limit,
                type=exa_type,
                contents={"highlights": True},
            )

            web_results = []
            for i, result in enumerate(response.results or []):
                highlights = result.highlights or []
                web_results.append(
                    {
                        "url": result.url or "",
                        "title": result.title or "",
                        "description": " ".join(highlights) if highlights else "",
                        "position": i + 1,
                    }
                )

            return {"success": True, "data": {"web": web_results}}
        except ValueError as exc:
            # Raised by _get_exa_client when EXA_API_KEY missing
            return {"success": False, "error": str(exc)}
        except ImportError as exc:
            return {"success": False, "error": f"Exa SDK not installed: {exc}"}
        except Exception as exc:  # noqa: BLE001 — surface as failure
            logger.warning("Exa search error: %s", exc)
            return {"success": False, "error": f"Exa search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Exa.

        Returns a list of result dicts shaped for the legacy LLM
        post-processing pipeline. On per-URL or whole-batch failure,
        results carry an ``error`` field rather than raising.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            logger.info("Exa extract: %d URL(s)", len(urls))
            response = _get_exa_client().get_contents(urls, text=True)

            results: List[Dict[str, Any]] = []
            for result in response.results or []:
                content = result.text or ""
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
            return results
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except ImportError as exc:
            return [
                {"url": u, "title": "", "content": "", "error": f"Exa SDK not installed: {exc}"}
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exa extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"Exa extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Exa",
            "badge": "paid",
            "tag": "Semantic + neural web search with content extraction.",
            "env_vars": [
                {
                    "key": "EXA_API_KEY",
                    "prompt": "Exa API key",
                    "url": "https://exa.ai",
                },
            ],
        }
