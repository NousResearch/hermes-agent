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

# Exa reference (docs.exa.ai/reference/search): the entity categories only accept a
# limited filter set, and sending the rest is a 400 for the entire call.
_UNSUPPORTED_FILTERS = {
    "company": frozenset({
        "exclude_domains", "start_published_date", "end_published_date",
        "start_crawl_date", "end_crawl_date",
    }),
    "people": frozenset({
        "exclude_domains", "start_published_date", "end_published_date",
        "start_crawl_date", "end_crawl_date",
    }),
}

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
        """Return True when ``EXA_API_KEY`` is set to a non-empty value."""
        from agent.web_search_provider import get_provider_env

        return bool(get_provider_env("EXA_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute an Exa search.

        Returns ``{"success": True, "data": {"web": [{...}, ...]}}`` on
        success, ``{"success": False, "error": str}`` on failure (incl.
        missing API key and SDK install errors).
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("Exa search: '%s' (limit=%d)", query, limit)
            response = _get_exa_client().search(
                query,
                num_results=limit,
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

    def advanced_search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute an Exa search with the full advanced filter set.

        Mirrors the ``web_search_advanced_exa`` MCP tool natively so the
        capability works without an MCP server. Accepts the same filters the
        Exa SDK ``search()`` exposes: ``include_domains``, ``exclude_domains``,
        ``start_published_date``/``end_published_date``,
        ``start_crawl_date``/``end_crawl_date``, ``category``, ``type``
        (auto/fast/instant), ``include_text``/``exclude_text``,
        ``user_location``, ``num_results``, ``enable_highlights``,
        ``enable_summary``, and ``subpages``.

        Returns ``{"success": True, "data": {"web": [...], "searchTime": ...}}``
        on success, ``{"success": False, "error": str}`` on failure.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("Exa advanced search: '%s'", query)

            # No contents.text: it is billed per page and the result mapper below
            # never reads result.text — highlights/summary/subpages carry the content.
            contents: Dict[str, Any] = {}
            if kwargs.get("enable_highlights"):
                contents["highlights"] = True
            if kwargs.get("enable_summary"):
                contents["summary"] = True
            if kwargs.get("subpages"):
                contents["subpages"] = int(kwargs["subpages"])

            search_kwargs: Dict[str, Any] = {
                "num_results": int(kwargs.get("num_results", 10)),
                "contents": contents,
            }
            category = kwargs.get("category")
            unsupported = _UNSUPPORTED_FILTERS.get(category, frozenset()) if category else frozenset()
            for key in (
                "include_domains",
                "exclude_domains",
                "start_published_date",
                "end_published_date",
                "start_crawl_date",
                "end_crawl_date",
                "category",
                "type",
                "include_text",
                "exclude_text",
                "user_location",
            ):
                if kwargs.get(key) is None:
                    continue
                if key in unsupported:
                    # Exa 400s on these for company/people; drop them rather than
                    # fail the whole call (the filter is a narrowing hint).
                    logger.info("Exa: dropping %s, unsupported for category=%s", key, category)
                    continue
                search_kwargs[key] = kwargs[key]

            response = _get_exa_client().search(query, **search_kwargs)

            web_results = []
            for i, result in enumerate(response.results or []):
                highlights = result.highlights or []
                item: Dict[str, Any] = {
                    "url": result.url or "",
                    "title": result.title or "",
                    "description": " ".join(highlights) if highlights else "",
                    "position": i + 1,
                }
                if kwargs.get("enable_summary"):
                    item["summary"] = result.summary or ""
                if kwargs.get("subpages"):
                    item["subpages"] = result.subpages or []
                web_results.append(item)

            return {
                "success": True,
                "data": {
                    "web": web_results,
                    "searchTime": getattr(response, "search_time", None),
                },
            }
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except ImportError as exc:
            return {"success": False, "error": f"Exa SDK not installed: {exc}"}
        except Exception as exc:  # noqa: BLE001 — surface as failure
            logger.warning("Exa advanced search error: %s", exc)
            return {"success": False, "error": f"Exa advanced search failed: {exc}"}

    def agent_run(
        self,
        instructions: str,
        *,
        output_schema: Optional[Dict[str, Any]] = None,
        model: str = "exa-research-fast",
        poll_interval: int = 2000,
        timeout_ms: int = 600000,
    ) -> Dict[str, Any]:
        """Run an Exa Agent (multi-step research) natively.

        Mirrors the ``agent_run`` MCP tool. Creates an Agent run on
        ``/agent/runs`` and polls until it reaches a terminal state, returning
        the output plus status/cost. ``model`` is accepted for API
        compatibility; the Agent API derives effort from the run payload.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("Exa agent run: '%s'", instructions)
            client = _get_exa_client()

            payload: Dict[str, Any] = {"query": instructions}
            if output_schema is not None:
                payload["outputSchema"] = output_schema

            created = client.request("/agent/runs", data=payload, method="POST")
            run_id = created.get("id") or created.get("runId")
            if not run_id:
                return {"success": False, "error": f"Agent run creation failed: {created}"}

            import time

            deadline = time.monotonic() + timeout_ms / 1000.0
            while True:
                if is_interrupted():
                    return {"success": False, "error": "Interrupted", "run_id": run_id}
                status_resp = client.request(
                    f"/agent/runs/{run_id}", method="GET"
                )
                status = status_resp.get("status", "running")
                if status in ("completed", "succeeded", "failed", "cancelled", "canceled"):
                    break
                if time.monotonic() > deadline:
                    return {
                        "success": False,
                        "error": f"Agent run timed out after {timeout_ms}ms",
                        "run_id": run_id,
                        "status": status,
                    }
                time.sleep(poll_interval / 1000.0)

            result = {
                "success": status in ("completed", "succeeded"),
                "run_id": run_id,
                "status": status,
                "output": status_resp.get("output"),
            }
            if status in ("failed", "cancelled", "canceled"):
                result["error"] = status_resp.get("error") or f"agent run {status}"
            cost = status_resp.get("costDollars") or status_resp.get("cost_dollars")
            if cost is not None:
                result["cost_dollars"] = cost
            return result
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except ImportError as exc:
            return {"success": False, "error": f"Exa SDK not installed: {exc}"}
        except Exception as exc:  # noqa: BLE001 — surface as failure
            logger.warning("Exa agent run error: %s", exc)
            return {"success": False, "error": f"Exa agent run failed: {exc}"}

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
