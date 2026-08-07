"""SearXNG search — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Same JSON
API call (``/search?format=json``), same result normalization. The legacy
in-tree module ``tools.web_providers.searxng`` was removed in the same
commit that moved this code under ``plugins/``; this file is now the
canonical implementation.

Search-only — SearXNG aggregates results from upstream engines but does not
fetch/extract arbitrary URLs. ``supports_extract()`` returns False.

Config keys this provider responds to::

    web:
      search_backend: "searxng"     # explicit per-capability
      backend: "searxng"            # shared fallback

Env var::

    SEARXNG_URL=http://localhost:8080
"""

from __future__ import annotations

import logging
import math
import os
import unicodedata
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_MAX_UNRESPONSIVE_ENGINES = 10
_MAX_UNRESPONSIVE_ENTRIES_TO_INSPECT = 100
_MAX_ENGINE_NAME_LENGTH = 80
_MAX_ENGINE_REASON_LENGTH = 200


def _sanitize_diagnostic_text(value: Any, max_length: int) -> tuple[str, bool]:
    """Return bounded, flattened diagnostic text plus a truncation indicator."""
    if not isinstance(value, str):
        return "", False
    inspection_limit = max_length * 4
    bounded = value[:inspection_limit]
    printable = "".join(
        " " if unicodedata.category(char).startswith("C") else char for char in bounded
    )
    normalized = " ".join(printable.split())
    truncated = len(value) > inspection_limit or len(normalized) > max_length
    return normalized[:max_length].rstrip(), truncated


def _result_score(result: Dict[str, Any]) -> float:
    """Return a sortable score without trusting the upstream value's type."""
    try:
        score = float(result.get("score", 0))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _searxng_url() -> str:
    """Return SEARXNG_URL from Hermes config-aware env, falling back to process env."""
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value("SEARXNG_URL")
    except Exception:
        val = None
    if val is None:
        val = os.getenv("SEARXNG_URL", "")
    return (val or "").strip()


class SearXNGWebSearchProvider(WebSearchProvider):
    """Search via a user-hosted SearXNG instance."""

    @property
    def name(self) -> str:
        return "searxng"

    @property
    def display_name(self) -> str:
        return "SearXNG"

    def is_available(self) -> bool:
        """Return True when ``SEARXNG_URL`` is set."""
        return bool(_searxng_url())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search against the configured SearXNG instance."""
        import httpx

        base_url = _searxng_url().rstrip("/")
        if not base_url:
            return {"success": False, "error": "SEARXNG_URL is not set"}

        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }

        try:
            resp = httpx.get(
                f"{base_url}/search",
                params=params,
                timeout=15,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("SearXNG HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"SearXNG returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("SearXNG request error: %s", exc)
            return {
                "success": False,
                "error": f"Could not reach SearXNG at {base_url}: {exc}",
            }

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("SearXNG response parse error: %s", exc)
            return {
                "success": False,
                "error": "Could not parse SearXNG response as JSON",
            }

        if not isinstance(data, dict):
            logger.warning("SearXNG returned an invalid top-level JSON payload")
            return {
                "success": False,
                "error": "SearXNG returned an invalid JSON response",
            }

        if "results" not in data:
            logger.warning("SearXNG response did not contain a results field")
            return {
                "success": False,
                "error": "SearXNG returned an invalid response without results",
            }

        raw_results = data["results"]
        if not isinstance(raw_results, list):
            logger.warning("SearXNG returned an invalid results payload")
            return {
                "success": False,
                "error": "SearXNG returned an invalid results payload",
            }

        unresponsive_engines = []
        unresponsive_engines_truncated = False
        unresponsive_engine_diagnostics_invalid = False
        raw_unresponsive_engines = data.get("unresponsive_engines", [])
        if isinstance(raw_unresponsive_engines, (list, tuple)):
            if len(raw_unresponsive_engines) > _MAX_UNRESPONSIVE_ENTRIES_TO_INSPECT:
                unresponsive_engines_truncated = True
            for entry in raw_unresponsive_engines[
                :_MAX_UNRESPONSIVE_ENTRIES_TO_INSPECT
            ]:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    unresponsive_engine_diagnostics_invalid = True
                    continue
                engine, reason = entry[:2]
                engine, engine_truncated = _sanitize_diagnostic_text(
                    engine, _MAX_ENGINE_NAME_LENGTH
                )
                reason, reason_truncated = _sanitize_diagnostic_text(
                    reason, _MAX_ENGINE_REASON_LENGTH
                )
                if engine_truncated or reason_truncated:
                    unresponsive_engines_truncated = True
                if not engine or not reason:
                    unresponsive_engine_diagnostics_invalid = True
                    continue
                if len(unresponsive_engines) < _MAX_UNRESPONSIVE_ENGINES:
                    unresponsive_engines.append({
                        "engine": engine,
                        "reason": reason,
                    })
                else:
                    unresponsive_engines_truncated = True
        elif "unresponsive_engines" in data:
            unresponsive_engine_diagnostics_invalid = True

        valid_results = []
        malformed_results = 0
        for result in raw_results:
            if not isinstance(result, dict):
                malformed_results += 1
                continue
            url = result.get("url")
            if not isinstance(url, str) or not url.strip():
                malformed_results += 1
                continue
            valid_results.append(result)

        if malformed_results:
            logger.warning(
                "SearXNG ignored %d malformed or URL-less result entries",
                malformed_results,
            )

        # SearXNG may return a score field; sort descending and cap to limit.
        sorted_results = sorted(
            valid_results,
            key=_result_score,
            reverse=True,
        )[:limit]

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("content", "")),
                "position": i + 1,
            }
            for i, r in enumerate(sorted_results)
        ]

        logger.info(
            "SearXNG search '%s': %d results (from %d raw, limit %d)",
            query,
            len(web_results),
            len(raw_results),
            limit,
        )

        result_data: Dict[str, Any] = {"web": web_results}
        diagnostic_parts = [
            f"{item['engine']} ({item['reason']})" for item in unresponsive_engines
        ]
        if unresponsive_engines_truncated:
            diagnostic_parts.append("additional engine diagnostics omitted")
        if unresponsive_engine_diagnostics_invalid:
            diagnostic_parts.append("some engine diagnostics were invalid")

        if diagnostic_parts:
            diagnostic = "; ".join(diagnostic_parts)
            if not web_results:
                logger.warning(
                    "SearXNG search failed with no usable results; %s",
                    diagnostic,
                )
                return {
                    "success": False,
                    "error": f"SearXNG returned no usable results; {diagnostic}",
                }

            result_data.update({
                "degraded": True,
                "unresponsive_engines": unresponsive_engines,
            })
            if unresponsive_engines_truncated:
                result_data["unresponsive_engines_truncated"] = True
            if unresponsive_engine_diagnostics_invalid:
                result_data["unresponsive_engine_diagnostics_invalid"] = True
            logger.warning(
                "SearXNG search returned degraded results; %s",
                diagnostic,
            )

        if not web_results and malformed_results:
            logger.warning(
                "SearXNG search failed with no usable results; "
                "%d malformed result entries",
                malformed_results,
            )
            return {
                "success": False,
                "error": (
                    "SearXNG returned no usable results because "
                    f"{malformed_results} result entries were invalid"
                ),
            }

        return {"success": True, "data": result_data}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "SearXNG",
            "badge": "free · self-hosted",
            "tag": "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.",
            "env_vars": [
                {
                    "key": "SEARXNG_URL",
                    "prompt": "SearXNG instance URL (e.g. http://localhost:8080)",
                    "url": "https://searx.space/",
                },
            ],
        }
