"""MiniMax CLI web search — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`.

Capabilities advertised:
- ``supports_search()``  -> True  (``mmx search query``)
- ``supports_extract()`` -> False (mmx-cli has no extract command)

How it works:
- Auth is via the existing ``MINIMAX_API_KEY`` (set in ``~/.hermes/.env``).
  The CLI persists its own copy to ``~/.mmx/config.json`` after the first
  ``mmx auth login --api-key ...`` call. Subsequent calls do not need the
  env var — the CLI reads from its config.
- Billing is against the MiniMax Token Plan weekly quota (96% general
  remaining, 90% weekly as of 2026-06-05). No separate API key, no
  per-call cost beyond the Token Plan.
- The provider shells out via :func:`subprocess.run` with a 60s timeout
  and parses the JSON envelope returned by ``mmx search query``.

Config keys this provider responds to::

    web:
      search_backend: "mmx_cli"     # use mmx-cli for search
      extract_backend: "tavily"     # keep an extract-capable backend
      backend: "tavily"             # shared fallback (search-only if no override)

Failure modes:
- ``mmx`` binary missing             -> is_available() returns False (provider
                                        does not register, falls back to next
                                        available backend).
- ``mmx auth`` failed (no key)       -> search() returns ``{success: False,
                                        error: ...}``; caller falls back to
                                        next backend in the chain.
- Token Plan quota exhausted         -> search() returns ``{success: False,
                                        error: "quota exceeded"}``; same
                                        fallback path.
- ``mmx search query`` non-zero exit -> search() surfaces stderr in error.
- JSON parse error                   -> search() returns ``{success: False,
                                        error: "mmx returned non-JSON"}``.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 60


def _mmx_is_installed() -> bool:
    """Return True if the ``mmx`` binary is on PATH.

    Cheap — no subprocess, no network I/O. Safe to call from
    :meth:`is_available`.
    """
    return shutil.which("mmx") is not None


def _mmx_search(query: str, limit: int) -> Dict[str, Any]:
    """Shell out to ``mmx search query`` and return the parsed JSON envelope.

    Raises ``RuntimeError`` on non-zero exit; the caller converts that
    into a typed error response. Raises ``ValueError`` if the output
    isn't valid JSON.
    """
    safe_limit = max(1, min(int(limit), 20))
    cmd = [
        "mmx", "search", "query",
        "--q", query,
        "--limit", str(safe_limit),
        "--output", "json",
        "--quiet",
        "--non-interactive",
    ]
    logger.info("mmx search: '%s' (limit=%d)", query, safe_limit)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"mmx search query exited {result.returncode}: {stderr or 'no stderr'}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"mmx returned non-JSON output: {exc}") from exc


def _normalize_mmx_response(response: Dict[str, Any], limit: int) -> Dict[str, Any]:
    """Map mmx ``search query`` response to ``{success, data: {web: [...]}}``.

    The mmx envelope is::

        {
          "organic": [
            {"title": str, "link": str, "snippet": str, "date": str},
            ...
          ],
          ...other fields we ignore
        }

    Returns the same shape the hermes web_search tool expects from any
    provider (Tavily, Exa, etc.).
    """
    web_results: List[Dict[str, Any]] = []
    for i, result in enumerate(response.get("organic", [])):
        if i >= limit:
            break
        web_results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "description": result.get("snippet", ""),
                "position": i + 1,
                # mmx-specific extra — preserved for callers that want it
                "date": result.get("date", ""),
            }
        )
    return {"success": True, "data": {"web": web_results}}


class MMXCliWebSearchProvider(WebSearchProvider):
    """MiniMax CLI web search provider.

    Search-only. Use together with an extract-capable backend (Tavily,
    Firecrawl, etc.) when content extraction is also needed.
    """

    @property
    def name(self) -> str:
        return "mmx_cli"

    @property
    def display_name(self) -> str:
        return "MiniMax CLI (Token Plan)"

    def is_available(self) -> bool:
        """Return True when the ``mmx`` binary is on PATH.

        We do NOT call ``mmx auth status`` here — the comment on
        :class:`WebSearchProvider` is explicit that ``is_available`` must
        avoid network I/O. Auth failures surface naturally in ``search()``
        as a typed error response, and the caller falls back to the next
        backend in the chain.
        """
        return _mmx_is_installed()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute an mmx search and return normalized results."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}
        except ImportError:
            # tools.interrupt is hermes-agent internal; if it's not on
            # sys.path during unit tests, we just skip the interrupt check.
            pass

        try:
            raw = _mmx_search(query, limit)
            return _normalize_mmx_response(raw, limit)
        except subprocess.TimeoutExpired:
            logger.warning("mmx search timed out after %ds", _TIMEOUT_SECONDS)
            return {
                "success": False,
                "error": f"mmx search timed out after {_TIMEOUT_SECONDS}s",
            }
        except FileNotFoundError:
            # Edge case: shutil.which found mmx at registration time but
            # the binary disappeared before the subprocess call.
            return {
                "success": False,
                "error": "mmx binary not found at call time (was it uninstalled?)",
            }
        except RuntimeError as exc:
            # Non-zero exit from mmx — auth failure, quota exceeded, etc.
            logger.warning("mmx search error: %s", exc)
            return {"success": False, "error": f"mmx search failed: {exc}"}
        except ValueError as exc:
            logger.warning("mmx returned unparseable output: %s", exc)
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — last-resort catch
            logger.exception("Unexpected mmx search error")
            return {"success": False, "error": f"mmx search crashed: {exc}"}

    def get_setup_schema(self) -> Dict[str, Any]:
        """Schema for ``hermes setup`` wizard prompts."""
        return {
            "name": "MiniMax CLI (Token Plan)",
            "badge": "token-plan",
            "tag": "Web search via the MiniMax CLI. Uses your existing "
                   "MINIMAX_API_KEY and Token Plan weekly quota. Search only "
                   "(no content extraction) — pair with an extract backend.",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax API key (Token Plan)",
                    "url": "https://api.minimax.io",
                    "shared_with": "model provider",
                },
            ],
            "post_install": [
                "npm install -g mmx-cli",
                "mmx auth login --api-key $MINIMAX_API_KEY",
                "mmx quota   # verify Token Plan is active",
            ],
        }
