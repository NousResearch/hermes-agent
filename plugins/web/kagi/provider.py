"""Kagi web search provider.

Uses the existing ``kagi_search.py`` script which scrapes Kagi's
``/html/search`` endpoint via a session cookie.  Because Kagi's Search
API is still in closed beta, this provider shells out to the script
rather than calling the API directly.

Configuration::

    # ~/.hermes/.env
    KAGI_SESSION=<session_cookie>

    # ~/.hermes/config.yaml
    web:
      search_backend: kagi
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_SCRIPT = Path(os.path.expanduser(
    "~/.hermes/skills/research/kagi-search/scripts/kagi_search.py"
))


def _has_kagi_session() -> bool:
    """Return True when KAGI_SESSION is available (env var or .env file)."""
    env_path = Path(os.path.expanduser("~/.hermes/.env"))
    try:
        for line in env_path.read_text().splitlines():
            if line.startswith("KAGI_SESSION=") and line.split("=", 1)[1].strip():
                return True
    except (OSError, FileNotFoundError):
        pass
    return bool(os.getenv("KAGI_SESSION", "").strip())


class KagiWebSearchProvider(WebSearchProvider):
    """Search via Kagi's session-cookie-backed /html/search endpoint.

    Delegates to ``kagi_search.py --json`` which handles cookie loading
    and HTML parsing.  Only search is supported — there is no extract
    or crawl capability.
    """

    @property
    def name(self) -> str:
        return "kagi"

    @property
    def display_name(self) -> str:
        return "Kagi"

    def is_available(self) -> bool:
        return _has_kagi_session() and _SCRIPT.exists()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "success": False,
                "error": "KAGI_SESSION not found or kagi_search.py missing",
            }

        try:
            result = subprocess.run(
                ["python3", str(_SCRIPT), "--json", "--limit", str(limit), query],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Kagi search timed out (30s)"}
        except Exception as exc:
            return {"success": False, "error": f"Kagi search failed: {exc}"}

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"kagi_search.py exited {result.returncode}: {result.stderr.strip()}",
            }

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"success": False, "error": "Could not parse Kagi response"}

        results = data.get("results", [])[:limit]
        web_results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("snippet", ""),
                "position": i + 1,
            }
            for i, r in enumerate(results)
        ]

        logger.info(
            "Kagi search '%s': %d results (limit %d)",
            query, len(web_results), limit,
        )
        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Kagi",
            "badge": "free · session cookie · search only",
            "tag": "Search via Kagi session cookie — set KAGI_SESSION in ~/.hermes/.env",
            "env_vars": [
                {
                    "key": "KAGI_SESSION",
                    "prompt": "Kagi session cookie",
                    "url": "https://kagi.com",
                },
            ],
        }
