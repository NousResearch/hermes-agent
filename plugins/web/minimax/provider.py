"""MiniMax search — plugin form using direct API calls.

Subclasses the plugin-facing :class:`agent.web_search_provider.WebSearchProvider`.
Uses direct API calls to MiniMax search endpoint.

API key resolution (Hermes standard):
  - MINIMAX_API_KEY env var (set via `hermes tools`)
  - MINIMAX_CN_API_KEY env var (set via `hermes tools`)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _load_hermes_dotenv():
    """Load ~/.hermes/.env if not already loaded."""
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value


# Load .env on module import
_load_hermes_dotenv()


# MiniMax search API endpoint
_SEARCH_ENDPOINT = "https://api.minimax.io/v1/coding_plan/search"
_CN_SEARCH_ENDPOINT = "https://api.minimaxi.com/v1/coding_plan/search"


def _load_api_key() -> tuple[str | None, str | None]:
    """Load API key and region.

    Resolution order:
      1. MINIMAX_API_KEY env var (Global region)
      2. MINIMAX_CN_API_KEY env var (CN region)

    Region config:
      - Reads web.region from ~/.hermes/config.yaml
      - Defaults to 'global' if not set

    Returns:
        tuple of (api_key, region) — region is 'cn' or 'global'
    """
    # Check region config first
    region = _load_region_config()

    # Try region-specific key first
    if region == "cn":
        api_key = os.getenv("MINIMAX_CN_API_KEY")
        if api_key:
            return api_key, "cn"
        # Fallback to global key
        api_key = os.getenv("MINIMAX_API_KEY")
        if api_key:
            return api_key, "global"
    else:
        # Global region
        api_key = os.getenv("MINIMAX_API_KEY")
        if api_key:
            return api_key, "global"
        # Fallback to CN key
        api_key = os.getenv("MINIMAX_CN_API_KEY")
        if api_key:
            return api_key, "cn"

    return None, None


def _load_region_config() -> str:
    """Load region from Hermes config.yaml (web.region).

    Defaults to 'global' if not set.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        web_cfg = cfg.get("web", {})
        region = web_cfg.get("region", "global")
        if region in ("cn", "global"):
            return region
    except Exception:
        pass
    return "global"


def _get_search_endpoint(region: str) -> str:
    """Get the appropriate search endpoint based on region."""
    if region == "cn":
        return _CN_SEARCH_ENDPOINT
    return _SEARCH_ENDPOINT


class MiniMaxWebSearchProvider(WebSearchProvider):
    """MiniMax web search provider using direct API calls.

    API key source:
      - MINIMAX_API_KEY env var (set via `hermes tools`)
    """

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax"

    def is_available(self) -> bool:
        """Return True when a MiniMax API key is configured."""
        api_key, _ = _load_api_key()
        return api_key is not None

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a MiniMax search via direct API call.

        Direct API call:
          - POST to /v1/coding_plan/search with { q: query }
          - Returns raw results with title, link, snippet, date
        """
        api_key, region = _load_api_key()

        if not api_key:
            return {
                "success": False,
                "error": (
                    "No MiniMax API key found.\n"
                    "Set MINIMAX_API_KEY env var via `hermes tools`"
                ),
            }

        endpoint = _get_search_endpoint(region)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "q": query,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(endpoint, headers=headers, json=body)
        except httpx.TimeoutException:
            logger.warning("MiniMax search timed out")
            return {"success": False, "error": "MiniMax search timed out after 30s"}
        except Exception as exc:
            logger.warning("MiniMax search HTTP error: %s", exc)
            return {"success": False, "error": f"MiniMax search failed: {exc}"}

        if response.status_code != 200:
            logger.warning("MiniMax search HTTP %d: %s", response.status_code, response.text[:200])
            return {"success": False, "error": f"MiniMax search failed with status {response.status_code}"}

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            logger.warning("MiniMax search JSON parse error: %s", exc)
            return {"success": False, "error": f"Invalid JSON response: {exc}"}

        # Check for API-level errors
        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code") and base_resp["status_code"] != 0:
            error_msg = base_resp.get("status_msg", f"API error {base_resp['status_code']}")
            return {"success": False, "error": f"MiniMax search API error: {error_msg}"}

        # Return all results without limiting
        organic = data.get("organic", [])
        logger.info("MiniMax search '%s': %d results", query, len(organic))
        return {
            "success": True,
            "organic": organic,
            "base_resp": data.get("base_resp", {"status_code": 0, "status_msg": "success"}),
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MiniMax",
            "badge": "search only",
            "tag": "Search via MiniMax AI platform",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax API key",
                    "url": "https://platform.minimax.io/",
                },
            ],
            "region_selection": [
                {"name": "Global (api.minimax.io)", "value": "global"},
                {"name": "China (api.minimaxi.com)", "value": "cn"},
            ],
            "region_env_var_map": {
                "global": "MINIMAX_API_KEY",
                "cn": "MINIMAX_CN_API_KEY",
            },
        }
