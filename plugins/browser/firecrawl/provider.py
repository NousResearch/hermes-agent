"""Firecrawl cloud browser (``/v2/browser`` only; the web plugin under ``plugins/web/firecrawl/``
shares ``FIRECRAWL_API_KEY``). Config ``browser.cloud_provider: "firecrawl"`` (explicit only — not
in the legacy auto-detect walk). Env: ``FIRECRAWL_API_KEY``, ``FIRECRAWL_API_URL`` (default
https://api.firecrawl.dev), ``FIRECRAWL_BROWSER_TTL`` (default 300 s)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from agent.secret_scope import get_secret
from plugins.browser._common import CloudBrowserProvider

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.firecrawl.dev"


class FirecrawlBrowserProvider(CloudBrowserProvider):
    """Firecrawl (https://firecrawl.dev) cloud browser backend."""

    provider_id = "firecrawl"
    label = "Firecrawl"
    release_method = "delete"
    release_path = "/v2/browser/{session_id}"
    create_label_suffix = " browser"
    setup_tag = "Cloud browser with remote execution"
    setup_env_vars = [
        {"key": "FIRECRAWL_API_KEY", "prompt": "Firecrawl API key", "url": "https://firecrawl.dev"},
    ]

    def _api_url(self) -> str:
        # Per-profile like the key: the scoped key must not be sent to the default profile's endpoint.
        return get_secret("FIRECRAWL_API_URL", "") or _BASE_URL

    def _get_config_or_none(self) -> Optional[dict[str, Any]]:
        return {"base_url": self._api_url()} if get_secret("FIRECRAWL_API_KEY") else None

    def _get_config(self) -> dict[str, Any]:
        # Never raises: a missing key surfaces from _headers() inside the request try-block, so
        # close_session logs it as an exception (legacy behaviour).
        return {"base_url": self._api_url()}

    def _headers(self, config: Optional[dict[str, Any]] = None) -> dict[str, str]:
        api_key = get_secret("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError(
                "FIRECRAWL_API_KEY environment variable is required. "
                "Get your key at https://firecrawl.dev")
        return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def create_session(self, task_id: str) -> dict[str, object]:
        try:
            ttl = int(os.environ.get("FIRECRAWL_BROWSER_TTL", "300"))
        except (ValueError, TypeError):
            ttl = 300

        response = self._post_create(f"{self._api_url()}/v2/browser", self._headers(), {"ttl": ttl})
        self._check_created(response)
        data = response.json()
        session_name = self._session_name(task_id)
        logger.info("Created Firecrawl browser session %s", session_name)
        return {
            "session_name": session_name,
            "bb_session_id": data["id"],
            "cdp_url": data["cdpUrl"],
            "features": {"firecrawl": True},
        }
