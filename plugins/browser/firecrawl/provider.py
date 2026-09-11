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
        return os.environ.get("FIRECRAWL_API_URL", _BASE_URL)

    def _get_config_or_none(self) -> Optional[Dict[str, Any]]:
        return {"base_url": self._api_url()} if get_secret("FIRECRAWL_API_KEY") else None

    def _get_config(self) -> Dict[str, Any]:
        # Never raises: a missing key surfaces from _headers() inside the request try-block, so
        # close_session logs it as an exception (legacy behaviour).
        return {"base_url": self._api_url()}

    def _headers(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        api_key = get_secret("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError(
                "FIRECRAWL_API_KEY environment variable is required. "
                "Get your key at https://firecrawl.dev")
        return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def create_session(self, task_id: str) -> Dict[str, object]:
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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import requests  # noqa: F401,E402
import uuid  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'BrowserProvider': ('agent.browser_provider', 'BrowserProvider'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
