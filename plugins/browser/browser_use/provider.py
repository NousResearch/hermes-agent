"""Browser Use Cloud with direct credentials and a durable employee profile."""

from __future__ import annotations

import logging
from typing import Any, Dict


from agent.secret_scope import get_secret
from plugins.browser._common import CloudBrowserProvider

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.browser-use.com/api/v3"


class BrowserUseBrowserProvider(CloudBrowserProvider):
    """Browser Use (https://browser-use.com) cloud browser backend."""

    provider_id = "browser-use"
    label = "Browser Use"
    release_method = "patch"
    release_path = "/browsers/{session_id}"
    # Hidden from the picker (its "Browser Use" row activates tools/browser_use_cli.py); stays
    # registered for the Nous gateway path and legacy cloud_provider configs.
    setup_tag = None

    def is_available(self) -> bool:
        return self._get_config_or_none(refresh_token=False) is not None

    def _get_config_or_none(self, *, refresh_token: bool = True):
        api_key = get_secret("BROWSER_USE_API_KEY")
        return {"api_key": api_key, "base_url": _BASE_URL, "managed_mode": False} if api_key else None

    def _get_config(self) -> Dict[str, Any]:
        config = self._get_config_or_none()
        if config is None:
            raise ValueError("Browser Use requires a direct BROWSER_USE_API_KEY credential.")
        return config

    def _headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        return {"Content-Type": "application/json", "X-Browser-Use-API-Key": config["api_key"]}

    def _release_body(self, config: Dict[str, Any]) -> Dict[str, object]:
        return {"action": "stop"}

    def create_session(self, task_id: str) -> Dict[str, object]:
        config = self._get_config()
        headers = self._headers(config)
        from .profile import cloud_profile
        payload = {"profileId": cloud_profile(config["base_url"], headers)}
        response = self._post_create(f"{config['base_url']}/browsers", headers, payload)
        self._check_created(response)
        session_data = response.json()
        session_name = self._session_name(task_id)
        logger.info("Created Browser Use session %s", session_name)
        return {
            "session_name": session_name,
            "bb_session_id": session_data["id"],
            "cdp_url": session_data.get("cdpUrl") or session_data.get("connectUrl") or "",
            # Fixed server-side lifetime: keep the API's authority so an expired CDP endpoint is retired.
            "expires_at": session_data.get("timeoutAt"),
            "features": {"browser_use": True},
        }


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402


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
