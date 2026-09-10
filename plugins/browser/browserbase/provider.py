"""Browserbase cloud browser provider: direct ``BROWSERBASE_API_KEY`` + ``BROWSERBASE_PROJECT_ID``
only (the Nous subscription routes through Browser Use). Config ``browser.cloud_provider:
"browserbase"``; knobs ``BROWSERBASE_BASE_URL``, ``BROWSERBASE_PROXIES`` (true),
``BROWSERBASE_ADVANCED_STEALTH`` (false), ``BROWSERBASE_KEEP_ALIVE`` (true),
``BROWSERBASE_SESSION_TIMEOUT`` (seconds, max 21600)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from agent.secret_scope import get_secret
from plugins.browser._common import CloudBrowserProvider

logger = logging.getLogger(__name__)

# 402 fallbacks, in retry order: (session_config key, warning logged when dropped).
_PAID_FEATURE_FALLBACKS = (
    ("keepAlive", "keepAlive may require paid plan (402), retrying without it. "
                  "Sessions may timeout during long operations."),
    ("proxies", "Proxies unavailable (402), retrying without proxies. "
                "Bot detection may be less effective."))


class BrowserbaseBrowserProvider(CloudBrowserProvider):
    """Browserbase (https://browserbase.com) cloud browser backend."""

    provider_id = "browserbase"
    label = "Browserbase"
    release_method = "post"
    release_path = "/v1/sessions/{session_id}"
    missing_credentials_error = (
        "Browserbase requires BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID "
        "environment variables.")
    close_fail_fmt = "Failed to close session %s: HTTP %s - %s"
    setup_tag = "Cloud browser with stealth and proxies"
    setup_env_vars = [
        {"key": "BROWSERBASE_API_KEY", "prompt": "Browserbase API key", "url": "https://browserbase.com"},
        {"key": "BROWSERBASE_PROJECT_ID", "prompt": "Browserbase project ID"},
    ]

    def _get_config_or_none(self) -> Optional[Dict[str, Any]]:
        api_key = get_secret("BROWSERBASE_API_KEY")
        project_id = get_secret("BROWSERBASE_PROJECT_ID")
        if not (api_key and project_id):
            return None
        return {
            "api_key": api_key,
            "project_id": project_id,
            "base_url": os.environ.get("BROWSERBASE_BASE_URL", "https://api.browserbase.com").rstrip("/"),
        }

    def _headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        return {"Content-Type": "application/json", "X-BB-API-Key": config["api_key"]}

    def _release_headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        return {"X-BB-API-Key": config["api_key"], "Content-Type": "application/json"}

    def _release_body(self, config: Dict[str, Any]) -> Dict[str, object]:
        return {"projectId": config["project_id"], "status": "REQUEST_RELEASE"}

    def create_session(self, task_id: str) -> Dict[str, object]:
        config = self._get_config()
        enable_proxies = os.environ.get("BROWSERBASE_PROXIES", "true").lower() != "false"
        enable_advanced_stealth = os.environ.get("BROWSERBASE_ADVANCED_STEALTH", "false").lower() == "true"
        enable_keep_alive = os.environ.get("BROWSERBASE_KEEP_ALIVE", "true").lower() != "false"
        custom_timeout_ms = os.environ.get("BROWSERBASE_SESSION_TIMEOUT")

        session_config: Dict[str, object] = {"projectId": config["project_id"]}
        if enable_keep_alive:
            session_config["keepAlive"] = True
        if custom_timeout_ms:
            try:
                timeout_val = int(custom_timeout_ms)
                if timeout_val > 0:
                    session_config["timeout"] = timeout_val
            except ValueError:
                logger.warning("Invalid BROWSERBASE_SESSION_TIMEOUT value: %s", custom_timeout_ms)
        if enable_proxies:
            session_config["proxies"] = True
        if enable_advanced_stealth:
            session_config["browserSettings"] = {"advancedStealth": True}

        url = f"{config['base_url']}/v1/sessions"
        headers = self._headers(config)
        response = self._post_create(url, headers, session_config)

        # 402 — paid features unavailable: drop keepAlive, then proxies, and retry.
        dropped = set()
        for key, warning in _PAID_FEATURE_FALLBACKS:
            if response.status_code == 402 and key in session_config:
                dropped.add(key)
                logger.warning(warning)
                session_config.pop(key)
                response = self._post_create(url, headers, session_config)
        self._check_created(response)

        session_data = response.json()
        session_name = self._session_name(task_id)
        features_enabled = {
            "basic_stealth": True,
            "proxies": enable_proxies and "proxies" not in dropped,
            "advanced_stealth": enable_advanced_stealth,
            "keep_alive": enable_keep_alive and "keepAlive" not in dropped,
            "custom_timeout": bool(custom_timeout_ms) and "timeout" in session_config,
        }
        feature_str = ", ".join(k for k, v in features_enabled.items() if v)
        logger.info("Created Browserbase session %s with features: %s", session_name, feature_str)
        return {
            "session_name": session_name,
            "bb_session_id": session_data["id"],
            "cdp_url": session_data["connectUrl"],
            "features": features_enabled,
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
