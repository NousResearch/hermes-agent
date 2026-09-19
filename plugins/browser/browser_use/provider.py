"""Browser Use cloud browser provider — the only backend with dual auth: direct
``BROWSER_USE_API_KEY`` (https://browser-use.com) or the managed Nous tool gateway (bills to a
Nous subscription). Direct first, managed second, unless ``tool_gateway.browser: gateway`` flips
it. Config: ``browser.cloud_provider: "browser-use"``."""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, Optional

import requests

from agent.secret_scope import get_secret
from plugins.browser._common import CloudBrowserProvider

logger = logging.getLogger(__name__)

# Managed-mode create idempotency keys: the gateway answers retried POSTs with 409 "already in
# progress", so the original key is forwarded; cleared on success or terminal failure.
_pending_create_keys: Dict[str, str] = {}
_pending_create_keys_lock = threading.Lock()

_BASE_URL = "https://api.browser-use.com/api/v4"
_DEFAULT_MANAGED_TIMEOUT_MINUTES = 5
_DEFAULT_MANAGED_PROXY_COUNTRY_CODE = "us"


def _get_or_create_pending_create_key(task_id: str) -> str:
    with _pending_create_keys_lock:
        existing = _pending_create_keys.get(task_id)
        if existing:
            return existing
        created = f"browser-use-session-create:{uuid.uuid4().hex}"
        _pending_create_keys[task_id] = created
        return created


def _clear_pending_create_key(task_id: str) -> None:
    with _pending_create_keys_lock:
        _pending_create_keys.pop(task_id, None)


def _should_preserve_pending_create_key(response: requests.Response) -> bool:
    """Keep the key when retryable: any 5xx, or a 409 saying the original request is still in
    flight. Other 4xx (auth, bad request) won't succeed on retry, so the key is dropped."""
    if response.status_code >= 500:
        return True
    if response.status_code != 409:
        return False
    try:
        payload = response.json()
    except Exception:
        return False
    error = payload.get("error") if isinstance(payload, dict) else None
    return isinstance(error, dict) and "already in progress" in str(error.get("message") or "").lower()


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

    def _get_config_or_none(self, *, refresh_token: bool = True) -> Optional[Dict[str, Any]]:
        # Lazy: managed_tool_gateway pulls in the Nous auth stack direct-key users never need.
        from tools.managed_tool_gateway import peek_nous_access_token, resolve_managed_tool_gateway
        from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, read_selection

        def _managed_config() -> Optional[Dict[str, Any]]:
            # Keep availability scans off the synchronous OAuth refresh path.
            managed = resolve_managed_tool_gateway(
                "browser-use", token_reader=None if refresh_token else peek_nous_access_token)
            if managed is None:
                return None
            return {
                "api_key": managed.nous_user_token,
                "base_url": managed.gateway_origin.rstrip("/"),
                "managed_mode": True,
            }

        api_key = get_secret("BROWSER_USE_API_KEY")
        selected = read_selection("browser")
        direct = {"api_key": api_key, "base_url": _BASE_URL, "managed_mode": False}

        # Strict: "nous" (or legacy use_gateway: true) → managed ONLY; any other stored selection →
        # direct ONLY (no silent managed fallback); never-configured → direct if present, else managed.
        if selected == NOUS_MANAGED_PROVIDER:
            return _managed_config()
        if selected is not None:
            return direct if api_key else None
        return direct if api_key else _managed_config()

    def _get_config(self) -> Dict[str, Any]:
        from tools.tool_backend_helpers import (
            NOUS_MANAGED_PROVIDER, managed_nous_tools_enabled, read_selection, selection_error)

        config = self._get_config_or_none()
        if config is not None:
            return config
        selected = read_selection("browser")
        if selected == NOUS_MANAGED_PROVIDER:
            raise ValueError(selection_error(
                "browser", NOUS_MANAGED_PROVIDER,
                "the Nous Tool Gateway is not available (not entitled or unreachable)"))
        if selected is not None:
            raise ValueError(selection_error("browser", selected, "BROWSER_USE_API_KEY is not set"))
        if managed_nous_tools_enabled():
            raise ValueError(
                "Browser Use requires either a direct BROWSER_USE_API_KEY "
                "credential or a managed Browser Use gateway configuration.")
        raise ValueError("Browser Use requires a direct BROWSER_USE_API_KEY credential.")

    def _headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        return {"Content-Type": "application/json", "X-Browser-Use-API-Key": config["api_key"]}

    def _release_body(self, config: Dict[str, Any]) -> Dict[str, object]:
        return {"action": "stop"}

    def create_session(self, task_id: str) -> Dict[str, object]:
        config = self._get_config()
        managed_mode = bool(config.get("managed_mode"))

        headers = self._headers(config)
        if managed_mode:
            headers["X-Idempotency-Key"] = _get_or_create_pending_create_key(task_id)
        # Short gateway sessions: billing authorization must not default to a long Browser-Use timeout.
        payload = (
            {"timeout": _DEFAULT_MANAGED_TIMEOUT_MINUTES, "proxyCountryCode": _DEFAULT_MANAGED_PROXY_COUNTRY_CODE}
            if managed_mode else {})
        # Managed mode propagates network errors raw (retry with the preserved key); direct wraps them.
        response = self._post_create(
            f"{config['base_url']}/browsers", headers, payload, wrap_errors=not managed_mode)
        if not response.ok and managed_mode and not _should_preserve_pending_create_key(response):
            _clear_pending_create_key(task_id)
        self._check_created(response)

        session_data = response.json()
        if managed_mode:
            _clear_pending_create_key(task_id)
        session_name = self._session_name(task_id)
        logger.info("Created Browser Use session %s", session_name)
        return {
            "session_name": session_name,
            "bb_session_id": session_data["id"],
            "cdp_url": session_data.get("cdpUrl") or session_data.get("connectUrl") or "",
            # Fixed server-side lifetime: keep the API's authority so an expired CDP endpoint is retired.
            "expires_at": session_data.get("timeoutAt"),
            "features": {"browser_use": True},
            "external_call_id": response.headers.get("x-external-call-id") if managed_mode else None,
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
