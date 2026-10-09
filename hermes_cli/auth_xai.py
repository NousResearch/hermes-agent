"""xAI Grok OAuth: token store, discovery, refresh, device-code login.

Split out of ``hermes_cli/auth.py``; origin helpers are imported lazily per function so
``hermes_cli.auth.<helper>`` patches still intercept and no cycle forms.
"""

from __future__ import annotations
from auth.providers.xai import _save_xai_oauth_tokens, _token_pair, _xai_access_token_is_expiring, _xai_oauth_inference_base_url, _xai_oauth_request_device_code, _xai_tokens_from_payload


import logging
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
from urllib.parse import urlparse
from auth.providers.codex import _load_auth_store_maybe_locked, _refresh_payload_access_token
from auth.store import AUTH_LOCK_TIMEOUT_SECONDS
from auth.errors import AuthError
from auth.constants import DEFAULT_XAI_OAUTH_BASE_URL, DEVICE_CODE_GRANT_TYPE, XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS, XAI_OAUTH_CLIENT_ID, XAI_OAUTH_DEVICE_CODE_URL, XAI_OAUTH_DISCOVERY_URL, XAI_OAUTH_SCOPE, _FORM_JSON_HEADERS, _xai_err, httpx
from utils import env_float

if TYPE_CHECKING:  # annotation-only; the runtime import would be a cycle
    from hermes_cli.auth import ProviderConfig


def _login_xai_oauth(args, pconfig: ProviderConfig, *, force_new_login: bool = False) -> None:
    from hermes_cli.auth_device_flow import _is_remote_session, _offer_existing_oauth_credentials, _print_login_success
    from hermes_cli.auth import _update_config_for_provider
    from hermes_cli.auth_xai import _xai_oauth_device_code_login
    from auth.providers.xai import resolve_xai_oauth_runtime_credentials
    from auth.sources import unsuppress_credential_source
    del pconfig

    if not force_new_login and _offer_existing_oauth_credentials(
        "xai-oauth",
        resolve=resolve_xai_oauth_runtime_credentials,
        is_expiring=_xai_access_token_is_expiring,
        display_name="xAI OAuth",
        default_base_url=DEFAULT_XAI_OAUTH_BASE_URL,
    ):
        return

    print()
    print("Signing in to xAI Grok OAuth (SuperGrok / Premium+)...")
    print("(Hermes creates its own local OAuth session)")
    print()

    timeout_seconds = float(getattr(args, "timeout", None) or 20.0)
    open_browser = not getattr(args, "no_browser", False)
    if _is_remote_session():
        open_browser = False

    creds = _xai_oauth_device_code_login(timeout_seconds=timeout_seconds, open_browser=open_browser)
    _save_xai_oauth_tokens(
        creds["tokens"], discovery=creds.get("discovery"),
        redirect_uri=creds.get("redirect_uri", ""), last_refresh=creds.get("last_refresh"),
        auth_mode="oauth_device_code",
    )
    # Explicit re-login re-enables the credential: clear the ``device_code`` suppression marker left
    # by ``hermes auth remove xai-oauth``. Deliberately NOT inside _save_xai_oauth_tokens — the
    # refresh hot path shares that helper and must never mutate suppression state.
    unsuppress_credential_source("xai-oauth", "device_code")
    config_path = _update_config_for_provider("xai-oauth", creds.get("base_url", DEFAULT_XAI_OAUTH_BASE_URL))
    _print_login_success("xai-oauth", config_path, show_auth_state=True)


def _xai_oauth_device_code_login(*, timeout_seconds: float = 20.0, open_browser: bool = True) -> Dict[str, Any]:
    from hermes_cli.auth_device_flow import _can_open_graphical_browser, _is_remote_session, _print_device_code_instructions
    from auth.oauth import _utc_now_z
    from auth.providers.xai import _xai_oauth_discovery, _xai_oauth_poll_device_token
    discovery = _xai_oauth_discovery(timeout_seconds)
    timeout = httpx.Timeout(max(20.0, timeout_seconds))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        device_data = _xai_oauth_request_device_code(client)
        interval = int(device_data["interval"])
        _print_device_code_instructions(
            str(device_data.get("verification_uri_complete") or device_data["verification_uri"]),
            str(device_data["user_code"]),
            open_browser=open_browser and not _is_remote_session() and _can_open_graphical_browser(),
            swallow_open_errors=True,
        )
        print(f"Waiting for approval (polling every {max(1, interval)}s)...")
        payload = _xai_oauth_poll_device_token(
            client, token_endpoint=discovery["token_endpoint"],
            device_code=str(device_data["device_code"]), expires_in=int(device_data["expires_in"]),
            poll_interval=interval,
        )

    access_token, refresh_token = _token_pair(payload)
    if not access_token or not refresh_token:
        raise _xai_err("xAI device-code token response was missing required tokens.", "xai_device_token_invalid")
    return {
        "tokens": _xai_tokens_from_payload(payload, access_token, refresh_token),
        "discovery": discovery, "redirect_uri": "", "base_url": _xai_oauth_inference_base_url(),
        "last_refresh": _utc_now_z(), "source": "oauth-device-code",
    }
