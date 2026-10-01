"""MiniMax OAuth (user-code grant) login, refresh and runtime credentials.

Split out of ``hermes_cli/auth.py``; origin helpers are imported lazily per function so
``hermes_cli.auth.<helper>`` patches still intercept and no cycle forms.
"""

from __future__ import annotations
from auth.providers.minimax import _minimax_expiry_fields, _minimax_poll_token


import logging
import base64
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
from auth.errors import AuthError
from auth.constants import MINIMAX_OAUTH_GRANT_TYPE, MINIMAX_OAUTH_REFRESH_SKEW_SECONDS, MINIMAX_OAUTH_SCOPE, _FORM_JSON_HEADERS, _minimax_err, httpx

if TYPE_CHECKING:  # annotation-only; the runtime import would be a cycle
    from hermes_cli.auth import ProviderConfig


def _minimax_oauth_login(*, region: str = "global", open_browser: bool = True, timeout_seconds: float = 15.0) -> Dict[str, Any]:
    """Run MiniMax OAuth flow, persist tokens, return auth state dict."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.auth_device_flow import _can_open_graphical_browser, _is_remote_session, _print_device_code_instructions
    from auth.providers.minimax import _minimax_pkce_pair, _minimax_request_user_code, _minimax_save_auth_state
    pconfig = PROVIDER_REGISTRY["minimax-oauth"]
    if region == "cn":
        portal_base_url = pconfig.extra["cn_portal_base_url"]
        inference_base_url = pconfig.extra["cn_inference_base_url"]
    else:
        portal_base_url = pconfig.portal_base_url
        inference_base_url = pconfig.inference_base_url

    verifier, challenge, state = _minimax_pkce_pair()

    if _is_remote_session():
        open_browser = False

    print(f"Starting Hermes login via MiniMax ({region}) OAuth...")
    print(f"Portal: {portal_base_url}")

    with httpx.Client(timeout=httpx.Timeout(timeout_seconds), headers={"Accept": "application/json"},
                      follow_redirects=True) as client:
        code_data = _minimax_request_user_code(
            client, portal_base_url=portal_base_url, client_id=pconfig.client_id, code_challenge=challenge, state=state,
        )
        _print_device_code_instructions(
            str(code_data["verification_uri"]), str(code_data["user_code"]),
            open_browser=open_browser and _can_open_graphical_browser(),
        )

        interval_raw = code_data.get("interval")
        print("Waiting for approval...")

        token_data = _minimax_poll_token(
            client, portal_base_url=portal_base_url, client_id=pconfig.client_id,
            user_code=str(code_data["user_code"]), code_verifier=verifier,
            expired_in=int(code_data["expired_in"]),
            interval_ms=int(interval_raw) if interval_raw is not None else None,
        )

    auth_state = {
        "provider": "minimax-oauth",
        "region": region,
        "portal_base_url": portal_base_url,
        "inference_base_url": inference_base_url,
        "client_id": pconfig.client_id,
        "scope": MINIMAX_OAUTH_SCOPE,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "resource_url": token_data.get("resource_url"),
        **_minimax_expiry_fields(token_data["expired_in"]),
    }

    _minimax_save_auth_state(auth_state)
    print("\u2713 MiniMax OAuth login successful.")
    if msg := token_data.get("notification_message"):
        print(f"Note from MiniMax: {msg}")
    return auth_state


def _login_minimax_oauth(args, pconfig: ProviderConfig) -> None:
    """CLI entry for MiniMax OAuth login."""
    from hermes_cli.auth_error_copy import format_auth_error
    try:
        _minimax_oauth_login(
            region=getattr(args, "region", None) or "global",
            open_browser=not getattr(args, "no_browser", False),
            timeout_seconds=getattr(args, "timeout", None) or 15.0,
        )
    except AuthError as exc:
        print(format_auth_error(exc))
        raise SystemExit(1)
