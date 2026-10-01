"""OpenAI Codex OAuth: token store, refresh, quota probe, device-code login.

Tokens live in ~/.hermes/auth.json, NOT ~/.codex/: Hermes keeps its own Codex OAuth session
separate from the Codex CLI / VS Code extension so one app's refresh-token rotation cannot
invalidate the other's session.

Split out of ``hermes_cli/auth.py``; origin helpers are imported lazily inside each function
so ``hermes_cli.auth.<name>`` patches still intercept (and no import cycle).
"""

from __future__ import annotations
from hermes_cli.version_info import get_version_info
CODEX_OAUTH_USER_AGENT = f"hermes-cli/{get_version_info().base_version}"

from auth.providers.codex import _codex_base_url, _parse_retry_after_seconds, _stripped, logger
from auth.providers.codex_http import _codex_http_client, _codex_login_post, _is_transient_transport_error, _ssl_interop_hint


import logging
import hashlib
import json
import os
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple
from auth.token_validation import _decode_jwt_claims
from auth.errors import AuthError
from auth.constants import CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS, CODEX_OAUTH_CLIENT_ID, CODEX_OAUTH_TOKEN_URL, CODEX_RATE_LIMITED_CODE, DEFAULT_CODEX_BASE_URL, _codex_err, httpx
from auth.store import AUTH_LOCK_TIMEOUT_SECONDS
from utils import env_float

if TYPE_CHECKING:  # annotation-only; the runtime import would be a cycle
    from hermes_cli.auth import ProviderConfig

# Log-record parity with the origin module (caplog tests pin "hermes_cli.auth").


# ``{relogin}`` is filled at raise time with the profile-aware sign-in command: a bare
# ``hermes auth`` from a named profile re-signs the ROOT store (93889b770da, #114012).


def _codex_pool_route_base_url(entry_base_url: Optional[str] = "") -> str:
    """Base URL the chat route sends a pooled Codex credential to — the same rule
    ``runtime_provider._pool_entry_mode_and_url`` applies (``HERMES_CODEX_BASE_URL`` > ``model.base_url``
    while the row still carries the canonical URL > the row's own URL). A pooled gateway key belongs to
    that host only; composing it with the ambient default sends it to chatgpt.com (#121486)."""
    base = _stripped(entry_base_url).rstrip("/")
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.runtime_provider import _pool_entry_mode_and_url
        model_cfg = load_config_readonly().get("model")
        return _pool_entry_mode_and_url(
            "openai-codex", None, model_cfg if isinstance(model_cfg, dict) else {}, "", base)[1]
    except Exception:
        logger.debug("Codex pool route base resolution failed", exc_info=True)
        # Profile-scoped override only (never the raw process env: a multiplexed sibling's gateway).
        with suppress(Exception):
            from agent.secret_scope import get_secret_str
            base = _stripped(get_secret_str("HERMES_CODEX_BASE_URL", "")).rstrip("/") or base
        return base or DEFAULT_CODEX_BASE_URL


  # real OAuth/device-auth payloads are a few hundred bytes


# Throttle for the live Codex quota probe. It runs on the hot credential-selection path while the
# pool is exhausted, so without a floor a busy gateway would hammer the usage endpoint per call.
  # 5 minutes


def _login_openai_codex(args, pconfig: ProviderConfig, *, force_new_login: bool = False) -> None:
    """OpenAI Codex login: device code by default, browser PKCE when opted in (``--browser`` /
    ``auth.codex_login_flow``). Tokens stored in ~/.hermes/auth.json."""
    from auth.providers.codex import _codex_access_token_is_expiring, _import_codex_cli_tokens, _save_codex_tokens, resolve_codex_runtime_credentials
    from hermes_cli.auth_device_flow import _offer_existing_oauth_credentials, _print_login_success, _prompt_yes_no
    from hermes_cli.auth import _update_config_for_provider
    from hermes_cli.auth_codex_browser import codex_oauth_login
    del pconfig  # kept for parity with other provider login helpers
    if not force_new_login:
        if _offer_existing_oauth_credentials(
            "openai-codex", resolve=resolve_codex_runtime_credentials,
            is_expiring=_codex_access_token_is_expiring, display_name="Codex",
            default_base_url=DEFAULT_CODEX_BASE_URL,
            expired_notice="Existing Codex credentials are expired. Starting fresh login..."):
            return
        cli_tokens = _import_codex_cli_tokens()
        if cli_tokens:
            print("Found existing Codex CLI credentials at ~/.codex/auth.json")
            print("Hermes will create its own session to avoid conflicts with Codex CLI / VS Code.")
            if _prompt_yes_no(
                "Import these credentials? (a separate login is recommended) [y/N]: ", default="n"):
                _save_codex_tokens(cli_tokens)
                config_path = _update_config_for_provider("openai-codex", _codex_base_url())
                print()
                print("Credentials imported. Note: if Codex CLI refreshes its token,")
                print("Hermes will keep working independently with its own session.")
                print(f"  Config updated: {config_path} (model.provider=openai-codex)")
                return

    # Run a fresh OAuth flow — Hermes gets its own session (device code unless the user opted in
    # to the browser flow).
    print()
    creds = codex_oauth_login(args)
    _save_codex_tokens(creds["tokens"], creds.get("last_refresh"))
    config_path = _update_config_for_provider(
        "openai-codex", creds.get("base_url", DEFAULT_CODEX_BASE_URL))
    _print_login_success("openai-codex", config_path, show_auth_state=True)


def _codex_device_code_login() -> Dict[str, Any]:
    """Run the OpenAI device code login flow and return credentials dict."""
    from auth.providers.codex_device import _codex_exchange_authorization_code, _codex_poll_authorization_code, _codex_request_device_code
    from auth.oauth import _utc_now_z
    issuer, client_id = "https://auth.openai.com", CODEX_OAUTH_CLIENT_ID
    device_data = _codex_request_device_code(issuer, client_id, on_progress=print)
    user_code = device_data["user_code"]

    # Step 2: Show user the code
    print("To continue, follow these steps:\n")
    print("  1. Open this URL in your browser:")
    print(f"     \033[94m{issuer}/codex/device\033[0m\n")
    print("  2. Enter this code:")
    print(f"     \033[94m{user_code}\033[0m\n")
    print("Waiting for sign-in... (press Ctrl+C to cancel)")
    try:
        code_resp = _codex_poll_authorization_code(
            issuer, device_auth_id=device_data["device_auth_id"], user_code=user_code,
            poll_interval=device_data["interval"], on_progress=print)
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)
    tokens = _codex_exchange_authorization_code(issuer, client_id, code_resp)
    # Return tokens for the caller to persist (never writes to ~/.codex/)
    return {
        "tokens": {
            "access_token": tokens.get("access_token", ""),
            "refresh_token": tokens.get("refresh_token", "")},
        "base_url": _codex_base_url(), "last_refresh": _utc_now_z(), "auth_mode": "chatgpt",
        "source": "device-code"}
