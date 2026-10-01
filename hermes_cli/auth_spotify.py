"""Spotify OAuth (loopback PKCE) login, refresh and runtime credentials.

Re-exported from ``hermes_cli/auth.py`` (patch targets unchanged); origin helpers are imported
lazily per function so ``hermes_cli.auth.<helper>`` patches still intercept and no cycle forms.
"""

from __future__ import annotations
from hermes_cli.config_credentials import credential_pool_environment as _phase6_auth_environment

from auth.providers.spotify import _spotify_accounts_base_url, _spotify_api_base_url, _spotify_build_authorize_url, _spotify_client_id, _spotify_redirect_uri, _spotify_scope_string, _spotify_token_payload_to_state, _spotify_token_post, _spotify_wait_for_callback


import logging
import uuid
import webbrowser
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode, urlparse
from auth.errors import AuthError
from auth.constants import DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL, DEFAULT_SPOTIFY_API_BASE_URL, DEFAULT_SPOTIFY_REDIRECT_URI, DEFAULT_SPOTIFY_SCOPE, SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS, SPOTIFY_DASHBOARD_URL, SPOTIFY_DOCS_URL, _spotify_err, httpx
from auth.oauth import _bind_loopback_callback_server, _make_loopback_callback_handler, _pkce_code_challenge, _pkce_code_verifier, _serve_loopback_callback


def _spotify_interactive_setup(redirect_uri_hint: str) -> str:
    """Walk the user through creating a Spotify developer app; persist the client_id to ~/.hermes/.env."""
    from hermes_cli.auth_device_flow import _is_remote_session
    from hermes_cli.config import save_env_value
    print(
        f"\n{'=' * 70}\nSpotify first-time setup\n{'=' * 70}\n\n"
        "Spotify requires every user to register their own lightweight\n"
        "developer app. This takes about two minutes and only has to be\n"
        "done once per machine.\n\n"
        f"Full guide: {SPOTIFY_DOCS_URL}\n\n"
        "Steps:\n"
        f"  1. Opening {SPOTIFY_DASHBOARD_URL} in your browser...\n"
        "  2. Click 'Create app' and fill in:\n"
        "       App name:     anything (e.g. hermes-agent)\n"
        "       Description:  anything\n"
        f"       Redirect URI: {redirect_uri_hint}\n"
        "       API/SDK:      Web API\n"
        "  3. Agree to the terms, click Save.\n"
        "  4. Open the app's Settings page and copy the Client ID.\n"
        "  5. Paste it below.\n"
    )

    if not _is_remote_session():
        try:
            webbrowser.open(SPOTIFY_DASHBOARD_URL)
        except Exception:
            pass

    from hermes_cli.cli_output import line_input
    try:
        raw = line_input("Spotify Client ID: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit("Spotify setup cancelled.")

    if not raw:
        print(f"\nNo Client ID entered. See {SPOTIFY_DOCS_URL} for the full guide.")
        raise SystemExit("Spotify setup cancelled: empty Client ID.")

    # Persist so later runs skip the wizard; only pin a NON-default redirect URI.
    save_env_value("HERMES_SPOTIFY_CLIENT_ID", raw)
    if redirect_uri_hint and redirect_uri_hint != DEFAULT_SPOTIFY_REDIRECT_URI:
        save_env_value("HERMES_SPOTIFY_REDIRECT_URI", redirect_uri_hint)

    print("\nSaved HERMES_SPOTIFY_CLIENT_ID to ~/.hermes/.env\n")
    return raw


def login_spotify_command(args) -> None:
    from hermes_cli.auth_device_flow import _can_open_graphical_browser, _is_remote_session, _print_loopback_ssh_hint
    from auth.provider_state import save_provider_auth_state, get_provider_auth_state
    existing_state = get_provider_auth_state("spotify") or {}

    # No client_id anywhere -> wizard instead of "HERMES_SPOTIFY_CLIENT_ID is required".
    try:
        client_id = _spotify_client_id(getattr(args, "client_id", None), existing_state, environment=_phase6_auth_environment())
    except AuthError as exc:
        if getattr(exc, "code", "") != "spotify_client_id_missing":
            raise
        client_id = _spotify_interactive_setup(
            redirect_uri_hint=getattr(args, "redirect_uri", None) or DEFAULT_SPOTIFY_REDIRECT_URI,
        )

    redirect_uri = _spotify_redirect_uri(getattr(args, "redirect_uri", None), existing_state, environment=_phase6_auth_environment())
    scope = _spotify_scope_string(getattr(args, "scope", None) or existing_state.get("scope"))
    accounts_base_url = _spotify_accounts_base_url(existing_state, environment=_phase6_auth_environment())
    api_base_url = _spotify_api_base_url(existing_state, environment=_phase6_auth_environment())
    open_browser = not getattr(args, "no_browser", False)

    code_verifier = _pkce_code_verifier()
    state_nonce = uuid.uuid4().hex
    authorize_url = _spotify_build_authorize_url(
        client_id=client_id, redirect_uri=redirect_uri, scope=scope, state=state_nonce,
        code_challenge=_pkce_code_challenge(code_verifier), accounts_base_url=accounts_base_url,
    )

    print(
        f"Starting Spotify PKCE login...\nClient ID: {client_id}\nRedirect URI: {redirect_uri}\n"
        "Make sure this redirect URI is allow-listed in your Spotify app settings.\n\n"
        f"Open this URL to authorize Hermes:\n{authorize_url}\n\nFull setup guide: {SPOTIFY_DOCS_URL}\n"
    )

    _print_loopback_ssh_hint(redirect_uri, docs_url=SPOTIFY_DOCS_URL)

    if open_browser and not _is_remote_session() and _can_open_graphical_browser():
        try:
            opened = webbrowser.open(authorize_url)
        except Exception:
            opened = False
        print(
            "Browser opened for Spotify authorization." if opened
            else "Could not open the browser automatically; use the URL above."
        )

    callback = _spotify_wait_for_callback(redirect_uri, timeout_seconds=float(getattr(args, "timeout", None) or 180.0))
    if callback.get("error"):
        raise SystemExit(f"Spotify authorization failed: {callback.get('error_description') or callback['error']}")
    if callback.get("state") != state_nonce:
        raise SystemExit("Spotify authorization failed: state mismatch.")

    token_payload = _spotify_token_post(
        accounts_base_url,
        {
            "client_id": client_id, "grant_type": "authorization_code",
            "code": str(callback.get("code") or ""), "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        },
        timeout_seconds=float(getattr(args, "timeout", None) or 20.0),
        what="token exchange", failed_code="spotify_token_exchange_failed",
        invalid_code="spotify_token_exchange_invalid",
        invalid_message="Spotify token response did not include an access_token.",
    )
    spotify_state = _spotify_token_payload_to_state(
        token_payload, client_id=client_id, redirect_uri=redirect_uri, requested_scope=scope,
        accounts_base_url=accounts_base_url, api_base_url=api_base_url,
    )

    saved_to = save_provider_auth_state("spotify", spotify_state, set_active=False)

    print(
        f"Spotify login successful!\n  Auth state: {saved_to}\n"
        f"  Provider state saved under providers.spotify\n  Docs: {SPOTIFY_DOCS_URL}"
    )
