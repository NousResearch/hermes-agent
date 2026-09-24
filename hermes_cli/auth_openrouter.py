"""OpenRouter OAuth PKCE login (``hermes auth add openrouter --type oauth``).

Contract: https://openrouter.ai/docs/guides/overview/auth/oauth. The browser is sent to
``/auth?callback_url=...&code_challenge=...&code_challenge_method=S256``; the redirect carries
``?code=``; ``POST /api/v1/auth/keys`` swaps ``{code, code_verifier, code_challenge_method}`` for
``{"key": "sk-or-v1-..."}`` — a plain user-controlled API key, no refresh token. OpenRouter echoes no
``state`` parameter, so the CSRF nonce rides in the loopback callback PATH: a redirect to any other
path is a 404 and never reaches the exchange.
"""

from __future__ import annotations

import secrets
import webbrowser
from typing import Any, Dict
from urllib.parse import urlencode

from hermes_cli.auth_constants import (
    OPENROUTER_AUTH_KEYS_URL, OPENROUTER_AUTH_URL, OPENROUTER_OAUTH_DOCS_URL, _openrouter_err, httpx)
from hermes_cli.auth_device_flow import (
    _bind_loopback_callback_server, _can_open_graphical_browser, _is_remote_session,
    _make_loopback_callback_handler, _pkce_code_challenge, _pkce_code_verifier, _serve_loopback_callback)

_ERROR_BODY_LIMIT = 2048


def _openrouter_exchange_code(code: str, code_verifier: str, *, timeout_seconds: float = 20.0) -> str:
    """Exchange the authorization code for an API key; the key never enters a log or error message."""
    try:
        response = httpx.post(
            OPENROUTER_AUTH_KEYS_URL, json={
                "code": code, "code_verifier": code_verifier, "code_challenge_method": "S256"},
            headers={"Content-Type": "application/json"}, timeout=timeout_seconds)
    except Exception as exc:
        raise _openrouter_err(f"OpenRouter code exchange failed: {exc}", "openrouter_token_exchange_failed") from exc

    if response.status_code == 403:
        raise _openrouter_err(
            "OpenRouter rejected the authorization code (invalid, already used, or older than 10 minutes). "
            "Run the login again.", "openrouter_token_exchange_denied", relogin=True)
    if response.status_code >= 400:
        detail = response.text.strip()[:_ERROR_BODY_LIMIT]
        raise _openrouter_err(
            f"OpenRouter code exchange failed (HTTP {response.status_code})." + (f" Response: {detail}" if detail else ""),
            "openrouter_token_exchange_failed")
    try:
        payload = response.json()
    except ValueError as exc:
        raise _openrouter_err(
            "OpenRouter code exchange returned a non-JSON body.", "openrouter_token_exchange_invalid") from exc
    key = str(payload.get("key") or "").strip() if isinstance(payload, dict) else ""
    if not key:
        raise _openrouter_err(
            "OpenRouter code exchange response did not include a 'key'.", "openrouter_token_exchange_invalid")
    return key


def _openrouter_headless_code(auth_url: str) -> str:
    """Remote/SSH: OpenRouter shows the code on screen when ``callback_url`` is omitted; the user pastes it."""
    from hermes_cli.secret_prompt import masked_secret_prompt
    print(
        "Remote session detected — using OpenRouter's headless flow.\n"
        f"Open this URL in a browser on any machine, authorize, then paste the code shown:\n  {auth_url}\n")
    code = masked_secret_prompt("Authorization code: ").strip()
    if not code:
        raise _openrouter_err("No authorization code entered.", "openrouter_auth_no_code")
    return code


def _openrouter_loopback_code(auth_url_params: Dict[str, str], *, open_browser: bool, timeout_seconds: float) -> str:
    nonce = secrets.token_urlsafe(16)
    path = f"/callback/{nonce}"
    handler_cls, result = _make_loopback_callback_handler(path, display_name="OpenRouter")
    server = _bind_loopback_callback_server(
        "127.0.0.1", 0, handler_cls, err=_openrouter_err, bind_failed_code="openrouter_callback_bind_failed")
    redirect_uri = f"http://127.0.0.1:{server.server_address[1]}{path}"
    auth_url = f"{OPENROUTER_AUTH_URL}?{urlencode({'callback_url': redirect_uri, **auth_url_params})}"

    print(f"Open this URL to authorize Hermes with OpenRouter:\n  {auth_url}\n\nDocs: {OPENROUTER_OAUTH_DOCS_URL}")
    if open_browser and _can_open_graphical_browser():
        try:
            opened = webbrowser.open(auth_url)
        except Exception:
            opened = False
        print("Browser opened for OpenRouter authorization." if opened
              else "Could not open the browser automatically; use the URL above.")
    print("Waiting for the OpenRouter callback...")
    callback = _serve_loopback_callback(
        server, result, timeout_seconds=timeout_seconds, err=_openrouter_err,
        timeout_code="openrouter_callback_timeout")
    if callback.get("error"):
        raise _openrouter_err(
            f"OpenRouter authorization failed: {callback.get('error_description') or callback['error']}",
            "openrouter_auth_denied")
    code = str(callback.get("code") or "").strip()
    if not code:
        raise _openrouter_err("OpenRouter callback did not carry an authorization code.", "openrouter_auth_no_code")
    return code


def _openrouter_pkce_login(*, open_browser: bool = True, timeout_seconds: float = 300.0) -> Dict[str, Any]:
    """Run the PKCE flow and return ``{"api_key": ...}`` for the credential-pool add path."""
    code_verifier = _pkce_code_verifier()
    params = {"code_challenge": _pkce_code_challenge(code_verifier), "code_challenge_method": "S256"}
    if _is_remote_session():
        code = _openrouter_headless_code(f"{OPENROUTER_AUTH_URL}?{urlencode({**params, 'key_label': 'hermes-agent'})}")
    else:
        code = _openrouter_loopback_code(params, open_browser=open_browser, timeout_seconds=timeout_seconds)
    print("Exchanging the authorization code for an OpenRouter API key...")
    return {"api_key": _openrouter_exchange_code(code, code_verifier)}
