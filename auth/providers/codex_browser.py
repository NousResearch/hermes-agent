"""codex browser protocol/lifecycle responsibilities."""

from __future__ import annotations
from auth.providers.codex_device import _codex_login_rate_limited_error

from typing import Any, Dict
from urllib.parse import urlencode
from auth.constants import CODEX_OAUTH_CLIENT_ID, CODEX_OAUTH_TOKEN_URL, _codex_err

CODEX_OAUTH_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"

CODEX_OAUTH_BROWSER_SCOPE = "openid profile email offline_access"


def _codex_browser_authorize_url(
    *, redirect_uri: str, state: str, code_challenge: str
) -> str:
    return f"{CODEX_OAUTH_AUTHORIZE_URL}?" + urlencode({
        "response_type": "code",
        "client_id": CODEX_OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": CODEX_OAUTH_BROWSER_SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "state": state,
    })


def _codex_browser_exchange_code(
    code: str, *, redirect_uri: str, code_verifier: str
) -> Dict[str, Any]:
    """Swap the authorization code for tokens at the token endpoint the device flow also uses."""
    from auth.providers.codex_http import _codex_login_post

    token_resp = _codex_login_post(
        CODEX_OAUTH_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": CODEX_OAUTH_CLIENT_ID,
            "code_verifier": code_verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        failure=("Token exchange failed", "token_exchange_failed"),
    )
    if token_resp.status_code == 429:
        raise _codex_login_rate_limited_error(
            token_resp, during=" during token exchange"
        )
    if token_resp.status_code != 200:
        raise _codex_err(
            f"Token exchange returned status {token_resp.status_code}.",
            "token_exchange_error",
        )
    tokens = token_resp.json()
    if not tokens.get("access_token", ""):
        raise _codex_err(
            "Token exchange did not return an access_token.",
            "token_exchange_no_access_token",
        )
    return tokens
