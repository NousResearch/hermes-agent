"""Helpers for using existing Codex CLI credentials from ~/.codex."""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CODEX_HOME = Path.home() / ".codex"
CODEX_AUTH_FILE = CODEX_HOME / "auth.json"

CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"


def _read_codex_auth() -> Dict[str, Any]:
    if not CODEX_AUTH_FILE.exists():
        return {}
    try:
        raw = json.loads(CODEX_AUTH_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _write_codex_auth(data: Dict[str, Any]) -> None:
    try:
        CODEX_AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        CODEX_AUTH_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        CODEX_AUTH_FILE.chmod(0o600)
    except Exception as e:
        logger.debug("Could not persist refreshed Codex auth: %s", e)


def get_codex_auth_mode() -> Optional[str]:
    """Return configured auth mode from ~/.codex/auth.json."""
    mode = _read_codex_auth().get("auth_mode")
    if mode in ("api_key", "chatgpt"):
        return mode
    return None


def _jwt_claims(token: str) -> Dict[str, Any]:
    """Decode the payload section of a JWT without signature verification."""
    if not token or token.count(".") != 2:
        return {}
    payload = token.split(".")[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def _extract_account_id(claims: Dict[str, Any]) -> Optional[str]:
    """Extract ChatGPT account ID from JWT claims (multiple known locations)."""
    # Nested under OpenAI auth namespace
    nested = claims.get("https://api.openai.com/auth", {})
    if nested.get("chatgpt_account_id"):
        return nested["chatgpt_account_id"]
    # Direct claim
    if claims.get("chatgpt_account_id"):
        return claims["chatgpt_account_id"]
    # First org ID
    orgs = claims.get("organizations", [])
    if orgs and isinstance(orgs[0], dict) and orgs[0].get("id"):
        return orgs[0]["id"]
    return None


def _token_expired(token: str, skew_seconds: int = 90) -> bool:
    exp = _jwt_claims(token).get("exp")
    if not isinstance(exp, (int, float)):
        return False
    return exp <= (datetime.now(tz=timezone.utc).timestamp() + skew_seconds)


def get_codex_openai_api_key() -> Optional[str]:
    """Return OPENAI_API_KEY stored by Codex when auth_mode=api_key."""
    if get_codex_auth_mode() != "api_key":
        return None
    return _read_codex_auth().get("OPENAI_API_KEY") or None


def refresh_codex_chatgpt_auth(refresh_token: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Refresh ChatGPT access token via stored Codex refresh token."""
    auth = _read_codex_auth()
    tokens = auth.get("tokens")
    if not isinstance(tokens, dict):
        return None

    refresh = refresh_token or tokens.get("refresh_token")
    if not refresh:
        return None

    try:
        import requests
        resp = requests.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.debug("Failed to refresh Codex access token: %s", e)
        return None

    access_token = payload.get("access_token")
    if not access_token:
        return None

    refresh_token_new = payload.get("refresh_token") or refresh
    id_token = payload.get("id_token") or tokens.get("id_token", "")

    account_id = (
        _extract_account_id(_jwt_claims(id_token))
        or _extract_account_id(_jwt_claims(access_token))
        or tokens.get("account_id")
    )
    if not account_id:
        return None

    auth["tokens"] = {
        "id_token": id_token,
        "access_token": access_token,
        "refresh_token": refresh_token_new,
        "account_id": account_id,
    }
    auth["auth_mode"] = "chatgpt"
    auth["last_refresh"] = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    _write_codex_auth(auth)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token_new,
        "account_id": account_id,
    }


def get_codex_chatgpt_auth(refresh_if_expiring: bool = True) -> Optional[Dict[str, str]]:
    """Return ChatGPT token credentials from Codex auth (refreshes if needed)."""
    auth = _read_codex_auth()
    if auth.get("auth_mode") != "chatgpt":
        return None

    tokens = auth.get("tokens")
    if not isinstance(tokens, dict):
        return None

    access = tokens.get("access_token")
    refresh = tokens.get("refresh_token")
    account = tokens.get("account_id")
    if not (access and refresh and account):
        return None

    if refresh_if_expiring and _token_expired(access):
        refreshed = refresh_codex_chatgpt_auth(refresh_token=refresh)
        if refreshed:
            return refreshed

    return {"access_token": access, "refresh_token": refresh, "account_id": account}


def has_codex_credentials() -> bool:
    """Return True when ~/.codex contains usable API-key or ChatGPT auth."""
    mode = get_codex_auth_mode()
    if mode == "api_key":
        return get_codex_openai_api_key() is not None
    if mode == "chatgpt":
        return get_codex_chatgpt_auth(refresh_if_expiring=False) is not None
    return False
