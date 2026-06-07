from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import uuid
import webbrowser
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode

import httpx


CURSOR_LOGIN_URL = "https://cursor.com/loginDeepControl"
CURSOR_POLL_URL = "https://api2.cursor.sh/auth/poll"
CURSOR_REFRESH_URL = "https://api2.cursor.sh/auth/exchange_user_api_key"
CURSOR_TOKEN_EXPIRY_SKEW_SECONDS = 5 * 60

OAuthCredentials = Dict[str, Any]


def _b64url_no_padding(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate_cursor_auth_params() -> Dict[str, str]:
    verifier = _b64url_no_padding(secrets.token_bytes(32))
    challenge = _b64url_no_padding(hashlib.sha256(verifier.encode("utf-8")).digest())
    auth_uuid = str(uuid.uuid4())
    query = urlencode({
        "challenge": challenge,
        "uuid": auth_uuid,
        "mode": "login",
        "redirectTarget": "cli",
    })
    login_url = f"{CURSOR_LOGIN_URL}?{query}"
    return {
        "verifier": verifier,
        "challenge": challenge,
        "uuid": auth_uuid,
        "login_url": login_url,
    }


def _extract_cursor_tokens(payload: Any) -> Dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("Cursor OAuth response was not a JSON object.")

    access_token = str(
        payload.get("access_token")
        or payload.get("accessToken")
        or payload.get("token")
        or ""
    ).strip()
    refresh_token = str(
        payload.get("refresh_token")
        or payload.get("refreshToken")
        or access_token
        or ""
    ).strip()
    if not access_token:
        raise ValueError("Cursor OAuth response missing access token.")
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }


def poll_cursor_auth(auth_uuid: str, verifier: str) -> Dict[str, str]:
    delay_seconds = 1.0
    deadline = time.time() + 300.0
    last_error: Optional[Exception] = None

    while time.time() < deadline:
        try:
            response = httpx.get(
                CURSOR_POLL_URL,
                params={"uuid": auth_uuid, "verifier": verifier},
                timeout=20.0,
            )
        except Exception as exc:
            last_error = exc
        else:
            if response.status_code == 404:
                time.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 1.5, 5.0)
                continue
            if response.status_code >= 400:
                raise ValueError(
                    f"Cursor OAuth poll failed with HTTP {response.status_code}: {response.text.strip()}"
                )
            return _extract_cursor_tokens(response.json())

        time.sleep(delay_seconds)
        delay_seconds = min(delay_seconds * 1.5, 5.0)

    if last_error is not None:
        raise TimeoutError(f"Cursor OAuth poll timed out after transport errors: {last_error}")
    raise TimeoutError("Cursor OAuth poll timed out waiting for approval.")


def get_token_expiry(token: str) -> Optional[int]:
    if not isinstance(token, str) or "." not in token:
        return None
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))
        exp = payload.get("exp")
        if isinstance(exp, (int, float)):
            return int(exp)
    except Exception:
        return None
    return None


def is_cursor_token_expiring_soon(token: str) -> bool:
    exp = get_token_expiry(token)
    if exp is None:
        return True
    return exp <= int(time.time()) + CURSOR_TOKEN_EXPIRY_SKEW_SECONDS


def refresh_cursor_token(token: str) -> OAuthCredentials:
    response = httpx.post(
        CURSOR_REFRESH_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={},
        timeout=20.0,
    )
    if response.status_code >= 400:
        raise ValueError(
            f"Cursor token refresh failed with HTTP {response.status_code}: {response.text.strip()}"
        )
    tokens = _extract_cursor_tokens(response.json())
    return {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires_at": get_token_expiry(tokens["access_token"]),
    }


def login_cursor(
    on_auth_url: Callable[[str], None],
    on_poll_start: Optional[Callable[[], None]] = None,
) -> OAuthCredentials:
    params = generate_cursor_auth_params()
    on_auth_url(params["login_url"])
    try:
        webbrowser.open(params["login_url"])
    except Exception:
        pass
    if on_poll_start is not None:
        on_poll_start()
    tokens = poll_cursor_auth(params["uuid"], params["verifier"])
    return {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires_at": get_token_expiry(tokens["access_token"]),
    }
