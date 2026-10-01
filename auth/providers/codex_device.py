"""codex device protocol/lifecycle responsibilities."""

from __future__ import annotations
from auth.providers.codex_http import (
    _codex_login_post,
    _codex_http_client,
    _is_transient_transport_error,
    _ssl_interop_hint,
)
from auth.providers.codex import _parse_retry_after_seconds

import time
from typing import Any, Dict
from auth.errors import AuthError
from auth.constants import (
    CODEX_OAUTH_TOKEN_URL,
    CODEX_RATE_LIMITED_CODE,
    _codex_err,
    httpx,
)


def _codex_login_rate_limited_error(
    response: "httpx.Response", *, during: str = ""
) -> AuthError:
    """AuthError for a 429 from OpenAI's device-auth endpoints (throttle, not credential fault)."""
    # Upstream rate-limit / usage-quota exhaustion on the token endpoint. The stored refresh token is still
    # valid here — re-authenticating cannot lift a quota cap. Classify distinctly from auth failures so
    # callers surface a "retry later" notice instead of a misleading "run hermes auth" prompt (see issue
    # #32790).
    retry_after = _parse_retry_after_seconds(getattr(response, "headers", None))
    wait_hint = (
        f" Try again in about {retry_after}s."
        if retry_after is not None
        else " Wait a minute and run the login again."
    )
    return _codex_err(
        f"OpenAI is rate-limiting Codex login requests (HTTP 429){during}. "
        f"This is a temporary throttle on OpenAI's side, not a credential problem.{wait_hint}",
        CODEX_RATE_LIMITED_CODE,
    )


def _codex_request_device_code(
    issuer: str, client_id: str, *, on_progress=None
) -> Dict[str, Any]:
    """Step 1 of the Codex device flow: request a user code, retrying capped on HTTP 429.

    OpenAI rate-limits this request when login is attempted too often from one IP/account — retry
    with capped backoff (honoring ``Retry-After``) before surfacing an actionable message.
    """
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        resp = _codex_login_post(
            f"{issuer}/api/accounts/deviceauth/usercode",
            json={"client_id": client_id},
            headers={"Content-Type": "application/json"},
            failure=("Failed to request device code", "device_code_request_failed"),
        )
        if resp.status_code != 429:
            break
        if attempt < max_attempts:
            # Exponential backoff (2s, 4s, 8s) capped, preferring the server's Retry-After.
            retry_after = _parse_retry_after_seconds(getattr(resp, "headers", None))
            delay = max(
                1, min(int(retry_after if retry_after is not None else 2**attempt), 60)
            )
            if on_progress is not None:
                on_progress(
                    f"OpenAI is rate-limiting login requests (429); retrying in {delay}s..."
                )
            time.sleep(delay)
    if resp.status_code == 429:
        raise _codex_login_rate_limited_error(resp)
    if resp.status_code != 200:
        raise _codex_err(
            f"Device code request returned status {resp.status_code}.",
            "device_code_request_error",
        )
    device_data = resp.json()
    device_data["interval"] = max(3, int(device_data.get("interval", "5")))
    if not device_data.get("user_code", "") or not device_data.get(
        "device_auth_id", ""
    ):
        raise _codex_err(
            "Device code response missing required fields.", "device_code_incomplete"
        )
    return device_data


def _codex_poll_authorization_code(
    issuer: str,
    *,
    device_auth_id: str,
    user_code: str,
    poll_interval: int,
    on_progress=None,
) -> Dict[str, Any]:
    """Step 3 of the Codex device flow: poll until sign-in completes (403/404 = still pending)."""
    max_wait = 15 * 60  # 15 minutes
    max_consecutive_blips = (
        6  # survives transient drops, still fails fast on a dead network
    )
    start = time.monotonic()
    code_resp = None
    try:
        with _codex_http_client(timeout=httpx.Timeout(15.0)) as client:
            consecutive_blips = 0
            while time.monotonic() - start < max_wait:
                time.sleep(poll_interval)
                try:
                    poll_resp = client.post(
                        f"{issuer}/api/accounts/deviceauth/token",
                        json={"device_auth_id": device_auth_id, "user_code": user_code},
                        headers={"Content-Type": "application/json"},
                    )
                except Exception as exc:
                    if not _is_transient_transport_error(exc):
                        raise _codex_err(
                            f"Device auth polling request failed: {exc}{_ssl_interop_hint(exc)}",
                            "device_code_poll_error",
                        ) from exc
                    consecutive_blips += 1
                    if consecutive_blips >= max_consecutive_blips:
                        raise _codex_err(
                            f"Device auth polling request failed after {consecutive_blips} consecutive"
                            f" transport errors: {exc}{_ssl_interop_hint(exc)}",
                            "device_code_poll_error",
                        ) from exc
                    if on_progress is not None:
                        on_progress(
                            "Transient network error while waiting for sign-in; retrying..."
                        )
                    continue
                consecutive_blips = 0
                if poll_resp.status_code == 200:
                    code_resp = poll_resp.json()
                    break
                if poll_resp.status_code not in {
                    403,
                    404,
                }:  # 403/404 = user hasn't finished yet
                    raise _codex_err(
                        f"Device auth polling returned status {poll_resp.status_code}.",
                        "device_code_poll_error",
                    )
    except KeyboardInterrupt:
        raise
    if code_resp is None:
        raise _codex_err("Login timed out after 15 minutes.", "device_code_timeout")
    return code_resp


def _codex_exchange_authorization_code(
    issuer: str, client_id: str, code_resp: Dict[str, Any]
) -> Dict[str, Any]:
    """Step 4 of the Codex device flow: swap the authorization code for tokens."""
    authorization_code = code_resp.get("authorization_code", "")
    code_verifier = code_resp.get("code_verifier", "")
    if not authorization_code or not code_verifier:
        raise _codex_err(
            "Device auth response missing authorization_code or code_verifier.",
            "device_code_incomplete_exchange",
        )
    token_resp = _codex_login_post(
        CODEX_OAUTH_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": f"{issuer}/deviceauth/callback",
            "client_id": client_id,
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
