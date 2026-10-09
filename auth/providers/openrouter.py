"""OpenRouter PKCE token exchange."""

from __future__ import annotations


from auth.constants import OPENROUTER_AUTH_KEYS_URL, _openrouter_err, httpx


_ERROR_BODY_LIMIT = 2048


def _openrouter_exchange_code(
    code: str, code_verifier: str, *, timeout_seconds: float = 20.0
) -> str:
    """Exchange the authorization code for an API key; the key never enters a log or error message."""
    try:
        response = httpx.post(
            OPENROUTER_AUTH_KEYS_URL,
            json={
                "code": code,
                "code_verifier": code_verifier,
                "code_challenge_method": "S256",
            },
            headers={"Content-Type": "application/json"},
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise _openrouter_err(
            f"OpenRouter code exchange failed: {exc}",
            "openrouter_token_exchange_failed",
        ) from exc

    if response.status_code == 403:
        raise _openrouter_err(
            "OpenRouter rejected the authorization code (invalid, already used, or older than 10 minutes). "
            "Run the login again.",
            "openrouter_token_exchange_denied",
            relogin=True,
        )
    if response.status_code >= 400:
        detail = response.text.strip()[:_ERROR_BODY_LIMIT]
        raise _openrouter_err(
            f"OpenRouter code exchange failed (HTTP {response.status_code})."
            + (f" Response: {detail}" if detail else ""),
            "openrouter_token_exchange_failed",
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise _openrouter_err(
            "OpenRouter code exchange returned a non-JSON body.",
            "openrouter_token_exchange_invalid",
        ) from exc
    key = str(payload.get("key") or "").strip() if isinstance(payload, dict) else ""
    if not key:
        raise _openrouter_err(
            "OpenRouter code exchange response did not include a 'key'.",
            "openrouter_token_exchange_invalid",
        )
    return key
