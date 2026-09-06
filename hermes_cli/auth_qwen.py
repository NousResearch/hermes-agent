"""Qwen OAuth (qwen-cli token file) runtime credentials and status.

Split out of ``hermes_cli/auth.py``; origin helpers are imported lazily per function so
``hermes_cli.auth.<helper>`` patches still intercept and no cycle forms.
"""

from __future__ import annotations

import logging
import json
import os
import time
from pathlib import Path
from typing import Any, Dict
from hermes_cli.auth_constants import (
    AuthError, DEFAULT_QWEN_BASE_URL, QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS, QWEN_OAUTH_CLIENT_ID,
    QWEN_OAUTH_TOKEN_URL, _FORM_JSON_HEADERS, _qwen_err, httpx,
)

logger = logging.getLogger("hermes_cli.auth")

_RERUN = "Re-run 'qwen auth qwen-oauth'."


def _qwen_cli_auth_path() -> Path:
    return Path.home() / ".qwen" / "oauth_creds.json"


def _read_qwen_cli_tokens() -> Dict[str, Any]:
    from hermes_cli.auth import _qwen_cli_auth_path
    auth_path = _qwen_cli_auth_path()
    if not auth_path.exists():
        raise _qwen_err("Qwen CLI credentials not found. Run 'qwen auth qwen-oauth' first.", "qwen_auth_missing")
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _qwen_err(
            f"Failed to read Qwen CLI credentials from {auth_path}: {exc}", "qwen_auth_read_failed",
        ) from exc
    if not isinstance(data, dict):
        raise _qwen_err(f"Invalid Qwen CLI credentials in {auth_path}.", "qwen_auth_invalid")
    return data


def _save_qwen_cli_tokens(tokens: Dict[str, Any]) -> Path:
    from hermes_cli.auth import _qwen_cli_auth_path, _write_private_file_atomic
    auth_path = _qwen_cli_auth_path()
    _write_private_file_atomic(auth_path, json.dumps(tokens, indent=2, sort_keys=True) + "\n")
    return auth_path


def _qwen_access_token_is_expiring(expiry_date_ms: Any, skew_seconds: int = QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS) -> bool:
    try:
        expiry_ms = int(expiry_date_ms)
    except Exception:
        return True
    return (time.time() + max(0, int(skew_seconds))) * 1000 >= expiry_ms


def _refresh_qwen_cli_tokens(tokens: Dict[str, Any], timeout_seconds: float = 20.0) -> Dict[str, Any]:
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    if not refresh_token:
        raise _qwen_err(f"Qwen OAuth refresh token missing. {_RERUN}", "qwen_refresh_token_missing")

    try:
        response = httpx.post(
            QWEN_OAUTH_TOKEN_URL, headers=_FORM_JSON_HEADERS,
            data={"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": QWEN_OAUTH_CLIENT_ID},
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise _qwen_err(f"Qwen OAuth refresh failed: {exc}", "qwen_refresh_failed") from exc

    if response.status_code >= 400:
        body = response.text.strip()
        raise _qwen_err(
            f"Qwen OAuth refresh failed. {_RERUN}" + (f" Response: {body}" if body else ""), "qwen_refresh_failed",
        )

    try:
        payload = response.json()
    except Exception as exc:
        raise _qwen_err(f"Qwen OAuth refresh returned invalid JSON: {exc}", "qwen_refresh_invalid_json") from exc

    if not isinstance(payload, dict) or not str(payload.get("access_token", "") or "").strip():
        raise _qwen_err("Qwen OAuth refresh response missing access_token.", "qwen_refresh_invalid_response")

    try:
        expires_in_seconds = int(payload.get("expires_in"))
    except Exception:
        expires_in_seconds = 6 * 60 * 60

    refreshed = {
        "access_token": str(payload.get("access_token", "") or "").strip(),
        "refresh_token": str(payload.get("refresh_token", refresh_token) or refresh_token).strip(),
        "token_type": str(payload.get("token_type", tokens.get("token_type", "Bearer")) or "Bearer").strip() or "Bearer",
        "resource_url": str(payload.get("resource_url", tokens.get("resource_url", "portal.qwen.ai")) or "portal.qwen.ai").strip(),
        "expiry_date": int(time.time() * 1000) + max(1, expires_in_seconds) * 1000,
    }
    _save_qwen_cli_tokens(refreshed)
    return refreshed


def _mark_qwen_oauth_active(creds: Dict[str, Any]) -> None:
    """Set active_provider to qwen-oauth with a minimal state entry (tokens stay in the Qwen CLI file)."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store, _save_provider_state
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state: Dict[str, Any] = {"base_url": str(creds["base_url"])} if creds.get("base_url") else {}
        _save_provider_state(auth_store, "qwen-oauth", state)
        _save_auth_store(auth_store)


def resolve_qwen_runtime_credentials(
    *, force_refresh: bool = False, refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    from hermes_cli.auth import _qwen_cli_auth_path, _refresh_qwen_cli_tokens
    tokens = _read_qwen_cli_tokens()
    should_refresh = bool(force_refresh)
    if not should_refresh and refresh_if_expiring:
        should_refresh = _qwen_access_token_is_expiring(tokens.get("expiry_date"), refresh_skew_seconds)
    if should_refresh:
        tokens = _refresh_qwen_cli_tokens(tokens)
    access_token = str(tokens.get("access_token", "") or "").strip()
    if not access_token:
        raise _qwen_err(f"Qwen OAuth access token missing. {_RERUN}", "qwen_access_token_missing")

    return {
        "provider": "qwen-oauth",
        "base_url": os.getenv("HERMES_QWEN_BASE_URL", "").strip().rstrip("/") or DEFAULT_QWEN_BASE_URL,
        "api_key": access_token, "source": "qwen-cli", "expires_at_ms": tokens.get("expiry_date"),
        "auth_file": str(_qwen_cli_auth_path()),
    }


def get_qwen_auth_status() -> Dict[str, Any]:
    from hermes_cli.auth import _qwen_cli_auth_path, resolve_qwen_runtime_credentials
    auth_path = _qwen_cli_auth_path()
    try:
        # Refresh-validate: otherwise stale CLI tokens read as "logged in" and break `hermes model`.
        creds = resolve_qwen_runtime_credentials(refresh_if_expiring=True)
        return {
            "logged_in": True, "auth_file": str(auth_path), "source": creds.get("source"),
            "api_key": creds.get("api_key"), "expires_at_ms": creds.get("expires_at_ms"),
        }
    except AuthError as exc:
        return {"logged_in": False, "auth_file": str(auth_path), "error": str(exc)}
