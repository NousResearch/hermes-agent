"""OpenAI Codex (ChatGPT-plan) provider auth.

F-C2 step 2 — first per-provider extraction from ``hermes_cli/auth/__init__.py``.

Hermes maintains its own Codex OAuth session separate from the Codex CLI
and the VS Code extension. Keeping a distinct token store in
``~/.hermes/auth.json`` (instead of sharing ``~/.codex/auth.json``)
prevents refresh-token rotation from one app silently invalidating
another app's session.

Public API (exposed via ``hermes_cli.auth`` re-exports):

    resolve_codex_runtime_credentials — main resolver, called by model
        routing and runtime_provider.py to mint a usable access token.
    get_codex_auth_status — status-panel snapshot for `hermes status`
        and doctor.

Internal helpers (also re-exported so tests that monkey-patch
``hermes_cli.auth.codex._refresh_codex_auth_tokens`` keep working):

    _codex_access_token_is_expiring
    _read_codex_tokens
    _save_codex_tokens
    refresh_codex_oauth_pure
    _refresh_codex_auth_tokens
    _import_codex_cli_tokens

Shared primitives (``AuthError``, the auth-store lock, load/save
helpers, ``_auth_file_path``, ``_decode_jwt_claims``) stay in the
package root (``hermes_cli.auth.__init__``) and are imported here.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from hermes_cli.auth import (
    AUTH_LOCK_TIMEOUT_SECONDS,
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    DEFAULT_CODEX_BASE_URL,
    AuthError,
    ProviderConfig,
    _auth_file_path,
    _auth_store_lock,
    _decode_jwt_claims,
    _load_auth_store,
    _load_provider_state,
    _save_auth_store,
    _save_provider_state,
    _update_config_for_provider,
)

logger = logging.getLogger(__name__)

__all__ = [
    "_codex_access_token_is_expiring",
    "_read_codex_tokens",
    "_save_codex_tokens",
    "refresh_codex_oauth_pure",
    "_refresh_codex_auth_tokens",
    "_import_codex_cli_tokens",
    "resolve_codex_runtime_credentials",
    "get_codex_auth_status",
    "_login_openai_codex",
    "_codex_device_code_login",
]


def _codex_access_token_is_expiring(access_token: Any, skew_seconds: int) -> bool:
    claims = _decode_jwt_claims(access_token)
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        return False
    return float(exp) <= (time.time() + max(0, int(skew_seconds)))


def _read_codex_tokens(*, _lock: bool = True) -> Dict[str, Any]:
    """Read Codex OAuth tokens from Hermes auth store (~/.hermes/auth.json).

    Returns dict with 'tokens' (access_token, refresh_token) and 'last_refresh'.
    Raises AuthError if no Codex tokens are stored.
    """
    if _lock:
        with _auth_store_lock():
            auth_store = _load_auth_store()
    else:
        auth_store = _load_auth_store()
    state = _load_provider_state(auth_store, "openai-codex")
    if not state:
        raise AuthError(
            "No Codex credentials stored. Run `hermes login` to authenticate.",
            provider="openai-codex",
            code="codex_auth_missing",
            relogin_required=True,
        )
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise AuthError(
            "Codex auth state is missing tokens. Run `hermes login` to re-authenticate.",
            provider="openai-codex",
            code="codex_auth_invalid_shape",
            relogin_required=True,
        )
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise AuthError(
            "Codex auth is missing access_token. Run `hermes login` to re-authenticate.",
            provider="openai-codex",
            code="codex_auth_missing_access_token",
            relogin_required=True,
        )
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError(
            "Codex auth is missing refresh_token. Run `hermes login` to re-authenticate.",
            provider="openai-codex",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )
    return {
        "tokens": tokens,
        "last_refresh": state.get("last_refresh"),
    }


def _save_codex_tokens(tokens: Dict[str, str], last_refresh: Optional[str] = None) -> None:
    """Save Codex OAuth tokens to Hermes auth store (~/.hermes/auth.json)."""
    if last_refresh is None:
        last_refresh = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "openai-codex") or {}
        state["tokens"] = tokens
        state["last_refresh"] = last_refresh
        state["auth_mode"] = "chatgpt"
        _save_provider_state(auth_store, "openai-codex", state)
        _save_auth_store(auth_store)


def refresh_codex_oauth_pure(
    access_token: str,
    refresh_token: str,
    *,
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    """Refresh Codex OAuth tokens without mutating Hermes auth state."""
    del access_token  # Access token is only used by callers to decide whether to refresh.
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError(
            "Codex auth is missing refresh_token. Run `hermes login` to re-authenticate.",
            provider="openai-codex",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )

    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )

    if response.status_code != 200:
        code = "codex_refresh_failed"
        message = f"Codex token refresh failed with status {response.status_code}."
        relogin_required = False
        try:
            err = response.json()
            if isinstance(err, dict):
                err_code = err.get("error")
                if isinstance(err_code, str) and err_code.strip():
                    code = err_code.strip()
                err_desc = err.get("error_description") or err.get("message")
                if isinstance(err_desc, str) and err_desc.strip():
                    message = f"Codex token refresh failed: {err_desc.strip()}"
        except Exception:
            pass
        if code in {"invalid_grant", "invalid_token", "invalid_request"}:
            relogin_required = True
        raise AuthError(
            message,
            provider="openai-codex",
            code=code,
            relogin_required=relogin_required,
        )

    try:
        refresh_payload = response.json()
    except Exception as exc:
        raise AuthError(
            "Codex token refresh returned invalid JSON.",
            provider="openai-codex",
            code="codex_refresh_invalid_json",
            relogin_required=True,
        ) from exc

    refreshed_access = refresh_payload.get("access_token")
    if not isinstance(refreshed_access, str) or not refreshed_access.strip():
        raise AuthError(
            "Codex token refresh response was missing access_token.",
            provider="openai-codex",
            code="codex_refresh_missing_access_token",
            relogin_required=True,
        )

    updated = {
        "access_token": refreshed_access.strip(),
        "refresh_token": refresh_token.strip(),
        "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    next_refresh = refresh_payload.get("refresh_token")
    if isinstance(next_refresh, str) and next_refresh.strip():
        updated["refresh_token"] = next_refresh.strip()
    return updated


def _refresh_codex_auth_tokens(
    tokens: Dict[str, str],
    timeout_seconds: float,
) -> Dict[str, str]:
    """Refresh Codex access token using the refresh token.

    Saves the new tokens to Hermes auth store automatically.
    """
    refreshed = refresh_codex_oauth_pure(
        str(tokens.get("access_token", "") or ""),
        str(tokens.get("refresh_token", "") or ""),
        timeout_seconds=timeout_seconds,
    )
    updated_tokens = dict(tokens)
    updated_tokens["access_token"] = refreshed["access_token"]
    updated_tokens["refresh_token"] = refreshed["refresh_token"]

    _save_codex_tokens(updated_tokens)
    return updated_tokens


def _import_codex_cli_tokens() -> Optional[Dict[str, str]]:
    """Try to read tokens from ~/.codex/auth.json (Codex CLI shared file).

    Returns tokens dict if valid, None otherwise. Does NOT write to the shared file.
    """
    codex_home = os.getenv("CODEX_HOME", "").strip()
    if not codex_home:
        codex_home = str(Path.home() / ".codex")
    auth_path = Path(codex_home).expanduser() / "auth.json"
    if not auth_path.is_file():
        return None
    try:
        payload = json.loads(auth_path.read_text())
        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            return None
        if not tokens.get("access_token") or not tokens.get("refresh_token"):
            return None
        return dict(tokens)
    except Exception:
        return None


def resolve_codex_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    """Resolve runtime credentials from Hermes's own Codex token store."""
    try:
        data = _read_codex_tokens()
    except AuthError as orig_err:
        # Only attempt migration when there are NO tokens stored at all
        # (code == "codex_auth_missing"), not when tokens exist but are invalid.
        if orig_err.code != "codex_auth_missing":
            raise

        # Migration: user had Codex as active provider with old storage (~/.codex/).
        cli_tokens = _import_codex_cli_tokens()
        if cli_tokens:
            logger.info("Migrating Codex credentials from ~/.codex/ to Hermes auth store")
            print("⚠️  Migrating Codex credentials to Hermes's own auth store.")
            print("   This avoids conflicts with Codex CLI and VS Code.")
            print("   Run `hermes login` to create a fully independent session.\n")
            _save_codex_tokens(cli_tokens)
            data = _read_codex_tokens()
        else:
            raise
    tokens = dict(data["tokens"])
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_timeout_seconds = float(os.getenv("HERMES_CODEX_REFRESH_TIMEOUT_SECONDS", "20"))

    should_refresh = bool(force_refresh)
    if (not should_refresh) and refresh_if_expiring:
        should_refresh = _codex_access_token_is_expiring(access_token, refresh_skew_seconds)
    if should_refresh:
        # Re-read under lock to avoid racing with other Hermes processes
        with _auth_store_lock(timeout_seconds=max(float(AUTH_LOCK_TIMEOUT_SECONDS), refresh_timeout_seconds + 5.0)):
            data = _read_codex_tokens(_lock=False)
            tokens = dict(data["tokens"])
            access_token = str(tokens.get("access_token", "") or "").strip()

            should_refresh = bool(force_refresh)
            if (not should_refresh) and refresh_if_expiring:
                should_refresh = _codex_access_token_is_expiring(access_token, refresh_skew_seconds)

            if should_refresh:
                tokens = _refresh_codex_auth_tokens(tokens, refresh_timeout_seconds)
                access_token = str(tokens.get("access_token", "") or "").strip()

    base_url = (
        os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
        or DEFAULT_CODEX_BASE_URL
    )

    return {
        "provider": "openai-codex",
        "base_url": base_url,
        "api_key": access_token,
        "source": "hermes-auth-store",
        "last_refresh": data.get("last_refresh"),
        "auth_mode": "chatgpt",
    }


def get_codex_auth_status() -> Dict[str, Any]:
    """Status snapshot for Codex auth."""
    try:
        creds = resolve_codex_runtime_credentials()
        return {
            "logged_in": True,
            "auth_store": str(_auth_file_path()),
            "last_refresh": creds.get("last_refresh"),
            "auth_mode": creds.get("auth_mode"),
            "source": creds.get("source"),
        }
    except AuthError as exc:
        return {
            "logged_in": False,
            "auth_store": str(_auth_file_path()),
            "error": str(exc),
        }


def _login_openai_codex(args, pconfig: ProviderConfig) -> None:
    """OpenAI Codex login via device code flow. Tokens stored in ~/.hermes/auth.json."""

    # Check for existing Hermes-owned credentials
    try:
        existing = resolve_codex_runtime_credentials()
        print("Existing Codex credentials found in Hermes auth store.")
        try:
            reuse = input("Use existing credentials? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            reuse = "y"
        if reuse in ("", "y", "yes"):
            config_path = _update_config_for_provider("openai-codex", existing.get("base_url", DEFAULT_CODEX_BASE_URL))
            print()
            print("Login successful!")
            print(f"  Config updated: {config_path} (model.provider=openai-codex)")
            return
    except AuthError:
        pass

    # Check for existing Codex CLI tokens we can import
    cli_tokens = _import_codex_cli_tokens()
    if cli_tokens:
        print("Found existing Codex CLI credentials at ~/.codex/auth.json")
        print("Hermes will create its own session to avoid conflicts with Codex CLI / VS Code.")
        try:
            do_import = input("Import these credentials? (a separate login is recommended) [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            do_import = "n"
        if do_import in ("y", "yes"):
            _save_codex_tokens(cli_tokens)
            base_url = os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/") or DEFAULT_CODEX_BASE_URL
            config_path = _update_config_for_provider("openai-codex", base_url)
            print()
            print("Credentials imported. Note: if Codex CLI refreshes its token,")
            print("Hermes will keep working independently with its own session.")
            print(f"  Config updated: {config_path} (model.provider=openai-codex)")
            return

    # Run a fresh device code flow — Hermes gets its own OAuth session
    print()
    print("Signing in to OpenAI Codex...")
    print("(Hermes creates its own session — won't affect Codex CLI or VS Code)")
    print()

    creds = _codex_device_code_login()

    # Save tokens to Hermes auth store
    _save_codex_tokens(creds["tokens"], creds.get("last_refresh"))
    config_path = _update_config_for_provider("openai-codex", creds.get("base_url", DEFAULT_CODEX_BASE_URL))
    print()
    print("Login successful!")
    from hermes_constants import display_hermes_home as _dhh
    print(f"  Auth state: {_dhh()}/auth.json")
    print(f"  Config updated: {config_path} (model.provider=openai-codex)")


def _codex_device_code_login() -> Dict[str, Any]:
    """Run the OpenAI device code login flow and return credentials dict."""
    import time as _time

    issuer = "https://auth.openai.com"
    client_id = CODEX_OAUTH_CLIENT_ID

    # Step 1: Request device code
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            resp = client.post(
                f"{issuer}/api/accounts/deviceauth/usercode",
                json={"client_id": client_id},
                headers={"Content-Type": "application/json"},
            )
    except Exception as exc:
        raise AuthError(
            f"Failed to request device code: {exc}",
            provider="openai-codex", code="device_code_request_failed",
        )

    if resp.status_code != 200:
        raise AuthError(
            f"Device code request returned status {resp.status_code}.",
            provider="openai-codex", code="device_code_request_error",
        )

    device_data = resp.json()
    user_code = device_data.get("user_code", "")
    device_auth_id = device_data.get("device_auth_id", "")
    poll_interval = max(3, int(device_data.get("interval", "5")))

    if not user_code or not device_auth_id:
        raise AuthError(
            "Device code response missing required fields.",
            provider="openai-codex", code="device_code_incomplete",
        )

    # Step 2: Show user the code
    print("To continue, follow these steps:\n")
    print("  1. Open this URL in your browser:")
    print(f"     \033[94m{issuer}/codex/device\033[0m\n")
    print("  2. Enter this code:")
    print(f"     \033[94m{user_code}\033[0m\n")
    print("Waiting for sign-in... (press Ctrl+C to cancel)")

    # Step 3: Poll for authorization code
    max_wait = 15 * 60  # 15 minutes
    start = _time.monotonic()
    code_resp = None

    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            while _time.monotonic() - start < max_wait:
                _time.sleep(poll_interval)
                poll_resp = client.post(
                    f"{issuer}/api/accounts/deviceauth/token",
                    json={"device_auth_id": device_auth_id, "user_code": user_code},
                    headers={"Content-Type": "application/json"},
                )

                if poll_resp.status_code == 200:
                    code_resp = poll_resp.json()
                    break
                elif poll_resp.status_code in (403, 404):
                    continue  # User hasn't completed login yet
                else:
                    raise AuthError(
                        f"Device auth polling returned status {poll_resp.status_code}.",
                        provider="openai-codex", code="device_code_poll_error",
                    )
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)

    if code_resp is None:
        raise AuthError(
            "Login timed out after 15 minutes.",
            provider="openai-codex", code="device_code_timeout",
        )

    # Step 4: Exchange authorization code for tokens
    authorization_code = code_resp.get("authorization_code", "")
    code_verifier = code_resp.get("code_verifier", "")
    redirect_uri = f"{issuer}/deviceauth/callback"

    if not authorization_code or not code_verifier:
        raise AuthError(
            "Device auth response missing authorization_code or code_verifier.",
            provider="openai-codex", code="device_code_incomplete_exchange",
        )

    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            token_resp = client.post(
                CODEX_OAUTH_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": authorization_code,
                    "redirect_uri": redirect_uri,
                    "client_id": client_id,
                    "code_verifier": code_verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except Exception as exc:
        raise AuthError(
            f"Token exchange failed: {exc}",
            provider="openai-codex", code="token_exchange_failed",
        )

    if token_resp.status_code != 200:
        raise AuthError(
            f"Token exchange returned status {token_resp.status_code}.",
            provider="openai-codex", code="token_exchange_error",
        )

    tokens = token_resp.json()
    access_token = tokens.get("access_token", "")
    refresh_token = tokens.get("refresh_token", "")

    if not access_token:
        raise AuthError(
            "Token exchange did not return an access_token.",
            provider="openai-codex", code="token_exchange_no_access_token",
        )

    # Return tokens for the caller to persist (no longer writes to ~/.codex/)
    base_url = (
        os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
        or DEFAULT_CODEX_BASE_URL
    )

    return {
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
        },
        "base_url": base_url,
        "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "auth_mode": "chatgpt",
        "source": "device-code",
    }
