"""Sign in with ChatGPT (SIWC) plan usage: opt-in ChatGPT Plus/Pro/Team quota on api.openai.com.

Provider openai-chatgpt. Tokens live in ~/.hermes/auth.json under their OWN provider key,
never under openai-codex, so the single-use refresh rotation of one session cannot revoke
the other. The credential routes ONLY to https://api.openai.com/v1 (Responses API); the
Codex backend (chatgpt.com/backend-api/codex) keeps using the untouched Codex credential.

Opt-in is the use_direct marker (chatgpt.tokens.use.direct): registration (login or
_save_siwc_tokens) sets it; resolve_siwc_runtime_credentials refuses to resolve without
it. Login reuses the ChatGPT PKCE/OIDC browser flow; the OAuth client id is dynamically
overridable via HERMES_SIWC_CLIENT_ID (dynamic_agent_client).

Split out following the hermes_cli/auth_codex.py pattern; origin helpers are imported lazily
inside each function so hermes_cli.auth.<name> patches still intercept (and no import cycle).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from hermes_cli.auth_constants import (
    _decode_jwt_claims, AUTH_LOCK_TIMEOUT_SECONDS, AuthError,
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS, CODEX_OAUTH_USER_AGENT,
    DEFAULT_CHATGPT_PLAN_BASE_URL, SIWC_OAUTH_CLIENT_ID, SIWC_OAUTH_TOKEN_URL,
    _siwc_err, httpx)
from utils import env_float

# Log-record parity with the origin module (caplog tests pin hermes_cli.auth).
logger = logging.getLogger("hermes_cli.auth")

SIWC_PROVIDER_ID = "openai-chatgpt"
SIWC_AUTH_MODE = "chatgpt-plan"

_MISSING_ACCESS_TOKEN_MSG = "ChatGPT plan auth is missing access_token. Run {relogin} to re-authenticate."
_MISSING_REFRESH_TOKEN_MSG = "ChatGPT plan auth is missing refresh_token. Run {relogin} to re-authenticate."
_NO_CREDENTIALS_MSG = "No ChatGPT plan credentials stored. Run {relogin} to authenticate."
_NOT_OPTED_IN_MSG = (
    "ChatGPT plan usage is not opted in. Run {relogin} to register "
    "Sign in with ChatGPT (chatgpt.tokens.use.direct).")


def _siwc_relogin_command() -> str:
    from agent.turn_failure_copy import oauth_relogin_command

    return oauth_relogin_command(SIWC_PROVIDER_ID)


def _siwc_oauth_client_id() -> str:
    """Dynamically registered agent client id: HERMES_SIWC_CLIENT_ID wins."""
    return os.getenv("HERMES_SIWC_CLIENT_ID", "").strip() or SIWC_OAUTH_CLIENT_ID


def _siwc_base_url() -> str:
    return os.getenv("HERMES_SIWC_BASE_URL", "").strip().rstrip("/") or DEFAULT_CHATGPT_PLAN_BASE_URL


def _stripped(value: Any) -> str:
    return str(value or "").strip()


def _siwc_access_token_is_expiring(access_token: Any, skew_seconds: int) -> bool:
    exp = _decode_jwt_claims(access_token).get("exp")
    return isinstance(exp, (int, float)) and float(exp) <= (time.time() + max(0, int(skew_seconds)))


def _siwc_runtime_result(
    api_key: str, *, source: str, last_refresh: Optional[str], base_url: Optional[str] = None) -> Dict[str, Any]:
    return {
        "provider": SIWC_PROVIDER_ID, "base_url": base_url or _siwc_base_url(), "api_key": api_key,
        "source": source, "last_refresh": last_refresh, "auth_mode": SIWC_AUTH_MODE}


def _read_siwc_tokens(*, _lock: bool = True) -> Dict[str, Any]:
    """Read SIWC tokens from the Hermes auth store (providers.openai-chatgpt)."""
    from hermes_cli.auth import _load_auth_store, _auth_store_lock, _load_provider_state, _nonempty_str
    if _lock:
        with _auth_store_lock():
            auth_store = _load_auth_store()
    else:
        auth_store = _load_auth_store()
    state = _load_provider_state(auth_store, SIWC_PROVIDER_ID)
    if not state:
        raise _siwc_err(_NO_CREDENTIALS_MSG.format(relogin=_siwc_relogin_command()),
                        "siwc_auth_missing", relogin=True)
    if not state.get("use_direct"):
        raise _siwc_err(_NOT_OPTED_IN_MSG.format(relogin=_siwc_relogin_command()),
                        "siwc_not_opted_in", relogin=True)
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise _siwc_err(_NO_CREDENTIALS_MSG.format(relogin=_siwc_relogin_command()),
                        "siwc_auth_invalid_shape", relogin=True)
    if not _nonempty_str(tokens.get("access_token")):
        raise _siwc_err(_MISSING_ACCESS_TOKEN_MSG.format(relogin=_siwc_relogin_command()),
                        "siwc_auth_missing_access_token", relogin=True)
    if not _nonempty_str(tokens.get("refresh_token")):
        raise _siwc_err(_MISSING_REFRESH_TOKEN_MSG.format(relogin=_siwc_relogin_command()),
                        "siwc_auth_missing_refresh_token", relogin=True)
    return {"tokens": dict(tokens), "last_refresh": state.get("last_refresh")}


def _save_siwc_tokens(
    tokens: Dict[str, str], last_refresh: str = None, label: str = None, *,
    set_active: bool = True) -> None:
    """Save SIWC tokens under providers.openai-chatgpt and opt in (use_direct).

    Never touches providers.openai-codex: the two sessions rotate independently.
    """
    from hermes_cli.auth import (
        _provider_state_transaction, _store_provider_state, _save_auth_store, _utc_now_z)
    if last_refresh is None:
        last_refresh = _utc_now_z()
    with _provider_state_transaction(SIWC_PROVIDER_ID) as (auth_store, state, _source):
        state = dict(state) if state else {}
        state.update(tokens=dict(tokens), last_refresh=last_refresh,
                     auth_mode=SIWC_AUTH_MODE, use_direct=True)
        if label and str(label).strip():
            state["label"] = str(label).strip()
        _store_provider_state(auth_store, SIWC_PROVIDER_ID, state, set_active=set_active)
        _save_auth_store(auth_store)


def refresh_siwc_oauth_pure(
    access_token: str, refresh_token: str, *, timeout_seconds: float = 20.0) -> Dict[str, Any]:
    """Refresh SIWC tokens without mutating Hermes auth state."""
    from hermes_cli.auth import _nonempty_str, _utc_now_z
    del access_token
    if not _nonempty_str(refresh_token):
        raise _siwc_err(_MISSING_REFRESH_TOKEN_MSG.format(relogin=_siwc_relogin_command()),
                        "siwc_auth_missing_refresh_token", relogin=True)
    from hermes_cli.auth_codex import _codex_http_client, _codex_quota_exhausted_error, _codex_refresh_failure_error
    with _codex_http_client(
        timeout=httpx.Timeout(max(5.0, float(timeout_seconds))),
        headers={"Accept": "application/json", "User-Agent": CODEX_OAUTH_USER_AGENT}) as client:
        response = client.post(
            SIWC_OAUTH_TOKEN_URL, headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token", "refresh_token": refresh_token,
                "client_id": _siwc_oauth_client_id()})
    if response.status_code == 429:
        raise _codex_quota_exhausted_error(None)
    if response.status_code != 200:
        raise _codex_refresh_failure_error(response)
    try:
        payload = response.json()
    except Exception:
        raise _siwc_err("ChatGPT plan token refresh returned invalid JSON.",
                        "siwc_refresh_invalid_json")
    refreshed_access = _stripped(payload.get("access_token") if isinstance(payload, dict) else "")
    if not refreshed_access:
        raise _siwc_err("ChatGPT plan token refresh response was missing access_token.",
                        "siwc_refresh_missing_access_token", relogin=True)
    updated = {
        "access_token": refreshed_access, "refresh_token": refresh_token.strip(),
        "last_refresh": _utc_now_z()}
    next_refresh = payload.get("refresh_token") if isinstance(payload, dict) else None
    if isinstance(next_refresh, str) and next_refresh.strip():
        updated["refresh_token"] = next_refresh.strip()
    return updated


def _refresh_siwc_auth_tokens(tokens: Dict[str, str], timeout_seconds: float) -> Dict[str, str]:
    """Refresh the SIWC access token; rotates ONLY the openai-chatgpt chain."""
    from hermes_cli.auth import _provider_state_transaction, _save_siwc_tokens
    lock_timeout = max(float(AUTH_LOCK_TIMEOUT_SECONDS), float(timeout_seconds) + 5.0)
    with _provider_state_transaction(SIWC_PROVIDER_ID, lock_timeout) as (_store, state, _source):
        stored = (state or {}).get("tokens")
        stored = stored if isinstance(stored, dict) else {}
        stored_at, stored_rt = _stripped(stored.get("access_token")), _stripped(stored.get("refresh_token"))
        if stored_at and stored_rt and stored_rt != _stripped(tokens.get("refresh_token")):
            logger.info("ChatGPT plan refresh token already rotated by a peer; adopting the stored pair.")
            return {**tokens, "access_token": stored_at, "refresh_token": stored_rt}
        refreshed = refresh_siwc_oauth_pure(
            str(tokens.get("access_token", "") or ""), str(tokens.get("refresh_token", "") or ""),
            timeout_seconds=timeout_seconds)
        updated_tokens = {
            **tokens, "access_token": refreshed["access_token"],
            "refresh_token": refreshed["refresh_token"]}
        _save_siwc_tokens(updated_tokens)
    return updated_tokens


def resolve_siwc_runtime_credentials(
    *, force_refresh: bool = False, refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    read_only: bool = False) -> Dict[str, Any]:
    """Resolve runtime credentials from the SIWC token store.

    read_only=True reports stored state as-is: no refresh, no write, and it wins over
    force_refresh. Always routes to https://api.openai.com/v1; the Codex backend is
    never consulted.
    """
    from hermes_cli.auth import _auth_store_lock
    try:
        if read_only:
            data = _read_siwc_tokens(_lock=False)
        else:
            with _auth_store_lock():
                data = _read_siwc_tokens(_lock=False)
    except AuthError:
        raise
    tokens = dict(data["tokens"])
    access_token = _stripped(tokens.get("access_token"))
    refresh_timeout_seconds = env_float("HERMES_SIWC_REFRESH_TIMEOUT_SECONDS", 20)

    def _should_refresh(token: str) -> bool:
        if read_only:
            return False
        return bool(force_refresh) or (
            refresh_if_expiring and _siwc_access_token_is_expiring(token, refresh_skew_seconds))

    if _should_refresh(access_token):
        lock_timeout = max(float(AUTH_LOCK_TIMEOUT_SECONDS), refresh_timeout_seconds + 5.0)
        with _auth_store_lock(timeout_seconds=lock_timeout):
            data = _read_siwc_tokens(_lock=False)
            tokens = dict(data["tokens"])
            if _should_refresh(_stripped(tokens.get("access_token"))):
                tokens = _refresh_siwc_auth_tokens(tokens, refresh_timeout_seconds)
            access_token = _stripped(tokens.get("access_token"))
    return _siwc_runtime_result(
        access_token, source="hermes-auth-store", last_refresh=data.get("last_refresh"))


def get_siwc_auth_status() -> Dict[str, Any]:
    """Status snapshot for SIWC auth. Read-only: never refreshes or persists."""
    from hermes_cli.auth import _pool_first_oauth_status
    return _pool_first_oauth_status(
        SIWC_PROVIDER_ID, is_expiring=_siwc_access_token_is_expiring, auth_mode=SIWC_AUTH_MODE,
        resolve=lambda: resolve_siwc_runtime_credentials(read_only=True))


def _login_openai_chatgpt(args, pconfig, *, force_new_login: bool = False) -> None:
    """SIWC login: ChatGPT PKCE/OIDC browser flow, stored as its own opted-in credential."""
    from hermes_cli.auth import (
        _offer_existing_oauth_credentials, _print_login_success, _update_config_for_provider,
        resolve_siwc_runtime_credentials)
    from hermes_cli.auth_codex_browser import codex_oauth_login
    del pconfig
    if not force_new_login:
        if _offer_existing_oauth_credentials(
            SIWC_PROVIDER_ID, resolve=resolve_siwc_runtime_credentials,
            is_expiring=_siwc_access_token_is_expiring, display_name="ChatGPT plan",
            default_base_url=DEFAULT_CHATGPT_PLAN_BASE_URL,
            expired_notice="Existing ChatGPT plan credentials are expired. Starting fresh login..."):
            return

    print()
    creds = codex_oauth_login(args)
    _save_siwc_tokens(creds["tokens"], creds.get("last_refresh"))
    config_path = _update_config_for_provider(SIWC_PROVIDER_ID, _siwc_base_url())
    _print_login_success(SIWC_PROVIDER_ID, config_path, show_auth_state=True)
