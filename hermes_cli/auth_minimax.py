"""MiniMax OAuth (user-code grant) login, refresh and runtime credentials.

Split out of ``hermes_cli/auth.py``; origin helpers are imported lazily per function so
``hermes_cli.auth.<helper>`` patches still intercept and no cycle forms.
"""

from __future__ import annotations

import logging
import base64
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
from hermes_cli.auth_constants import (
    AuthError, MINIMAX_OAUTH_GRANT_TYPE, MINIMAX_OAUTH_REFRESH_SKEW_SECONDS, MINIMAX_OAUTH_SCOPE,
    _FORM_JSON_HEADERS, _minimax_err, httpx,
)

if TYPE_CHECKING:  # annotation-only; the runtime import would be a cycle
    from hermes_cli.auth import ProviderConfig
logger = logging.getLogger("hermes_cli.auth")

_MINIMAX_OAUTH_ERROR_BODY_LIMIT = 16 * 1024


def _minimax_response_error_text(response: httpx.Response, *, limit: int = _MINIMAX_OAUTH_ERROR_BODY_LIMIT) -> str:
    """Return a bounded error body from a streamed MiniMax OAuth response."""
    limit = max(0, int(limit))
    try:
        if getattr(response, "is_stream_consumed", False):
            text = response.text
            return text[:limit] + ("...[truncated]" if len(text) > limit else "")
        # Read at most limit+1 bytes so truncation can be detected without buffering the whole body.
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            chunks.append(chunk[: limit + 1 - total])
            total += len(chunks[-1])
            if total > limit:
                break
        raw = b"".join(chunks)
        text = raw[:limit].decode(response.encoding or "utf-8", errors="replace")
        return text + ("...[truncated]" if len(raw) > limit else "")
    finally:
        response.close()


def _minimax_post_form(client: httpx.Client, url: str, *, data: Dict[str, Any], headers: Dict[str, str]) -> httpx.Response:
    """POST a MiniMax OAuth form without eagerly reading error bodies."""
    response = client.send(client.build_request("POST", url, data=data, headers=headers), stream=True)
    if response.status_code == 200:
        response.read()
    return response


def _minimax_pkce_pair() -> tuple:
    """Generate (code_verifier, code_challenge_S256, state) for MiniMax OAuth."""
    import secrets
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge, secrets.token_urlsafe(16)


def _minimax_request_user_code(
    client: httpx.Client, *, portal_base_url: str, client_id: str, code_challenge: str, state: str
) -> Dict[str, Any]:
    response = _minimax_post_form(
        client,
        f"{portal_base_url}/oauth/code",
        data={
            "response_type": "code", "client_id": client_id, "scope": MINIMAX_OAUTH_SCOPE,
            "code_challenge": code_challenge, "code_challenge_method": "S256", "state": state,
        },
        headers={**_FORM_JSON_HEADERS, "x-request-id": str(uuid.uuid4())},
    )
    if response.status_code != 200:
        body = _minimax_response_error_text(response)
        raise _minimax_err(f"MiniMax OAuth authorization failed: {body or response.reason_phrase}", "authorization_failed")
    payload = response.json()
    for field in ("user_code", "verification_uri", "expired_in"):
        if field not in payload:
            raise _minimax_err(f"MiniMax OAuth response missing field: {field}", "authorization_incomplete")
    if payload.get("state") != state:
        raise _minimax_err("MiniMax OAuth state mismatch (possible CSRF).", "state_mismatch")
    return payload


def _minimax_expired_in_looks_like_unix_ms(expired_in: int, *, now_ms: int) -> bool:
    """True if ``expired_in`` is plausibly a unix-ms absolute time (vs TTL seconds)."""
    return int(expired_in) > (now_ms // 2)


def _minimax_resolve_token_expiry_unix(expired_in: int, *, now: datetime) -> float:
    """Return access-token expiry as unix seconds (MiniMax uses ms epoch or TTL seconds)."""
    raw = int(expired_in)
    if _minimax_expired_in_looks_like_unix_ms(raw, now_ms=int(now.timestamp() * 1000)):
        return raw / 1000.0
    return now.timestamp() + max(1, raw)


def _minimax_expiry_fields(expired_in: Any) -> Dict[str, Any]:
    """``obtained_at`` / ``expires_at`` / ``expires_in`` derived from a MiniMax ``expired_in``."""
    now = datetime.now(timezone.utc)
    expires_at_unix = _minimax_resolve_token_expiry_unix(int(expired_in), now=now)
    return {
        "obtained_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(expires_at_unix, tz=timezone.utc).isoformat(),
        "expires_in": max(0, int(expires_at_unix - now.timestamp())),
    }


def _minimax_poll_token(
    client: httpx.Client, *, portal_base_url: str, client_id: str,
    user_code: str, code_verifier: str, expired_in: int, interval_ms: Optional[int],
) -> Dict[str, Any]:
    # expired_in is a unix-ms timestamp upstream (OpenClaw) but small values are TTL seconds.
    deadline = _minimax_resolve_token_expiry_unix(expired_in, now=datetime.now(timezone.utc))
    interval = max(2.0, (interval_ms or 2000) / 1000.0)

    while time.time() < deadline:
        response = _minimax_post_form(
            client,
            f"{portal_base_url}/oauth/token",
            data={
                "grant_type": MINIMAX_OAUTH_GRANT_TYPE, "client_id": client_id,
                "user_code": user_code, "code_verifier": code_verifier,
            },
            headers=_FORM_JSON_HEADERS,
        )
        if response.status_code != 200:
            error_text = _minimax_response_error_text(response)
            try:
                payload = json.loads(error_text) if error_text else {}
            except Exception:
                payload = {}
            msg = (payload.get("base_resp", {}) or {}).get("status_msg") or error_text
            raise _minimax_err(f"MiniMax OAuth error: {msg or 'unknown'}", "token_exchange_failed")
        try:
            payload = response.json() if response.text else {}
        except Exception:
            payload = {}

        status = payload.get("status")
        if status == "error":
            raise _minimax_err("MiniMax OAuth reported an error. Please try again later.", "authorization_denied")
        if status == "success":
            if not all(payload.get(k) for k in ("access_token", "refresh_token", "expired_in")):
                raise _minimax_err("MiniMax OAuth success payload missing required token fields.", "token_incomplete")
            return payload
        # "pending" or any other status -> keep polling
        time.sleep(interval)

    raise _minimax_err("MiniMax OAuth timed out before authorization completed.", "timeout")


def _minimax_save_auth_state(
    auth_state: Dict[str, Any],
    *,
    source_path: Optional[Path] = None,
    set_active: bool = True,
) -> None:
    """Persist MiniMax state to its owning store without creating a profile shadow."""
    from hermes_cli.auth import (
        _auth_file_path,
        _load_auth_store,
        _persist_provider_state_to_store,
        _same_path,
        _save_active_provider_state,
        _save_auth_store,
        _store_provider_state,
    )

    active_path = _auth_file_path()
    if source_path is not None and not _same_path(source_path, active_path):
        _persist_provider_state_to_store(
            "minimax-oauth", auth_state, source_path, set_active=set_active
        )
        return
    if set_active:
        _save_active_provider_state("minimax-oauth", auth_state)
        return
    auth_store = _load_auth_store()
    _store_provider_state(
        auth_store, "minimax-oauth", dict(auth_state), set_active=False
    )
    _save_auth_store(auth_store)


def _minimax_oauth_login(*, region: str = "global", open_browser: bool = True, timeout_seconds: float = 15.0) -> Dict[str, Any]:
    """Run MiniMax OAuth flow, persist tokens, return auth state dict."""
    from hermes_cli.auth import PROVIDER_REGISTRY, _can_open_graphical_browser, _is_remote_session, _minimax_pkce_pair, _minimax_request_user_code, _minimax_save_auth_state, _print_device_code_instructions
    pconfig = PROVIDER_REGISTRY["minimax-oauth"]
    if region == "cn":
        portal_base_url = pconfig.extra["cn_portal_base_url"]
        inference_base_url = pconfig.extra["cn_inference_base_url"]
    else:
        portal_base_url = pconfig.portal_base_url
        inference_base_url = pconfig.inference_base_url

    verifier, challenge, state = _minimax_pkce_pair()

    if _is_remote_session():
        open_browser = False

    print(f"Starting Hermes login via MiniMax ({region}) OAuth...")
    print(f"Portal: {portal_base_url}")

    with httpx.Client(timeout=httpx.Timeout(timeout_seconds), headers={"Accept": "application/json"},
                      follow_redirects=True) as client:
        code_data = _minimax_request_user_code(
            client, portal_base_url=portal_base_url, client_id=pconfig.client_id, code_challenge=challenge, state=state,
        )
        _print_device_code_instructions(
            str(code_data["verification_uri"]), str(code_data["user_code"]),
            open_browser=open_browser and _can_open_graphical_browser(),
        )

        interval_raw = code_data.get("interval")
        print("Waiting for approval...")

        token_data = _minimax_poll_token(
            client, portal_base_url=portal_base_url, client_id=pconfig.client_id,
            user_code=str(code_data["user_code"]), code_verifier=verifier,
            expired_in=int(code_data["expired_in"]),
            interval_ms=int(interval_raw) if interval_raw is not None else None,
        )

    auth_state = {
        "provider": "minimax-oauth",
        "region": region,
        "portal_base_url": portal_base_url,
        "inference_base_url": inference_base_url,
        "client_id": pconfig.client_id,
        "scope": MINIMAX_OAUTH_SCOPE,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "resource_url": token_data.get("resource_url"),
        **_minimax_expiry_fields(token_data["expired_in"]),
    }

    _minimax_save_auth_state(auth_state)
    print("\u2713 MiniMax OAuth login successful.")
    if msg := token_data.get("notification_message"):
        print(f"Note from MiniMax: {msg}")
    return auth_state


def refresh_minimax_oauth_pure(
    state: Dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
) -> Dict[str, Any]:
    """Exchange one MiniMax refresh token without mutating an auth store."""
    refresh_token = state.get("refresh_token")
    if not refresh_token:
        raise _minimax_err(
            "MiniMax OAuth state has no refresh_token; please re-login.",
            "no_refresh_token",
            relogin=True,
        )
    portal_base_url = state.get("portal_base_url")
    if not portal_base_url:
        raise _minimax_err(
            "MiniMax OAuth state has no portal_base_url; please re-login.",
            "no_portal_base_url",
            relogin=True,
        )
    with httpx.Client(
        timeout=httpx.Timeout(timeout_seconds), follow_redirects=True
    ) as client:
        response = _minimax_post_form(
            client,
            f"{portal_base_url}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "client_id": state["client_id"],
                "refresh_token": refresh_token,
            },
            headers=_FORM_JSON_HEADERS,
        )
        if response.status_code != 200:
            body = _minimax_response_error_text(response)
            relogin = any(
                marker in body.lower()
                for marker in (
                    "invalid_grant",
                    "refresh_token_reused",
                    "invalid_refresh_token",
                )
            )
            raise _minimax_err(
                f"MiniMax OAuth refresh failed: {body or response.reason_phrase}",
                "refresh_failed",
                relogin=relogin,
            )
    payload = response.json()
    if payload.get("status") != "success" or not payload.get("access_token"):
        raise _minimax_err(
            "MiniMax OAuth refresh did not return a complete success response.",
            "refresh_failed",
            relogin=True,
        )
    if payload.get("expired_in") is None:
        raise _minimax_err(
            "MiniMax OAuth refresh response omitted token expiry.",
            "refresh_failed",
            relogin=True,
        )
    return {
        "access_token": payload["access_token"],
        # A successful MiniMax response may omit a replacement refresh token.
        "refresh_token": payload.get("refresh_token") or refresh_token,
        **_minimax_expiry_fields(payload["expired_in"]),
    }


def _is_terminal_minimax_oauth_refresh_error(exc: Exception) -> bool:
    """Whether retrying the same MiniMax refresh grant cannot succeed."""
    return (
        isinstance(exc, AuthError)
        and exc.provider == "minimax-oauth"
        and exc.code
        in {
            "refresh_failed",
            "no_refresh_token",
            "no_portal_base_url",
            "invalid_grant",
            "invalid_token",
            "refresh_token_reused",
        }
        and bool(exc.relogin_required)
    )


def _minimax_state_needs_refresh(state: Dict[str, Any], *, force: bool) -> bool:
    try:
        expires_at = datetime.fromisoformat(state.get("expires_at", "")).timestamp()
    except Exception:
        expires_at = 0.0
    return force or (expires_at - time.time()) <= MINIMAX_OAUTH_REFRESH_SKEW_SECONDS


def _refresh_minimax_oauth_state(
    state: Dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
    force: bool = False,
    source_path: Optional[Path] = None,
    set_active: bool = False,
    require_persisted: bool = False,
) -> Dict[str, Any]:
    """Refresh under the authoritative auth-store transaction and save through."""
    from hermes_cli.auth import (
        AUTH_LOCK_TIMEOUT_SECONDS,
        _load_auth_store,
        _load_provider_state_with_source,
        _minimax_oauth_quarantine_on_terminal_refresh,
        _minimax_save_auth_state,
        _provider_state_transaction,
        refresh_minimax_oauth_pure,
    )

    if not state.get("refresh_token"):
        raise _minimax_err(
            "MiniMax OAuth state has no refresh_token; please re-login.",
            "no_refresh_token",
            relogin=True,
        )
    if not _minimax_state_needs_refresh(state, force=force):
        return state

    persisted, discovered_path = _load_provider_state_with_source(
        _load_auth_store(), "minimax-oauth"
    )
    if persisted is None and source_path is None:
        if require_persisted:
            raise _minimax_err(
                "MiniMax OAuth session was removed; please re-login.",
                "not_logged_in",
                relogin=True,
            )
        # Preserve the direct helper contract used by setup/tests where the
        # supplied state has not yet been materialized in auth.json.
        updated = dict(state)
        updated.update(
            refresh_minimax_oauth_pure(state, timeout_seconds=timeout_seconds)
        )
        _minimax_save_auth_state(updated)
        return updated

    lock_timeout = max(float(AUTH_LOCK_TIMEOUT_SECONDS), timeout_seconds + 5.0)
    with _provider_state_transaction(
        "minimax-oauth", timeout_seconds=lock_timeout
    ) as (_auth_store, authoritative, authoritative_path):
        if not isinstance(authoritative, dict):
            raise _minimax_err(
                "MiniMax OAuth session was removed; please re-login.",
                "not_logged_in",
                relogin=True,
            )
        if not _minimax_state_needs_refresh(authoritative, force=force):
            return authoritative
        try:
            refreshed = refresh_minimax_oauth_pure(
                authoritative, timeout_seconds=timeout_seconds
            )
        except AuthError as exc:
            if _is_terminal_minimax_oauth_refresh_error(exc):
                quarantined = dict(authoritative)
                _minimax_oauth_quarantine_on_terminal_refresh(
                    quarantined, exc, persist=False
                )
                try:
                    _minimax_save_auth_state(
                        quarantined,
                        source_path=authoritative_path or discovered_path,
                        set_active=False,
                    )
                except Exception:
                    logger.error(
                        "MiniMax OAuth terminal quarantine could not be persisted",
                        exc_info=True,
                    )
            raise

        updated = dict(authoritative)
        updated.update(refreshed)
        try:
            _minimax_save_auth_state(
                updated,
                source_path=authoritative_path or discovered_path,
                set_active=set_active,
            )
        except Exception as save_exc:
            persist_error = _minimax_err(
                "MiniMax OAuth rotated its refresh token but the replacement "
                "could not be persisted; please re-login.",
                "credential_persist_failed",
                relogin=True,
            )
            quarantined = dict(authoritative)
            _minimax_oauth_quarantine_on_terminal_refresh(
                quarantined, persist_error, persist=False
            )
            try:
                _minimax_save_auth_state(
                    quarantined,
                    source_path=authoritative_path or discovered_path,
                    set_active=False,
                )
            except Exception:
                logger.error(
                    "MiniMax OAuth rotation and fail-closed quarantine both failed to persist",
                    exc_info=True,
                )
            raise persist_error from save_exc
        return updated


def _minimax_oauth_quarantine_on_terminal_refresh(
    state: Dict[str, Any],
    exc: AuthError,
    *,
    persist: bool = True,
    source_path: Optional[Path] = None,
    set_active: bool = False,
) -> None:
    """Strip a terminal grant and persist the quarantine to its owning store."""
    from hermes_cli.auth import (
        _load_auth_store,
        _load_provider_state_with_source,
        _minimax_save_auth_state,
        _quarantine_flat_oauth_state,
    )

    if not (exc.relogin_required and state.get("refresh_token")):
        return
    _quarantine_flat_oauth_state(state, "minimax-oauth", exc)
    if not persist:
        return
    if source_path is None:
        _persisted, source_path = _load_provider_state_with_source(
            _load_auth_store(), "minimax-oauth"
        )
    try:
        if source_path is None:
            if not set_active:
                # No authoritative store remains. Never resurrect the stale
                # in-memory grant into the active profile.
                return
            # Historical direct-call seam for setup/unit callers whose state
            # was intentionally not materialized yet.
            _minimax_save_auth_state(state)
        else:
            _minimax_save_auth_state(
                state, source_path=source_path, set_active=False
            )
    except Exception:
        logger.debug(
            "MiniMax OAuth: failed to persist quarantined state", exc_info=True
        )


def _minimax_fresh_state() -> Dict[str, Any]:
    """Load MiniMax state with source identity, refresh, and quarantine terminal failures."""
    from hermes_cli.auth import (
        _load_auth_store,
        _load_provider_state_with_source,
        _refresh_minimax_oauth_state,
        get_provider_auth_state,
    )

    state, source_path = _load_provider_state_with_source(
        _load_auth_store(), "minimax-oauth"
    )
    if state is None:
        # Preserve the public monkeypatch seam used by callers/tests that
        # supply an unmaterialized state directly.
        state = get_provider_auth_state("minimax-oauth")
        if state is not None:
            persisted, persisted_path = _load_provider_state_with_source(
                _load_auth_store(), "minimax-oauth"
            )
            if persisted is not None:
                state, source_path = persisted, persisted_path
    if not state or not state.get("access_token"):
        raise _minimax_err(
            "Not logged into MiniMax OAuth. Run `hermes model` and select MiniMax (OAuth).", "not_logged_in", relogin=True,
        )
    try:
        if source_path is not None:
            return _refresh_minimax_oauth_state(
                state,
                source_path=source_path,
                require_persisted=True,
            )
        return _refresh_minimax_oauth_state(state)
    except AuthError as exc:
        if exc.code != "not_logged_in":
            _minimax_oauth_quarantine_on_terminal_refresh(
                state, exc, set_active=True
            )
        raise


def build_minimax_oauth_token_provider() -> Callable[[], str]:
    """Zero-arg callable yielding a fresh MiniMax access token.

    The Anthropic SDK caches ``api_key`` at construction; MiniMax tokens live ~15 minutes, so a
    static bearer would start 401-ing mid-session.
    """
    def _provide() -> str:
        token = _minimax_fresh_state().get("access_token")
        if not token:
            raise _minimax_err("MiniMax OAuth state has no access_token after refresh.", "no_access_token", relogin=True)
        return token

    return _provide


def resolve_minimax_oauth_runtime_credentials(
    *, min_token_ttl_seconds: int = MINIMAX_OAUTH_REFRESH_SKEW_SECONDS,
    as_token_provider: bool = False,
) -> Dict[str, Any]:
    """Return {provider, api_key, base_url, source}; string ``api_key`` by default (``hermes status`` contract)."""
    state = _minimax_fresh_state()
    return {
        "provider": "minimax-oauth",
        "api_key": build_minimax_oauth_token_provider() if as_token_provider else state["access_token"],
        "base_url": state["inference_base_url"].rstrip("/"),
        "source": "oauth",
    }


def _login_minimax_oauth(args, pconfig: ProviderConfig) -> None:
    """CLI entry for MiniMax OAuth login."""
    from hermes_cli.auth import format_auth_error
    try:
        _minimax_oauth_login(
            region=getattr(args, "region", None) or "global",
            open_browser=not getattr(args, "no_browser", False),
            timeout_seconds=getattr(args, "timeout", None) or 15.0,
        )
    except AuthError as exc:
        print(format_auth_error(exc))
        raise SystemExit(1)
