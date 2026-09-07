"""Nous Portal OAuth: device-code login, refresh, shared-store mirroring, JWT selection, status.

Split out of ``hermes_cli/auth.py``; origin helpers are imported lazily inside each function
so ``hermes_cli.auth.<name>`` patches still intercept (and no import cycle).
"""

from __future__ import annotations

import logging
import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional
from urllib.parse import urlparse
from hermes_cli.auth_codex import _pool_entries
from hermes_cli.auth_constants import (
    _decode_jwt_claims, AUTH_LOCK_TIMEOUT_SECONDS, AuthError, DEFAULT_NOUS_CLIENT_ID,
    DEFAULT_NOUS_INFERENCE_URL, DEFAULT_NOUS_PORTAL_URL, DEFAULT_NOUS_SCOPE,
    DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS, NOUS_AUTH_PATH_INVOKE_JWT, NOUS_BILLING_MANAGE_SCOPE,
    NOUS_DEVICE_CODE_SOURCE, NOUS_INFERENCE_INVOKE_SCOPE, NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
    _nous_err, httpx)

if TYPE_CHECKING:  # annotation-only; the runtime import would be a cycle
    from hermes_cli.auth import ProviderConfig

# Log-record parity with the origin module (caplog tests pin "hermes_cli.auth").
logger = logging.getLogger("hermes_cli.auth")

_UNUSABLE_JWT_RELOGIN = "Re-authenticate with: hermes auth add nous"


def _unusable_invoke_jwt_error(reason: str, *, no_refresh_token: bool = False) -> AuthError:
    """Shared ``relogin=True`` error for an access token that is not a usable inference JWT."""
    detail = " and no refresh token is available" if no_refresh_token else ""
    return _nous_err(
        f"Nous Portal access token is not a usable inference JWT ({reason}){detail}. "
        f"{_UNUSABLE_JWT_RELOGIN}",
        reason, relogin=True)


def _token_fingerprint(token: Any) -> Optional[str]:
    """Return a short hash fingerprint for telemetry without leaking token bytes."""
    cleaned = token.strip() if isinstance(token, str) else ""
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12] if cleaned else None


def _oauth_trace(event: str, *, sequence_id: Optional[str] = None, **fields: Any) -> None:
    if os.getenv("HERMES_OAUTH_TRACE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    payload: Dict[str, Any] = {"event": event}
    if sequence_id:
        payload["sequence_id"] = sequence_id
    payload.update(fields)
    logger.info("oauth_trace %s", json.dumps(payload, sort_keys=True, ensure_ascii=False))


def _iso_after(now: datetime, ttl_seconds: int) -> str:
    """ISO timestamp *ttl_seconds* after *now* (UTC)."""
    return datetime.fromtimestamp(now.timestamp() + ttl_seconds, tz=timezone.utc).isoformat()


# Nous agent-key slots; a fresh login persists them as None, quarantine strips them.
_NOUS_EMPTY_AGENT_KEY_FIELDS: Dict[str, Any] = {
    "agent_key": None, "agent_key_id": None, "agent_key_expires_at": None,
    "agent_key_expires_in": None, "agent_key_reused": None, "agent_key_obtained_at": None}

_NOUS_STALE_PORTAL_HOSTS: FrozenSet[str] = frozenset({"api.nousresearch.com"})


def _portal_entitlement_message(capability: str) -> str:
    """Portal entitlement notice for *capability* (fresh account data), "" when unavailable."""
    from hermes_cli.nous_account import (
        format_nous_portal_entitlement_message, get_nous_portal_account_info)
    account_info = get_nous_portal_account_info(force_fresh=True)
    return format_nous_portal_entitlement_message(account_info, capability=capability) or ""


def _format_nous_entitlement_auth_error(error: AuthError) -> str:
    with suppress(Exception):
        if message := _portal_entitlement_message("Nous model access"):
            return message
    return f"{error} Check credits or billing in Nous Portal, then retry."


def _migrate_stale_nous_portal_url(providers: Dict[str, Any]) -> None:
    nous = providers.get("nous")
    if not isinstance(nous, dict):
        return
    stored = (nous.get("portal_base_url") or "").strip()
    if stored and urlparse(stored).hostname in _NOUS_STALE_PORTAL_HOSTS:
        logger.warning(
            "auth: migrating stale nous portal_base_url %s -> %s", stored, DEFAULT_NOUS_PORTAL_URL)
        nous["portal_base_url"] = DEFAULT_NOUS_PORTAL_URL


# Allowlist of hosts the Nous Portal proxy will forward inference JWTs to — a bearer sent anywhere
# else would leak. Consulted only for URLs from the NETWORK side (Portal refresh responses);
# the NOUS_INFERENCE_BASE_URL env override bypasses it (documented dev/staging escape hatch, the
# user set it themselves).
_ALLOWED_NOUS_INFERENCE_HOSTS: FrozenSet[str] = frozenset({"inference-api.nousresearch.com"})


def _validate_nous_inference_url_from_network(url: Optional[str]) -> Optional[str]:
    """Validate a Portal-returned inference URL against the host allowlist.

    Defense-in-depth: a compromised refresh response (MITM, response injection) could otherwise
    redirect every proxy request — bearing the user's inference JWT — to an attacker endpoint.
    """
    cleaned = url.strip() if isinstance(url, str) else ""
    if not cleaned:
        return None
    try:
        parsed = urlparse(cleaned)
    except Exception:
        return None
    if parsed.scheme != "https":
        logger.warning(
            "nous: refusing non-https inference URL scheme %r from Portal response", parsed.scheme)
        return None
    if parsed.hostname not in _ALLOWED_NOUS_INFERENCE_HOSTS:
        logger.warning(
            "nous: refusing inference URL host %r from Portal response "
            "(not in allowlist); falling back to default",
            parsed.hostname)
        return None
    return cleaned.rstrip("/")


def _nous_inference_env_override() -> Optional[str]:
    """User-set ``NOUS_INFERENCE_BASE_URL`` override (trailing slash stripped) or None.

    Documented dev/staging escape hatch; the env source is trusted, so unlike Portal-returned URLs
    it is intentionally NOT gated by the network host allowlist.
    """
    from hermes_cli.auth import _optional_base_url
    return _optional_base_url(os.getenv("NOUS_INFERENCE_BASE_URL"))


def _nous_portal_env_override() -> Optional[str]:
    """``HERMES_PORTAL_BASE_URL`` / ``NOUS_PORTAL_BASE_URL`` override or None.

    Documented dev/staging escape hatch (e.g. hosted agents on the staging Portal). Trusted env
    source: must NOT be gated by ``_NOUS_PORTAL_ALLOWED_HOSTS``, which rejects untrusted
    NETWORK-provided values persisted to auth.json, not operator config.
    """
    from hermes_cli.auth import _optional_base_url
    return _optional_base_url(
        os.getenv("HERMES_PORTAL_BASE_URL") or os.getenv("NOUS_PORTAL_BASE_URL"))


def _scope_values(raw_scope: Any) -> set[str]:
    # OAuth token responses return a space-separated string; collections are kept for JWT ``scp``
    # claims and older stored fixtures.
    scopes: set[str] = set()
    if isinstance(raw_scope, str):
        scopes.update(part for part in raw_scope.replace(",", " ").split() if part.strip())
    elif isinstance(raw_scope, (list, tuple, set, frozenset)):
        scopes.update(*(_scope_values(item) for item in raw_scope if isinstance(item, str)))
    return scopes


def _nous_invoke_jwt_status(
    token: Any, *, scope: Any = None, expires_at: Any = None,
    min_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS) -> Optional[str]:
    """Return None when the token can be used for inference, else a reason."""
    from hermes_cli.auth import _is_expiring
    claims = _decode_jwt_claims(token)
    if not claims:
        return "access_token_not_jwt"
    scopes = (_scope_values(scope) | _scope_values(claims.get("scope"))
              | _scope_values(claims.get("scp")))
    if NOUS_INFERENCE_INVOKE_SCOPE not in scopes:
        return "missing_inference_invoke_scope"
    exp = claims.get("exp")
    skew = max(0, int(min_ttl_seconds))
    if isinstance(exp, (int, float)):
        return "invoke_jwt_expiring" if float(exp) <= (time.time() + skew) else None
    return "invoke_jwt_expiry_unknown_or_expiring" if _is_expiring(expires_at, skew) else None


def _nous_invoke_jwt_is_usable(
    token: Any, *, scope: Any = None, expires_at: Any = None,
    min_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS) -> bool:
    from hermes_cli.auth import _nous_invoke_jwt_status
    return _nous_invoke_jwt_status(
        token, scope=scope, expires_at=expires_at, min_ttl_seconds=min_ttl_seconds) is None


def _state_invoke_jwt_status(state: Dict[str, Any], token: Any) -> Optional[str]:
    """``_nous_invoke_jwt_status`` for *token* using *state*'s scope / expires_at (patchable)."""
    from hermes_cli.auth import _nous_invoke_jwt_status
    return _nous_invoke_jwt_status(
        token, scope=state.get("scope"), expires_at=state.get("expires_at"))


def _assert_nous_inference_jwt_usable(state: Dict[str, Any], *, access_token: Any = None) -> None:
    token = state.get("access_token") if access_token is None else access_token
    reason = _state_invoke_jwt_status(state, token)
    if reason is not None:
        raise _unusable_invoke_jwt_error(reason)


def _remaining_ttl(expires_at: Any, fallback_expires_in: Any) -> int:
    """Seconds until *expires_at* (ISO), else *fallback_expires_in* coerced to a TTL."""
    from hermes_cli.auth import _coerce_ttl_seconds, _parse_iso_timestamp
    expires_epoch = _parse_iso_timestamp(expires_at)
    if expires_epoch is not None:
        return max(0, int(expires_epoch - time.time()))
    return _coerce_ttl_seconds(fallback_expires_in)


def _nous_jwt_expires_at(token: Any, fallback_expires_at: Any = None) -> Optional[str]:
    claims = _decode_jwt_claims(token)
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        with suppress(Exception):
            return datetime.fromtimestamp(float(exp), tz=timezone.utc).isoformat()
    return fallback_expires_at if isinstance(fallback_expires_at, str) else None


def _set_nous_agent_key_from_invoke_jwt(
    state: Dict[str, Any], *, obtained_at: Optional[str] = None) -> None:
    from hermes_cli.auth import _nonempty_str
    access_token = state.get("access_token")
    if not _nonempty_str(access_token):
        return
    existing_obtained_at = state.get("agent_key_obtained_at")
    if not obtained_at:
        reuse = state.get("agent_key") == access_token and _nonempty_str(existing_obtained_at)
        obtained_at = existing_obtained_at if reuse else datetime.now(timezone.utc).isoformat()
    expires_at = _nous_jwt_expires_at(access_token, state.get("expires_at"))
    expires_in = _remaining_ttl(expires_at, state.get("expires_in"))
    if expires_at:
        state["expires_at"] = expires_at
        state["expires_in"] = expires_in
    state.update(
        agent_key=access_token, agent_key_id=None, agent_key_expires_at=expires_at,
        agent_key_expires_in=expires_in, agent_key_reused=False, agent_key_obtained_at=obtained_at)


def _select_nous_invoke_jwt(
    state: Dict[str, Any], *, access_token: Any = None, sequence_id: Optional[str] = None) -> None:
    from hermes_cli.auth import _nonempty_str
    if _nonempty_str(access_token):
        state["access_token"] = access_token
    _set_nous_agent_key_from_invoke_jwt(state)
    logger.debug("Nous inference auth: using NAS invoke JWT")
    _oauth_trace(
        "nous_invoke_jwt_selected", sequence_id=sequence_id,
        access_token_fp=_token_fingerprint(state.get("access_token")))


# Derived from expires_at/JWT exp and tick down between reads; persisting only these changes makes
# auth.json noisy and defeats the mtime-keyed auth-status cache.
_NOUS_EFFECTIVE_STATE_IGNORED_KEYS = frozenset({"expires_in", "agent_key_expires_in"})


def _nous_effective_provider_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in state.items() if k not in _NOUS_EFFECTIVE_STATE_IGNORED_KEYS}


NOUS_SHARED_STORE_FILENAME = "nous_auth.json"
_nous_shared_lock_holder = threading.local()


def _nous_shared_auth_dir() -> Path:
    """Directory of the shared Nous token store: ``HERMES_SHARED_AUTH_DIR`` or ``<root>/shared/``.

    Outside any named profile so all profiles share it (``hermes --profile X auth add nous --type
    oauth`` one-tap imports it). Written on login AND every runtime refresh so the refresh_token
    stays current across profiles; a stale token just falls back to device-code.
    """
    override = os.getenv("HERMES_SHARED_AUTH_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "shared"


def _nous_shared_store_path() -> Path:
    path = _nous_shared_auth_dir() / NOUS_SHARED_STORE_FILENAME
    # Seat belt (mirrors the _auth_file_path() guard): under pytest, refuse a path under the real
    # user's Hermes root so a test that forgot HERMES_SHARED_AUTH_DIR fails loudly instead of
    # corrupting cross-profile state.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        from hermes_constants import get_default_hermes_root
        real_home_shared = (
            get_default_hermes_root() / "shared" / NOUS_SHARED_STORE_FILENAME).resolve(strict=False)
        try:
            resolved = path.resolve(strict=False)
        except Exception:
            resolved = path
        if resolved == real_home_shared:
            raise RuntimeError(
                f"Refusing to touch real user shared Nous auth store during test run: "
                f"{path}. Set HERMES_SHARED_AUTH_DIR to a tmp_path in your test fixture.")
    return path


@contextmanager
def _nous_shared_store_lock(timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS):
    """Cross-profile lock for the shared Nous OAuth store.

    Lock ordering invariant: if both this and ``_auth_store_lock`` need to be held, acquire
    ``_auth_store_lock`` FIRST. All runtime refresh paths follow this order.
    """
    from hermes_cli.auth import _file_lock
    try:
        lock_path = _nous_shared_store_path().with_suffix(".lock")
    except RuntimeError:
        yield  # No HERMES_HOME yet (pre-setup): fall through without locking.
        return
    with _file_lock(
        lock_path, _nous_shared_lock_holder, timeout_seconds,
        "Timed out waiting for shared Nous auth lock"):
        yield


def _shared_lock_timeout(timeout_seconds: float) -> float:
    return max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)


# OAuth fields mirrored between a profile's Nous state and the shared cross-profile store.
_NOUS_SHARED_STATE_KEYS = (
    "access_token", "refresh_token", "token_type", "scope", "client_id", "portal_base_url",
    "inference_base_url", "obtained_at", "expires_at")


def _merge_shared_nous_oauth_state(state: Dict[str, Any]) -> bool:
    """Copy fresher shared OAuth tokens into a profile-local Nous state."""
    from hermes_cli.auth import _nonempty_str, _parse_iso_timestamp, _read_shared_nous_state
    shared = _read_shared_nous_state() or {}
    shared_refresh = shared.get("refresh_token")
    if not _nonempty_str(shared_refresh):
        return False
    shared_access_exp = _parse_iso_timestamp(shared.get("expires_at")) or 0.0
    local_access_exp = _parse_iso_timestamp(state.get("expires_at")) or 0.0
    refresh_changed = shared_refresh.strip() != str(state.get("refresh_token") or "").strip()
    if not refresh_changed and not shared_access_exp > local_access_exp:
        return False
    for key in _NOUS_SHARED_STATE_KEYS:
        value = shared.get(key)
        if value not in {None, ""}:
            state[key] = value
    return True


def _nous_shared_shape(src: Dict[str, Any]) -> Dict[str, Any]:
    """The defaulted OAuth core (tokens + routing + expiry) shared across profiles."""
    return {
        "access_token": src.get("access_token"), "refresh_token": src.get("refresh_token"),
        "token_type": src.get("token_type") or "Bearer",
        "scope": src.get("scope") or DEFAULT_NOUS_SCOPE,
        "client_id": src.get("client_id") or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": src.get("portal_base_url") or DEFAULT_NOUS_PORTAL_URL,
        "inference_base_url": src.get("inference_base_url") or DEFAULT_NOUS_INFERENCE_URL,
        "obtained_at": src.get("obtained_at"), "expires_at": src.get("expires_at")}


def _write_shared_nous_state(state: Dict[str, Any]) -> None:
    """Persist a minimal copy of the Nous OAuth state to the shared store.

    Best-effort: failures are logged and swallowed; per-profile auth.json stays the source of truth.
    """
    from hermes_cli.auth import _nonempty_str, _write_private_file_atomic
    refresh_token = state.get("refresh_token")
    # No refresh_token = nothing worth sharing across profiles
    if not (_nonempty_str(refresh_token) and _nonempty_str(state.get("access_token"))):
        return
    shared = {
        "_schema": 1, **_nous_shared_shape(state),
        "updated_at": datetime.now(timezone.utc).isoformat()}
    try:
        with _nous_shared_store_lock():
            path = _nous_shared_store_path()
            _write_private_file_atomic(
                path, json.dumps(shared, indent=2, sort_keys=True), replace=os.replace)
        _oauth_trace(
            "nous_shared_store_written", path=str(path),
            refresh_token_fp=_token_fingerprint(refresh_token))
    except Exception as exc:
        logger.debug("Failed to write shared Nous auth store: %s", exc)


def _read_shared_nous_state() -> Optional[Dict[str, Any]]:
    """Shared Nous OAuth state when present and well-formed, else None.

    None (missing / unreadable / malformed / lacking tokens) means "fall through to device-code".
    """
    from hermes_cli.auth import _nonempty_str
    try:
        path = _nous_shared_store_path()
    except RuntimeError:
        return None  # Test seat belt tripped — treat as missing
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.debug("Shared Nous auth store at %s is unreadable: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    has_tokens = (
        _nonempty_str(payload.get("refresh_token")) and _nonempty_str(payload.get("access_token")))
    return payload if has_tokens else None


def _clear_shared_nous_state(reason: str) -> None:
    """Remove the shared Nous OAuth store after a terminal token failure."""
    try:
        with _nous_shared_store_lock():
            _nous_shared_store_path().unlink(missing_ok=True)
        _oauth_trace("nous_shared_store_cleared", reason=reason)
    except Exception as exc:
        logger.debug("Failed to clear shared Nous auth store: %s", exc)


def _quarantine_forensics(state: Dict[str, Any], error: AuthError, reason: str) -> Dict[str, Any]:
    """Redaction-safe forensic record for a quarantine: fingerprints, sizes and booleans only.

    NEVER include a raw token/agent_key (credential-shaped literals get corrupted in logs). The
    12-char SHA-256 prefix correlates to NAS's refreshTokenHash without leaking the secret;
    provenance is client_id + agent_key_id (Nous state has no session_id).
    """
    from hermes_cli.auth import _auth_file_path
    forensic: Dict[str, Any] = {
        "reason": reason, "error_code": error.code, "client_id": state.get("client_id"),
        "agent_key_id": state.get("agent_key_id"),
        "refresh_token_fp": _token_fingerprint(state.get("refresh_token"))}
    # On-disk integrity of the auth store at the moment of quarantine.
    try:
        auth_path = _auth_file_path()
        forensic["auth_json_path"] = str(auth_path)
        try:
            st = os.stat(auth_path)
            forensic.update(
                auth_json_size=st.st_size, auth_json_mtime=st.st_mtime, auth_json_exists=True)
        except FileNotFoundError:
            forensic["auth_json_exists"] = False
    except Exception as exc:  # pragma: no cover - never let logging break quarantine
        forensic["auth_json_stat_error"] = repr(exc)
    # Was the token already past its own expiry when it was rejected?
    already_expired: Optional[bool] = None
    expires_at_raw = state.get("expires_at")
    if isinstance(expires_at_raw, str) and expires_at_raw:
        try:
            parsed = datetime.fromisoformat(expires_at_raw)
            already_expired = (parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)
                               < datetime.now(timezone.utc))
        except ValueError:
            already_expired = None
    forensic["token_already_expired"] = already_expired
    return forensic


def _quarantine_nous_oauth_state(state: Dict[str, Any], error: AuthError, *, reason: str) -> None:
    """Keep routing metadata but remove dead OAuth material so it is not replayed.

    Only for terminal errors (``*_refresh_failed`` = HTTP 400/401/403 invalid_grant / revoked /
    refresh_token_reused; ``*_auth_missing_refresh_token``), all ``relogin_required=True`` —
    transient 429/5xx never quarantine.
    """
    from hermes_cli.auth import (
        _FLAT_OAUTH_TOKEN_KEYS, _last_auth_error_marker, invalidate_nous_auth_status_cache)
    # Forensics BEFORE clearing token material: a hosted agent quarantined here is otherwise only
    # visible as a later "No access token found" WARNING, too late to root-cause. Managed log
    # drains may be WARNING-only, so this MUST be logger.warning.
    logger.warning(
        "Nous OAuth state quarantined (terminal auth death): %s",
        json.dumps(_quarantine_forensics(state, error, reason), sort_keys=True, ensure_ascii=False))
    for key in (*_FLAT_OAUTH_TOKEN_KEYS, *_NOUS_EMPTY_AGENT_KEY_FIELDS):
        state.pop(key, None)
    state["last_auth_error"] = _last_auth_error_marker("nous", error, reason=reason)
    _clear_shared_nous_state(reason)
    invalidate_nous_auth_status_cache()


def _quarantine_nous_pool_entries(
    auth_store: Dict[str, Any], error: AuthError, *, reason: str) -> bool:
    """Remove singleton-seeded Nous pool entries that contain dead OAuth state."""
    entries = _pool_entries(auth_store, "nous")
    if entries is None:
        return False
    singleton_sources = {NOUS_DEVICE_CODE_SOURCE, f"manual:{NOUS_DEVICE_CODE_SOURCE}"}
    retained = [
        e for e in entries if not (isinstance(e, dict) and e.get("source") in singleton_sources)]
    removed = len(retained) != len(entries)
    if removed:
        auth_store["credential_pool"]["nous"] = retained
        _oauth_trace("nous_pool_device_code_quarantined", reason=reason, error_code=error.code)
    return removed


def _try_import_shared_nous_state(*, timeout_seconds: float = 15.0) -> Optional[Dict[str, Any]]:
    """Rehydrate Nous OAuth state from the shared store via a forced refresh.

    Returns auth_state ready for ``persist_nous_credentials()``; None on any failure (expired
    token, portal unreachable) so the caller falls through to device-code.
    """
    from hermes_cli.auth import (
        _read_shared_nous_state, _write_shared_nous_state, refresh_nous_oauth_from_state,
        _is_terminal_nous_refresh_error)
    try:
        with _nous_shared_store_lock(timeout_seconds=_shared_lock_timeout(timeout_seconds)):
            shared = _read_shared_nous_state()
            if not shared:
                return None
            # Full state dict so refresh_nous_oauth_from_state has every field it needs.
            state: Dict[str, Any] = {
                **_nous_shared_shape(shared), "agent_key": None, "agent_key_expires_at": None,
                "tls": {"insecure": False, "ca_bundle": None}}
            refreshed = refresh_nous_oauth_from_state(
                state, timeout_seconds=timeout_seconds, force_refresh=True,
                on_state_update=lambda updated, _reason: _write_shared_nous_state(updated))
            _write_shared_nous_state(refreshed)
    except Exception as exc:
        is_auth = isinstance(exc, AuthError)
        _oauth_trace(
            "nous_shared_import_failed", error_type=type(exc).__name__,
            **({"error_code": getattr(exc, "code", None)} if is_auth else {}))
        if is_auth and _is_terminal_nous_refresh_error(exc):
            _clear_shared_nous_state("shared_import_terminal_refresh_failure")
        logger.debug("Shared Nous import failed: %s", exc)
        return None
    return refreshed


def _refresh_access_token(
    *, client: httpx.Client, portal_base_url: str, client_id: str, refresh_token: str,
) -> Dict[str, Any]:
    response = client.post(
        f"{portal_base_url}/api/oauth/token",
        headers={"x-nous-refresh-token": refresh_token},
        data={"grant_type": "refresh_token", "client_id": client_id})
    if response.status_code == 200:
        payload = response.json()
        if "access_token" not in payload:
            raise _nous_err("Refresh response missing access_token", "invalid_token", relogin=True)
        return payload
    try:
        error_payload = response.json()
    except Exception as exc:
        raise _nous_err("Refresh token exchange failed", relogin=True) from exc
    code = str(error_payload.get("error", "invalid_grant"))
    description = str(error_payload.get("error_description") or "Refresh token exchange failed")
    relogin = code in {"invalid_grant", "invalid_token", "refresh_token_reused"}
    # OAuth 2.1 "refresh token reuse": an external process (health check, monitoring tool, custom
    # self-heal hook) redeemed Hermes's refresh_token without persisting the rotated token, so the
    # server retired the original and revoked the whole session chain as a token-theft signal.
    if code == "refresh_token_reused" or "reuse" in description.lower():
        description = (
            "Nous Portal detected refresh-token reuse and revoked this session.\n"
            "This usually means an external process (monitoring script, "
            "custom self-heal hook, or another Hermes install sharing "
            "~/.hermes/auth.json) called POST /api/oauth/token with Hermes's "
            "refresh token without persisting the rotated token back.\n"
            "Nous refresh tokens are single-use — only Hermes may call the "
            "refresh endpoint. For health checks, use `hermes auth status` "
            "instead.\n"
            "Re-authenticate with: hermes auth add nous")
        relogin = True
    raise _nous_err(description, code, relogin=relogin)


def _refresh_nous_or_quarantine(
    *, client: httpx.Client, auth_store: Dict[str, Any], state: Dict[str, Any],
    portal_base_url: str, client_id: str, refresh_token: str, reason: str,
    persist: Callable[[], None]) -> Dict[str, Any]:
    """Redeem the refresh token; on terminal failure quarantine state + pool, persist, re-raise."""
    from hermes_cli.auth import _refresh_access_token, _is_terminal_nous_refresh_error
    try:
        return _refresh_access_token(
            client=client, portal_base_url=portal_base_url, client_id=client_id,
            refresh_token=refresh_token)
    except AuthError as exc:
        if _is_terminal_nous_refresh_error(exc):
            _quarantine_nous_oauth_state(state, exc, reason=reason)
            _quarantine_nous_pool_entries(auth_store, exc, reason=reason)
            persist()
        raise


def _apply_nous_refreshed_tokens(
    state: Dict[str, Any], refreshed: Dict[str, Any], refresh_token: str, *,
    inference_base_url: Optional[str] = None) -> None:
    """Write a successful Nous token-refresh payload into *state* (tokens + expiry fields).

    *inference_base_url*, when given, is the healed network-provenance URL to persist alongside
    the rotated tokens (key order in auth.json is preserved from the original login shape).
    """
    from hermes_cli.auth import _coerce_ttl_seconds
    now = datetime.now(timezone.utc)
    access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
    state["access_token"] = refreshed["access_token"]
    state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
    state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
    state["scope"] = refreshed.get("scope") or state.get("scope")
    if inference_base_url is not None:
        state["inference_base_url"] = inference_base_url
    state["obtained_at"] = now.isoformat()
    state["expires_in"] = access_ttl
    state["expires_at"] = _iso_after(now, access_ttl)


def _healed_nous_inference_url(refreshed: Dict[str, Any]) -> str:
    """Validated network-provenance inference URL from a refresh payload, healed to the default.

    A Portal URL rejected by the allowlist resets to the production default instead of leaving a
    previously-persisted bad host (e.g. a stale staging URL) in place — otherwise a poisoned
    auth.json re-validates to None on every refresh and silently re-uses the dead endpoint.
    """
    url = _validate_nous_inference_url_from_network(refreshed.get("inference_base_url"))
    return url or DEFAULT_NOUS_INFERENCE_URL


def _nous_http_client(timeout_seconds: float, verify: Any) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(timeout_seconds), headers={"Accept": "application/json"},
        verify=verify)


def _model_priority(mid: str) -> tuple:
    """Sort key: opus > pro > haiku/flash > sonnet (sonnet is cheap/fast; best model first)."""
    low = mid.lower()
    rank = (0 if "opus" in low else 1 if "pro" in low and "sonnet" not in low
            else 3 if "sonnet" in low else 2)
    return (rank, mid)


def fetch_nous_models(
    *, inference_base_url: str, api_key: str, timeout_seconds: float = 15.0,
    verify: bool | str = True) -> List[str]:
    """Fetch available model IDs from the Nous inference API."""
    from hermes_cli.auth import _nonempty_str
    with _nous_http_client(timeout_seconds, verify) as client:
        response = client.get(
            f"{inference_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"})
    if response.status_code != 200:
        description = f"/models request failed with status {response.status_code}"
        try:
            err = response.json()
            description = str(err.get("error_description") or err.get("error") or description)
        except Exception as e:
            logger.debug("Could not parse error response JSON: %s", e)
        raise _nous_err(description, "models_fetch_failed")
    data = response.json().get("data")
    if not isinstance(data, list):
        return []
    model_ids: List[str] = []
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else None
        # Hermes models aren't reliable for agentic tool-calling
        if _nonempty_str(model_id) and "hermes" not in model_id.lower():
            model_ids.append(model_id.strip())
    model_ids.sort(key=_model_priority)
    return list(dict.fromkeys(model_ids))


def _agent_key_is_usable(state: Dict[str, Any], min_ttl_seconds: int) -> bool:
    from hermes_cli.auth import _nonempty_str
    key = state.get("agent_key")
    return _nonempty_str(key) and _nous_invoke_jwt_is_usable(
        key, scope=state.get("scope"), expires_at=state.get("agent_key_expires_at"),
        min_ttl_seconds=max(0, int(min_ttl_seconds)))


def refresh_nous_oauth_pure(
    access_token: str, refresh_token: str, client_id: str, portal_base_url: str,
    inference_base_url: str, *, token_type: str = "Bearer", scope: str = DEFAULT_NOUS_SCOPE,
    obtained_at: Optional[str] = None, expires_at: Optional[str] = None,
    agent_key: Optional[str] = None, agent_key_expires_at: Optional[str] = None,
    timeout_seconds: float = 15.0, insecure: Optional[bool] = None, ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    on_state_update: Optional[Callable[[Dict[str, Any], str], None]] = None) -> Dict[str, Any]:
    """Refresh Nous OAuth state without mutating auth.json directly.

    ``on_state_update`` fires after a successful access-token refresh so callers owning persistent
    state can save the rotated refresh token before later validation can fail.
    """
    return refresh_nous_oauth_from_state(
        {
            "access_token": access_token, "refresh_token": refresh_token, "client_id": client_id,
            "portal_base_url": portal_base_url, "inference_base_url": inference_base_url,
            "token_type": token_type, "scope": scope, "obtained_at": obtained_at,
            "expires_at": expires_at, "agent_key": agent_key,
            "agent_key_expires_at": agent_key_expires_at,
            "tls": {"insecure": insecure, "ca_bundle": ca_bundle}},
        timeout_seconds=timeout_seconds, force_refresh=force_refresh,
        on_state_update=on_state_update)


def refresh_nous_oauth_from_state(
    src: Dict[str, Any], *, timeout_seconds: float = 15.0, force_refresh: bool = False,
    on_state_update: Optional[Callable[[Dict[str, Any], str], None]] = None) -> Dict[str, Any]:
    """Refresh Nous OAuth from a state dict (defaults filled in) without mutating auth.json."""
    from hermes_cli.auth import (
        _assert_nous_inference_jwt_usable, _refresh_access_token, _resolve_verify,
        _select_nous_invoke_jwt)
    tls = src.get("tls") or {}
    insecure, ca_bundle = tls.get("insecure"), tls.get("ca_bundle")
    state: Dict[str, Any] = {
        "access_token": src.get("access_token", ""), "refresh_token": src.get("refresh_token", ""),
        "client_id": src.get("client_id") or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": (src.get("portal_base_url") or DEFAULT_NOUS_PORTAL_URL).rstrip("/"),
        "inference_base_url": (
            src.get("inference_base_url") or DEFAULT_NOUS_INFERENCE_URL).rstrip("/"),
        "token_type": src.get("token_type") or "Bearer",
        "scope": src.get("scope") or DEFAULT_NOUS_SCOPE,
        "obtained_at": src.get("obtained_at"), "expires_at": src.get("expires_at"),
        "agent_key": src.get("agent_key"), "agent_key_expires_at": src.get("agent_key_expires_at"),
        "tls": {"insecure": bool(insecure), "ca_bundle": ca_bundle}}
    verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
    with _nous_http_client(timeout_seconds or 15.0, verify) as client:
        current_invoke_jwt_status = _state_invoke_jwt_status(state, state.get("access_token"))
        if force_refresh or current_invoke_jwt_status is not None:
            refresh_token_value = state.get("refresh_token")
            if not isinstance(refresh_token_value, str) or not refresh_token_value:
                if current_invoke_jwt_status is not None:
                    raise _unusable_invoke_jwt_error(
                        current_invoke_jwt_status, no_refresh_token=True)
                raise _nous_err("No refresh token is available for Nous Portal.", relogin=True)
            refreshed = _refresh_access_token(
                client=client, portal_base_url=state["portal_base_url"],
                client_id=state["client_id"], refresh_token=refresh_token_value)
            _apply_nous_refreshed_tokens(
                state, refreshed, refresh_token_value,
                inference_base_url=_healed_nous_inference_url(refreshed))
            if on_state_update is not None:
                on_state_update(dict(state), "post_refresh_access_token")
        _assert_nous_inference_jwt_usable(state)
        _select_nous_invoke_jwt(state)
    return state


def persist_nous_credentials(creds: Dict[str, Any], *, label: Optional[str] = None):
    """Persist Nous OAuth credentials as the singleton provider state.

    Nous credentials are read from ``providers.nous`` (401 recovery, pool seeding) AND
    ``credential_pool.nous`` (runtime ``pool.select()``); a pool-only write broke expiry recovery.
    So: write the singleton, mirror to the shared store, then ``load_pool("nous")`` upserts the
    canonical ``device_code`` entry in place. ``label`` rides in the singleton so re-seeding keeps
    it.
    """
    from hermes_cli.auth import _save_active_provider_state, _write_shared_nous_state
    from agent.credential_pool import load_pool
    state = dict(creds)
    if label and str(label).strip():
        state["label"] = str(label).strip()
    _save_active_provider_state("nous", state)
    _write_shared_nous_state(state)
    pool = load_pool("nous")
    return next((e for e in pool.entries() if e.source == NOUS_DEVICE_CODE_SOURCE), None)


def _sync_nous_pool_from_auth_store() -> None:
    """Best-effort pool reseed after providers.nous changes; never fail login."""
    try:
        from agent.credential_pool import load_pool
        load_pool("nous")
    except Exception as exc:
        logger.debug("Failed to sync Nous credential pool from auth store: %s", exc)


def _nous_effective_routing(state: Dict[str, Any]) -> tuple[str, str, str, str]:
    """``(portal_url, stored_inference_url, effective_inference_url, client_id)`` from *state*.

    The stored inference URL is re-validated network-provenance (persisted); the effective one
    layers the runtime-only ``NOUS_INFERENCE_BASE_URL`` override on top and is never persisted.
    """
    from hermes_cli.auth import _NOUS_PORTAL_ALLOWED_HOSTS, _optional_base_url
    portal_url = (
        _optional_base_url(state.get("portal_base_url")) or os.getenv("HERMES_PORTAL_BASE_URL")
        or os.getenv("NOUS_PORTAL_BASE_URL") or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
    # A persisted/stale portal_base_url is where the refresh token gets POSTed — reject any host
    # outside the allowlist so a poisoned value can't exfiltrate the bearer, healing to the
    # default. Trusted operator env overrides bypass this network-value gate.
    env_portal_override = _nous_portal_env_override()
    if env_portal_override:
        portal_url = env_portal_override.rstrip("/")
    else:
        parsed_portal_url = urlparse(portal_url)
        portal_host, scheme = parsed_portal_url.hostname, parsed_portal_url.scheme
        trusted_scheme = scheme == "https" or (
            scheme == "http" and portal_host in {"localhost", "127.0.0.1"})
        if not portal_host or portal_host not in _NOUS_PORTAL_ALLOWED_HOSTS or not trusted_scheme:
            logger.warning(
                "auth: ignoring invalid portal_base_url %r "
                "(host %r or scheme not allowed), using default",
                portal_url, portal_host)
            portal_url = DEFAULT_NOUS_PORTAL_URL
    stored_inference_url = (
        _validate_nous_inference_url_from_network(
            _optional_base_url(state.get("inference_base_url")))
        or DEFAULT_NOUS_INFERENCE_URL)
    return (
        portal_url, stored_inference_url, _nous_inference_env_override() or stored_inference_url,
        str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID))


class _NousRuntimeResolve:
    """Working set for one ``resolve_nous_runtime_credentials`` call.

    Holds the token pair + routing tuple that shared-store merges / refreshes replace mid-flight.
    ``persist`` skips writes where only derived TTL countdowns changed (keeps the mtime-keyed
    auth-status cache warm) and mirrors every real write to the shared store (best-effort).
    """

    def __init__(
        self, auth_store: Dict[str, Any], state: Dict[str, Any], state_source_path: Optional[Path],
        *, force_refresh: bool, stale_access_token: Optional[str], timeout_seconds: float) -> None:
        self.auth_store, self.state, self._source_path = auth_store, state, state_source_path
        self.force_refresh, self.stale_access_token = force_refresh, stale_access_token
        self.timeout_seconds = timeout_seconds
        self.sequence_id = uuid.uuid4().hex[:12]
        self._persisted_state = dict(state)
        self.persisted_any = False
        self.access_token = state.get("access_token")
        self.refresh_token = state.get("refresh_token")
        self._reload_routing()

    def _reload_routing(self) -> None:
        (self.portal_base_url, self.stored_inference_base_url, self.inference_base_url,
         self.client_id) = _nous_effective_routing(self.state)

    def persist(self, reason: str) -> None:
        from hermes_cli.auth import _save_provider_state_to_source, _write_shared_nous_state
        state = self.state
        persisted = _nous_effective_provider_state(self._persisted_state)
        if _nous_effective_provider_state(state) == persisted:
            _oauth_trace("nous_state_persist_skipped", sequence_id=self.sequence_id, reason=reason)
            return
        try:
            _save_provider_state_to_source(self.auth_store, "nous", state, self._source_path)
        except Exception as exc:
            _oauth_trace(
                "nous_state_persist_failed", sequence_id=self.sequence_id, reason=reason,
                error_type=type(exc).__name__)
            raise
        _oauth_trace(
            "nous_state_persisted", sequence_id=self.sequence_id, reason=reason,
            refresh_token_fp=_token_fingerprint(state.get("refresh_token")),
            access_token_fp=_token_fingerprint(state.get("access_token")))
        self._persisted_state = dict(state)
        self.persisted_any = True
        _write_shared_nous_state(state)

    def shared_lock(self):
        return _nous_shared_store_lock(timeout_seconds=_shared_lock_timeout(self.timeout_seconds))

    def has_access_token(self) -> bool:
        return isinstance(self.access_token, str) and bool(self.access_token)

    def invoke_jwt_status(self) -> Optional[str]:
        return _state_invoke_jwt_status(self.state, self.access_token)

    def merge_shared(self) -> bool:
        """Adopt fresher shared-store tokens (caller holds the shared lock). True when merged."""
        if not _merge_shared_nous_oauth_state(self.state):
            return False
        self.access_token = self.state.get("access_token")
        self.refresh_token = self.state.get("refresh_token")
        self._reload_routing()
        return True

    def skip_refresh_if_peer_rotated(self) -> None:
        """Skip the refresh when a peer already rotated the grant.

        Under the store lock: if the bearer that failed upstream is no longer the one on disk and
        the on-disk one is usable, adopt it — never re-POST the shared grant.
        """
        token = self.access_token
        if (self.force_refresh and self.stale_access_token and isinstance(token, str) and token
                and token != self.stale_access_token and self.invoke_jwt_status() is None):
            _oauth_trace(
                "refresh_skipped_peer_rotated", sequence_id=self.sequence_id,
                access_token_fp=_token_fingerprint(token))
            self.force_refresh = False

    def refresh(self, client: httpx.Client, invoke_jwt_status: Optional[str]) -> None:
        """Redeem the refresh token, apply + persist the rotated pair (caller holds both locks)."""
        if not isinstance(self.refresh_token, str) or not self.refresh_token:
            raise _unusable_invoke_jwt_error(
                invoke_jwt_status or "force_refresh", no_refresh_token=True)
        refresh_reason = (
            "force_refresh" if self.force_refresh else (invoke_jwt_status or "access_unusable"))
        _oauth_trace(
            "refresh_start", sequence_id=self.sequence_id, reason=refresh_reason,
            refresh_token_fp=_token_fingerprint(self.refresh_token))
        refreshed = _refresh_nous_or_quarantine(
            client=client, auth_store=self.auth_store, state=self.state,
            portal_base_url=self.portal_base_url, client_id=self.client_id,
            refresh_token=self.refresh_token, reason="runtime_access_refresh_failure",
            persist=lambda: self.persist("terminal_runtime_access_refresh_failure"))
        previous_refresh_token = self.refresh_token
        # The validated, network-provenance URL is what gets persisted (with the rotated tokens,
        # so a later JWT validation failure cannot leave the stores on stale metadata). The
        # NOUS_INFERENCE_BASE_URL env override is layered on for the client/return value only.
        self.stored_inference_base_url = _healed_nous_inference_url(refreshed)
        self.inference_base_url = _nous_inference_env_override() or self.stored_inference_base_url
        _apply_nous_refreshed_tokens(
            self.state, refreshed, self.refresh_token,
            inference_base_url=self.stored_inference_base_url)
        self.access_token = self.state["access_token"]
        self.refresh_token = self.state["refresh_token"]
        _oauth_trace(
            "refresh_success", sequence_id=self.sequence_id, reason=refresh_reason,
            previous_refresh_token_fp=_token_fingerprint(previous_refresh_token),
            new_refresh_token_fp=_token_fingerprint(self.refresh_token))
        # Persist immediately so validation failures cannot drop rotated refresh tokens.
        self.persist("post_refresh_access_token")

    def ensure_usable_access_token(self, client: httpx.Client) -> None:
        """Merge from the shared store / refresh until the access token is a usable invoke JWT."""
        if not self.has_access_token():
            with self.shared_lock():
                if self.merge_shared():
                    self.persist("runtime_shared_merge_missing_access_token")
        if not self.has_access_token():
            raise _nous_err("No access token found for Nous Portal login.", relogin=True)
        invoke_jwt_status = self.invoke_jwt_status()
        self.skip_refresh_if_peer_rotated()
        if not (self.force_refresh or invoke_jwt_status is not None):
            return
        with self.shared_lock():
            if self.merge_shared():
                invoke_jwt_status = self.invoke_jwt_status()
                self.persist("post_shared_merge_access_unusable")
                self.skip_refresh_if_peer_rotated()
            if self.force_refresh or invoke_jwt_status is not None:
                self.refresh(client, invoke_jwt_status)


def resolve_nous_runtime_credentials(
    *, timeout_seconds: float = 15.0, insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None, force_refresh: bool = False,
    stale_access_token: Optional[str] = None) -> Dict[str, Any]:
    """Resolve Nous inference credentials for runtime use (refreshing under the auth-store lock).

    ``stale_access_token`` is the bearer that just failed upstream (401): with ``force_refresh``,
    the refresh POST is skipped if the store (re-read under the lock) already holds a *different*
    usable token — a peer won the rotation; adopt it rather than invalidate a sibling's token.
    """
    from hermes_cli.auth import (
        _assert_nous_inference_jwt_usable, _auth_file_path, _provider_state_transaction,
        _resolve_verify, _select_nous_invoke_jwt, _sync_nous_pool_from_auth_store,
        _tls_state_from_verify)
    with _provider_state_transaction("nous") as (auth_store, state, state_source_path):
        if not state:
            raise _nous_err("Hermes is not logged into Nous Portal.", relogin=True)
        run = _NousRuntimeResolve(
            auth_store, state, state_source_path, force_refresh=force_refresh,
            stale_access_token=stale_access_token, timeout_seconds=timeout_seconds)
        verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
        _oauth_trace(
            "nous_runtime_credentials_start", sequence_id=run.sequence_id,
            refresh_token_fp=_token_fingerprint(state.get("refresh_token")))
        with _nous_http_client(timeout_seconds or 15.0, verify) as client:
            run.ensure_usable_access_token(client)
            _assert_nous_inference_jwt_usable(state, access_token=run.access_token)
            _select_nous_invoke_jwt(
                state, access_token=run.access_token, sequence_id=run.sequence_id)
            # Persist routing and TLS metadata for non-interactive refresh — the validated,
            # network-provenance URL, NEVER the env override (a runtime-only overlay; persisting
            # it would leak a dev/staging host into auth.json and survive unsetting it).
            state.update(
                portal_base_url=run.portal_base_url, client_id=run.client_id,
                inference_base_url=run.stored_inference_base_url,
                tls=_tls_state_from_verify(verify))
        run.persist("resolve_nous_runtime_credentials_final")
    if run.persisted_any:
        _sync_nous_pool_from_auth_store()
    api_key = state.get("agent_key")
    if not isinstance(api_key, str) or not api_key:
        raise _nous_err("Failed to resolve a Nous inference API key", "server_error")
    expires_at = state.get("agent_key_expires_at")
    return {
        "provider": "nous", "base_url": run.inference_base_url, "api_key": api_key,
        "key_id": state.get("agent_key_id"), "expires_at": expires_at,
        "expires_in": _remaining_ttl(expires_at, state.get("agent_key_expires_in")),
        "source": NOUS_AUTH_PATH_INVOKE_JWT,
        # Public semantic source label; the concrete store is exposed separately for diagnostics.
        # Refresh persistence uses state_source_path internally and must not overload this field.
        "auth_path": NOUS_AUTH_PATH_INVOKE_JWT,
        "state_path": str(state_source_path or _auth_file_path())}


def _empty_nous_auth_status() -> Dict[str, Any]:
    return {
        "logged_in": False, "portal_base_url": None, "inference_base_url": None,
        "access_expires_at": None, "agent_key_expires_at": None, "has_refresh_token": False,
        "inference_credential_present": False, "credential_source": None}


def _snapshot_nous_pool_status() -> Dict[str, Any]:
    """Best-effort status from the credential pool.

    Fallback only: the auth-store provider state is the runtime source of truth because it is what
    ``resolve_nous_runtime_credentials()`` refreshes.
    """
    from hermes_cli.auth import _parse_iso_timestamp
    try:
        from agent.credential_pool import load_pool
        pool = load_pool("nous")
        entries = list(pool.entries()) if pool and pool.has_credentials() else []
        if not entries:
            return _empty_nous_auth_status()
        entry = max(entries, key=lambda e: (
            _parse_iso_timestamp(getattr(e, "agent_key_expires_at", None)) or 0.0,
            _parse_iso_timestamp(getattr(e, "expires_at", None)) or 0.0,
            -int(getattr(e, "priority", 0) or 0)))
        attr = lambda name, default=None: getattr(entry, name, default)  # noqa: E731
        if not attr("runtime_api_key"):
            return _empty_nous_auth_status()
        access_token, refresh_token = attr("access_token"), attr("refresh_token")
        auth_type = str(attr("auth_type", "") or "").strip().lower()
        is_portal_oauth = bool(access_token) and (
            auth_type.startswith("oauth") or bool(refresh_token))
        label = attr("label", "unknown")
        return {
            "logged_in": is_portal_oauth,
            "portal_base_url": (
                (attr("portal_base_url") or DEFAULT_NOUS_PORTAL_URL) if is_portal_oauth else None),
            "inference_base_url": (
                attr("inference_base_url") or attr("runtime_base_url") or attr("base_url")),
            "access_token": access_token if is_portal_oauth else None,
            "access_expires_at": attr("expires_at"),
            "agent_key_expires_at": attr("agent_key_expires_at"),
            "has_refresh_token": bool(refresh_token), "inference_credential_present": True,
            "credential_source": f"pool:{label}", "source": f"pool:{label}"}
    except Exception:
        return _empty_nous_auth_status()


def _nous_status_from_state(
    state: Dict[str, Any], *, logged_in: bool, source: str) -> Dict[str, Any]:
    """Auth-store-backed Nous status snapshot (shared by the live and refresh-free variants)."""
    access_token = state.get("access_token")
    return {
        "logged_in": logged_in, "portal_base_url": state.get("portal_base_url"),
        "inference_base_url": state.get("inference_base_url"),
        "access_expires_at": state.get("expires_at"),
        "agent_key_expires_at": state.get("agent_key_expires_at"),
        "has_refresh_token": bool(state.get("refresh_token")), "access_token": access_token,
        "inference_credential_present": bool(access_token or state.get("agent_key")),
        "credential_source": "auth_store", "source": source}


def _compute_nous_auth_status() -> Dict[str, Any]:
    """Uncached implementation of get_nous_auth_status(). See that function."""
    from hermes_cli.auth import get_provider_auth_state, resolve_nous_runtime_credentials
    state = get_provider_auth_state("nous")
    if not state:
        return _snapshot_nous_pool_status()
    base_status = _nous_status_from_state(
        state, logged_in=bool(state.get("access_token")), source="auth_store")
    try:
        creds = resolve_nous_runtime_credentials()
        refreshed_state = get_provider_auth_state("nous") or state
        base_status.update({
            "logged_in": True,
            "portal_base_url": (
                refreshed_state.get("portal_base_url") or base_status.get("portal_base_url")),
            "inference_base_url": (
                creds.get("base_url") or refreshed_state.get("inference_base_url")
                or base_status.get("inference_base_url")),
            "access_expires_at": (
                refreshed_state.get("expires_at") or base_status.get("access_expires_at")),
            "agent_key_expires_at": (
                creds.get("expires_at") or refreshed_state.get("agent_key_expires_at")
                or base_status.get("agent_key_expires_at")),
            "has_refresh_token": bool(refreshed_state.get("refresh_token")),
            "inference_credential_present": True, "credential_source": "auth_store",
            "source": f"runtime:{creds.get('source', 'portal')}", "key_id": creds.get("key_id")})
    except AuthError as exc:
        base_status.update({
            "logged_in": False, "error": str(exc),
            "relogin_required": bool(getattr(exc, "relogin_required", False)),
            "error_code": getattr(exc, "code", None)})
    return base_status


def _terminal_quarantine_marker(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The persisted ``last_auth_error`` when it is a terminal quarantine with no credential left.

    Only terminal while there is no usable credential: if a later login repopulated tokens the stale
    marker must not keep reporting terminal.
    """
    last_err = state.get("last_auth_error")
    if (isinstance(last_err, dict) and last_err.get("relogin_required")
            and not (state.get("access_token") or state.get("refresh_token"))):
        return last_err
    return None


def get_nous_auth_status_local() -> Dict[str, Any]:
    """Refresh-free Nous auth snapshot for read-only display surfaces.

    NEVER calls ``resolve_nous_runtime_credentials()`` (no refresh POST / single-use token spent);
    ``logged_in`` = usable invoke JWT, or a refresh token not terminally quarantined — not proof
    the server still accepts it.
    """
    from hermes_cli.auth import get_provider_auth_state
    try:
        state = get_provider_auth_state("nous")
    except Exception:
        state = None
    if not state:
        return _snapshot_nous_pool_status()
    jwt_reason = _state_invoke_jwt_status(state, state.get("access_token"))
    last_err = _terminal_quarantine_marker(state)
    logged_in = (jwt_reason is None) or (bool(state.get("refresh_token")) and last_err is None)
    status = _nous_status_from_state(state, logged_in=logged_in, source="auth_store_local")
    if last_err is not None:
        status.update(
            relogin_required=True, error_code=last_err.get("code"),
            error=last_err.get("message") or "re-login required")
    return status


# Enum values reported on the dashboard /api/status as ``nous_session_valid``. NAS's health sweep
# re-mints the bootstrap session ONLY on "terminal"; "valid" and "unknown" are no-ops. Keep this
# set small and stable — NAS parses it permissively, so new members are non-breaking but rare.
NOUS_SESSION_VALID = "valid"
NOUS_SESSION_TERMINAL = "terminal"
NOUS_SESSION_UNKNOWN = "unknown"


def get_nous_session_validity() -> str:
    """Classify the Nous bootstrap session for the dashboard /api/status probe.

    Local auth-store state only; polled frequently, so it never resolves or refreshes. ANTI-FLAP:
    only a *terminal* failure maps to "terminal" — a rotation blip, network error, or expiring
    token must NOT (that would trigger a spurious NAS re-mint on a healthy box).
    """
    from hermes_cli.auth import get_provider_auth_state
    try:
        state = get_provider_auth_state("nous")
    except Exception:
        state = None
    if not state:
        return NOUS_SESSION_UNKNOWN
    # The persisted quarantine marker (`last_auth_error.relogin_required=True`, written when the
    # refresh path clears dead tokens) is the strongest, most stable terminal signal — report
    # "terminal" even after the in-memory AuthError is long gone.
    if _terminal_quarantine_marker(state) is not None:
        return NOUS_SESSION_TERMINAL
    if _state_invoke_jwt_status(state, state.get("access_token")) is None:
        return NOUS_SESSION_VALID
    # Missing, malformed, expired, or merely expiring credentials are not proof of a terminal
    # session. Runtime paths own refreshes; the health endpoint stays side-effect free.
    return NOUS_SESSION_UNKNOWN


def _pool_first_oauth_status(
    provider_id: str, *, is_expiring: Callable[[str, int], bool], auth_mode: str,
    resolve: Callable[[], Dict[str, Any]],
    on_pool_miss: Optional[Callable[[], Optional[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """Status snapshot for a store-backed OAuth provider (Codex, xAI).

    Pool first (where `hermes auth` / `hermes model` store device_code tokens), then
    *on_pool_miss* for a pool-derived degraded status, then the legacy state via *resolve*.
    """
    from hermes_cli.auth import _auth_file_path
    try:
        from agent.credential_pool import load_pool
        pool = load_pool(provider_id)
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                api_key = (
                    getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", ""))
                if api_key and not is_expiring(api_key, 0):
                    return {
                        "logged_in": True, "auth_store": str(_auth_file_path()),
                        "last_refresh": getattr(entry, "last_refresh", None),
                        "auth_mode": auth_mode,
                        "source": f"pool:{getattr(entry, 'label', 'unknown')}", "api_key": api_key}
            if on_pool_miss is not None and (degraded := on_pool_miss()):
                return degraded
    except Exception:
        pass
    try:
        creds = resolve()
        return {
            "logged_in": True, "auth_store": str(_auth_file_path()),
            "last_refresh": creds.get("last_refresh"),
            "auth_mode": creds.get("auth_mode"), "source": creds.get("source"),
            "api_key": creds.get("api_key")}
    except AuthError as exc:
        return {"logged_in": False, "auth_store": str(_auth_file_path()), "error": str(exc)}


def _nous_device_code_login(
    *, portal_base_url: Optional[str] = None, inference_base_url: Optional[str] = None,
    client_id: Optional[str] = None, scope: Optional[str] = None, open_browser: bool = True,
    timeout_seconds: float = 15.0, insecure: bool = False, ca_bundle: Optional[str] = None,
    on_verification: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
    """Run the Nous device-code flow and return full OAuth state without persisting."""
    from hermes_cli.auth import (
        PROVIDER_REGISTRY, _coerce_ttl_seconds, _is_remote_session, _optional_base_url,
        _poll_for_token, _print_device_code_instructions, _request_device_code,
        _tls_state_from_verify, format_auth_error, refresh_nous_oauth_from_state)
    pconfig = PROVIDER_REGISTRY["nous"]
    portal_base_url = (
        portal_base_url or os.getenv("HERMES_PORTAL_BASE_URL") or os.getenv("NOUS_PORTAL_BASE_URL")
        or pconfig.portal_base_url).rstrip("/")
    requested_inference_url = (
        inference_base_url or os.getenv("NOUS_INFERENCE_BASE_URL")
        or pconfig.inference_base_url).rstrip("/")
    client_id = client_id or pconfig.client_id
    scope = scope or pconfig.scope
    verify: bool | str = False if insecure else (ca_bundle if ca_bundle else True)
    if _is_remote_session():
        open_browser = False
    print(f"Starting Hermes login via {pconfig.name}...")
    print(f"Portal: {portal_base_url}")
    if insecure:
        print("TLS verification: disabled (--insecure)")
    elif ca_bundle:
        print(f"TLS verification: custom CA bundle ({ca_bundle})")
    with _nous_http_client(timeout_seconds, verify) as client:
        device_data = _request_device_code(
            client=client, portal_base_url=portal_base_url, client_id=client_id, scope=scope)
        verification_url = str(device_data["verification_uri_complete"])
        user_code = str(device_data["user_code"])
        expires_in = int(device_data["expires_in"])
        interval = int(device_data["interval"])
        _print_device_code_instructions(
            verification_url, user_code, open_browser=open_browser, failure_dash="—")
        # Out-of-band consumer (e.g. the TUI gateway, whose stdout is a JSON-RPC pipe): fired AFTER
        # the print/browser block and BEFORE polling so it can render the link while we wait.
        if on_verification is not None:
            with suppress(Exception):
                on_verification(verification_url, user_code)
        effective_interval = max(1, min(interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS))
        print(f"Waiting for approval (polling every {effective_interval}s)...")
        token_data = _poll_for_token(
            client=client, portal_base_url=portal_base_url, client_id=client_id,
            device_code=str(device_data["device_code"]), expires_in=expires_in,
            poll_interval=interval)
    now = datetime.now(timezone.utc)
    token_expires_in = _coerce_ttl_seconds(token_data.get("expires_in", 0))
    resolved_inference_url = (
        _optional_base_url(token_data.get("inference_base_url")) or requested_inference_url)
    if resolved_inference_url != requested_inference_url:
        print(f"Using portal-provided inference URL: {resolved_inference_url}")
    auth_state = {
        "portal_base_url": portal_base_url, "inference_base_url": resolved_inference_url,
        "client_id": client_id, "scope": token_data.get("scope") or scope,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "obtained_at": now.isoformat(), "expires_at": _iso_after(now, token_expires_in),
        "expires_in": token_expires_in, "tls": _tls_state_from_verify(verify),
        **_NOUS_EMPTY_AGENT_KEY_FIELDS}
    try:
        return refresh_nous_oauth_from_state(
            auth_state, timeout_seconds=timeout_seconds, force_refresh=False)
    except AuthError as exc:
        if exc.code == "subscription_required":
            portal_url = auth_state.get("portal_base_url", DEFAULT_NOUS_PORTAL_URL).rstrip("/")
            print()
            print(format_auth_error(exc))
            print(f"  Subscribe here: {portal_url}/billing")
            print()
            print("After subscribing, run `hermes model` again to finish setup.")
            raise SystemExit(1)
        raise


def _mirror_nous_state_best_effort(auth_state: Dict[str, Any]) -> None:
    """Mirror to the shared store + reseed the pool, swallowing all errors (same as _login_nous)."""
    from hermes_cli.auth import _sync_nous_pool_from_auth_store, _write_shared_nous_state
    with suppress(Exception):
        _write_shared_nous_state(auth_state)
    with suppress(Exception):
        _sync_nous_pool_from_auth_store()


def step_up_nous_billing_scope(
    *, open_browser: bool = True, timeout_seconds: float = 15.0,
    on_verification: Optional[Callable[[str, str], None]] = None) -> bool:
    """Re-run the device flow requesting ``billing:manage`` (step-up on 403 insufficient_scope).

    The user must be ADMIN/OWNER and select "Allow Remote Spending" in the portal, else the
    server silently downscopes and this returns False. Persists like ``_login_nous`` minus the
    model picker.
    """
    from hermes_cli.auth import (
        PROVIDER_REGISTRY, _nous_device_code_login, _save_active_provider_state,
        get_provider_auth_state)
    prior = get_provider_auth_state("nous") or {}
    pconfig = PROVIDER_REGISTRY["nous"]
    # Step-up scope: existing scopes (if any) + billing:manage, deduped, order-stable. Falls back
    # to the standard inference+tool+billing set.
    _raw_scope = prior.get("scope")
    prior_scope = _raw_scope.split() if isinstance(_raw_scope, str) else []
    requested = list(dict.fromkeys([
        *(prior_scope or [NOUS_INFERENCE_INVOKE_SCOPE, "tool:invoke"]), NOUS_BILLING_MANAGE_SCOPE]))
    auth_state = _nous_device_code_login(
        portal_base_url=prior.get("portal_base_url") or None,
        inference_base_url=prior.get("inference_base_url") or None,
        client_id=prior.get("client_id") or pconfig.client_id, scope=" ".join(requested),
        open_browser=open_browser, timeout_seconds=timeout_seconds, on_verification=on_verification)
    _save_active_provider_state("nous", auth_state)
    _mirror_nous_state_best_effort(auth_state)
    granted = auth_state.get("scope")
    return isinstance(granted, str) and NOUS_BILLING_MANAGE_SCOPE in granted.split()


def _pick_nous_model_after_login(
    auth_state: Dict[str, Any], inference_base_url: str) -> Optional[str]:
    """Fetch the curated Nous model list (tier/policy-filtered) and run the interactive picker.

    Returns the selected model id, or None when the user skipped / nothing was selectable.
    Raises on any fetch failure so the caller can print the "Login succeeded, but..." notice.
    """
    from hermes_cli.auth import _prompt_model_selection
    runtime_key = auth_state.get("agent_key") or auth_state.get("access_token")
    if not isinstance(runtime_key, str) or not runtime_key:
        raise _nous_err("No runtime API key available to fetch models", "invalid_token")
    from hermes_cli.models import (
        get_curated_nous_model_ids,
        check_nous_free_tier,
        partition_nous_models_by_tier,
        union_with_portal_free_recommendations,
        union_with_portal_paid_recommendations,
    )
    from hermes_cli.models_pricing import (
        get_pricing_for_provider,
        nous_policy_allowed_ids,
        restrict_to_nous_policy,
    )
    model_ids = get_curated_nous_model_ids()
    _portal = auth_state.get("portal_base_url", "")
    print()
    unavailable_models: list = []
    unavailable_message = ""
    _policy_narrowed = False
    if model_ids:
        pricing = get_pricing_for_provider("nous")
        # Force fresh account data so recent credit purchases are reflected immediately.
        free_tier = check_nous_free_tier(force_fresh=True)
        # Narrow before the tier split, so a rescued id still has to pass the free/paid predicate.
        _policy_allowed = nous_policy_allowed_ids()
        if free_tier:
            with suppress(Exception):
                unavailable_message = _portal_entitlement_message("paid Nous models")
        # The Portal's free/paidRecommendedModels endpoint is the source of truth for what's
        # available *right now*: newly-launched models show without a CLI release.
        union = (
            union_with_portal_free_recommendations if free_tier
            else union_with_portal_paid_recommendations)
        model_ids, pricing = union(model_ids, pricing, _portal)
        _before_policy = model_ids
        model_ids = restrict_to_nous_policy(model_ids, _policy_allowed, rescue_empty=True)
        _policy_narrowed = model_ids != _before_policy
        if free_tier:
            model_ids, unavailable_models = partition_nous_models_by_tier(
                model_ids, pricing, free_tier=True)
    if model_ids:
        from hermes_cli.nous_account import nous_policy_notice
        _policy_notice = nous_policy_notice(removed=_policy_narrowed)
        if _policy_notice:
            print(_policy_notice)
        print(
            f"Showing {len(model_ids)} curated models — "
            "use \"Enter custom model name\" for others.")
        return _prompt_model_selection(
            model_ids, pricing=pricing, unavailable_models=unavailable_models, portal_url=_portal,
            unavailable_message=unavailable_message, confirm_provider="nous",
            confirm_base_url=inference_base_url, confirm_api_key=runtime_key)
    if unavailable_models:
        _url = (_portal or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
        print("No free models currently available.")
        print(unavailable_message or f"Upgrade at {_url} to access paid models.")
    else:
        print("No curated models available for Nous Portal.")
    return None


def _offer_shared_nous_import(timeout_seconds: float) -> Optional[Dict[str, Any]]:
    """Codex-style auto-import: offer to rehydrate a Nous credential from another profile.

    Checks the shared store before launching a fresh device-code flow. Returns the refreshed
    auth state when the user accepted and the import succeeded, else None.
    """
    from hermes_cli.auth import _prompt_yes_no, _read_shared_nous_state
    shared = _read_shared_nous_state()
    if not shared:
        return None
    try:
        shared_path = _nous_shared_store_path()
    except RuntimeError:
        shared_path = None
    print()
    print(f"Found existing Nous OAuth credentials at {shared_path}" if shared_path
          else "Found existing shared Nous OAuth credentials")
    if not _prompt_yes_no("Import these credentials? [Y/n]: ", default="y"):
        return None
    print("Rehydrating Nous session from shared credentials...")
    auth_state = _try_import_shared_nous_state(timeout_seconds=timeout_seconds)
    if auth_state is None:
        print("Could not refresh shared credentials — falling back to device-code login.")
    return auth_state


def _restore_active_provider(prior_active_provider: Any) -> None:
    """Undo the ``active_provider="nous"`` that ``_save_provider_state`` wrote during login."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
    with _auth_store_lock():
        auth_store = _load_auth_store()
        if prior_active_provider:
            auth_store["active_provider"] = prior_active_provider
        else:
            auth_store.pop("active_provider", None)
        _save_auth_store(auth_store)


def _login_nous(args, pconfig: ProviderConfig) -> None:
    """Nous Portal device authorization flow."""
    from hermes_cli.auth import (
        _auth_store_lock, _load_auth_store, _nous_device_code_login, _save_active_provider_state,
        _save_model_choice, _sync_nous_pool_from_auth_store, _update_config_for_provider,
        _write_shared_nous_state, format_auth_error)
    timeout_seconds = getattr(args, "timeout", None) or 15.0
    ca_bundle = (
        getattr(args, "ca_bundle", None) or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE"))
    try:
        auth_state = _offer_shared_nous_import(timeout_seconds)
        if auth_state is None:
            auth_state = _nous_device_code_login(
                portal_base_url=getattr(args, "portal_url", None),
                inference_base_url=getattr(args, "inference_url", None),
                client_id=getattr(args, "client_id", None) or pconfig.client_id,
                scope=getattr(args, "scope", None),
                open_browser=not getattr(args, "no_browser", False),
                timeout_seconds=timeout_seconds, insecure=bool(getattr(args, "insecure", False)),
                ca_bundle=ca_bundle)
        inference_base_url = auth_state["inference_base_url"]
        # Snapshot BEFORE _save_provider_state overwrites active_provider to "nous", so a
        # model-picker "Skip (keep current)" can restore the user's previous provider.
        with _auth_store_lock():
            prior_active_provider = _load_auth_store().get("active_provider")
        saved_to = _save_active_provider_state("nous", auth_state)
        # Mirror to the shared store so other profiles can one-tap import (best-effort inside).
        _write_shared_nous_state(auth_state)
        _sync_nous_pool_from_auth_store()
        print()
        print("Login successful!")
        print(f"  Auth state: {saved_to}")
        # Pick the model BEFORE writing the provider to config.yaml so config is never half-updated.
        selected_model = None
        try:
            selected_model = _pick_nous_model_after_login(auth_state, inference_base_url)
        except Exception as exc:
            message = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
            print()
            print(f"Login succeeded, but could not fetch available models. Reason: {message}")
        # No model (Skip, fetch failed, nothing curated): keep the previous provider rather than
        # switch to Nous with a mismatched model; the Nous tokens stay saved for future use.
        if not selected_model:
            _restore_active_provider(prior_active_provider)
            print()
            print("No provider change. Nous credentials saved for future use.")
            print("  Run `hermes model` again to switch to Nous Portal.")
            return
        config_path = _update_config_for_provider(
            "nous", inference_base_url, default_model=selected_model)
        _save_model_choice(selected_model)
        print(f"Default model set to: {selected_model}")
        print(f"  Config updated: {config_path} (model.provider=nous)")
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Login failed: {exc}")
        raise SystemExit(1)
