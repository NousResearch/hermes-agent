"""nous store protocol/lifecycle responsibilities."""

from __future__ import annotations
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from auth.providers.codex import _pool_entries
from auth.errors import AuthError
from auth.constants import (
    DEFAULT_NOUS_CLIENT_ID,
    DEFAULT_NOUS_INFERENCE_URL,
    DEFAULT_NOUS_SCOPE,
    DEFAULT_NOUS_WELCOME_URL,
    NOUS_DEVICE_CODE_SOURCE,
)
from auth.store import AUTH_LOCK_TIMEOUT_SECONDS
from auth.store_migrations import DEFAULT_NOUS_PORTAL_URL

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
            get_default_hermes_root() / "shared" / NOUS_SHARED_STORE_FILENAME
        ).resolve(strict=False)
        try:
            resolved = path.resolve(strict=False)
        except Exception:
            resolved = path
        if resolved == real_home_shared:
            raise RuntimeError(
                f"Refusing to touch real user shared Nous auth store during test run: "
                f"{path}. Set HERMES_SHARED_AUTH_DIR to a tmp_path in your test fixture."
            )
    return path


@contextmanager
def _nous_shared_store_lock(timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS):
    """Cross-profile lock for the shared Nous OAuth store.

    Lock ordering invariant: if both this and ``_auth_store_lock`` need to be held, acquire
    ``_auth_store_lock`` FIRST. All runtime refresh paths follow this order.
    """
    from auth.store import _file_lock

    try:
        lock_path = _nous_shared_store_path().with_suffix(".lock")
    except RuntimeError:
        yield  # No HERMES_HOME yet (pre-setup): fall through without locking.
        return
    with _file_lock(
        lock_path,
        _nous_shared_lock_holder,
        timeout_seconds,
        "Timed out waiting for shared Nous auth lock",
    ):
        yield


def _shared_lock_timeout(timeout_seconds: float) -> float:
    return max(timeout_seconds + 5.0, AUTH_LOCK_TIMEOUT_SECONDS)


_NOUS_SHARED_STATE_KEYS = (
    "access_token",
    "refresh_token",
    "token_type",
    "scope",
    "client_id",
    "portal_base_url",
    "inference_base_url",
    "obtained_at",
    "expires_at",
    # Guest (``auth_method: anonymous``) identity: the ``anon_`` credential is the refresh material.
    "auth_method",
    "account_tier",
    "anon_token",
    "user_id",
    "org_id",
)


def _merge_shared_nous_oauth_state(state: Dict[str, Any]) -> bool:
    """Copy fresher shared OAuth tokens into a profile-local Nous state."""
    from auth.token_validation import _nonempty_str, _parse_iso_timestamp
    from auth.providers.nous_store import _read_shared_nous_state

    shared = _read_shared_nous_state() or {}
    shared_refresh = shared.get("refresh_token")
    if not _nonempty_str(shared_refresh):
        return False  # a free-tier identity has no refresh token; nothing to merge into an OAuth state
    shared_access_exp = _parse_iso_timestamp(shared.get("expires_at")) or 0.0
    local_access_exp = _parse_iso_timestamp(state.get("expires_at")) or 0.0
    refresh_changed = (
        shared_refresh.strip() != str(state.get("refresh_token") or "").strip()
    )
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
        "access_token": src.get("access_token"),
        "refresh_token": src.get("refresh_token"),
        "token_type": src.get("token_type") or "Bearer",
        "scope": src.get("scope") or DEFAULT_NOUS_SCOPE,
        "client_id": src.get("client_id") or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": src.get("portal_base_url") or DEFAULT_NOUS_PORTAL_URL,
        # A guest's route defaults to the welcome host: the paid host cross-refuses its JWT.
        "inference_base_url": src.get("inference_base_url")
        or (
            DEFAULT_NOUS_WELCOME_URL
            if src.get("auth_method") == "anonymous"
            else DEFAULT_NOUS_INFERENCE_URL
        ),
        "obtained_at": src.get("obtained_at"),
        "expires_at": src.get("expires_at"),
        **{
            k: src[k]
            for k in ("auth_method", "account_tier", "anon_token", "user_id", "org_id")
            if src.get(k) not in (None, "")
        },
    }


def _write_shared_nous_state(state: Dict[str, Any]) -> None:
    """Persist a minimal copy of the Nous OAuth state to the shared store.

    Best-effort: failures are logged and swallowed; per-profile auth.json stays the source of truth.
    """
    from auth.providers.nous import _oauth_trace, _token_fingerprint, logger
    from auth.token_validation import _nonempty_str
    from auth.store import _save_private_json

    refresh_token = state.get("refresh_token")
    # Nothing worth sharing without refresh material: an OAuth refresh_token (with its access token),
    # or a guest's anon_ credential, which is the whole identity and may not have been exchanged yet.
    is_guest = _nonempty_str(state.get("anon_token"))
    if not is_guest and not (
        _nonempty_str(refresh_token) and _nonempty_str(state.get("access_token"))
    ):
        return
    shared = {
        "_schema": 1,
        **_nous_shared_shape(state),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with _nous_shared_store_lock():
            path = _nous_shared_store_path()
            _save_private_json(path, shared, sort_keys=True)
        _oauth_trace(
            "nous_shared_store_written",
            path=str(path),
            refresh_token_fp=_token_fingerprint(refresh_token),
        )
    except Exception as exc:
        logger.debug("Failed to write shared Nous auth store: %s", exc)


def _read_shared_nous_state() -> Optional[Dict[str, Any]]:
    """Shared Nous OAuth state when present and well-formed, else None.

    None (missing / unreadable / malformed / lacking tokens) means "fall through to device-code".
    """
    from auth.providers.nous import logger
    from auth.token_validation import _nonempty_str

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
    has_tokens = _nonempty_str(payload.get("anon_token")) or (
        _nonempty_str(payload.get("access_token"))
        and _nonempty_str(payload.get("refresh_token"))
    )
    return payload if has_tokens else None


def _clear_shared_nous_state(reason: str) -> None:
    """Remove the shared Nous OAuth store after a terminal token failure."""
    from auth.providers.nous import _oauth_trace, logger

    try:
        with _nous_shared_store_lock():
            _nous_shared_store_path().unlink(missing_ok=True)
        _oauth_trace("nous_shared_store_cleared", reason=reason)
    except Exception as exc:
        logger.debug("Failed to clear shared Nous auth store: %s", exc)


def _quarantine_nous_oauth_state(
    state: Dict[str, Any], error: AuthError, *, reason: str
) -> None:
    """Keep routing metadata but remove dead OAuth material so it is not replayed.

    Only for terminal errors (``*_refresh_failed`` = HTTP 400/401/403 invalid_grant / revoked /
    refresh_token_reused; ``*_auth_missing_refresh_token``), all ``relogin_required=True`` —
    transient 429/5xx never quarantine.
    """
    from auth.providers.nous import (
        _NOUS_EMPTY_AGENT_KEY_FIELDS,
        _quarantine_forensics,
        logger,
    )
    from auth.providers.nous_status import invalidate_nous_auth_status_cache
    from auth.oauth import _FLAT_OAUTH_TOKEN_KEYS, _last_auth_error_marker

    # Forensics BEFORE clearing token material: a hosted agent quarantined here is otherwise only
    # visible as a later "No access token found" WARNING, too late to root-cause. Managed log
    # drains may be WARNING-only, so this MUST be logger.warning.
    logger.warning(
        "Nous OAuth state quarantined (terminal auth death): %s",
        json.dumps(
            _quarantine_forensics(state, error, reason),
            sort_keys=True,
            ensure_ascii=False,
        ),
    )
    for key in (*_FLAT_OAUTH_TOKEN_KEYS, *_NOUS_EMPTY_AGENT_KEY_FIELDS):
        state.pop(key, None)
    state["last_auth_error"] = _last_auth_error_marker("nous", error, reason=reason)
    _clear_shared_nous_state(reason)
    invalidate_nous_auth_status_cache()


def _quarantine_nous_pool_entries(
    auth_store: Dict[str, Any], error: AuthError, *, reason: str
) -> bool:
    """Remove singleton-seeded Nous pool entries that contain dead OAuth state."""
    from auth.providers.nous import _oauth_trace

    entries = _pool_entries(auth_store, "nous")
    if entries is None:
        return False
    singleton_sources = {NOUS_DEVICE_CODE_SOURCE, f"manual:{NOUS_DEVICE_CODE_SOURCE}"}
    retained = [
        e
        for e in entries
        if not (isinstance(e, dict) and e.get("source") in singleton_sources)
    ]
    removed = len(retained) != len(entries)
    if removed:
        auth_store["credential_pool"]["nous"] = retained
        _oauth_trace(
            "nous_pool_device_code_quarantined", reason=reason, error_code=error.code
        )
    return removed


def _try_import_shared_nous_state(
    *, timeout_seconds: float = 15.0
) -> Optional[Dict[str, Any]]:
    """Rehydrate Nous OAuth state from the shared store via a forced refresh.

    Returns auth_state ready for ``persist_nous_credentials()``; None on any failure (expired
    token, portal unreachable) so the caller falls through to device-code.
    """
    from auth.providers.nous import _oauth_trace, logger, refresh_nous_oauth_from_state
    from auth.providers.nous_store import (
        _read_shared_nous_state,
        _write_shared_nous_state,
    )
    from auth.oauth import _is_terminal_nous_refresh_error

    try:
        with _nous_shared_store_lock(
            timeout_seconds=_shared_lock_timeout(timeout_seconds)
        ):
            shared = _read_shared_nous_state()
            if not shared:
                return None
            # Full state dict so refresh_nous_oauth_from_state has every field it needs.
            state: Dict[str, Any] = {
                **_nous_shared_shape(shared),
                "agent_key": None,
                "agent_key_expires_at": None,
                "tls": {"insecure": False, "ca_bundle": None},
            }
            refreshed = refresh_nous_oauth_from_state(
                state,
                timeout_seconds=timeout_seconds,
                force_refresh=True,
                on_state_update=lambda updated, _reason: _write_shared_nous_state(
                    updated
                ),
            )
            _write_shared_nous_state(refreshed)
    except Exception as exc:
        is_auth = isinstance(exc, AuthError)
        _oauth_trace(
            "nous_shared_import_failed",
            error_type=type(exc).__name__,
            **({"error_code": getattr(exc, "code", None)} if is_auth else {}),
        )
        if is_auth and _is_terminal_nous_refresh_error(exc):
            _clear_shared_nous_state("shared_import_terminal_refresh_failure")
        logger.debug("Shared Nous import failed: %s", exc)
        return None
    return refreshed
