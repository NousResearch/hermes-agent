"""nous status protocol/lifecycle responsibilities."""

from __future__ import annotations
import auth.store as auth_storage
from typing import Tuple
import time
from typing import Any, Callable, Dict, Optional
from auth.errors import AuthError
from auth.store_migrations import DEFAULT_NOUS_PORTAL_URL


def _snapshot_nous_pool_status(*, environment) -> Dict[str, Any]:
    """Best-effort status from the credential pool.

    Fallback only: the auth-store provider state is the runtime source of truth because it is what
    ``resolve_nous_runtime_credentials()`` refreshes.
    """
    from auth.providers.nous import _empty_nous_auth_status

    environment.require_current_scope()
    pass
    from auth.token_validation import _parse_iso_timestamp

    try:
        from auth.credential_pool import load_pool

        pool = load_pool("nous", environment=environment)
        entries = list(pool.entries()) if pool and pool.has_credentials() else []
        if not entries:
            return _empty_nous_auth_status()
        entry = max(
            entries,
            key=lambda e: (
                _parse_iso_timestamp(getattr(e, "agent_key_expires_at", None)) or 0.0,
                _parse_iso_timestamp(getattr(e, "expires_at", None)) or 0.0,
                -int(getattr(e, "priority", 0) or 0),
            ),
        )
        attr = lambda name, default=None: getattr(entry, name, default)  # noqa: E731
        if not attr("runtime_api_key"):
            return _empty_nous_auth_status()
        access_token, refresh_token = attr("access_token"), attr("refresh_token")
        auth_type = str(attr("auth_type", "") or "").strip().lower()
        is_portal_oauth = bool(access_token) and (
            auth_type.startswith("oauth") or bool(refresh_token)
        )
        label = attr("label", "unknown")
        return {
            "logged_in": is_portal_oauth,
            "portal_base_url": (
                (attr("portal_base_url") or DEFAULT_NOUS_PORTAL_URL)
                if is_portal_oauth
                else None
            ),
            "inference_base_url": (
                attr("inference_base_url")
                or attr("runtime_base_url")
                or attr("base_url")
            ),
            "access_token": access_token if is_portal_oauth else None,
            "access_expires_at": attr("expires_at"),
            "agent_key_expires_at": attr("agent_key_expires_at"),
            "has_refresh_token": bool(refresh_token),
            "inference_credential_present": True,
            "credential_source": f"pool:{label}",
            "source": f"pool:{label}",
        }
    except Exception:
        return _empty_nous_auth_status()


def _compute_nous_auth_status(*, environment) -> Dict[str, Any]:
    """Uncached implementation of get_nous_auth_status(). See that function."""
    from auth.providers.nous import (
        _nous_status_from_state,
        resolve_nous_runtime_credentials,
    )

    environment.require_current_scope()
    from auth.provider_state import get_provider_auth_state

    state = get_provider_auth_state("nous")
    if not state:
        return _snapshot_nous_pool_status(environment=environment)
    base_status = _nous_status_from_state(
        state, logged_in=bool(state.get("access_token")), source="auth_store"
    )
    try:
        creds = resolve_nous_runtime_credentials(environment=environment)
        refreshed_state = get_provider_auth_state("nous") or state
        base_status.update({
            "logged_in": True,
            "portal_base_url": (
                refreshed_state.get("portal_base_url")
                or base_status.get("portal_base_url")
            ),
            "inference_base_url": (
                creds.get("base_url")
                or refreshed_state.get("inference_base_url")
                or base_status.get("inference_base_url")
            ),
            "access_expires_at": (
                refreshed_state.get("expires_at")
                or base_status.get("access_expires_at")
            ),
            "agent_key_expires_at": (
                creds.get("expires_at")
                or refreshed_state.get("agent_key_expires_at")
                or base_status.get("agent_key_expires_at")
            ),
            "has_refresh_token": bool(refreshed_state.get("refresh_token")),
            "inference_credential_present": True,
            "credential_source": "auth_store",
            "source": f"runtime:{creds.get('source', 'portal')}",
            "key_id": creds.get("key_id"),
        })
    except AuthError as exc:
        base_status.update({
            "logged_in": False,
            "error": str(exc),
            "relogin_required": bool(getattr(exc, "relogin_required", False)),
            "error_code": getattr(exc, "code", None),
        })
    return base_status


def get_nous_auth_status_local(*, environment) -> Dict[str, Any]:
    """Refresh-free Nous auth snapshot for read-only display surfaces.

    NEVER calls ``resolve_nous_runtime_credentials()`` (no refresh POST / single-use token spent);
    ``logged_in`` = usable invoke JWT, or a refresh token not terminally quarantined — not proof
    the server still accepts it.
    """
    from auth.providers.nous import (
        _nous_status_from_state,
        _state_invoke_jwt_status,
        _terminal_quarantine_marker,
    )

    environment.require_current_scope()
    from auth.provider_state import get_provider_auth_state

    try:
        state = get_provider_auth_state("nous")
    except Exception:
        state = None
    if not state:
        return _snapshot_nous_pool_status(environment=environment)
    jwt_reason = _state_invoke_jwt_status(state, state.get("access_token"))
    last_err = _terminal_quarantine_marker(state)
    logged_in = (jwt_reason is None) or (
        bool(state.get("refresh_token")) and last_err is None
    )
    status = _nous_status_from_state(
        state, logged_in=logged_in, source="auth_store_local"
    )
    if last_err is not None:
        status.update(
            relogin_required=True,
            error_code=last_err.get("code"),
            error=last_err.get("message") or "re-login required",
        )
    return status


NOUS_SESSION_VALID = "valid"

NOUS_SESSION_TERMINAL = "terminal"

NOUS_SESSION_UNKNOWN = "unknown"


def get_nous_session_validity() -> str:
    """Classify the Nous bootstrap session for the dashboard /api/status probe.

    Local auth-store state only; polled frequently, so it never resolves or refreshes. ANTI-FLAP:
    only a *terminal* failure maps to "terminal" — a rotation blip, network error, or expiring
    token must NOT (that would trigger a spurious NAS re-mint on a healthy box).
    """
    from auth.providers.nous import (
        _state_invoke_jwt_status,
        _terminal_quarantine_marker,
    )
    from auth.provider_state import get_provider_auth_state

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
    provider_id: str,
    *,
    is_expiring: Callable[[str, int], bool],
    auth_mode: str,
    resolve: Callable[[], Dict[str, Any]],
    on_pool_miss: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
    environment,
) -> Dict[str, Any]:
    """Status snapshot for a store-backed OAuth provider (Codex, xAI).

    Pool first (where `hermes auth` / `hermes model` store device_code tokens), then
    *on_pool_miss* for a pool-derived degraded status, then the legacy state via *resolve*.

    The pool read is an observation (``peek``), not a lease: ``select()`` refreshes an expiring
    single-use token and, when that speculative POST fails transiently, benches the entry with a
    persisted cooldown — every credential-gated listing (``/model`` picker, doctor) then shows the
    provider as unconfigured while the runtime resolver still serves it. Refreshing stays with the
    runtime resolver reached through *resolve*, whose failures persist nothing.
    """
    environment.require_current_scope()
    pass
    from auth.store import _auth_file_path

    try:
        from auth.credential_pool import load_pool

        pool = load_pool(provider_id, environment=environment)
        if pool and pool.has_credentials():
            entry = pool.peek()
            if entry is not None:
                api_key = getattr(entry, "runtime_api_key", None) or getattr(
                    entry, "access_token", ""
                )
                if api_key and not is_expiring(api_key, 0):
                    return {
                        "logged_in": True,
                        "auth_store": str(_auth_file_path()),
                        "last_refresh": getattr(entry, "last_refresh", None),
                        "auth_mode": auth_mode,
                        "source": f"pool:{getattr(entry, 'label', 'unknown')}",
                        "api_key": api_key,
                        # The host this entry's key belongs to, so a caller never pairs it with
                        # another provider default (#121486).
                        "base_url": str(
                            getattr(entry, "runtime_base_url", None)
                            or getattr(entry, "base_url", None)
                            or ""
                        ).rstrip("/"),
                    }
            if on_pool_miss is not None and (degraded := on_pool_miss()):
                return degraded
    except Exception:
        pass
    try:
        creds = resolve()
        return {
            "logged_in": True,
            "auth_store": str(_auth_file_path()),
            "last_refresh": creds.get("last_refresh"),
            "auth_mode": creds.get("auth_mode"),
            "source": creds.get("source"),
            "api_key": creds.get("api_key"),
            "base_url": creds.get("base_url") or "",
        }
    except AuthError as exc:
        return {
            "logged_in": False,
            "auth_store": str(_auth_file_path()),
            "error": str(exc),
        }


_NOUS_AUTH_STATUS_CACHE_TTL = 15.0

_nous_auth_status_cache: Optional[
    Tuple[float, str, Optional[float], Dict[str, Any]]
] = None


def _auth_file_cache_key() -> Tuple[str, Optional[float]]:
    auth_file = auth_storage._auth_file_path()
    try:
        return auth_storage._resolved_key(auth_file), auth_file.stat().st_mtime
    except Exception:  # missing file included: key without an mtime
        return auth_storage._resolved_key(auth_file), None


def invalidate_nous_auth_status_cache() -> None:
    """Clear the get_nous_auth_status() memo (for code paths that mutate Nous auth state without
    touching auth.json, e.g. tests; login/logout invalidate via the mtime check automatically)."""
    global _nous_auth_status_cache
    _nous_auth_status_cache = None


def get_nous_auth_status(*, environment) -> Dict[str, Any]:
    """Status snapshot for Nous auth, memoised ~15s keyed on the auth.json mtime.

    Prefers the auth-store provider state (the live source of truth for refresh) and validates it by
    resolving runtime credentials so revoked refresh sessions do not show up as a healthy login."""
    environment.require_current_scope()
    global _nous_auth_status_cache
    now = time.monotonic()
    auth_file_key, mtime = _auth_file_cache_key()
    cached = _nous_auth_status_cache
    if (
        cached is not None
        and cached[1:3] == (auth_file_key, mtime)
        and (now - cached[0]) < _NOUS_AUTH_STATUS_CACHE_TTL
    ):
        return dict(cached[3])
    status = _compute_nous_auth_status(environment=environment)
    _nous_auth_status_cache = (now, auth_file_key, mtime, dict(status))
    return status
