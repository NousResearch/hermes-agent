"""Canonical nous authentication mechanics; no CLI dependencies."""

from __future__ import annotations
import auth.store as auth_storage
import auth.provider_state as auth_provider_state
import auth.store_migrations as auth_store_migrations
from auth.oauth import _optional_base_url, _resolve_verify, _tls_state_from_verify
from auth.constants import ACCESS_TOKEN_REFRESH_SKEW_SECONDS
from auth.token_validation import _is_expiring
from hermes_constants import hermes_home_key


import logging

import hashlib

import json

import os

import threading

import time

import uuid

from contextlib import suppress

from datetime import datetime, timezone

from pathlib import Path

from typing import Any, Callable, Dict, FrozenSet, Optional

from urllib.parse import urlparse


from auth.token_validation import _decode_jwt_claims

from auth.errors import AuthError

from auth.constants import (
    DEFAULT_NOUS_CLIENT_ID,
    DEFAULT_NOUS_INFERENCE_URL,
    DEFAULT_NOUS_SCOPE,
    DEFAULT_NOUS_WELCOME_URL,
    NOUS_AUTH_PATH_INVOKE_JWT,
    NOUS_DEVICE_CODE_SOURCE,
    _nous_err,
    httpx,
)


from auth.store_migrations import DEFAULT_NOUS_PORTAL_URL

logger = logging.getLogger("hermes_cli.auth")

_UNUSABLE_JWT_RELOGIN = "Re-authenticate with: hermes auth add nous"


def _unusable_invoke_jwt_error(
    reason: str, *, no_refresh_token: bool = False
) -> AuthError:
    """Shared ``relogin=True`` error for an access token that is not a usable inference JWT.

    With no refresh token the failure is a state-shape one (nothing to redeem), so it carries the
    terminal ``nous_auth_missing_refresh_token`` code the pool recognises instead of the JWT
    ``reason``, which would bench the row as a transient outage (#113718).
    """
    detail = " and no refresh token is available" if no_refresh_token else ""
    return _nous_err(
        f"Nous Portal access token is not a usable inference JWT ({reason}){detail}. "
        f"{_UNUSABLE_JWT_RELOGIN}",
        "nous_auth_missing_refresh_token" if no_refresh_token else reason,
        relogin=True,
    )


def _token_fingerprint(token: Any) -> Optional[str]:
    """Return a short hash fingerprint for telemetry without leaking token bytes."""
    cleaned = token.strip() if isinstance(token, str) else ""
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12] if cleaned else None


def _oauth_trace(
    event: str, *, sequence_id: Optional[str] = None, **fields: Any
) -> None:
    if os.getenv("HERMES_OAUTH_TRACE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    payload: Dict[str, Any] = {"event": event}
    if sequence_id:
        payload["sequence_id"] = sequence_id
    payload.update(fields)
    logger.info(
        "oauth_trace %s", json.dumps(payload, sort_keys=True, ensure_ascii=False)
    )


def _iso_after(now: datetime, ttl_seconds: int) -> str:
    """ISO timestamp *ttl_seconds* after *now* (UTC)."""
    return datetime.fromtimestamp(
        now.timestamp() + ttl_seconds, tz=timezone.utc
    ).isoformat()


_NOUS_EMPTY_AGENT_KEY_FIELDS: Dict[str, Any] = {
    "agent_key": None,
    "agent_key_id": None,
    "agent_key_expires_at": None,
    "agent_key_expires_in": None,
    "agent_key_reused": None,
    "agent_key_obtained_at": None,
}


def _portal_entitlement_message(capability: str, *, environment) -> str:
    """Ask the application for existing account-entitlement presentation."""
    environment.require_current_scope()
    if environment.entitlement_message is None:
        return ""
    return environment.entitlement_message(capability) or ""


def _format_nous_entitlement_auth_error(error: AuthError, *, environment) -> str:
    environment.require_current_scope()
    with suppress(Exception):
        if message := _portal_entitlement_message(
            "Nous model access", environment=environment
        ):
            return message
    return f"{error} Check credits or billing in Nous Portal, then retry."


_ALLOWED_NOUS_INFERENCE_HOSTS: FrozenSet[str] = frozenset({
    "inference-api.nousresearch.com",
    # Free-tier (anonymous) host: serves the single ``nous/welcome`` model.
    "welcome-api.nousresearch.com",
})


def _nous_inference_host_allowed(hostname: Optional[str]) -> bool:
    """Production hosts always; otherwise only the host the operator named in
    ``NOUS_INFERENCE_BASE_URL``.

    A non-production Portal's refresh response names that environment's inference gateway. The
    Portal-returned value is network provenance, so it does not get bearer-receive authority on
    its own — not even for a Nous-owned host: the operator's explicit override is the authority,
    and the network value is accepted exactly when it agrees with it. Then the persisted endpoint,
    the pricing scope and the proxy all follow the environment the operator chose, and the
    per-turn "refusing inference URL host" warning stops.
    """
    if hostname in _ALLOWED_NOUS_INFERENCE_HOSTS:
        return True
    if not hostname:
        return False
    override = _nous_inference_env_override()
    return override is not None and urlparse(override).hostname == hostname


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
            "nous: refusing non-https inference URL scheme %r from Portal response",
            parsed.scheme,
        )
        return None
    if not _nous_inference_host_allowed(parsed.hostname):
        logger.warning(
            "nous: refusing inference URL host %r from Portal response "
            "(not in allowlist); falling back to default",
            parsed.hostname,
        )
        return None
    return cleaned.rstrip("/")


def _scoped_operator_override(*names: str) -> Optional[str]:
    """The first set operator routing override among ``names`` (``NOUS_INFERENCE_BASE_URL``,
    ``HERMES_PORTAL_BASE_URL`` / its ``NOUS_PORTAL_BASE_URL`` alias), resolved through the profile
    secret scope, or None.

    ``get_secret`` already reads ``os.environ`` for a single-profile process, so the only time it
    raises is a multi-profile call that has lost its profile scope. That call has no authority to
    route on the launch profile's value — returning the ambient env there would send a secondary's
    tokens to the launch profile's Portal or inference host — so the override is simply absent.
    Absent is not free: default routing then applies, so a non-production deployment's token is
    spent against the production hosts. One WARNING per lost-scope event names the override; the
    downstream "ignoring invalid portal_base_url" line never says which caller lost its scope.
    """
    from agent.secret_scope import UnscopedSecretError, get_secret

    try:
        for name in names:
            value = get_secret(name)
            if value:
                return value
        return None
    except UnscopedSecretError:
        logger.warning(
            "nous: %s unreadable — no profile secret scope on a multiplexed call; treating the "
            "override as absent (default routing applies). The caller needs a profile scope binding.",
            "/".join(names),
        )
        return None


def _nous_inference_env_override() -> Optional[str]:
    """User-set ``NOUS_INFERENCE_BASE_URL`` override (trailing slash stripped) or None.

    Documented dev/staging escape hatch; the env source is trusted, so unlike Portal-returned URLs
    it is intentionally NOT gated by the network host allowlist. Read through the profile-aware
    resolver so a multiplexed profile uses its own override and never inherits the default
    profile's process-wide value (#65941).
    """
    from auth.oauth import _optional_base_url

    return _optional_base_url(_scoped_operator_override("NOUS_INFERENCE_BASE_URL"))


def _nous_portal_env_override() -> Optional[str]:
    """``HERMES_PORTAL_BASE_URL`` / ``NOUS_PORTAL_BASE_URL`` override or None.

    Documented dev/staging escape hatch (e.g. hosted agents on the staging Portal). Trusted env
    source: must NOT be gated by ``_NOUS_PORTAL_ALLOWED_HOSTS``, which rejects untrusted
    NETWORK-provided values persisted to auth.json, not operator config. Read through the
    profile secret scope like ``_nous_inference_env_override``: it is reached on every routed
    turn (``_nous_effective_routing``), and a raw environ read would POST a multiplexed
    secondary's refresh token to the DEFAULT profile's Portal.
    """
    from auth.oauth import _optional_base_url

    return _optional_base_url(
        _scoped_operator_override("HERMES_PORTAL_BASE_URL", "NOUS_PORTAL_BASE_URL")
    )


def _state_invoke_jwt_status(state: Dict[str, Any], token: Any) -> Optional[str]:
    """``_nous_invoke_jwt_status`` for *token* using *state*'s scope / expires_at (patchable)."""
    return _nous_invoke_jwt_status(
        token, scope=state.get("scope"), expires_at=state.get("expires_at")
    )


def _assert_nous_inference_jwt_usable(
    state: Dict[str, Any], *, access_token: Any = None
) -> None:
    token = state.get("access_token") if access_token is None else access_token
    reason = _state_invoke_jwt_status(state, token)
    if reason is not None:
        raise _unusable_invoke_jwt_error(reason)


def _remaining_ttl(expires_at: Any, fallback_expires_in: Any) -> int:
    """Seconds until *expires_at* (ISO), else *fallback_expires_in* coerced to a TTL."""
    from auth.oauth import _coerce_ttl_seconds
    from auth.token_validation import _parse_iso_timestamp

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
    state: Dict[str, Any], *, obtained_at: Optional[str] = None
) -> None:
    from auth.token_validation import _nonempty_str

    access_token = state.get("access_token")
    if not _nonempty_str(access_token):
        return
    existing_obtained_at = state.get("agent_key_obtained_at")
    if not obtained_at:
        reuse = state.get("agent_key") == access_token and _nonempty_str(
            existing_obtained_at
        )
        obtained_at = (
            existing_obtained_at if reuse else datetime.now(timezone.utc).isoformat()
        )
    expires_at = _nous_jwt_expires_at(access_token, state.get("expires_at"))
    expires_in = _remaining_ttl(expires_at, state.get("expires_in"))
    if expires_at:
        state["expires_at"] = expires_at
        state["expires_in"] = expires_in
    state.update(
        agent_key=access_token,
        agent_key_id=None,
        agent_key_expires_at=expires_at,
        agent_key_expires_in=expires_in,
        agent_key_reused=False,
        agent_key_obtained_at=obtained_at,
    )


def _select_nous_invoke_jwt(
    state: Dict[str, Any],
    *,
    access_token: Any = None,
    sequence_id: Optional[str] = None,
) -> None:
    from auth.token_validation import _nonempty_str

    if _nonempty_str(access_token):
        state["access_token"] = access_token
    _set_nous_agent_key_from_invoke_jwt(state)
    logger.debug("Nous inference auth: using NAS invoke JWT")
    _oauth_trace(
        "nous_invoke_jwt_selected",
        sequence_id=sequence_id,
        access_token_fp=_token_fingerprint(state.get("access_token")),
    )


_NOUS_EFFECTIVE_STATE_IGNORED_KEYS = frozenset({"expires_in", "agent_key_expires_in"})


def _nous_effective_provider_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v for k, v in state.items() if k not in _NOUS_EFFECTIVE_STATE_IGNORED_KEYS
    }


def _quarantine_forensics(
    state: Dict[str, Any], error: AuthError, reason: str
) -> Dict[str, Any]:
    """Redaction-safe forensic record for a quarantine: fingerprints, sizes and booleans only.

    NEVER include a raw token/agent_key (credential-shaped literals get corrupted in logs). The
    12-char SHA-256 prefix correlates to NAS's refreshTokenHash without leaking the secret;
    provenance is client_id + agent_key_id (Nous state has no session_id).
    """
    from auth.store import _auth_file_path

    forensic: Dict[str, Any] = {
        "reason": reason,
        "error_code": error.code,
        "client_id": state.get("client_id"),
        "agent_key_id": state.get("agent_key_id"),
        "refresh_token_fp": _token_fingerprint(state.get("refresh_token")),
    }
    # On-disk integrity of the auth store at the moment of quarantine.
    try:
        auth_path = _auth_file_path()
        forensic["auth_json_path"] = str(auth_path)
        try:
            st = os.stat(auth_path)
            forensic.update(
                auth_json_size=st.st_size,
                auth_json_mtime=st.st_mtime,
                auth_json_exists=True,
            )
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
            already_expired = parsed.replace(
                tzinfo=parsed.tzinfo or timezone.utc
            ) < datetime.now(timezone.utc)
        except ValueError:
            already_expired = None
    forensic["token_already_expired"] = already_expired
    return forensic


def _refresh_access_token(
    *,
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    refresh_token: str,
) -> Dict[str, Any]:
    response = client.post(
        f"{portal_base_url}/api/oauth/token",
        headers={"x-nous-refresh-token": refresh_token},
        data={"grant_type": "refresh_token", "client_id": client_id},
    )
    if response.status_code == 200:
        payload = response.json()
        if "access_token" not in payload:
            raise _nous_err(
                "Refresh response missing access_token", "invalid_token", relogin=True
            )
        return payload
    if 500 <= response.status_code <= 599:
        raise AuthError(
            f"Nous Portal is temporarily unavailable (HTTP {response.status_code}).",
            provider="nous",
            code="temporarily_unavailable",
            retryable=True,
        )
    # Vercel's Security Checkpoint in front of the Portal answers non-browser clients with a
    # 403 (``x-vercel-mitigated: deny``) or 429 (``challenge``) page (#120602). That is the edge
    # refusing the request, not the token endpoint rejecting the grant, so keep the credentials
    # instead of forcing a re-login.
    mitigated = (
        response.headers.get("x-vercel-mitigated")
        if response.status_code in {403, 429}
        else None
    )
    if mitigated:
        from agent.retry_utils import parse_retry_after_seconds

        raise AuthError(
            f"Nous Portal's edge firewall challenged the token refresh (HTTP {response.status_code}, "
            f"x-vercel-mitigated={mitigated}). Credentials kept; try again shortly.",
            provider="nous",
            code="upstream_blocked",
            retryable=True,
            retry_after=parse_retry_after_seconds(response.headers),
        )
    from auth.errors import _OAUTH_GRANT_DEAD_CODES

    try:
        error_payload = response.json()
    except Exception:
        error_payload = {}
    if not isinstance(error_payload, dict):
        error_payload = {}
    # Only an explicit OAuth grant-dead code is terminal: a 429/404 gateway body without an
    # ``error`` key says nothing about the refresh token, so it must not wipe credentials.
    # A 401/403 without an ``error`` code still means the token endpoint rejected the refresh
    # token, so it is reported as ``invalid_grant`` (terminal) rather than left unclassified.
    raw_code = error_payload.get("error")
    if raw_code is None and response.status_code in {401, 403}:
        raw_code = "invalid_grant"
    code = None if raw_code is None else str(raw_code)
    description = str(
        error_payload.get("error_description") or "Refresh token exchange failed"
    )
    relogin = code in _OAUTH_GRANT_DEAD_CODES
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
            "Re-authenticate with: hermes auth add nous"
        )
        relogin = True
    raise _nous_err(description, code, relogin=relogin)


def _refresh_nous_or_quarantine(
    *,
    client: httpx.Client,
    auth_store: Dict[str, Any],
    state: Dict[str, Any],
    portal_base_url: str,
    client_id: str,
    refresh_token: str,
    reason: str,
    persist: Callable[[], None],
) -> Dict[str, Any]:
    """Redeem the refresh token; on terminal failure quarantine state + pool, persist, re-raise."""
    from auth.providers.nous_store import (
        _quarantine_nous_oauth_state,
        _quarantine_nous_pool_entries,
    )
    from auth.providers.nous import _refresh_access_token
    from auth.oauth import _is_terminal_nous_refresh_error

    try:
        return _refresh_access_token(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            refresh_token=refresh_token,
        )
    except AuthError as exc:
        if _is_terminal_nous_refresh_error(exc):
            _quarantine_nous_oauth_state(state, exc, reason=reason)
            _quarantine_nous_pool_entries(auth_store, exc, reason=reason)
            persist()
        raise


def _apply_nous_refreshed_tokens(
    state: Dict[str, Any],
    refreshed: Dict[str, Any],
    refresh_token: str,
    *,
    inference_base_url: Optional[str] = None,
) -> None:
    """Write a successful Nous token-refresh payload into *state* (tokens + expiry fields).

    *inference_base_url*, when given, is the healed network-provenance URL to persist alongside
    the rotated tokens (key order in auth.json is preserved from the original login shape).
    """
    from auth.oauth import _coerce_ttl_seconds

    now = datetime.now(timezone.utc)
    access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
    state["access_token"] = refreshed["access_token"]
    state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
    state["token_type"] = (
        refreshed.get("token_type") or state.get("token_type") or "Bearer"
    )
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
        timeout=httpx.Timeout(timeout_seconds),
        headers={"Accept": "application/json"},
        verify=verify,
    )


def _agent_key_is_usable(state: Dict[str, Any], min_ttl_seconds: int) -> bool:
    from auth.token_validation import _nonempty_str

    key = state.get("agent_key")
    return _nonempty_str(key) and _nous_invoke_jwt_is_usable(
        key,
        scope=state.get("scope"),
        expires_at=state.get("agent_key_expires_at"),
        min_ttl_seconds=max(0, int(min_ttl_seconds)),
    )


def refresh_nous_oauth_pure(
    access_token: str,
    refresh_token: str,
    client_id: str,
    portal_base_url: str,
    inference_base_url: str,
    *,
    token_type: str = "Bearer",
    scope: str = DEFAULT_NOUS_SCOPE,
    obtained_at: Optional[str] = None,
    expires_at: Optional[str] = None,
    agent_key: Optional[str] = None,
    agent_key_expires_at: Optional[str] = None,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    on_state_update: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> Dict[str, Any]:
    """Refresh Nous OAuth state without mutating auth.json directly.

    ``on_state_update`` fires after a successful access-token refresh so callers owning persistent
    state can save the rotated refresh token before later validation can fail.
    """
    return refresh_nous_oauth_from_state(
        {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "client_id": client_id,
            "portal_base_url": portal_base_url,
            "inference_base_url": inference_base_url,
            "token_type": token_type,
            "scope": scope,
            "obtained_at": obtained_at,
            "expires_at": expires_at,
            "agent_key": agent_key,
            "agent_key_expires_at": agent_key_expires_at,
            "tls": {"insecure": insecure, "ca_bundle": ca_bundle},
        },
        timeout_seconds=timeout_seconds,
        force_refresh=force_refresh,
        on_state_update=on_state_update,
    )


def refresh_nous_oauth_from_state(
    src: Dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
    force_refresh: bool = False,
    on_state_update: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> Dict[str, Any]:
    """Refresh Nous OAuth from a state dict (defaults filled in) without mutating auth.json."""
    from auth.providers.nous import (
        _assert_nous_inference_jwt_usable,
        _refresh_access_token,
        _select_nous_invoke_jwt,
    )
    from auth.oauth import _resolve_verify

    tls = src.get("tls") or {}
    insecure, ca_bundle = tls.get("insecure"), tls.get("ca_bundle")
    state: Dict[str, Any] = {
        "access_token": src.get("access_token", ""),
        "refresh_token": src.get("refresh_token", ""),
        "client_id": src.get("client_id") or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": (
            src.get("portal_base_url") or DEFAULT_NOUS_PORTAL_URL
        ).rstrip("/"),
        "inference_base_url": (
            src.get("inference_base_url") or DEFAULT_NOUS_INFERENCE_URL
        ).rstrip("/"),
        "token_type": src.get("token_type") or "Bearer",
        "scope": src.get("scope") or DEFAULT_NOUS_SCOPE,
        "obtained_at": src.get("obtained_at"),
        "expires_at": src.get("expires_at"),
        "agent_key": src.get("agent_key"),
        "agent_key_expires_at": src.get("agent_key_expires_at"),
        "tls": {"insecure": bool(insecure), "ca_bundle": ca_bundle},
    }
    verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
    with _nous_http_client(timeout_seconds or 15.0, verify) as client:
        current_invoke_jwt_status = _state_invoke_jwt_status(
            state, state.get("access_token")
        )
        if force_refresh or current_invoke_jwt_status is not None:
            refresh_token_value = state.get("refresh_token")
            if not isinstance(refresh_token_value, str) or not refresh_token_value:
                if current_invoke_jwt_status is not None:
                    raise _unusable_invoke_jwt_error(
                        current_invoke_jwt_status, no_refresh_token=True
                    )
                raise _nous_err(
                    "No refresh token is available for Nous Portal.",
                    "nous_auth_missing_refresh_token",
                    relogin=True,
                )
            refreshed = _refresh_access_token(
                client=client,
                portal_base_url=state["portal_base_url"],
                client_id=state["client_id"],
                refresh_token=refresh_token_value,
            )
            _apply_nous_refreshed_tokens(
                state,
                refreshed,
                refresh_token_value,
                inference_base_url=_healed_nous_inference_url(refreshed),
            )
            if on_state_update is not None:
                on_state_update(dict(state), "post_refresh_access_token")
        _assert_nous_inference_jwt_usable(state)
        _select_nous_invoke_jwt(state)
    return state


def persist_nous_credentials(
    creds: Dict[str, Any], *, label: Optional[str] = None, environment
):
    """Persist Nous OAuth credentials as the singleton provider state.

    Nous credentials are read from ``providers.nous`` (401 recovery, pool seeding) AND
    ``credential_pool.nous`` (runtime ``pool.select()``); a pool-only write broke expiry recovery.
    So: write the singleton, mirror to the shared store, then ``load_pool("nous")`` upserts the
    canonical ``device_code`` entry in place. ``label`` rides in the singleton so re-seeding keeps
    it.
    """
    from auth.providers.nous_store import _write_shared_nous_state

    environment.require_current_scope()
    pass
    from auth.provider_state import _save_active_provider_state
    from auth.credential_pool import load_pool

    state = dict(creds)
    if label and str(label).strip():
        state["label"] = str(label).strip()
    _save_active_provider_state("nous", state)
    _write_shared_nous_state(state)
    pool = load_pool("nous", environment=environment)
    return next(
        (e for e in pool.entries() if e.source == NOUS_DEVICE_CODE_SOURCE), None
    )


def _sync_nous_pool_from_auth_store(*, environment) -> None:
    """Best-effort pool reseed after providers.nous changes; never fail login."""
    environment.require_current_scope()
    pass
    try:
        from auth.credential_pool import load_pool

        load_pool("nous", environment=environment)
    except Exception as exc:
        logger.debug("Failed to sync Nous credential pool from auth store: %s", exc)


def _nous_effective_routing(state: Dict[str, Any]) -> tuple[str, str, str, str]:
    """``(portal_url, stored_inference_url, effective_inference_url, client_id)`` from *state*.

    The stored inference URL is re-validated network-provenance (persisted); the effective one
    layers the runtime-only ``NOUS_INFERENCE_BASE_URL`` override on top and is never persisted.
    """
    from auth.providers.nous import _NOUS_PORTAL_ALLOWED_HOSTS
    from auth.oauth import _optional_base_url

    portal_url = (
        _optional_base_url(state.get("portal_base_url")) or DEFAULT_NOUS_PORTAL_URL
    ).rstrip("/")
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
            scheme == "http" and portal_host in {"localhost", "127.0.0.1"}
        )
        if (
            not portal_host
            or portal_host not in _NOUS_PORTAL_ALLOWED_HOSTS
            or not trusted_scheme
        ):
            logger.warning(
                "auth: ignoring invalid portal_base_url %r "
                "(host %r or scheme not allowed), using default",
                portal_url,
                portal_host,
            )
            portal_url = DEFAULT_NOUS_PORTAL_URL
    # A guest never falls back to the paid host: the gateway cross-refuses an anonymous JWT there
    # (400 naming the welcome host), so an absent or disallowed URL heals to the welcome literal.
    from auth.providers.nous_guest import is_guest_state

    stored_inference_url = _validate_nous_inference_url_from_network(
        _optional_base_url(state.get("inference_base_url"))
    ) or (
        DEFAULT_NOUS_WELCOME_URL
        if is_guest_state(state)
        else DEFAULT_NOUS_INFERENCE_URL
    )
    return (
        portal_url,
        stored_inference_url,
        _nous_inference_env_override() or stored_inference_url,
        str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID),
    )


class _NousRuntimeResolve:
    """Working set for one ``resolve_nous_runtime_credentials`` call.

    Holds the token pair + routing tuple that shared-store merges / refreshes replace mid-flight.
    ``persist`` skips writes where only derived TTL countdowns changed (keeps the mtime-keyed
    auth-status cache warm) and mirrors every real write to the shared store (best-effort).
    """

    def __init__(
        self,
        auth_store: Dict[str, Any],
        state: Dict[str, Any],
        state_source_path: Optional[Path],
        *,
        force_refresh: bool,
        stale_access_token: Optional[str],
        timeout_seconds: float,
    ) -> None:
        self.auth_store, self.state, self._source_path = (
            auth_store,
            state,
            state_source_path,
        )
        self.force_refresh, self.stale_access_token = force_refresh, stale_access_token
        self.timeout_seconds = timeout_seconds
        self.sequence_id = uuid.uuid4().hex[:12]
        self._persisted_state = dict(state)
        self.persisted_any = False
        self.access_token = state.get("access_token")
        self.refresh_token = state.get("refresh_token")
        self._reload_routing()

    def _reload_routing(self) -> None:
        (
            self.portal_base_url,
            self.stored_inference_base_url,
            self.inference_base_url,
            self.client_id,
        ) = _nous_effective_routing(self.state)

    def persist(self, reason: str) -> None:
        from auth.providers.nous_store import _write_shared_nous_state
        from auth.provider_state import _save_provider_state_to_source

        state = self.state
        persisted = _nous_effective_provider_state(self._persisted_state)
        if _nous_effective_provider_state(state) == persisted:
            _oauth_trace(
                "nous_state_persist_skipped",
                sequence_id=self.sequence_id,
                reason=reason,
            )
            return
        try:
            _save_provider_state_to_source(
                self.auth_store, "nous", state, self._source_path
            )
        except Exception as exc:
            _oauth_trace(
                "nous_state_persist_failed",
                sequence_id=self.sequence_id,
                reason=reason,
                error_type=type(exc).__name__,
            )
            raise
        _oauth_trace(
            "nous_state_persisted",
            sequence_id=self.sequence_id,
            reason=reason,
            refresh_token_fp=_token_fingerprint(state.get("refresh_token")),
            access_token_fp=_token_fingerprint(state.get("access_token")),
        )
        self._persisted_state = dict(state)
        self.persisted_any = True
        _write_shared_nous_state(state)

    def shared_lock(self):
        from auth.providers.nous_store import (
            _nous_shared_store_lock,
            _shared_lock_timeout,
        )

        return _nous_shared_store_lock(
            timeout_seconds=_shared_lock_timeout(self.timeout_seconds)
        )

    def has_access_token(self) -> bool:
        return isinstance(self.access_token, str) and bool(self.access_token)

    def invoke_jwt_status(self) -> Optional[str]:
        return _state_invoke_jwt_status(self.state, self.access_token)

    def merge_shared(self) -> bool:
        """Adopt fresher shared-store tokens (caller holds the shared lock). True when merged."""
        from auth.providers.nous_store import _merge_shared_nous_oauth_state

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
        if (
            self.force_refresh
            and self.stale_access_token
            and isinstance(token, str)
            and token
            and token != self.stale_access_token
            and self.invoke_jwt_status() is None
        ):
            _oauth_trace(
                "refresh_skipped_peer_rotated",
                sequence_id=self.sequence_id,
                access_token_fp=_token_fingerprint(token),
            )
            self.force_refresh = False

    def refresh(self, client: httpx.Client, invoke_jwt_status: Optional[str]) -> None:
        """Redeem the refresh token, apply + persist the rotated pair (caller holds both locks)."""
        if not isinstance(self.refresh_token, str) or not self.refresh_token:
            raise _unusable_invoke_jwt_error(
                invoke_jwt_status or "force_refresh", no_refresh_token=True
            )
        refresh_reason = (
            "force_refresh"
            if self.force_refresh
            else (invoke_jwt_status or "access_unusable")
        )
        _oauth_trace(
            "refresh_start",
            sequence_id=self.sequence_id,
            reason=refresh_reason,
            refresh_token_fp=_token_fingerprint(self.refresh_token),
        )
        refreshed = _refresh_nous_or_quarantine(
            client=client,
            auth_store=self.auth_store,
            state=self.state,
            portal_base_url=self.portal_base_url,
            client_id=self.client_id,
            refresh_token=self.refresh_token,
            reason="runtime_access_refresh_failure",
            persist=lambda: self.persist("terminal_runtime_access_refresh_failure"),
        )
        previous_refresh_token = self.refresh_token
        # The validated, network-provenance URL is what gets persisted (with the rotated tokens,
        # so a later JWT validation failure cannot leave the stores on stale metadata). The
        # NOUS_INFERENCE_BASE_URL env override is layered on for the client/return value only.
        self.stored_inference_base_url = _healed_nous_inference_url(refreshed)
        self.inference_base_url = (
            _nous_inference_env_override() or self.stored_inference_base_url
        )
        _apply_nous_refreshed_tokens(
            self.state,
            refreshed,
            self.refresh_token,
            inference_base_url=self.stored_inference_base_url,
        )
        self.access_token = self.state["access_token"]
        self.refresh_token = self.state["refresh_token"]
        _oauth_trace(
            "refresh_success",
            sequence_id=self.sequence_id,
            reason=refresh_reason,
            previous_refresh_token_fp=_token_fingerprint(previous_refresh_token),
            new_refresh_token_fp=_token_fingerprint(self.refresh_token),
        )
        # Persist immediately so validation failures cannot drop rotated refresh tokens.
        self.persist("post_refresh_access_token")

    def ensure_usable_access_token(self, client: httpx.Client) -> None:
        """Merge from the shared store / refresh until the access token is a usable invoke JWT."""
        from auth.providers.nous_guest import is_guest_state, refresh_guest_state

        if is_guest_state(self.state):
            # Guest seam: the anon_ credential is the refresh material; re-exchange instead of
            # redeeming a rotating refresh token. Quarantine never applies to a guest.
            if self.force_refresh or self.invoke_jwt_status() is not None:
                refresh_guest_state(self.state, client)
                self.access_token = self.state["access_token"]
                self.stored_inference_base_url = (
                    self.state.get("inference_base_url")
                    or self.stored_inference_base_url
                )
                self.inference_base_url = (
                    _nous_inference_env_override() or self.stored_inference_base_url
                )
                self.persist("guest_exchange")
            return
        if not self.has_access_token():
            with self.shared_lock():
                if self.merge_shared():
                    self.persist("runtime_shared_merge_missing_access_token")
        if not self.has_access_token():
            raise _nous_err(
                "No access token found for Nous Portal login.",
                "nous_auth_missing_access_token",
                relogin=True,
            )
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
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    stale_access_token: Optional[str] = None,
    environment,
) -> Dict[str, Any]:
    """Resolve Nous inference credentials for runtime use (refreshing under the auth-store lock).

    A guest whose ``anon_`` credential NAS no longer knows (reaped or claimed) is retired and a new
    identity is set up once, transparently -- the one client rule covering both reap and claim.
    """
    environment.require_current_scope()
    from auth.providers.nous_guest import (
        AnonCredentialDead,
        clear_dead_guest,
        ensure_portal_identity,
    )

    try:
        return _resolve_nous_runtime_credentials(
            timeout_seconds=timeout_seconds,
            insecure=insecure,
            ca_bundle=ca_bundle,
            force_refresh=force_refresh,
            stale_access_token=stale_access_token,
            environment=environment,
        )
    except AnonCredentialDead as dead_exc:
        from auth.provider_state import get_provider_auth_state
        from auth.providers.nous_guest import ANON_ACCOUNT_LOCKED

        dead = get_provider_auth_state("nous") or {}
        clear_dead_guest(
            str(dead_exc.code or "anon_credential_dead"),
            dead_token=dead.get("anon_token"),
        )
        # A locked account is retired but never silently replaced: the way forward is a sign-in.
        if dead_exc.code == ANON_ACCOUNT_LOCKED:
            raise
        if (
            ensure_portal_identity(
                explicit=True, timeout_seconds=timeout_seconds, environment=environment
            )
            is None
        ):
            raise
        return _resolve_nous_runtime_credentials(
            timeout_seconds=timeout_seconds,
            insecure=insecure,
            ca_bundle=ca_bundle,
            environment=environment,
        )


def _resolve_nous_runtime_credentials(
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    stale_access_token: Optional[str] = None,
    environment,
) -> Dict[str, Any]:
    """Resolve Nous inference credentials for runtime use (refreshing under the auth-store lock).

    ``stale_access_token`` is the bearer that just failed upstream (401): with ``force_refresh``,
    the refresh POST is skipped if the store (re-read under the lock) already holds a *different*
    usable token — a peer won the rotation; adopt it rather than invalidate a sibling's token.
    """
    environment.require_current_scope()
    from auth.providers.nous import (
        _assert_nous_inference_jwt_usable,
        _select_nous_invoke_jwt,
        _sync_nous_pool_from_auth_store,
    )
    from auth.oauth import _resolve_verify, _tls_state_from_verify
    from auth.store import _auth_file_path
    from auth.provider_state import _provider_state_transaction

    with _provider_state_transaction("nous") as (auth_store, state, state_source_path):
        if not state:
            raise _nous_err(
                "Hermes is not logged into Nous Portal.",
                "nous_auth_missing",
                relogin=True,
            )
        run = _NousRuntimeResolve(
            auth_store,
            state,
            state_source_path,
            force_refresh=force_refresh,
            stale_access_token=stale_access_token,
            timeout_seconds=timeout_seconds,
        )
        verify = _resolve_verify(
            insecure=insecure, ca_bundle=ca_bundle, auth_state=state
        )
        _oauth_trace(
            "nous_runtime_credentials_start",
            sequence_id=run.sequence_id,
            refresh_token_fp=_token_fingerprint(state.get("refresh_token")),
        )
        with _nous_http_client(timeout_seconds or 15.0, verify) as client:
            run.ensure_usable_access_token(client)
            _assert_nous_inference_jwt_usable(state, access_token=run.access_token)
            _select_nous_invoke_jwt(
                state, access_token=run.access_token, sequence_id=run.sequence_id
            )
            # Persist routing and TLS metadata for non-interactive refresh — the validated,
            # network-provenance URL, NEVER the env override (a runtime-only overlay; persisting
            # it would leak a dev/staging host into auth.json and survive unsetting it).
            state.update(
                portal_base_url=run.portal_base_url,
                client_id=run.client_id,
                inference_base_url=run.stored_inference_base_url,
                tls=_tls_state_from_verify(verify),
            )
        run.persist("resolve_nous_runtime_credentials_final")
    if run.persisted_any:
        _sync_nous_pool_from_auth_store(environment=environment)
    api_key = state.get("agent_key")
    if not isinstance(api_key, str) or not api_key:
        raise _nous_err("Failed to resolve a Nous inference API key", "server_error")
    expires_at = state.get("agent_key_expires_at")
    return {
        "provider": "nous",
        "base_url": run.inference_base_url,
        "api_key": api_key,
        "key_id": state.get("agent_key_id"),
        "expires_at": expires_at,
        "expires_in": _remaining_ttl(expires_at, state.get("agent_key_expires_in")),
        "source": NOUS_AUTH_PATH_INVOKE_JWT,
        # Public semantic source label; the concrete store is exposed separately for diagnostics.
        # Refresh persistence uses state_source_path internally and must not overload this field.
        "auth_path": NOUS_AUTH_PATH_INVOKE_JWT,
        "state_path": str(state_source_path or _auth_file_path()),
    }


def _empty_nous_auth_status() -> Dict[str, Any]:
    return {
        "logged_in": False,
        "portal_base_url": None,
        "inference_base_url": None,
        "access_expires_at": None,
        "agent_key_expires_at": None,
        "has_refresh_token": False,
        "inference_credential_present": False,
        "credential_source": None,
    }


def _nous_status_from_state(
    state: Dict[str, Any], *, logged_in: bool, source: str
) -> Dict[str, Any]:
    """Auth-store-backed Nous status snapshot (shared by the live and refresh-free variants)."""
    from auth.providers.nous_guest import is_guest_state

    access_token = state.get("access_token")
    account_tier = state.get("account_tier")
    return {
        "logged_in": logged_in,
        "portal_base_url": state.get("portal_base_url"),
        "inference_base_url": state.get("inference_base_url"),
        "access_expires_at": state.get("expires_at"),
        "agent_key_expires_at": state.get("agent_key_expires_at"),
        "has_refresh_token": bool(state.get("refresh_token")),
        "access_token": access_token,
        "inference_credential_present": bool(access_token or state.get("agent_key")),
        "credential_source": "auth_store",
        "source": source,
        # Free tier: display surfaces render it with the free-tier copy, never as an account login.
        "account_tier": account_tier if isinstance(account_tier, str) else None,
        "free_tier": is_guest_state(state),
    }


def _terminal_quarantine_marker(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The persisted ``last_auth_error`` when it is a terminal quarantine with no credential left.

    Only terminal while there is no usable credential: if a later login repopulated tokens the stale
    marker must not keep reporting terminal.
    """
    last_err = state.get("last_auth_error")
    if (
        isinstance(last_err, dict)
        and last_err.get("relogin_required")
        and not (state.get("access_token") or state.get("refresh_token"))
    ):
        return last_err
    return None


from auth.token_validation import _nous_invoke_jwt_status, _nous_invoke_jwt_is_usable

_NOUS_PORTAL_ALLOWED_HOSTS: FrozenSet[str] = frozenset({
    "portal.nousresearch.com",
    "localhost",
    "127.0.0.1",
})

_RESOLVE_TOKEN_CACHE_LOCK = threading.Lock()

_RESOLVE_TOKEN_CACHE: "dict[str, tuple[float, str]]" = {}

_RESOLVE_TOKEN_CACHE_TTL_S = 5.0


def _nous_portal_base_url(state: Dict[str, Any]) -> str:
    """HERMES_PORTAL_BASE_URL / NOUS_PORTAL_BASE_URL is the trusted operator override and wins
    OUTRIGHT, bypassing the host allowlist (which exists to reject an untrusted network-provided
    value, not one the operator configured). Otherwise the stored/default value, allowlist-gated."""
    env_portal_override = _nous_portal_env_override()
    if env_portal_override:
        return env_portal_override.rstrip("/")
    portal_base_url = (
        _optional_base_url(state.get("portal_base_url"))
        or auth_store_migrations.DEFAULT_NOUS_PORTAL_URL
    )
    portal_base_url = portal_base_url.rstrip("/")
    host = urlparse(portal_base_url).hostname
    if host and host not in _NOUS_PORTAL_ALLOWED_HOSTS:
        logger.warning(
            "auth: ignoring invalid portal_base_url %r (host %r not in allowlist), using default",
            portal_base_url,
            host,
        )
        return auth_store_migrations.DEFAULT_NOUS_PORTAL_URL
    return portal_base_url


def resolve_nous_access_token(
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    refresh_skew_seconds: int = ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> str:
    """Resolve a refresh-aware Nous Portal access token for managed tool gateways."""
    # Only a default-TLS resolution is memoised; error paths never populate the memo.
    from auth.providers.nous_store import (
        _merge_shared_nous_oauth_state,
        _nous_shared_store_lock,
        _write_shared_nous_state,
    )

    memoable = not insecure and ca_bundle is None
    cache_key = hermes_home_key()
    if memoable:
        with _RESOLVE_TOKEN_CACHE_LOCK:
            cached = _RESOLVE_TOKEN_CACHE.get(cache_key)
        if (
            cached is not None
            and (time.monotonic() - cached[0]) < _RESOLVE_TOKEN_CACHE_TTL_S
        ):
            return cached[1]

    def _memo(token: str) -> str:
        if memoable:
            with _RESOLVE_TOKEN_CACHE_LOCK:
                _RESOLVE_TOKEN_CACHE[cache_key] = (time.monotonic(), token)
        return token

    with auth_provider_state._provider_state_transaction("nous") as (
        auth_store,
        state,
        state_source_path,
    ):
        if not state:
            raise _nous_err(
                "Hermes is not logged into Nous Portal.",
                "nous_auth_missing",
                relogin=True,
            )
        portal_base_url = _nous_portal_base_url(state)
        client_id = str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID)
        verify = _resolve_verify(
            insecure=insecure, ca_bundle=ca_bundle, auth_state=state
        )
        persist = lambda: auth_provider_state._save_provider_state_to_source(  # noqa: E731
            auth_store, "nous", state, state_source_path
        )

        lock_timeout = max(
            timeout_seconds + 5.0, auth_storage.AUTH_LOCK_TIMEOUT_SECONDS
        )
        with _nous_shared_store_lock(timeout_seconds=lock_timeout):
            from auth.providers.nous_guest import is_guest_state, refresh_guest_state

            if is_guest_state(state):
                # Guest seam: the anon_ credential is the identity; a first use has no access token
                # yet and an expired one is re-exchanged. No refresh token, no quarantine.
                access_token = state.get("access_token")
                if (
                    isinstance(access_token, str)
                    and access_token
                    and not _is_expiring(state.get("expires_at"), refresh_skew_seconds)
                ):
                    return _memo(access_token)
                with httpx.Client(
                    timeout=httpx.Timeout(timeout_seconds or 15.0),
                    headers={"Accept": "application/json"},
                    verify=verify,
                ) as client:
                    refresh_guest_state(state, client)
                persist()
                _write_shared_nous_state(state)
                return _memo(state["access_token"])

            merged_shared = _merge_shared_nous_oauth_state(state)
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")
            if not isinstance(access_token, str) or not access_token:
                raise _nous_err(
                    "No access token found for Nous Portal login.",
                    "nous_auth_missing_access_token",
                    relogin=True,
                )

            if not _is_expiring(state.get("expires_at"), refresh_skew_seconds):
                if merged_shared:
                    persist()
                # Memoise the valid-token fast path too: each check_fn otherwise pays two
                # cross-process file locks to get here. The token has >= refresh_skew_seconds (>=
                # 120s) of life, so a 5s memo can never serve an expired token.
                return _memo(access_token)

            if not isinstance(refresh_token, str) or not refresh_token:
                raise _nous_err(
                    "Session expired and no refresh token is available.",
                    "nous_auth_missing_refresh_token",
                    relogin=True,
                )

            with httpx.Client(
                timeout=httpx.Timeout(timeout_seconds or 15.0),
                headers={"Accept": "application/json"},
                verify=verify,
            ) as client:
                refreshed = _refresh_nous_or_quarantine(
                    client=client,
                    auth_store=auth_store,
                    state=state,
                    portal_base_url=portal_base_url,
                    client_id=client_id,
                    refresh_token=refresh_token,
                    reason="managed_access_token_refresh_failure",
                    persist=persist,
                )

            _apply_nous_refreshed_tokens(state, refreshed, refresh_token)
            state["portal_base_url"] = portal_base_url
            state["client_id"] = client_id
            state["tls"] = _tls_state_from_verify(verify)
            persist()
            _write_shared_nous_state(state)
            return _memo(state["access_token"])
