"""Nous Portal provider auth.

F-C2 step 4 — second per-provider extraction from
``hermes_cli/auth/__init__.py`` (following the F-C2 step 2 codex
extraction recipe).

Nous authentication is an OAuth 2.0 device-code flow with short-
lived "agent keys" minted from a long-lived refresh token. Hermes
keeps its own token store (``~/.hermes/auth.json``) separate from
any other Nous clients so rotating the refresh token on one app
cannot invalidate another.

Public API (re-exported from ``hermes_cli.auth`` via
``from .nous import *`` at the bottom of the package ``__init__``):

    fetch_nous_models                  — curated model list (GET /models).
    resolve_nous_access_token          — refresh-aware access token for
                                         managed tool gateways.
    refresh_nous_oauth_pure            — stateless refresh + mint without
                                         touching auth.json (used by the
                                         standalone helper wrapper below
                                         and by the webui bootstrap flow).
    refresh_nous_oauth_from_state      — thin wrapper over the pure variant.
    resolve_nous_runtime_credentials   — main runtime resolver (holds the
                                         auth-store lock, handles refresh
                                         and agent-key minting with a
                                         single retry on invalid_token).
    get_nous_auth_status               — snapshot for `hermes status`.

Internal helpers (also re-exported so module-level monkeypatch strings
like ``hermes_cli.auth._refresh_access_token`` keep resolving to the
same object that lives here):

    _request_device_code
    _poll_for_token
    _refresh_access_token
    _mint_agent_key
    _agent_key_is_usable
    _nous_device_code_login
    _login_nous

Test-patching gotcha: internal calls inside these functions use bare
names (e.g. ``_refresh_access_token(...)``). Once a call is dispatched
from *inside* nous.py, Python resolves the name in this module's own
namespace. Monkey-patching ``hermes_cli.auth._refresh_access_token``
(the package-level re-export) will not intercept those calls — patch
``hermes_cli.auth.nous._refresh_access_token`` instead. See
``tests/test_auth_nous_provider.py`` for the canonical pattern.

Shared primitives (``AuthError``, the auth-store lock,
``_auth_file_path``, OAuth trace helpers, TLS verify resolution,
``_update_config_for_provider``, etc.) stay in
``hermes_cli.auth.__init__`` and are imported here.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from hermes_cli.auth.types import NousCredentials
from hermes_cli.auth import (
    ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    DEFAULT_AGENT_KEY_MIN_TTL_SECONDS,
    DEFAULT_NOUS_CLIENT_ID,
    DEFAULT_NOUS_INFERENCE_URL,
    DEFAULT_NOUS_PORTAL_URL,
    DEFAULT_NOUS_SCOPE,
    DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS,
    PROVIDER_REGISTRY,
    AuthError,
    ProviderConfig,
    _auth_store_lock,
    _coerce_ttl_seconds,
    _is_expiring,
    _is_remote_session,
    _load_auth_store,
    _load_provider_state,
    _oauth_trace,
    _optional_base_url,
    _parse_iso_timestamp,
    _prompt_model_selection,
    _resolve_verify,
    _save_auth_store,
    _save_model_choice,
    _save_provider_state,
    _token_fingerprint,
    _update_config_for_provider,
    format_auth_error,
    get_provider_auth_state,
)

logger = logging.getLogger(__name__)

__all__ = [
    "_request_device_code",
    "_poll_for_token",
    "_refresh_access_token",
    "_mint_agent_key",
    "_agent_key_is_usable",
    "fetch_nous_models",
    "resolve_nous_access_token",
    "refresh_nous_oauth_pure",
    "refresh_nous_oauth_from_state",
    "resolve_nous_runtime_credentials",
    "get_nous_auth_status",
    "_nous_device_code_login",
    "_login_nous",
]


# =============================================================================
# OAuth Device Code Flow — generic, parameterized by provider.
# Currently only Nous uses this path; kept here (not in the package root) so
# every Nous-specific OAuth primitive is colocated and test monkey-patching
# is predictable.
# =============================================================================

def _request_device_code(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    scope: Optional[str],
) -> Dict[str, Any]:
    """POST to the device code endpoint. Returns device_code, user_code, etc."""
    response = client.post(
        f"{portal_base_url}/api/oauth/device/code",
        data={
            "client_id": client_id,
            **({"scope": scope} if scope else {}),
        },
    )
    response.raise_for_status()
    data = response.json()

    required_fields = [
        "device_code", "user_code", "verification_uri",
        "verification_uri_complete", "expires_in", "interval",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Device code response missing fields: {', '.join(missing)}")
    return data


def _poll_for_token(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    device_code: str,
    expires_in: int,
    poll_interval: int,
) -> Dict[str, Any]:
    """Poll the token endpoint until the user approves or the code expires."""
    deadline = time.time() + max(1, expires_in)
    current_interval = max(1, min(poll_interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS))

    while time.time() < deadline:
        response = client.post(
            f"{portal_base_url}/api/oauth/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": client_id,
                "device_code": device_code,
            },
        )

        if response.status_code == 200:
            payload = response.json()
            if "access_token" not in payload:
                raise ValueError("Token response did not include access_token")
            return payload

        try:
            error_payload = response.json()
        except Exception:
            response.raise_for_status()
            raise RuntimeError("Token endpoint returned a non-JSON error response")

        error_code = error_payload.get("error", "")
        if error_code == "authorization_pending":
            time.sleep(current_interval)
            continue
        if error_code == "slow_down":
            current_interval = min(current_interval + 1, 30)
            time.sleep(current_interval)
            continue

        description = error_payload.get("error_description") or "Unknown authentication error"
        raise RuntimeError(f"{error_code}: {description}")

    raise TimeoutError("Timed out waiting for device authorization")


# =============================================================================
# Nous Portal — token refresh, agent key minting, model discovery
# =============================================================================

def _refresh_access_token(
    *,
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    refresh_token: str,
) -> Dict[str, Any]:
    response = client.post(
        f"{portal_base_url}/api/oauth/token",
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        },
    )

    if response.status_code == 200:
        payload = response.json()
        if "access_token" not in payload:
            raise AuthError("Refresh response missing access_token",
                            provider="nous", code="invalid_token", relogin_required=True)
        return payload

    try:
        error_payload = response.json()
    except Exception as exc:
        raise AuthError("Refresh token exchange failed",
                        provider="nous", relogin_required=True) from exc

    code = str(error_payload.get("error", "invalid_grant"))
    description = str(error_payload.get("error_description") or "Refresh token exchange failed")
    relogin = code in {"invalid_grant", "invalid_token"}
    raise AuthError(description, provider="nous", code=code, relogin_required=relogin)


def _mint_agent_key(
    *,
    client: httpx.Client,
    portal_base_url: str,
    access_token: str,
    min_ttl_seconds: int,
) -> Dict[str, Any]:
    """Mint (or reuse) a short-lived inference API key."""
    response = client.post(
        f"{portal_base_url}/api/oauth/agent-key",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"min_ttl_seconds": max(60, int(min_ttl_seconds))},
    )

    if response.status_code == 200:
        payload = response.json()
        if "api_key" not in payload:
            raise AuthError("Mint response missing api_key",
                            provider="nous", code="server_error")
        return payload

    try:
        error_payload = response.json()
    except Exception as exc:
        raise AuthError("Agent key mint request failed",
                        provider="nous", code="server_error") from exc

    code = str(error_payload.get("error", "server_error"))
    description = str(error_payload.get("error_description") or "Agent key mint request failed")
    relogin = code in {"invalid_token", "invalid_grant"}
    raise AuthError(description, provider="nous", code=code, relogin_required=relogin)


def fetch_nous_models(
    *,
    inference_base_url: str,
    api_key: str,
    timeout_seconds: float = 15.0,
    verify: bool | str = True,
) -> List[str]:
    """Fetch available model IDs from the Nous inference API."""
    timeout = httpx.Timeout(timeout_seconds)
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        response = client.get(
            f"{inference_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )

    if response.status_code != 200:
        description = f"/models request failed with status {response.status_code}"
        try:
            err = response.json()
            description = str(err.get("error_description") or err.get("error") or description)
        except Exception as e:
            logger.debug("Could not parse error response JSON: %s", e)
        raise AuthError(description, provider="nous", code="models_fetch_failed")

    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    model_ids: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            mid = model_id.strip()
            # Skip Hermes models — they're not reliable for agentic tool-calling
            if "hermes" in mid.lower():
                continue
            model_ids.append(mid)

    # Sort: prefer opus > pro > haiku/flash > sonnet (sonnet is cheap/fast,
    # users who want the best model should see opus first).
    def _model_priority(mid: str) -> tuple:
        low = mid.lower()
        if "opus" in low:
            return (0, mid)
        if "pro" in low and "sonnet" not in low:
            return (1, mid)
        if "sonnet" in low:
            return (3, mid)
        return (2, mid)

    model_ids.sort(key=_model_priority)
    return list(dict.fromkeys(model_ids))


def _agent_key_is_usable(state: Dict[str, Any], min_ttl_seconds: int) -> bool:
    key = state.get("agent_key")
    if not isinstance(key, str) or not key.strip():
        return False
    return not _is_expiring(state.get("agent_key_expires_at"), min_ttl_seconds)


def resolve_nous_access_token(
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    refresh_skew_seconds: int = ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> str:
    """Resolve a refresh-aware Nous Portal access token for managed tool gateways."""
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "nous")

        if not state:
            raise AuthError(
                "Hermes is not logged into Nous Portal.",
                provider="nous",
                relogin_required=True,
            )

        portal_base_url = (
            _optional_base_url(state.get("portal_base_url"))
            or os.getenv("HERMES_PORTAL_BASE_URL")
            or os.getenv("NOUS_PORTAL_BASE_URL")
            or DEFAULT_NOUS_PORTAL_URL
        ).rstrip("/")
        client_id = str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID)
        verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)

        access_token = state.get("access_token")
        refresh_token = state.get("refresh_token")
        if not isinstance(access_token, str) or not access_token:
            raise AuthError(
                "No access token found for Nous Portal login.",
                provider="nous",
                relogin_required=True,
            )

        if not _is_expiring(state.get("expires_at"), refresh_skew_seconds):
            return access_token

        if not isinstance(refresh_token, str) or not refresh_token:
            raise AuthError(
                "Session expired and no refresh token is available.",
                provider="nous",
                relogin_required=True,
            )

        timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)
        with httpx.Client(
            timeout=timeout,
            headers={"Accept": "application/json"},
            verify=verify,
        ) as client:
            refreshed = _refresh_access_token(
                client=client,
                portal_base_url=portal_base_url,
                client_id=client_id,
                refresh_token=refresh_token,
            )

        now = datetime.now(timezone.utc)
        access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
        state["access_token"] = refreshed["access_token"]
        state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
        state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
        state["scope"] = refreshed.get("scope") or state.get("scope")
        state["obtained_at"] = now.isoformat()
        state["expires_in"] = access_ttl
        state["expires_at"] = datetime.fromtimestamp(
            now.timestamp() + access_ttl,
            tz=timezone.utc,
        ).isoformat()
        state["portal_base_url"] = portal_base_url
        state["client_id"] = client_id
        state["tls"] = {
            "insecure": verify is False,
            "ca_bundle": verify if isinstance(verify, str) else None,
        }
        _save_provider_state(auth_store, "nous", state)
        _save_auth_store(auth_store)
        return state["access_token"]


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
    min_key_ttl_seconds: int = DEFAULT_AGENT_KEY_MIN_TTL_SECONDS,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    force_mint: bool = False,
) -> Dict[str, Any]:
    """Refresh Nous OAuth state without mutating auth.json."""
    state: Dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "client_id": client_id or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": (portal_base_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/"),
        "inference_base_url": (inference_base_url or DEFAULT_NOUS_INFERENCE_URL).rstrip("/"),
        "token_type": token_type or "Bearer",
        "scope": scope or DEFAULT_NOUS_SCOPE,
        "obtained_at": obtained_at,
        "expires_at": expires_at,
        "agent_key": agent_key,
        "agent_key_expires_at": agent_key_expires_at,
        "tls": {
            "insecure": bool(insecure),
            "ca_bundle": ca_bundle,
        },
    }
    verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
    timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)

    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        if force_refresh or _is_expiring(state.get("expires_at"), ACCESS_TOKEN_REFRESH_SKEW_SECONDS):
            refreshed = _refresh_access_token(
                client=client,
                portal_base_url=state["portal_base_url"],
                client_id=state["client_id"],
                refresh_token=state["refresh_token"],
            )
            now = datetime.now(timezone.utc)
            access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
            state["access_token"] = refreshed["access_token"]
            state["refresh_token"] = refreshed.get("refresh_token") or state["refresh_token"]
            state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
            state["scope"] = refreshed.get("scope") or state.get("scope")
            refreshed_url = _optional_base_url(refreshed.get("inference_base_url"))
            if refreshed_url:
                state["inference_base_url"] = refreshed_url
            state["obtained_at"] = now.isoformat()
            state["expires_in"] = access_ttl
            state["expires_at"] = datetime.fromtimestamp(
                now.timestamp() + access_ttl, tz=timezone.utc
            ).isoformat()

        if force_mint or not _agent_key_is_usable(state, max(60, int(min_key_ttl_seconds))):
            mint_payload = _mint_agent_key(
                client=client,
                portal_base_url=state["portal_base_url"],
                access_token=state["access_token"],
                min_ttl_seconds=min_key_ttl_seconds,
            )
            now = datetime.now(timezone.utc)
            state["agent_key"] = mint_payload.get("api_key")
            state["agent_key_id"] = mint_payload.get("key_id")
            state["agent_key_expires_at"] = mint_payload.get("expires_at")
            state["agent_key_expires_in"] = mint_payload.get("expires_in")
            state["agent_key_reused"] = bool(mint_payload.get("reused", False))
            state["agent_key_obtained_at"] = now.isoformat()
            minted_url = _optional_base_url(mint_payload.get("inference_base_url"))
            if minted_url:
                state["inference_base_url"] = minted_url

    return state


def refresh_nous_oauth_from_state(
    state: Dict[str, Any],
    *,
    min_key_ttl_seconds: int = DEFAULT_AGENT_KEY_MIN_TTL_SECONDS,
    timeout_seconds: float = 15.0,
    force_refresh: bool = False,
    force_mint: bool = False,
) -> Dict[str, Any]:
    """Refresh Nous OAuth from a state dict. Thin wrapper around refresh_nous_oauth_pure."""
    tls = state.get("tls") or {}
    return refresh_nous_oauth_pure(
        state.get("access_token", ""),
        state.get("refresh_token", ""),
        state.get("client_id", "hermes-cli"),
        state.get("portal_base_url", DEFAULT_NOUS_PORTAL_URL),
        state.get("inference_base_url", DEFAULT_NOUS_INFERENCE_URL),
        token_type=state.get("token_type", "Bearer"),
        scope=state.get("scope", DEFAULT_NOUS_SCOPE),
        obtained_at=state.get("obtained_at"),
        expires_at=state.get("expires_at"),
        agent_key=state.get("agent_key"),
        agent_key_expires_at=state.get("agent_key_expires_at"),
        min_key_ttl_seconds=min_key_ttl_seconds,
        timeout_seconds=timeout_seconds,
        insecure=tls.get("insecure"),
        ca_bundle=tls.get("ca_bundle"),
        force_refresh=force_refresh,
        force_mint=force_mint,
    )


def resolve_nous_runtime_credentials(
    *,
    min_key_ttl_seconds: int = DEFAULT_AGENT_KEY_MIN_TTL_SECONDS,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_mint: bool = False,
) -> NousCredentials:
    """
    Resolve Nous inference credentials for runtime use.

    Ensures access_token is valid (refreshes if needed) and a short-lived
    inference key is present with minimum TTL (mints/reuses as needed).
    Concurrent processes coordinate through the auth store file lock.

    Returns dict with: provider, base_url, api_key, key_id, expires_at,
    expires_in, source ("cache" or "portal").
    """
    min_key_ttl_seconds = max(60, int(min_key_ttl_seconds))
    sequence_id = uuid.uuid4().hex[:12]

    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "nous")

        if not state:
            raise AuthError("Hermes is not logged into Nous Portal.",
                            provider="nous", relogin_required=True)

        portal_base_url = (
            _optional_base_url(state.get("portal_base_url"))
            or os.getenv("HERMES_PORTAL_BASE_URL")
            or os.getenv("NOUS_PORTAL_BASE_URL")
            or DEFAULT_NOUS_PORTAL_URL
        ).rstrip("/")
        inference_base_url = (
            _optional_base_url(state.get("inference_base_url"))
            or os.getenv("NOUS_INFERENCE_BASE_URL")
            or DEFAULT_NOUS_INFERENCE_URL
        ).rstrip("/")
        client_id = str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID)

        def _persist_state(reason: str) -> None:
            try:
                _save_provider_state(auth_store, "nous", state)
                _save_auth_store(auth_store)
            except Exception as exc:
                _oauth_trace(
                    "nous_state_persist_failed",
                    sequence_id=sequence_id,
                    reason=reason,
                    error_type=type(exc).__name__,
                )
                raise
            _oauth_trace(
                "nous_state_persisted",
                sequence_id=sequence_id,
                reason=reason,
                refresh_token_fp=_token_fingerprint(state.get("refresh_token")),
                access_token_fp=_token_fingerprint(state.get("access_token")),
            )

        verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
        timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)
        _oauth_trace(
            "nous_runtime_credentials_start",
            sequence_id=sequence_id,
            force_mint=bool(force_mint),
            min_key_ttl_seconds=min_key_ttl_seconds,
            refresh_token_fp=_token_fingerprint(state.get("refresh_token")),
        )

        with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")

            if not isinstance(access_token, str) or not access_token:
                raise AuthError("No access token found for Nous Portal login.",
                                provider="nous", relogin_required=True)

            # Step 1: refresh access token if expiring
            if _is_expiring(state.get("expires_at"), ACCESS_TOKEN_REFRESH_SKEW_SECONDS):
                if not isinstance(refresh_token, str) or not refresh_token:
                    raise AuthError("Session expired and no refresh token is available.",
                                    provider="nous", relogin_required=True)

                _oauth_trace(
                    "refresh_start",
                    sequence_id=sequence_id,
                    reason="access_expiring",
                    refresh_token_fp=_token_fingerprint(refresh_token),
                )
                refreshed = _refresh_access_token(
                    client=client, portal_base_url=portal_base_url,
                    client_id=client_id, refresh_token=refresh_token,
                )
                now = datetime.now(timezone.utc)
                access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
                previous_refresh_token = refresh_token
                state["access_token"] = refreshed["access_token"]
                state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
                state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
                state["scope"] = refreshed.get("scope") or state.get("scope")
                refreshed_url = _optional_base_url(refreshed.get("inference_base_url"))
                if refreshed_url:
                    inference_base_url = refreshed_url
                state["obtained_at"] = now.isoformat()
                state["expires_in"] = access_ttl
                state["expires_at"] = datetime.fromtimestamp(
                    now.timestamp() + access_ttl, tz=timezone.utc
                ).isoformat()
                access_token = state["access_token"]
                refresh_token = state["refresh_token"]
                _oauth_trace(
                    "refresh_success",
                    sequence_id=sequence_id,
                    reason="access_expiring",
                    previous_refresh_token_fp=_token_fingerprint(previous_refresh_token),
                    new_refresh_token_fp=_token_fingerprint(refresh_token),
                )
                # Persist immediately so downstream mint failures cannot drop rotated refresh tokens.
                _persist_state("post_refresh_access_expiring")

            # Step 2: mint agent key if missing/expiring
            used_cached_key = False
            mint_payload: Optional[Dict[str, Any]] = None

            if not force_mint and _agent_key_is_usable(state, min_key_ttl_seconds):
                used_cached_key = True
                _oauth_trace("agent_key_reuse", sequence_id=sequence_id)
            else:
                try:
                    _oauth_trace(
                        "mint_start",
                        sequence_id=sequence_id,
                        access_token_fp=_token_fingerprint(access_token),
                    )
                    mint_payload = _mint_agent_key(
                        client=client, portal_base_url=portal_base_url,
                        access_token=access_token, min_ttl_seconds=min_key_ttl_seconds,
                    )
                except AuthError as exc:
                    _oauth_trace(
                        "mint_error",
                        sequence_id=sequence_id,
                        code=exc.code,
                    )
                    # Retry path: access token may be stale server-side despite local checks
                    latest_refresh_token = state.get("refresh_token")
                    if (
                        exc.code in {"invalid_token", "invalid_grant"}
                        and isinstance(latest_refresh_token, str)
                        and latest_refresh_token
                    ):
                        _oauth_trace(
                            "refresh_start",
                            sequence_id=sequence_id,
                            reason="mint_retry_after_invalid_token",
                            refresh_token_fp=_token_fingerprint(latest_refresh_token),
                        )
                        refreshed = _refresh_access_token(
                            client=client, portal_base_url=portal_base_url,
                            client_id=client_id, refresh_token=latest_refresh_token,
                        )
                        now = datetime.now(timezone.utc)
                        access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
                        state["access_token"] = refreshed["access_token"]
                        state["refresh_token"] = refreshed.get("refresh_token") or latest_refresh_token
                        state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
                        state["scope"] = refreshed.get("scope") or state.get("scope")
                        refreshed_url = _optional_base_url(refreshed.get("inference_base_url"))
                        if refreshed_url:
                            inference_base_url = refreshed_url
                        state["obtained_at"] = now.isoformat()
                        state["expires_in"] = access_ttl
                        state["expires_at"] = datetime.fromtimestamp(
                            now.timestamp() + access_ttl, tz=timezone.utc
                        ).isoformat()
                        access_token = state["access_token"]
                        refresh_token = state["refresh_token"]
                        _oauth_trace(
                            "refresh_success",
                            sequence_id=sequence_id,
                            reason="mint_retry_after_invalid_token",
                            previous_refresh_token_fp=_token_fingerprint(latest_refresh_token),
                            new_refresh_token_fp=_token_fingerprint(refresh_token),
                        )
                        # Persist retry refresh immediately for crash safety and cross-process visibility.
                        _persist_state("post_refresh_mint_retry")

                        mint_payload = _mint_agent_key(
                            client=client, portal_base_url=portal_base_url,
                            access_token=access_token, min_ttl_seconds=min_key_ttl_seconds,
                        )
                    else:
                        raise

            if mint_payload is not None:
                now = datetime.now(timezone.utc)
                state["agent_key"] = mint_payload.get("api_key")
                state["agent_key_id"] = mint_payload.get("key_id")
                state["agent_key_expires_at"] = mint_payload.get("expires_at")
                state["agent_key_expires_in"] = mint_payload.get("expires_in")
                state["agent_key_reused"] = bool(mint_payload.get("reused", False))
                state["agent_key_obtained_at"] = now.isoformat()
                minted_url = _optional_base_url(mint_payload.get("inference_base_url"))
                if minted_url:
                    inference_base_url = minted_url
                _oauth_trace(
                    "mint_success",
                    sequence_id=sequence_id,
                    reused=bool(mint_payload.get("reused", False)),
                )

            # Persist routing and TLS metadata for non-interactive refresh/mint
            state["portal_base_url"] = portal_base_url
            state["inference_base_url"] = inference_base_url
            state["client_id"] = client_id
            state["tls"] = {
                "insecure": verify is False,
                "ca_bundle": verify if isinstance(verify, str) else None,
            }

        _persist_state("resolve_nous_runtime_credentials_final")

    api_key = state.get("agent_key")
    if not isinstance(api_key, str) or not api_key:
        raise AuthError("Failed to resolve a Nous inference API key",
                        provider="nous", code="server_error")

    expires_at = state.get("agent_key_expires_at")
    expires_epoch = _parse_iso_timestamp(expires_at)
    expires_in = (
        max(0, int(expires_epoch - time.time()))
        if expires_epoch is not None
        else _coerce_ttl_seconds(state.get("agent_key_expires_in"))
    )

    return {
        "provider": "nous",
        "base_url": inference_base_url,
        "api_key": api_key,
        "key_id": state.get("agent_key_id"),
        "expires_at": expires_at,
        "expires_in": expires_in,
        "source": "cache" if used_cached_key else "portal",
    }


# =============================================================================
# Status helper
# =============================================================================

def get_nous_auth_status() -> Dict[str, Any]:
    """Status snapshot for `hermes status` output."""
    state = get_provider_auth_state("nous")
    if not state:
        return {
            "logged_in": False,
            "portal_base_url": None,
            "inference_base_url": None,
            "access_expires_at": None,
            "agent_key_expires_at": None,
            "has_refresh_token": False,
        }
    return {
        "logged_in": bool(state.get("access_token")),
        "portal_base_url": state.get("portal_base_url"),
        "inference_base_url": state.get("inference_base_url"),
        "access_expires_at": state.get("expires_at"),
        "agent_key_expires_at": state.get("agent_key_expires_at"),
        "has_refresh_token": bool(state.get("refresh_token")),
    }


# =============================================================================
# Device code login flow + login_nous orchestration
# =============================================================================

def _nous_device_code_login(
    *,
    portal_base_url: Optional[str] = None,
    inference_base_url: Optional[str] = None,
    client_id: Optional[str] = None,
    scope: Optional[str] = None,
    open_browser: bool = True,
    timeout_seconds: float = 15.0,
    insecure: bool = False,
    ca_bundle: Optional[str] = None,
    min_key_ttl_seconds: int = 5 * 60,
) -> Dict[str, Any]:
    """Run the Nous device-code flow and return full OAuth state without persisting."""
    pconfig = PROVIDER_REGISTRY["nous"]
    portal_base_url = (
        portal_base_url
        or os.getenv("HERMES_PORTAL_BASE_URL")
        or os.getenv("NOUS_PORTAL_BASE_URL")
        or pconfig.portal_base_url
    ).rstrip("/")
    requested_inference_url = (
        inference_base_url
        or os.getenv("NOUS_INFERENCE_BASE_URL")
        or pconfig.inference_base_url
    ).rstrip("/")
    client_id = client_id or pconfig.client_id
    scope = scope or pconfig.scope
    timeout = httpx.Timeout(timeout_seconds)
    verify: bool | str = False if insecure else (ca_bundle if ca_bundle else True)

    if _is_remote_session():
        open_browser = False

    print(f"Starting Hermes login via {pconfig.name}...")
    print(f"Portal: {portal_base_url}")
    if insecure:
        print("TLS verification: disabled (--insecure)")
    elif ca_bundle:
        print(f"TLS verification: custom CA bundle ({ca_bundle})")

    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        device_data = _request_device_code(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            scope=scope,
        )

        verification_url = str(device_data["verification_uri_complete"])
        user_code = str(device_data["user_code"])
        expires_in = int(device_data["expires_in"])
        interval = int(device_data["interval"])

        print()
        print("To continue:")
        print(f"  1. Open: {verification_url}")
        print(f"  2. If prompted, enter code: {user_code}")

        if open_browser:
            opened = webbrowser.open(verification_url)
            if opened:
                print("  (Opened browser for verification)")
            else:
                print("  Could not open browser automatically — use the URL above.")

        effective_interval = max(1, min(interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS))
        print(f"Waiting for approval (polling every {effective_interval}s)...")

        token_data = _poll_for_token(
            client=client,
            portal_base_url=portal_base_url,
            client_id=client_id,
            device_code=str(device_data["device_code"]),
            expires_in=expires_in,
            poll_interval=interval,
        )

    now = datetime.now(timezone.utc)
    token_expires_in = _coerce_ttl_seconds(token_data.get("expires_in", 0))
    expires_at = now.timestamp() + token_expires_in
    resolved_inference_url = (
        _optional_base_url(token_data.get("inference_base_url"))
        or requested_inference_url
    )
    if resolved_inference_url != requested_inference_url:
        print(f"Using portal-provided inference URL: {resolved_inference_url}")

    auth_state = {
        "portal_base_url": portal_base_url,
        "inference_base_url": resolved_inference_url,
        "client_id": client_id,
        "scope": token_data.get("scope") or scope,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "obtained_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        "expires_in": token_expires_in,
        "tls": {
            "insecure": verify is False,
            "ca_bundle": verify if isinstance(verify, str) else None,
        },
        "agent_key": None,
        "agent_key_id": None,
        "agent_key_expires_at": None,
        "agent_key_expires_in": None,
        "agent_key_reused": None,
        "agent_key_obtained_at": None,
    }
    return refresh_nous_oauth_from_state(
        auth_state,
        min_key_ttl_seconds=min_key_ttl_seconds,
        timeout_seconds=timeout_seconds,
        force_refresh=False,
        force_mint=True,
    )


def _login_nous(args, pconfig: ProviderConfig) -> None:
    """Nous Portal device authorization flow."""
    timeout_seconds = getattr(args, "timeout", None) or 15.0
    insecure = bool(getattr(args, "insecure", False))
    ca_bundle = (
        getattr(args, "ca_bundle", None)
        or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
    )

    try:
        auth_state = _nous_device_code_login(
            portal_base_url=getattr(args, "portal_url", None) or pconfig.portal_base_url,
            inference_base_url=getattr(args, "inference_url", None) or pconfig.inference_base_url,
            client_id=getattr(args, "client_id", None) or pconfig.client_id,
            scope=getattr(args, "scope", None) or pconfig.scope,
            open_browser=not getattr(args, "no_browser", False),
            timeout_seconds=timeout_seconds,
            insecure=insecure,
            ca_bundle=ca_bundle,
            min_key_ttl_seconds=5 * 60,
        )
        inference_base_url = auth_state["inference_base_url"]

        with _auth_store_lock():
            auth_store = _load_auth_store()
            _save_provider_state(auth_store, "nous", auth_state)
            saved_to = _save_auth_store(auth_store)

        config_path = _update_config_for_provider("nous", inference_base_url)
        print()
        print("Login successful!")
        print(f"  Auth state: {saved_to}")
        print(f"  Config updated: {config_path} (model.provider=nous)")

        try:
            runtime_key = auth_state.get("agent_key") or auth_state.get("access_token")
            if not isinstance(runtime_key, str) or not runtime_key:
                raise AuthError(
                    "No runtime API key available to fetch models",
                    provider="nous",
                    code="invalid_token",
                )

            # Use curated model list (same as OpenRouter defaults) instead
            # of the full /models dump which returns hundreds of models.
            from hermes_cli.models import _PROVIDER_MODELS
            model_ids = _PROVIDER_MODELS.get("nous", [])

            print()
            if model_ids:
                print(f"Showing {len(model_ids)} curated models — use \"Enter custom model name\" for others.")
                selected_model = _prompt_model_selection(model_ids)
                if selected_model:
                    _save_model_choice(selected_model)
                    print(f"Default model set to: {selected_model}")
            else:
                print("No curated models available for Nous Portal.")
        except Exception as exc:
            message = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
            print()
            print(f"Login succeeded, but could not fetch available models. Reason: {message}")

    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Login failed: {exc}")
        raise SystemExit(1)
