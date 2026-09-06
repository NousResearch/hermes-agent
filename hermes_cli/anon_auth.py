"""Nous guest identity: the ``anonymous`` auth method of the ``nous`` provider.

A fresh install mints an anonymous Nous account (``POST /api/anonymous/create``) and exchanges its
``anon_`` credential for short-lived JWTs (``POST /api/anonymous/token``). The result is persisted
through the same ``persist_nous_credentials`` a real login uses, so it is the singleton
``providers.nous`` *and* ``active_provider`` -- the resolver ladder (``resolve_provider``) is
untouched; ``active_provider`` is already its last-resort rung, so any explicit provider (env key,
``model.provider``, OpenRouter pool) beats the guest for inference while the guest keeps carrying the
tool-gateway JWT for connectors.

Only two mechanics differ from an OAuth login and both are isolated behind ``is_guest_state``:
token acquisition (re-exchange the ``anon_`` credential; there is no refresh token) and routing
(the welcome inference host, single model ``nous/welcome``).

Users are never shown the words guest / anonymous / account for this state: surfaces say
"Nous · free tier". The one user-facing verb is ``hermes auth upgrade`` (sign in, keeping the
identity's connectors).

Lifecycle lives in ONE primitive, :func:`ensure_portal_identity`: adopt what the shared store already
holds, else mint under the shared-store lock. It is the only minter; nothing else calls
:func:`mint_guest`.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from hermes_cli.auth_constants import (
    AuthError, DEFAULT_NOUS_PORTAL_URL, _decode_jwt_claims, httpx)

logger = logging.getLogger("hermes_cli.auth")

ANON_AUTH_METHOD = "anonymous"
ANON_CLIENT_ID = "nas-anonymous"
ANON_ACCOUNT_TIER = "anonymous"
GUEST_MODEL = "nous/welcome"
ANON_SECRET_HEADER = "x-anonymous-api-secret"
# The shared secret gates the anonymous surface during its integration phase. It is a deployment
# secret (Sid's), read from the environment only.
ANON_SECRET_ENV = "HERMES_ANON_API_SECRET"
# Dev lever: "1" makes the guest carry inference even when explicit providers exist; "new" also
# bypasses the shared store and mints a fresh guest for this process. Overrides ``nous.guest: false``.
FORCE_GUEST_ENV = "HERMES_FORCE_GUEST"
GUEST_MINT_TIMEOUT_SECONDS = 5.0
# Copy shared by every surface that names the free tier (R-USR-1): never guest / anonymous / account.
FREE_TIER_LABEL = "Nous · free tier"
UPGRADE_HINT = "Run `hermes auth upgrade` to sign in with a Nous account."
FREE_TIER_NOT_SIGNED_IN = (
    "You're not signed in. Free inference and connectors are always on. "
    "Run `hermes auth` to sign in with a Nous account.")


class AnonCredentialDead(AuthError):
    """NAS no longer knows this ``anon_`` credential (reaped, or claimed into a real account).

    The one client rule for reap AND claim: mark dead, re-mint on the next need.
    """


def _anon_err(message: str, code: str) -> AuthError:
    return AuthError(message, code=code)


def force_guest_mode() -> str:
    """``""`` (off), ``"1"`` or ``"new"``; anything else truthy counts as ``"1"``."""
    raw = (os.environ.get(FORCE_GUEST_ENV) or "").strip().lower()
    if not raw or raw in {"0", "false", "no", "off"}:
        return ""
    return "new" if raw == "new" else "1"


def guest_enabled() -> bool:
    """``nous.guest`` (default True), overridden by the dev lever."""
    if force_guest_mode():
        return True
    try:
        from hermes_cli.config import load_config_readonly
        nous_cfg = load_config_readonly().get("nous")
    except Exception as exc:  # config unreadable: keep today's behaviour (no guest) rather than mint
        logger.debug("guest: config unreadable, treating nous.guest as false: %s", exc)
        return False
    if not isinstance(nous_cfg, dict):
        return True
    return bool(nous_cfg.get("guest", True))


def is_guest_state(state: Any) -> bool:
    return isinstance(state, dict) and state.get("auth_method") == ANON_AUTH_METHOD


def current_nous_state() -> Optional[Dict[str, Any]]:
    """The profile's ``providers.nous`` state without locking or network (status/picker reads)."""
    from hermes_cli.auth import _load_auth_store, _load_provider_state
    try:
        return _load_provider_state(_load_auth_store(), "nous")
    except Exception as exc:
        logger.debug("guest: auth store unreadable: %s", exc)
        return None


def has_guest() -> bool:
    return is_guest_state(current_nous_state())


def guest_carries_inference() -> bool:
    """True when the Nous provider selected for inference is the free tier (a guest state)."""
    return guest_enabled() and has_guest()


def anon_secret() -> str:
    return (os.environ.get(ANON_SECRET_ENV) or "").strip()


def _anon_headers() -> Dict[str, str]:
    headers = {"content-type": "application/json"}
    if secret := anon_secret():
        headers[ANON_SECRET_HEADER] = secret
    return headers


def _raise_for_anon_status(response: httpx.Response, *, action: str) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    error = str(payload.get("error") or "")
    if response.status_code in (200, 201):
        return payload
    if response.status_code == 404 and error == "unknown_token":
        raise AnonCredentialDead("Nous free-tier credential is no longer valid.", code="anon_credential_dead")
    if response.status_code == 401 and error == "invalid_shared_secret":
        raise _anon_err("Nous free tier is not open on this portal.", "anon_gate_closed")
    if response.status_code == 401:
        raise AnonCredentialDead("Nous free-tier credential was revoked.", code="anon_credential_dead")
    if response.status_code == 429:
        raise _anon_err("Nous free tier is rate limited; try again shortly.", "anon_rate_limited")
    if response.status_code == 403 and error in {"anonymous_accounts_disabled", "circuit_open"}:
        raise _anon_err("Nous free tier is currently disabled.", "anon_gate_closed")
    raise _anon_err(
        f"Nous free tier {action} failed ({response.status_code}{': ' + error if error else ''}).",
        "anon_server_error")


def mint_guest(client: httpx.Client, portal_base_url: str) -> Dict[str, Any]:
    """``POST /api/anonymous/create`` -> ``{user_id, org_id, token, idle_ttl_days}``. Token shown once."""
    response = client.post(f"{portal_base_url.rstrip('/')}/api/anonymous/create", headers=_anon_headers(), json={})
    payload = _raise_for_anon_status(response, action="sign-up")
    token = payload.get("token")
    if not isinstance(token, str) or not token.startswith("anon_"):
        raise _anon_err("Nous free tier sign-up returned no credential.", "anon_server_error")
    return payload


def exchange_anon_jwt(client: httpx.Client, portal_base_url: str, anon_token: str) -> Dict[str, Any]:
    """``POST /api/anonymous/token {token}`` -> ``{access_token, expires_in, inference_base_url, ...}``.

    Raises :class:`AnonCredentialDead` on 404 ``unknown_token`` / 401 (reaped or claimed).
    """
    response = client.post(
        f"{portal_base_url.rstrip('/')}/api/anonymous/token", headers=_anon_headers(), json={"token": anon_token})
    payload = _raise_for_anon_status(response, action="token exchange")
    if not isinstance(payload.get("access_token"), str) or not payload["access_token"]:
        raise _anon_err("Nous free tier token exchange returned no token.", "anon_server_error")
    return payload


def apply_exchange_to_state(state: Dict[str, Any], exchanged: Dict[str, Any]) -> None:
    """Write a fresh exchange result into a guest state in place (token, expiry, routing)."""
    from hermes_cli.auth_nous import _validate_nous_inference_url_from_network
    access_token = exchanged["access_token"]
    claims = _decode_jwt_claims(access_token)
    now = datetime.now(timezone.utc)
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        expires_at = datetime.fromtimestamp(float(exp), tz=timezone.utc)
    else:
        expires_at = now + timedelta(seconds=int(exchanged.get("expires_in") or 900))
    inference_url = _validate_nous_inference_url_from_network(exchanged.get("inference_base_url"))
    scope = claims.get("scope") or claims.get("scp") or state.get("scope")
    if isinstance(scope, (list, tuple)):
        scope = " ".join(str(s) for s in scope)
    state.update(
        access_token=access_token, token_type="Bearer", scope=scope,
        obtained_at=now.isoformat(), expires_at=expires_at.isoformat(),
        expires_in=max(0, int((expires_at - now).total_seconds())),
        account_tier=str(claims.get("account_tier") or ANON_ACCOUNT_TIER))
    if inference_url:
        state["inference_base_url"] = inference_url
    for key in ("user_id", "org_id"):
        if exchanged.get(key):
            state[key] = exchanged[key]
    state.pop("refresh_token", None)


def anon_state_from_exchange(minted: Dict[str, Any], exchanged: Dict[str, Any], *, portal_base_url: str) -> Dict[str, Any]:
    """The ``providers.nous`` shape for a guest. No ``refresh_token``: the ``anon_`` credential is it."""
    state: Dict[str, Any] = {
        "auth_method": ANON_AUTH_METHOD, "account_tier": ANON_ACCOUNT_TIER,
        "anon_token": minted["token"], "client_id": ANON_CLIENT_ID,
        "portal_base_url": portal_base_url.rstrip("/"),
        "user_id": minted.get("user_id"), "org_id": minted.get("org_id"),
        "idle_ttl_days": minted.get("idle_ttl_days"),
    }
    apply_exchange_to_state(state, exchanged)
    return state


def _portal_base_url() -> str:
    from hermes_cli.auth_nous import _nous_portal_env_override
    return (_nous_portal_env_override() or DEFAULT_NOUS_PORTAL_URL).rstrip("/")


def _mint_and_persist(*, timeout_seconds: float) -> Dict[str, Any]:
    from hermes_cli.auth_nous import _nous_http_client, persist_nous_credentials
    from hermes_cli.auth import _resolve_verify
    portal = _portal_base_url()
    verify = _resolve_verify(insecure=None, ca_bundle=None, auth_state=None)
    with _nous_http_client(timeout_seconds, verify) as client:
        minted = mint_guest(client, portal)
        exchanged = exchange_anon_jwt(client, portal, minted["token"])
    state = anon_state_from_exchange(minted, exchanged, portal_base_url=portal)
    persist_nous_credentials(state)
    logger.info("Nous free tier ready (identity minted)")
    return state


_background_lock = threading.Lock()
_background_started = False


def ensure_portal_identity(*, blocking: bool = True, timeout_seconds: float = GUEST_MINT_TIMEOUT_SECONDS) -> Optional[Dict[str, Any]]:
    """Make sure this profile has a Nous identity (guest or account); mint a guest only if the shared
    store has none. Returns the ``providers.nous`` state, or None (disabled / non-blocking / failed).

    Order: ``nous.guest`` gate -> profile state -> shared store (adopt) -> mint. The mint runs under
    the shared-store lock so two profiles booting together produce one identity, not two.
    Non-blocking mode runs the mint on a daemon thread and returns None immediately; a failure there
    is logged at DEBUG (the guest is a fallback; a fallback failing is not an error).
    """
    if not guest_enabled():
        return None
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _load_provider_state
    from hermes_cli.auth_nous import (
        _nous_shared_store_lock, _read_shared_nous_state, persist_nous_credentials)
    force = force_guest_mode()
    if force != "new":
        with _auth_store_lock():
            state = _load_provider_state(_load_auth_store(), "nous")
        if state:
            return state

    def _adopt_or_mint() -> Dict[str, Any]:
        with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds + 5.0, 30.0)):
            if force != "new":
                shared = _read_shared_nous_state()
                if shared:
                    persist_nous_credentials(dict(shared))
                    logger.debug("Nous identity adopted from the shared store")
                    return dict(shared)
            return _mint_and_persist(timeout_seconds=timeout_seconds)

    if blocking:
        return _adopt_or_mint()

    global _background_started
    with _background_lock:
        if _background_started:
            return None
        _background_started = True

    def _run() -> None:
        try:
            _adopt_or_mint()
        except Exception as exc:
            logger.debug("Nous free tier background setup skipped: %s", exc)

    threading.Thread(target=_run, name="nous-guest-identity", daemon=True).start()
    return None


def refresh_guest_state(state: Dict[str, Any], client: httpx.Client) -> None:
    """Token-acquisition seam for a guest: re-exchange the ``anon_`` credential in place.

    Raises :class:`AnonCredentialDead` when NAS no longer knows the credential; the caller owns
    re-minting (:func:`ensure_portal_identity` after clearing the dead state).
    """
    anon_token = state.get("anon_token")
    if not isinstance(anon_token, str) or not anon_token:
        raise AnonCredentialDead("Nous free-tier credential is missing.", code="anon_credential_dead")
    portal = (state.get("portal_base_url") or _portal_base_url()).rstrip("/")
    apply_exchange_to_state(state, exchange_anon_jwt(client, portal, anon_token))


def clear_dead_guest(reason: str) -> None:
    """Drop a dead guest from the profile store and the shared store so the next need re-mints."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _load_provider_state, _save_auth_store, _store_section
    from hermes_cli.auth_nous import _clear_shared_nous_state
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "nous")
        if is_guest_state(state):
            _store_section(auth_store, "providers").pop("nous", None)
            _store_section(auth_store, "credential_pool").pop("nous", None)
            if auth_store.get("active_provider") == "nous":
                auth_store["active_provider"] = None
            _save_auth_store(auth_store)
    _clear_shared_nous_state(reason)
    logger.info("Nous free-tier identity retired (%s); a new one is set up on next use", reason)
