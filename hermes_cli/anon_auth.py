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
import time
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
    """True when the profile's Nous identity is the free tier and the free tier is on.

    Profile-level: use for status, picker and notice surfaces. Routing decisions (which model a
    request may carry) must use :func:`route_is_welcome_host` on the SELECTED runtime instead: a
    credential-pool entry can pick a paid Nous key while the profile singleton is still a guest.
    """
    return guest_enabled() and has_guest()


WELCOME_HOSTS = frozenset({"welcome-api.nousresearch.com"})


def pin_model_for_route(provider: Any, base_url: Any, model: Any) -> Any:
    """Model policy at agent START: on the Nous welcome host the model is ``nous/welcome``; anywhere
    else the caller's model stands. Used once, when the route is first finalized. Mid-conversation
    route changes go through :func:`route_can_serve_model` instead: a conversation's model is never
    silently rewritten by a credential rotation.
    """
    if provider == "nous" and route_is_welcome_host(base_url):
        if model and model != GUEST_MODEL:
            logger.info("Nous free tier: using %s instead of configured model %s", GUEST_MODEL, model)
        return GUEST_MODEL
    return model


def route_can_serve_model(provider: Any, base_url: Any, model: Any) -> bool:
    """Eligibility for a credential ROTATION: the welcome host serves only ``nous/welcome``, so a
    conversation on any other model must not be rotated onto it (and a ``nous/welcome`` conversation
    may move to the portal host, which serves it too). Non-Nous routes are always eligible."""
    if provider != "nous" or not route_is_welcome_host(base_url):
        return True
    return not model or model == GUEST_MODEL


def route_is_welcome_host(base_url: Any) -> bool:
    """The routing predicate for the free tier: the welcome host serves exactly ``nous/welcome``.

    Keyed on the resolved endpoint, never on profile state, so a paid pool credential routed to the
    portal host keeps its model even when a guest singleton exists beside it.
    """
    from urllib.parse import urlparse
    try:
        host = (urlparse(str(base_url or "")).hostname or "").lower()
    except ValueError:
        return False
    return host in WELCOME_HOSTS


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


def _portal_base_url() -> str:
    from hermes_cli.auth_nous import _nous_portal_env_override
    return (_nous_portal_env_override() or DEFAULT_NOUS_PORTAL_URL).rstrip("/")


def _shared_identity_key(state: Any) -> Optional[str]:
    """Stable identity of a Nous credential: the anon_ token for a guest, the refresh token for an
    account. Used to decide whether two stores hold the SAME identity."""
    if not isinstance(state, dict):
        return None
    return state.get("anon_token") if is_guest_state(state) else state.get("refresh_token")


def _mint_locked(client: httpx.Client, portal: str, auth_store: Dict[str, Any]) -> Dict[str, Any]:
    """Mint under the caller's locks. The identity is persisted as soon as ``create`` succeeds, BEFORE
    the exchange: a 429 or timeout on the exchange must not lose a credential NAS still honours (the
    next attempt exchanges the stored one instead of minting again)."""
    from hermes_cli.auth import _save_provider_state, _save_auth_store
    from hermes_cli.auth_nous import _write_shared_nous_state
    minted = mint_guest(client, portal)
    state: Dict[str, Any] = {
        "auth_method": ANON_AUTH_METHOD, "account_tier": ANON_ACCOUNT_TIER,
        "anon_token": minted["token"], "client_id": ANON_CLIENT_ID,
        "portal_base_url": portal.rstrip("/"),
        "user_id": minted.get("user_id"), "org_id": minted.get("org_id"),
        "idle_ttl_days": minted.get("idle_ttl_days"),
    }
    _save_provider_state(auth_store, "nous", state)
    _save_auth_store(auth_store)
    _write_shared_nous_state(state)
    logger.info("Nous free tier ready (identity minted)")
    return state


_background_lock = threading.Lock()
_background_started = False
# Per-process memos for the blocking path. ``_mint_failed``: one failed mint is enough for a process
# (several bootstrap sites call in sequence; a 429 or a closed gate must not be hit twice);
# ``clear_dead_guest`` resets it because a retired credential is a reason to mint again.
# ``_forced_new_done``: ``HERMES_FORCE_GUEST=new`` re-mints once per process, not on every resolution.
_mint_failed = False
_forced_new_done = False


def _reconcile_and_provision(*, force: str, timeout_seconds: float) -> Dict[str, Any]:
    """The lifecycle body, run under profile lock THEN shared lock (the documented order).

    1. The shared store is the identity of record for this Hermes root. If it holds an identity
       that differs from the profile's, the profile adopts it (a stale guest never outlives a
       sibling profile's sign-in, and never overwrites it).
    2. Otherwise the profile's own identity stands.
    3. Nothing anywhere: mint, persisting the credential before exchanging it.
    ``force == "new"`` skips 1 and 2.
    """
    from hermes_cli.auth import (
        _auth_store_lock, _load_auth_store, _load_provider_state, _save_auth_store,
        _save_provider_state, _resolve_verify)
    from hermes_cli.auth_nous import (
        _nous_http_client, _nous_shared_store_lock, _read_shared_nous_state, _write_shared_nous_state)
    portal = _portal_base_url()
    with _auth_store_lock():
        auth_store = _load_auth_store()
        profile_state = _load_provider_state(auth_store, "nous")
        with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds, 5.0)):
            if force != "new":
                shared = _read_shared_nous_state()
                if shared and _shared_identity_key(shared) != _shared_identity_key(profile_state):
                    state = dict(shared)
                    _save_provider_state(auth_store, "nous", state)
                    _save_auth_store(auth_store)
                    logger.debug("Nous identity adopted from the shared store")
                    return state
                if profile_state:
                    if not shared:
                        _write_shared_nous_state(profile_state)
                    return profile_state
            verify = _resolve_verify(insecure=None, ca_bundle=None, auth_state=None)
            with _nous_http_client(timeout_seconds, verify) as client:
                return _mint_locked(client, portal, auth_store)


def ensure_portal_identity(*, blocking: bool = True, timeout_seconds: float = GUEST_MINT_TIMEOUT_SECONDS) -> Optional[Dict[str, Any]]:
    """Make sure this profile has a Nous identity (guest or account); mint a guest only if the shared
    store has none. Returns the ``providers.nous`` state, or None (disabled / non-blocking / failed).

    Order: ``nous.guest`` gate -> reconcile with the shared store -> mint. Locks are taken profile
    first, then shared, matching every other Nous path. Non-blocking mode runs on a daemon thread
    and returns None immediately; a failure there is logged at DEBUG (the guest is a fallback; a
    fallback failing is not an error).
    """
    global _mint_failed, _forced_new_done
    if not guest_enabled():
        return None
    force = force_guest_mode()
    if force == "new" and _forced_new_done:
        force = "1"
    if _mint_failed and force != "new" and not current_nous_state():
        return None  # this process already tried and failed; do not hammer the portal

    if blocking:
        try:
            result = _reconcile_and_provision(force=force, timeout_seconds=timeout_seconds)
        except Exception:
            _mint_failed = True
            raise
        if force == "new":
            _forced_new_done = True
        return result

    global _background_started
    with _background_lock:
        if _background_started:
            return None
        _background_started = True

    def _run() -> None:
        global _background_started
        try:
            _reconcile_and_provision(force=force, timeout_seconds=timeout_seconds)
        except Exception as exc:
            logger.debug("Nous free tier background setup skipped: %s", exc)
            # A transient failure must not consume the process's only attempt: release the latch
            # so a later non-blocking call can try again (still one setup in flight at a time).
            with _background_lock:
                _background_started = False

    try:
        threading.Thread(target=_run, name="nous-guest-identity", daemon=True).start()
    except Exception as exc:  # thread limit / interpreter shutdown: release so a later call can retry
        with _background_lock:
            _background_started = False
        logger.debug("Nous free tier background setup could not start: %s", exc)
    return None


def refresh_guest_state(state: Dict[str, Any], client: httpx.Client) -> None:
    """Token-acquisition seam for a guest: re-exchange the ``anon_`` credential in place.

    The portal URL is the resolver's canonical one (env override, else the validated stored URL,
    else the default), never a raw stored value on its own.
    Raises :class:`AnonCredentialDead` when NAS no longer knows the credential; the caller owns
    re-minting (:func:`ensure_portal_identity` after :func:`clear_dead_guest`).
    """
    anon_token = state.get("anon_token")
    if not isinstance(anon_token, str) or not anon_token:
        raise AnonCredentialDead("Nous free-tier credential is missing.", code="anon_credential_dead")
    from hermes_cli.auth import _nous_portal_base_url
    apply_exchange_to_state(state, exchange_anon_jwt(client, _nous_portal_base_url(state), anon_token))


def clear_dead_guest(reason: str, *, dead_token: Optional[str] = None) -> None:
    """Drop a dead guest so the next need re-mints.

    Only the identity that actually failed is removed: a stale profile whose credential NAS rejected
    must not erase a sibling profile's newer sign-in or replacement guest from the shared store. When
    *dead_token* is None the profile's current guest is treated as the failed one.
    """
    from hermes_cli.auth import (
        _auth_store_lock, _load_auth_store, _load_provider_state, _save_auth_store, _store_section)
    from hermes_cli.auth_nous import _clear_shared_nous_state, _nous_shared_store_lock, _read_shared_nous_state
    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "nous")
        if is_guest_state(state):
            token = dead_token or state.get("anon_token")
            if state.get("anon_token") == token:
                _store_section(auth_store, "providers").pop("nous", None)
                _store_section(auth_store, "credential_pool").pop("nous", None)
                if auth_store.get("active_provider") == "nous":
                    auth_store["active_provider"] = None
                _save_auth_store(auth_store)
        else:
            token = dead_token
        with _nous_shared_store_lock():
            shared = _read_shared_nous_state()
            if token and is_guest_state(shared) and shared.get("anon_token") == token:
                _clear_shared_nous_state(reason)
    global _mint_failed
    _mint_failed = False
    logger.info("Nous free-tier identity retired (%s); a new one is set up on next use", reason)


# One-time CLI notice: an install whose inference is carried by an explicit provider learns once that
# the free tier (inference + connectors) now exists. The flag lives on the guest state itself so it
# dies with the identity; a fresh guest (re-mint, new profile) may announce itself once more.
GUEST_NOTICE_FLAG = "guest_notice_shown"
FREE_TIER_AVAILABLE_NOTICE = (
    "Free Nous inference and connectors are now available. "
    "`hermes model` to try them, `hermes auth upgrade` to sign in.")


def guest_notice_pending() -> bool:
    """True when a guest identity exists and the one-time availability notice has not been shown."""
    state = current_nous_state()
    return is_guest_state(state) and not bool(state.get(GUEST_NOTICE_FLAG))


def mark_guest_notice_shown() -> bool:
    """Persist ``guest_notice_shown`` on the guest's ``providers.nous`` state (whichever store holds it).

    Returns True when a flag was written; False when there is no guest to mark."""
    from hermes_cli.auth import (
        _auth_file_path, _load_auth_store, _provider_state_transaction, _same_path, _save_auth_store,
        _store_section)
    with _provider_state_transaction("nous") as (auth_store, state, source_path):
        if not is_guest_state(state) or source_path is None:
            return False
        if state.get(GUEST_NOTICE_FLAG):
            return True
        state = dict(state)
        state[GUEST_NOTICE_FLAG] = True
        if _same_path(source_path, _auth_file_path()):
            _store_section(auth_store, "providers")["nous"] = state
            _save_auth_store(auth_store)
        else:
            source_store = _load_auth_store(source_path)
            _store_section(source_store, "providers")["nous"] = state
            _save_auth_store(source_store, target_path=source_path)
    return True


# --- ``hermes auth upgrade``: sign the guest into a real Nous account, keeping its connectors ---------
#
# Wire: the normal device-code flow, with a promotion intent registered on NAS BETWEEN the code
# request and the token poll (``POST /api/anonymous/promotion-intent {token, user_code, device_code}``).
# NAS then transfers the guest's connectors into whichever account approves that device code. We
# watch ``POST /api/anonymous/promotion-status {claim_code}`` until it leaves ``pending``; only a
# ``completed`` promotion is followed by the token grant, which ``persist_nous_credentials`` writes
# over the guest singleton and the shared store. The server never reports expiry: our own
# ``expires_in`` clock ends the wait. User-facing copy never says guest / anonymous / claim.

UPGRADE_START = "Sign in to keep your connectors and unlock more."
UPGRADE_ALREADY_SIGNED_IN = "Already signed in."
UPGRADE_DO_NOT_SHARE = "Do not share this code."
UPGRADE_TIMED_OUT = "Sign-in timed out; run the command again."
UPGRADE_NOT_COMPLETED = "Sign-in did not complete; run the command again."
UPGRADE_UNAVAILABLE = "The free tier is not available right now; run `hermes auth add nous` to sign in."
UPGRADE_REASON_COPY = {
    "user_declined": "Sign-in was rejected in the browser.",
    "superseded": "A newer sign-in code replaced this one.",
    "account_retired": "This free-tier identity was already used or expired; a new one is set up on next use.",
    "account_not_anonymous": "This free-tier identity was already used or expired; a new one is set up on next use.",
    "account_busy": "The transfer could not run; run the command again.",
}
_RETIRED_REASONS = frozenset({"account_retired", "account_not_anonymous"})
UPGRADED_AUTH_METHOD = "oauth_device_code"


def register_promotion_intent(
    client: httpx.Client, portal_base_url: str, anon_token: str, *, user_code: str, device_code: str,
) -> Dict[str, Any]:
    """``POST /api/anonymous/promotion-intent`` -> ``{claim_code, claim_url, expires_in, interval}``."""
    response = client.post(
        f"{portal_base_url.rstrip('/')}/api/anonymous/promotion-intent", headers=_anon_headers(),
        json={"token": anon_token, "user_code": user_code, "device_code": device_code})
    payload = _raise_for_anon_status(response, action="sign-in")
    if not isinstance(payload.get("claim_code"), str) or not payload["claim_code"]:
        raise _anon_err("Nous free tier sign-in returned no transfer code.", "anon_server_error")
    return payload


def _retry_after_seconds(response: httpx.Response, default: float) -> float:
    raw = (response.headers.get("retry-after") or "").strip()
    try:
        return max(0.0, float(raw)) if raw else default
    except ValueError:
        return default


def wait_for_promotion(
    client: httpx.Client, portal_base_url: str, claim_code: str, *, expires_in: int, interval: int,
) -> Dict[str, Any]:
    """Poll ``POST /api/anonymous/promotion-status`` until it leaves ``pending`` or our clock runs out.

    Returns the final status payload; ``{"status": "timeout"}`` when ``expires_in`` elapsed. 429 honours
    ``Retry-After``; other non-2xx statuses raise through :func:`_raise_for_anon_status`.
    """
    deadline = time.monotonic() + max(1, int(expires_in))
    wait = max(0, int(interval))
    while time.monotonic() < deadline:
        response = client.post(
            f"{portal_base_url.rstrip('/')}/api/anonymous/promotion-status", headers=_anon_headers(),
            json={"claim_code": claim_code})
        if response.status_code == 429:
            time.sleep(min(_retry_after_seconds(response, default=max(1, wait)), max(0.0, deadline - time.monotonic())))
            continue
        payload = _raise_for_anon_status(response, action="sign-in")
        if str(payload.get("status") or "unknown") != "pending":
            return payload
        time.sleep(wait)
    return {"status": "timeout"}


def _account_state_from_token(
    token_data: Dict[str, Any], *, portal_base_url: str, client_id: str, scope: Optional[str], verify: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    """The ``providers.nous`` shape for the signed-in account (same fields the device-code login writes)."""
    from hermes_cli.auth import PROVIDER_REGISTRY, _coerce_ttl_seconds, _optional_base_url, _tls_state_from_verify
    from hermes_cli.auth_nous import _NOUS_EMPTY_AGENT_KEY_FIELDS, _iso_after, refresh_nous_oauth_from_state
    now = datetime.now(timezone.utc)
    ttl = _coerce_ttl_seconds(token_data.get("expires_in", 0))
    inference_url = (
        _optional_base_url(token_data.get("inference_base_url"))
        or PROVIDER_REGISTRY["nous"].inference_base_url.rstrip("/"))
    state = {
        "portal_base_url": portal_base_url, "inference_base_url": inference_url,
        "client_id": client_id, "scope": token_data.get("scope") or scope,
        "token_type": token_data.get("token_type", "Bearer"),
        "access_token": token_data["access_token"], "refresh_token": token_data.get("refresh_token"),
        "obtained_at": now.isoformat(), "expires_at": _iso_after(now, ttl), "expires_in": ttl,
        "tls": _tls_state_from_verify(verify), **_NOUS_EMPTY_AGENT_KEY_FIELDS}
    state = refresh_nous_oauth_from_state(state, timeout_seconds=timeout_seconds, force_refresh=False)
    state["auth_method"] = UPGRADED_AUTH_METHOD
    return state


def settle_after_upgrade(account_state: Dict[str, Any]) -> Dict[str, Any]:
    """After a sign-in from the free tier persisted the account: move the config off the free tier's route.

    Picking the free-tier row may have written ``model.default: nous/welcome`` and ``model.base_url``
    = welcome host. An account cannot keep either: the welcome host refuses account tokens, and the
    portal host serves ``nous/welcome`` as a paid model. When the config is on the free tier's route,
    ``model.base_url`` becomes the account's inference host and ``model.default`` the recommended
    default for the account's tier (:func:`hermes_cli.models.recommended_nous_default_model`, the
    same pick as ``GET /api/model/recommended-default``), through the same config write a plain Nous
    login uses. A config on the user's own model and host is left alone.

    Every sign-in completion (CLI ``hermes auth upgrade``, the desktop poller) calls this once, after
    ``persist_nous_credentials``. Returns ``{"model": str, "changed": bool}``: ``model`` is the default
    the config now carries (``""`` when it carries none); ``changed`` says whether this call wrote it.
    Never raises: a failed pick or write is logged and reported as ``changed: False`` so the sign-in
    itself still counts.
    """
    from hermes_cli.config import load_config_readonly
    try:
        raw = load_config_readonly().get("model")
    except Exception as exc:
        logger.warning("sign-in completion: config unreadable, default model left as is: %s", exc)
        return {"model": "", "changed": False}
    model_cfg = raw if isinstance(raw, dict) else ({"default": raw} if isinstance(raw, str) else {})
    current = str(model_cfg.get("default") or "").strip()
    on_welcome_model = current == GUEST_MODEL
    on_welcome_host = route_is_welcome_host(model_cfg.get("base_url"))
    if not (on_welcome_model or on_welcome_host):
        return {"model": current, "changed": False}
    model = current
    if on_welcome_model:
        from hermes_cli.models import recommended_nous_default_model
        try:
            model = str(recommended_nous_default_model().get("model") or "")
        except Exception as exc:
            logger.debug("sign-in completion: recommended default unavailable: %s", exc)
            model = ""
    try:
        from hermes_cli.auth import _update_config_for_provider
        # One write: host and default move together, so a failure leaves the config as it was
        # rather than the account host paired with the welcome model. No eligible recommendation
        # (Portal unreachable, or the plan and org policy admit nothing) clears the default in that
        # same write; the runtime's silent default applies until the user picks one with `hermes model`.
        _update_config_for_provider(
            "nous", str(account_state.get("inference_base_url") or ""),
            default_model=model if on_welcome_model else None,
            clear_default=on_welcome_model and not model)
    except Exception as exc:
        logger.warning("sign-in completion: could not update the default model: %s", exc)
        return {"model": current, "changed": False}
    return {"model": model, "changed": True}


def _print_promotion_outcome(outcome: Dict[str, Any]) -> None:
    status = str(outcome.get("status") or "unknown")
    reason = str(outcome.get("reason") or "")
    if status == "timeout":
        print(UPGRADE_TIMED_OUT)
        return
    print(UPGRADE_REASON_COPY.get(reason, UPGRADE_NOT_COMPLETED))
    if reason in _RETIRED_REASONS:
        clear_dead_guest("retired")


def upgrade_guest(args) -> int:
    """``hermes auth upgrade``: sign in with a Nous account, transferring the free tier's connectors.

    Returns 0 on success (or when already signed in), 1 otherwise. Never persists anything unless the
    promotion completed AND the token grant succeeded.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY, _resolve_verify
    from hermes_cli.auth_device_flow import (
        _is_remote_session, _poll_for_token, _print_device_code_instructions, _request_device_code)
    from hermes_cli.auth_nous import _nous_http_client, persist_nous_credentials
    timeout_seconds = float(getattr(args, "timeout", None) or 15.0)
    open_browser = not getattr(args, "no_browser", False) and not _is_remote_session()
    state = current_nous_state()
    if state and not is_guest_state(state):
        print(UPGRADE_ALREADY_SIGNED_IN)
        return 0
    if not state:
        try:
            state = ensure_portal_identity(blocking=True, timeout_seconds=timeout_seconds)
        except AuthError as exc:
            print(f"{UPGRADE_UNAVAILABLE} ({exc})")
            return 1
        if not is_guest_state(state):
            print(UPGRADE_UNAVAILABLE)
            return 1
    anon_token = str(state.get("anon_token") or "")
    portal = (state.get("portal_base_url") or _portal_base_url()).rstrip("/")
    pconfig = PROVIDER_REGISTRY["nous"]
    client_id, scope = pconfig.client_id, pconfig.scope
    verify = _resolve_verify(insecure=None, ca_bundle=None, auth_state=None)
    print(UPGRADE_START)
    try:
        with _nous_http_client(timeout_seconds, verify) as client:
            device = _request_device_code(client, portal, client_id, scope)
            intent = register_promotion_intent(
                client, portal, anon_token, user_code=str(device["user_code"]),
                device_code=str(device["device_code"]))
            # The browser leg is the consent page for THIS sign-in (claim_url), not the generic
            # device page: it shows both identities and the Move button. Relative paths are
            # portal-relative.
            claim_url = str(intent.get("claim_url") or "")
            if claim_url.startswith("/"):
                claim_url = f"{portal}{claim_url}"
            _print_device_code_instructions(
                claim_url or str(device["verification_uri_complete"]), str(intent["claim_code"]),
                open_browser=open_browser, swallow_open_errors=True)
            print(f"  {UPGRADE_DO_NOT_SHARE}")
            expires_in = min(int(device["expires_in"]), int(intent.get("expires_in") or device["expires_in"]))
            interval = int(intent.get("interval") or device.get("interval") or 5)
            print("Waiting for sign-in...")
            outcome = wait_for_promotion(client, portal, intent["claim_code"], expires_in=expires_in, interval=interval)
            if str(outcome.get("status")) != "completed":
                _print_promotion_outcome(outcome)
                return 1
            token_data = _poll_for_token(
                client=client, portal_base_url=portal, client_id=client_id,
                device_code=str(device["device_code"]), expires_in=max(1, expires_in), poll_interval=interval)
        account_state = _account_state_from_token(
            token_data, portal_base_url=portal, client_id=client_id, scope=scope, verify=verify,
            timeout_seconds=timeout_seconds)
    except AnonCredentialDead:
        print(UPGRADE_REASON_COPY["account_retired"])
        clear_dead_guest("retired")
        return 1
    except TimeoutError:
        print(UPGRADE_TIMED_OUT)
        return 1
    except KeyboardInterrupt:
        print("\nSign-in cancelled.")
        return 130
    except Exception as exc:
        print(f"Sign-in failed: {exc}")
        return 1
    persist_nous_credentials(account_state)
    settled = settle_after_upgrade(account_state)
    email = str(outcome.get("account_email") or "").strip()
    print(f"Signed in as {email}. Your connectors are kept." if email else "Signed in. Your connectors are kept.")
    if settled["changed"]:
        print(f"Default model is now {settled['model']}." if settled["model"]
              else "No default model is set yet; run `hermes model` to pick one.")
    return 0
