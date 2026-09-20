"""Nous free-tier identity: the ``anonymous`` auth method of the ``nous`` provider.

The identity is created in exactly one place, at boot (``hermes_cli.free_tier_bootstrap``), and only
while ``HERMES_GUEST_ONBOARDING=1`` (see ``guest_enabled``). The bootstrap mints an anonymous Nous
account (``POST /api/anonymous/create``); its ``anon_`` credential is later exchanged for short-lived
JWTs (``POST /api/anonymous/token``). The result is persisted as the singleton ``providers.nous``; it
becomes ``active_provider`` only when the bootstrap's inventory found nothing else usable, so an
install with its own key keeps that key for inference and uses the identity for connectors only. In
the resolver ladder (``resolve_provider``) an existing free-tier identity sits directly above the
implicit AWS Bedrock chain (NS-829): any explicit provider (env key, ``model.provider``, OpenRouter
pool, a logged-in ``active_provider``) beats it, and the ladder never creates one.

Only two mechanics differ from an OAuth login and both are isolated behind ``is_guest_state``:
token acquisition (re-exchange the ``anon_`` credential; there is no refresh token) and routing
(the welcome inference host, single model ``nous/welcome``).

Users are never shown the words guest / anonymous / account for this state: surfaces say
"Nous · free tier". Two user-facing verbs reach the same flow, both keeping the identity's
connectors: ``hermes auth upgrade`` in a terminal and ``/login`` inside a chat.

Lifecycle lives in ONE primitive, :func:`ensure_portal_identity`: adopt what the shared store already
holds, else mint under the shared-store lock. It is the only minter; nothing else calls
:func:`mint_guest`.
"""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from agent.retry_utils import parse_retry_after_seconds
from hermes_cli.auth_constants import (
    AuthError, DEFAULT_NOUS_PORTAL_URL, DEFAULT_NOUS_WELCOME_URL, _decode_jwt_claims, httpx)

logger = logging.getLogger("hermes_cli.auth")

ANON_AUTH_METHOD = "anonymous"
ANON_CLIENT_ID = "nas-anonymous"
ANON_ACCOUNT_TIER = "anonymous"
GUEST_MODEL = "nous/welcome"
ANON_SECRET_HEADER = "x-anonymous-api-secret"
# The shared secret gates the anonymous surface during its integration phase. It is a deployment
# secret (Sid's), read from the environment only.
ANON_SECRET_ENV = "HERMES_ANON_API_SECRET"
# Launch gate for the whole free tier while it is pre-GA: exactly "1" turns it on for this process
# (CLI, gateway, serve backend alike); anything else leaves every surface behaving as if the free
# tier did not exist. ``guest_enabled`` is the only reader. Not a user preference: never written to
# config.yaml or .env, never shown in setup. Deleted at GA together with this comment.
GUEST_ONBOARDING_ENV = "HERMES_GUEST_ONBOARDING"
GUEST_MINT_TIMEOUT_SECONDS = 5.0
# Copy shared by every surface that names the free tier (R-USR-1): never guest / anonymous / account.
FREE_TIER_LABEL = "Nous · free tier"
# Used after retries are exhausted on the anonymous inference host. Keep this separate from
# structured refusal copy: a transport outage has no actionable sign-in or route verdict.
FREE_TIER_OUTAGE_COPY = ("The free model is having trouble responding right now. "
                         "Try sending your message again in a minute.")
ANON_GATE_CLOSED = "anon_gate_closed"
ANON_GATE_PAUSED = "anon_gate_paused"
ANON_RATE_LIMITED = "anon_rate_limited"
ANON_POW_REQUIRED = "anon_pow_required"
ANON_ACCOUNT_LOCKED = "anon_account_locked"
ANON_CREDENTIAL_DEAD = "anon_credential_dead"
ANON_SERVER_ERROR = "anon_server_error"
ANON_UNREACHABLE = "anon_unreachable"
ANON_TERMINAL_CODES = frozenset({ANON_GATE_CLOSED, ANON_POW_REQUIRED, ANON_ACCOUNT_LOCKED})
ANON_UNREACHABLE_CODES = frozenset({ANON_UNREACHABLE})
ANON_FAILURE_COPY = {
    ANON_GATE_CLOSED: "Nous free tier isn't available right now. Sign in with a Nous account to continue.",
    ANON_POW_REQUIRED: "The Nous server asked for a proof of work, but that isn't implemented yet.",
    ANON_ACCOUNT_LOCKED: "This Nous free-tier credential is no longer available. Sign in again.",
}
UPGRADE_HINT = "Run `hermes auth upgrade` to sign in with a Nous account, or /login inside a chat."
FREE_TIER_NOT_SIGNED_IN = (
    "You're not signed in. Free inference and connectors are always on. "
    "Run `hermes auth` to sign in with a Nous account.")


class AnonCredentialDead(AuthError):
    """NAS no longer knows this ``anon_`` credential (reaped, or claimed into a real account).

    The one client rule for reap AND claim: mark dead, re-mint on the next need.
    """


def _anon_err(message: str, code: str, *, retry_after: Optional[float] = None) -> AuthError:
    err = AuthError(message, code=code, retry_after=retry_after)
    err.retryable = code not in ANON_TERMINAL_CODES  # type: ignore[attr-defined]
    return err


def guest_enabled() -> bool:
    """The free tier is on for this process: the launch gate is set AND ``nous.guest`` (default
    True) has not switched it off. The only place either is read."""
    if (os.environ.get(GUEST_ONBOARDING_ENV) or "").strip() != "1":
        return False
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


def is_anonymous_request(provider: Any, api_key: Any) -> bool:
    """Classify UX from the credential actually sent, not saved profile state."""
    from hermes_cli.auth_constants import _decode_jwt_claims

    return provider == "nous" and _decode_jwt_claims(api_key).get("account_tier") == ANON_ACCOUNT_TIER


def is_anonymous_agent(agent: Any) -> bool:
    """Classify a live request using its current, potentially rotated credential."""
    return is_anonymous_request(getattr(agent, "provider", ""), getattr(agent, "api_key", None))


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
    extras = {part.strip().lower() for part in os.environ.get("HERMES_EXTRA_WELCOME_HOSTS", "").split(",")
              if part.strip()}
    return host in WELCOME_HOSTS or host in extras


def classify_mint_exception(exc: BaseException) -> AuthError:
    """Normalize account-service and transport failures for bootstrap/sign-in surfaces."""
    if isinstance(exc, AuthError):
        return exc
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, OSError)):
        return AuthError("The Nous service could not be reached; check your internet connection and try again.",
                         code=ANON_UNREACHABLE)
    return AuthError(str(exc), code=ANON_SERVER_ERROR)


def anon_failure_copy(code: str, *, retry_after: Any = 0) -> str:
    if code == ANON_GATE_CLOSED:
        return "The Nous account service is not available right now. Sign in with a Nous account to continue."
    if code == ANON_POW_REQUIRED:
        return ANON_FAILURE_COPY[ANON_POW_REQUIRED]
    if code == ANON_ACCOUNT_LOCKED:
        return ANON_FAILURE_COPY[ANON_ACCOUNT_LOCKED]
    if code in {ANON_RATE_LIMITED, ANON_GATE_PAUSED}:
        return f"The Nous service is busy. Try again in {friendly_wait(retry_after)}."
    return "Sign-in didn't finish. Try again whenever you're ready."


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
        raise AnonCredentialDead("Nous free-tier credential is no longer valid.", code=ANON_CREDENTIAL_DEAD)
    if response.status_code == 401 and error == "invalid_shared_secret":
        raise _anon_err("Nous free tier is not open on this portal.", ANON_GATE_CLOSED)
    if response.status_code == 401:
        raise AnonCredentialDead("Nous free-tier credential was revoked.", code=ANON_CREDENTIAL_DEAD)
    if response.status_code == 429:
        delay = _retry_after_seconds(response, 60.0)
        raise _anon_err(f"Nous free tier is rate limited; try again in {friendly_wait(delay)}.",
                        ANON_RATE_LIMITED, retry_after=delay)
    if response.status_code == 403 and error in {"anonymous_accounts_disabled", "circuit_open"}:
        raise _anon_err("Nous free tier is currently disabled.", ANON_GATE_CLOSED)
    if response.status_code == 403 and error == "account_locked":
        raise AnonCredentialDead("Nous free-tier credential is no longer valid.", code=ANON_ACCOUNT_LOCKED)
    if response.status_code == 428 and error == "pow_required":
        raise _anon_err("The Nous server asked for a proof of work, but that isn't implemented yet.",
                        ANON_POW_REQUIRED)
    if response.status_code == 503 and error == "temporarily_disabled":
        raise _anon_err("Nous free tier is temporarily paused; try again shortly.", ANON_GATE_PAUSED)
    if response.status_code == 404 and error == "not_found":
        raise _anon_err("Nous free tier is not open on this portal. Sign in with a Nous account instead.",
                        ANON_GATE_CLOSED)
    detail = "hiccup" if response.status_code >= 500 or not error else error
    raise _anon_err(f"Nous free tier {action} hit a server hiccup ({response.status_code}{': ' + detail}).",
                    ANON_SERVER_ERROR)


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
    # NAS names the welcome host on every exchange; absent (older NAS) or outside the allowlist
    # (a staging host without NOUS_INFERENCE_BASE_URL set), the literal stands in. Never the paid
    # host: the gateway cross-refuses an anonymous JWT there.
    inference_url = (_validate_nous_inference_url_from_network(exchanged.get("inference_base_url"))
                     or DEFAULT_NOUS_WELCOME_URL)
    scope = claims.get("scope") or claims.get("scp") or state.get("scope")
    if isinstance(scope, (list, tuple)):
        scope = " ".join(str(s) for s in scope)
    state.update(
        access_token=access_token, token_type="Bearer", scope=scope,
        obtained_at=now.isoformat(), expires_at=expires_at.isoformat(),
        expires_in=max(0, int((expires_at - now).total_seconds())),
        account_tier=str(claims.get("account_tier") or ANON_ACCOUNT_TIER))
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


def _mint_locked(
    client: httpx.Client, portal: str, auth_store: Dict[str, Any], *, carries_inference: bool = True,
) -> Dict[str, Any]:
    """Mint under the caller's locks. The identity is persisted as soon as ``create`` succeeds, BEFORE
    the exchange: a 429 or timeout on the exchange must not lose a credential NAS still honours (the
    next attempt exchanges the stored one instead of minting again).

    ``carries_inference`` decides whether the new identity also becomes ``active_provider``. The
    bootstrap passes False when its inventory found another usable provider: the identity exists for
    connectors, the user's own provider keeps carrying inference (NS-845 Q1.3)."""
    from hermes_cli.auth import _store_provider_state, _save_auth_store
    from hermes_cli.auth_nous import _write_shared_nous_state
    minted = mint_guest(client, portal)
    state: Dict[str, Any] = {
        "auth_method": ANON_AUTH_METHOD, "account_tier": ANON_ACCOUNT_TIER,
        "anon_token": minted["token"], "client_id": ANON_CLIENT_ID,
        "portal_base_url": portal.rstrip("/"),
        "user_id": minted.get("user_id"), "org_id": minted.get("org_id"),
        "idle_ttl_days": minted.get("idle_ttl_days"),
    }
    _store_provider_state(auth_store, "nous", state, set_active=carries_inference)
    _save_auth_store(auth_store)
    _write_shared_nous_state(state)
    logger.info("Nous free tier ready (identity minted)")
    return state


@dataclass(frozen=True)
class MintFailure:
    code: str
    message: str
    retryable: bool
    retry_after: float
    not_before: float

    def as_payload(self) -> Dict[str, Any]:
        remaining = max(0.0, self.not_before - time.monotonic()) if self.retryable else 0.0
        return {"error": self.message, "error_code": self.code, "retryable": self.retryable,
                "retry_after": int(round(remaining))}


# A failure memo is profile-scoped: one profile's disabled gate must not prevent another from
# setting up. Unscoped operation uses the launch-profile slot.
_mint_failure: Optional[MintFailure] = None
_mint_failures_by_home: Dict[str, MintFailure] = {}
_MINT_RETRY_LADDER = _UNREACHABLE_RETRY_LADDER = (15.0, 60.0, 300.0)
_unreachable_attempts = 0


def _mint_failure_for_profile() -> Optional[MintFailure]:
    from hermes_constants import get_hermes_home_override, hermes_home_key
    if get_hermes_home_override() is None:
        return _mint_failure
    return _mint_failures_by_home.get(hermes_home_key())


def _store_mint_failure(failure: Optional[MintFailure]) -> None:
    global _mint_failure
    from hermes_constants import get_hermes_home_override, hermes_home_key
    if get_hermes_home_override() is None:
        _mint_failure = failure
    elif failure is None:
        _mint_failures_by_home.pop(hermes_home_key(), None)
    else:
        _mint_failures_by_home[hermes_home_key()] = failure


def _record_mint_failure(exc: Exception) -> MintFailure:
    global _unreachable_attempts
    code = getattr(exc, "code", None)
    message = str(exc)
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, OSError)):
        code = ANON_UNREACHABLE
    if code == ANON_UNREACHABLE:
        retryable = True
        delay = _UNREACHABLE_RETRY_LADDER[min(_unreachable_attempts, len(_UNREACHABLE_RETRY_LADDER) - 1)]
        _unreachable_attempts += 1
    elif code in ANON_TERMINAL_CODES:
        retryable, delay = False, 0.0
    elif code == ANON_RATE_LIMITED:
        retryable, delay = True, max(0.0, float(getattr(exc, "retry_after", None) or 60.0))
    elif code == ANON_GATE_PAUSED:
        retryable, delay = True, max(60.0, float(getattr(exc, "retry_after", None) or 60.0))
    else:
        code = code or ANON_SERVER_ERROR
        retryable, delay = True, max(1.0, float(getattr(exc, "retry_after", None) or 15.0))
    failure = MintFailure(str(code), message, retryable, delay, time.monotonic() + delay)
    _store_mint_failure(failure)
    return failure


def last_mint_failure() -> Optional[Dict[str, Any]]:
    failure = _mint_failure_for_profile()
    return failure.as_payload() if failure is not None else None


def reset_mint_memo_for_tests() -> None:
    """Clear process-local mint cooldown state between isolated service tests."""
    global _mint_failure, _unreachable_attempts
    _mint_failure = None
    _mint_failures_by_home.clear()
    _unreachable_attempts = 0


def _mint_failed_for_profile() -> bool:
    failure = _mint_failure_for_profile()
    return bool(failure and (not failure.retryable or time.monotonic() < failure.not_before))


def _set_mint_failed(failed: bool) -> None:
    if not failed:
        _store_mint_failure(None)


def _reconcile_and_provision(*, timeout_seconds: float, carries_inference: bool = True) -> Optional[Dict[str, Any]]:
    """The lifecycle body, run under profile lock THEN shared lock (the documented order).

    1. The shared store is the identity of record for this Hermes root. If it holds an identity
       that differs from the profile's, the profile adopts it (a stale guest never outlives a
       sibling profile's sign-in, and never overwrites it). An adopted free-tier identity claims
       ``active_provider`` under the same rule as a mint; an adopted ACCOUNT always does (the user
       signed in somewhere on this machine).
    2. Otherwise the profile's own identity stands.
    3. Nothing anywhere: mint, persisting the credential before exchanging it.
    """
    from hermes_cli.auth import (
        _auth_store_lock, _load_auth_store, _load_provider_state, _save_auth_store,
        _store_provider_state, _resolve_verify)
    from hermes_cli.auth_nous import (
        _nous_http_client, _nous_shared_store_lock, _read_shared_nous_state, _write_shared_nous_state)
    portal = _portal_base_url()
    with _auth_store_lock():
        auth_store = _load_auth_store()
        profile_state = _load_provider_state(auth_store, "nous")
        with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds, 5.0)):
            shared = _read_shared_nous_state()
            if shared and _shared_identity_key(shared) != _shared_identity_key(profile_state):
                state = dict(shared)
                _store_provider_state(
                    auth_store, "nous", state,
                    set_active=carries_inference or not is_guest_state(state))
                _save_auth_store(auth_store)
                logger.debug("Nous identity adopted from the shared store")
                return state
            if profile_state:
                if not shared:
                    _write_shared_nous_state(profile_state)
                return profile_state
            verify = _resolve_verify(insecure=None, ca_bundle=None, auth_state=None)
            with _nous_http_client(timeout_seconds, verify) as client:
                return _mint_locked(client, portal, auth_store, carries_inference=carries_inference)


def ensure_portal_identity(
    *, explicit: bool, timeout_seconds: float = GUEST_MINT_TIMEOUT_SECONDS,
    carries_inference: bool = True, force: bool = False,
) -> Optional[Dict[str, Any]]:
    """Make sure this profile has a Nous identity (guest or account); mint a guest only if the shared
    store has none. Returns the ``providers.nous`` state, or None (disabled / failed once already).

    ``explicit`` is required and must be True: the only callers are the boot bootstrap
    (``free_tier_bootstrap.run_bootstrap``), the desktop's ``free_tier.provision`` retry, and the
    dead-credential replacements (``auth_nous.resolve_nous_runtime_credentials``,
    ``managed_tool_gateway._replace_dead_guest_token``). Nothing creates an identity as a side effect
    of reading status, resolving a provider or fetching a connector bearer (NS-845 Q1.2).

    Order: ``guest_enabled`` gate -> reconcile with the shared store -> mint. Locks are taken profile
    first, then shared, matching every other Nous path. ``carries_inference=False`` leaves
    ``active_provider`` alone (the identity is for connectors; another provider does inference).
    Blocking, bounded by ``timeout_seconds``; the bootstrap puts it on its own thread.
    """
    if not explicit:
        raise ValueError("ensure_portal_identity: only explicit creators may call this (explicit=True)")
    if not guest_enabled():
        return None
    if _mint_failed_for_profile() and not current_nous_state() and not force:
        return None  # this profile already tried and failed in this process; do not hammer the portal
    try:
        state = _reconcile_and_provision(
            timeout_seconds=timeout_seconds, carries_inference=carries_inference)
        if state is not None:
            _set_mint_failed(False)
            global _unreachable_attempts
            _unreachable_attempts = 0
        return state
    except Exception as exc:
        failure = _record_mint_failure(exc)
        normalized = classify_mint_exception(exc)
        normalized.retry_after = failure.retry_after
        normalized.retryable = failure.retryable  # type: ignore[attr-defined]
        if isinstance(exc, AuthError):
            raise normalized
        raise normalized from exc


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
    _set_mint_failed(False)
    logger.info("Nous free-tier identity retired (%s); a new one is set up on next use", reason)


# --- Gateway welcome-tier contract: structured refusals and the model-switch header ------------------
#
# The inference gateway answers a welcome-tier request it will not serve with a structured 429
# (``{status, message, reason, retry_after, alternates?, upgrade_url?}``), and a request on the wrong
# host with a 400 (or a 403 while the tier is dark) whose message names the right host. A NAMED
# account that still asks for ``nous/welcome`` is served the id's backing model and told what to
# switch to in the ``x-nous-model-switch`` response header. Every rule for reading those lives here;
# the error classifier and the turn loop only call in.

MODEL_SWITCH_HEADER = "x-nous-model-switch"
# Fairshare refusal reasons the welcome tier can answer with (api ``FairshareRefusalReason``).
WELCOME_REFUSAL_REASONS = frozenset(
    {"model_not_free", "feature_not_free", "at_capacity", "admission_closed", "rate_limited"})
# Reasons that mean "not on this tier, ever": no retry helps, only a sign-in or another provider.
WELCOME_TIER_GATE_REASONS = frozenset({"model_not_free", "feature_not_free"})
# Gateway messages (lowercased substrings) for a request on the wrong host or a dark tier.
_WELCOME_ROUTE_REFUSALS = (
    ("anonymous accounts must use", "anon_on_paid_host"),
    ("serves anonymous hermes agent accounts only", "named_on_welcome_host"),
    ("anonymous accounts are not accepted", "tier_disabled"),
)
_WELCOME_ROUTE_COPY = {
    "anon_on_paid_host": "The Nous free tier must use its own inference host ({host}); "
                         "Hermes is pointed at the paid one. Restart Hermes to re-read the route, "
                         "or unset NOUS_INFERENCE_BASE_URL if you set it.",
    "named_on_welcome_host": "This Nous account needs to reconnect to its account route. "
                             "Run /model and pick the Nous row again.",
    "tier_disabled": "The Nous free tier is switched off right now. {signin}",
}
_SIGNIN_CHAT = "Sign in with a Nous account for the full catalog: /login."
_SIGNIN_TERMINAL = "Sign in with a Nous account for the full catalog: `hermes auth upgrade`."


def friendly_wait(seconds: Any) -> str:
    """Render a retry delay as a short, rounded duration rather than raw seconds."""
    try:
        value = max(0.0, float(seconds or 0))
    except (TypeError, ValueError):
        value = 0.0
    if value <= 15:
        return "a few seconds"
    if value < 90:
        return "about a minute"
    if value < 3600:
        return f"about {int(round(value / 60))} minutes"
    hours = int(round(value / 3600))
    return "about an hour" if hours <= 1 else f"about {hours} hours"


def parse_welcome_refusal(body: Any) -> Optional[Dict[str, Any]]:
    """The structured welcome-tier refusal in a gateway 429 body, or None for any other shape.

    Returns ``{"reason", "retry_after", "alternates", "upgrade_url"}`` with ``retry_after`` an int
    of whole seconds (0 when the gateway sent none) and ``alternates`` a list of model ids.
    """
    if not isinstance(body, dict):
        return None
    reason = body.get("reason")
    if not isinstance(reason, str) or reason not in WELCOME_REFUSAL_REASONS:
        return None
    raw_retry = body.get("retry_after")
    try:
        retry_after = max(0, int(float(raw_retry))) if raw_retry not in (None, "") else 0
    except (TypeError, ValueError):
        retry_after = 0
    raw_alternates = body.get("alternates")
    alternates = [str(a) for a in raw_alternates if isinstance(a, str) and a] if isinstance(raw_alternates, list) else []
    upgrade_url = body.get("upgrade_url")
    return {"reason": reason, "retry_after": retry_after, "alternates": alternates,
            "upgrade_url": upgrade_url if isinstance(upgrade_url, str) else ""}


def welcome_refusal_copy(
    refusal: Dict[str, Any], *, model: str = "", in_chat: bool = True, door: bool = True,
) -> str:
    """User copy for a structured welcome-tier refusal: what happened and the one way forward.

    Never guest / anonymous / claim; ``in_chat`` picks ``/login`` over the terminal verb.
    ``door=False`` omits the sign-in tail when a surface renders it as a button."""
    signin = (_SIGNIN_CHAT if in_chat else _SIGNIN_TERMINAL) if door else ""
    reason = str(refusal.get("reason") or "")
    alternates = refusal.get("alternates") or []
    serves = alternates[0] if alternates else GUEST_MODEL
    retry = int(refusal.get("retry_after") or 0)
    wait = friendly_wait(retry) if retry > 0 else "a little while"
    if reason == "model_not_free":
        what = f"{model} isn't on the Nous free tier" if model else "That model isn't on the Nous free tier"
        return f"{what}; it serves {serves} only. {signin}".rstrip()
    if reason == "feature_not_free":
        return f"This feature isn't on the Nous free tier. {signin}".rstrip()
    if reason == "at_capacity":
        return ("Chatting without signing in is really busy right now. Sign in to skip the queue, "
                f"it's free, or try again in {wait}. {signin}").rstrip()
    if reason == "admission_closed":
        return f"The Nous free tier isn't admitting new sessions right now. Try again in {wait}. {signin}".rstrip()
    if reason == "rate_limited":
        return f"Nous free tier rate limit active \u2014 resets in {retry}s. {signin}".rstrip()
    return f"The Nous free tier refused this request ({reason}). {signin}".rstrip()


def welcome_route_refusal(status: Any, message: Any, base_url: Any = None) -> Optional[str]:
    """Which host cross-refusal a gateway 400/403 is, by its message; None for any other error.

    ``"anon_on_paid_host"``: a free-tier JWT reached the paid host. ``"named_on_welcome_host"``: an
    account or API key reached the free tier's host. ``"tier_disabled"``: the tier is dark
    (``WELCOME_MODE=off``). Each is deterministic for the request: retrying cannot help."""
    if status not in (400, 403):
        return None
    text = str(message or "").lower()
    kind = next((kind for needle, kind in _WELCOME_ROUTE_REFUSALS if needle in text), None)
    if kind is None and status == 403 and route_is_welcome_host(base_url):
        return "tier_disabled"
    return kind


def welcome_route_refusal_copy(kind: str, *, in_chat: bool = True, door: bool = True) -> str:
    template = _WELCOME_ROUTE_COPY.get(kind) or "The Nous inference gateway refused this route."
    return template.format(
        host=DEFAULT_NOUS_WELCOME_URL,
        signin=(_SIGNIN_CHAT if in_chat else _SIGNIN_TERMINAL) if door else "").rstrip()


def note_model_switch(agent: Any, headers: Any) -> Optional[str]:
    """Record the gateway's ``x-nous-model-switch`` header on *agent* for the next call, if present.

    The header arrives on a NAMED account's response that asked for ``nous/welcome`` (the gateway
    served the backing model and billed it normally): the free tier's model no longer belongs in
    this install's configuration. Recorded here, applied by :func:`apply_model_switch` between
    calls so a response still streaming is never re-labelled under itself. Returns the backing id.
    """
    if headers is None:
        return None
    value = None
    try:
        value = headers.get(MODEL_SWITCH_HEADER)
        if value is None and hasattr(headers, "items"):
            value = next((v for k, v in headers.items() if str(k).lower() == MODEL_SWITCH_HEADER), None)
    except Exception:
        return None
    backing = str(value or "").strip()
    if not backing:
        return None
    requested = str(getattr(agent, "model", "") or "")
    if backing == requested:
        return None
    try:
        agent._nous_pending_model_switch = (requested, backing)
    except Exception:
        return None
    return backing


def apply_model_switch(agent: Any) -> Optional[str]:
    """Move *agent* (and the config default, when it still names the switched id) to the backing
    model the gateway named. Returns the new model, or None when nothing was pending.

    Runs once per recorded header, between calls. The conversation keeps its history; only the id
    the next request carries changes, so a promoted account stops relying on the gateway's reverse
    map. The config write is the same one a sign-in completion uses, so ``hermes model`` and the
    gateway's config re-read agree with the live session.
    """
    pending = getattr(agent, "_nous_pending_model_switch", None)
    if not pending:
        return None
    agent._nous_pending_model_switch = None
    requested, backing = pending
    if str(getattr(agent, "model", "") or "") != requested:
        return None  # the session already moved (a /model, a sign-in sweep)
    agent.model = backing
    # The gateway's cache check compares agent.model with the config default and evicts on a
    # mismatch it did not cause; this pair names the move so the check can recognise exactly this
    # server-driven switch even when the config write below did not land.
    agent._nous_model_switch = (requested, backing)
    logger.info("Nous gateway asked to switch %s -> %s; applied for this session", requested, backing)
    try:
        from hermes_cli.config import load_config_readonly
        raw = load_config_readonly().get("model")
        model_cfg = raw if isinstance(raw, dict) else ({"default": raw} if isinstance(raw, str) else {})
        if str(model_cfg.get("default") or "").strip() == requested:
            from hermes_cli.auth import _update_config_for_provider
            _update_config_for_provider(
                "nous", str(getattr(agent, "base_url", "") or ""), default_model=backing)
            logger.info("Config default model moved %s -> %s", requested, backing)
    except Exception as exc:
        logger.debug("model switch: config default left as is: %s", exc)
    status = getattr(agent, "_buffer_status", None)
    if callable(status):
        try:
            status(f"Model is now {backing} (your account's model; {requested} is the free tier's).")
        except Exception:
            pass
    return backing


# One-time CLI notice: an install whose inference is carried by an explicit provider learns once that
# the free tier (inference + connectors) now exists. The flag lives on the guest state itself so it
# dies with the identity; a fresh guest (re-mint, new profile) may announce itself once more.
GUEST_NOTICE_FLAG = "guest_notice_shown"
FREE_TIER_AVAILABLE_NOTICE = (
    "Free Nous inference and connectors are now available. "
    "/model to try them, /login to sign in.")


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
    seconds = parse_retry_after_seconds(response.headers)
    return default if seconds is None else seconds


def _sleep_until(wake: float, cancelled: Optional[Callable[[], bool]]) -> bool:
    """Sleep until the monotonic time *wake*. Returns True when *cancelled* fired first.

    Without a hook this is one plain :func:`time.sleep`. With one the sleep is cut into <= 1 s
    ticks so an attempt stopped from outside ends in about a second instead of blocking to the
    sign-in code's own expiry.
    """
    if cancelled is None:
        remaining = wake - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return False
    while True:
        if cancelled():
            return True
        remaining = wake - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(1.0, remaining))


def wait_for_promotion(
    client: httpx.Client, portal_base_url: str, claim_code: str, *, expires_in: int, interval: int,
    cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Poll ``POST /api/anonymous/promotion-status`` until it leaves ``pending`` or our clock runs out.

    Returns the final status payload; ``{"status": "timeout"}`` when ``expires_in`` elapsed. 429 honours
    ``Retry-After``; other non-2xx statuses raise through :func:`_raise_for_anon_status`.

    *cancelled* is an optional hook a surface passes to stop an attempt it no longer wants (a newer
    sign-in replaced it, the user cancelled, the process is shutting down). It is polled at the top
    of every iteration and on a <= 1 s tick while sleeping; once it has fired this call returns
    ``{"status": "cancelled"}`` for every outcome except a ``completed`` transfer already in hand,
    which is reported so the caller's ``cancel_wins_after_promotion`` ruling can decide it.
    Passing nothing is today's behaviour.
    """
    deadline = time.monotonic() + max(1, int(expires_in))
    wait = max(0, int(interval))
    while time.monotonic() < deadline:
        if cancelled is not None and cancelled():
            return {"status": "cancelled"}
        response = client.post(
            f"{portal_base_url.rstrip('/')}/api/anonymous/promotion-status", headers=_anon_headers(),
            json={"claim_code": claim_code})
        if response.status_code == 429:
            retry = min(_retry_after_seconds(response, default=max(1, wait)),
                        max(0.0, deadline - time.monotonic()))
            if _sleep_until(time.monotonic() + retry, cancelled):
                return {"status": "cancelled"}
            continue
        payload = _raise_for_anon_status(response, action="sign-in")
        status = str(payload.get("status") or "unknown")
        if status != "pending":
            # A completed transfer is already committed on the account service; report it even when
            # the hook fired during this request. run_sign_in's cancel_wins_after_promotion
            # rules what each surface does with it. Every other terminal outcome loses to a cancel.
            if status != "completed" and cancelled is not None and cancelled():
                return {"status": "cancelled"}
            return payload
        if _sleep_until(time.monotonic() + wait, cancelled):
            return {"status": "cancelled"}
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


def _poll_for_token(*args, **kwargs) -> Dict[str, Any]:
    """Keep both the sign-in module seam and the device-flow seam live at call time."""
    from hermes_cli.auth_device_flow import _poll_for_token as poll
    return poll(*args, **kwargs)


def persist_nous_credentials(*args, **kwargs):
    """Keep the existing auth_nous persistence seam behind the sign-in entry point."""
    from hermes_cli.auth_nous import persist_nous_credentials as persist
    return persist(*args, **kwargs)


# Public sign-in imports remain here for existing callers and module-attribute patches.
# The flow imports this module only inside calls, so either module can be imported first.
from hermes_cli.anon_sign_in import (  # noqa: E402
    AlreadySignedIn as AlreadySignedIn,
    Code as Code,
    Completed as Completed,
    Declined as Declined,
    FREE_TIER_RATE_LIMIT_CHAT as FREE_TIER_RATE_LIMIT_CHAT,
    FREE_TIER_RATE_LIMIT_CARD as FREE_TIER_RATE_LIMIT_CARD,
    Failed as Failed,
    LOGIN_BUSY_ELSEWHERE as LOGIN_BUSY_ELSEWHERE,
    LOGIN_COMMAND as LOGIN_COMMAND,
    LOGIN_DM_ONLY as LOGIN_DM_ONLY,
    LOGIN_NOT_ALLOWED as LOGIN_NOT_ALLOWED,
    LOGIN_STARTING as LOGIN_STARTING,
    Retired as Retired,
    SignInState as SignInState,
    Superseded as Superseded,
    TimedOut as TimedOut,
    UPGRADE_ALREADY_SIGNED_IN as UPGRADE_ALREADY_SIGNED_IN,
    UPGRADE_CANCELLED as UPGRADE_CANCELLED,
    UPGRADE_DO_NOT_SHARE as UPGRADE_DO_NOT_SHARE,
    UPGRADE_NOT_COMPLETED as UPGRADE_NOT_COMPLETED,
    UPGRADE_NO_DEFAULT_CHAT as UPGRADE_NO_DEFAULT_CHAT,
    UPGRADE_NO_DEFAULT_TERMINAL as UPGRADE_NO_DEFAULT_TERMINAL,
    UPGRADE_REASON_COPY as UPGRADE_REASON_COPY,
    UPGRADE_START as UPGRADE_START,
    UPGRADE_TIMED_OUT as UPGRADE_TIMED_OUT,
    UPGRADE_UNAVAILABLE as UPGRADE_UNAVAILABLE,
    UPGRADE_UNAVAILABLE_CHAT as UPGRADE_UNAVAILABLE_CHAT,
    UPGRADE_WAITING as UPGRADE_WAITING,
    UPGRADE_WAITING_UP_TO as UPGRADE_WAITING_UP_TO,
    Unavailable as Unavailable,
    Waiting as Waiting,
    _RETIRED_REASONS as _RETIRED_REASONS,
    _default_persist_guard as _default_persist_guard,
    _outcome_state as _outcome_state,
    format_wait_line as format_wait_line,
    run_sign_in as run_sign_in,
)
from hermes_cli.anon_sign_in_cli import (  # noqa: E402
    drain_sign_in_copy as drain_sign_in_copy,
    render_sign_in_cli as render_sign_in_cli,
    render_sign_in_cli_code as render_sign_in_cli_code,
    upgrade_guest as upgrade_guest,
)

FREE_TIER_STATUS_LINE = f"{FREE_TIER_LABEL} \u00b7 {GUEST_MODEL} \u00b7 {LOGIN_COMMAND} to sign in"
