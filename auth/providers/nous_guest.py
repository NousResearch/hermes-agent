"""Canonical nous_guest authentication mechanics; no CLI dependencies."""

from __future__ import annotations

import logging

import math

import os

import time

from dataclasses import dataclass

from datetime import datetime, timedelta, timezone

from typing import Any, Dict, Optional

from agent.retry_utils import parse_retry_after_seconds

from auth.errors import AuthError

from auth.constants import DEFAULT_NOUS_WELCOME_URL, httpx

from auth.token_validation import _decode_jwt_claims

from auth.store_migrations import DEFAULT_NOUS_PORTAL_URL

logger = logging.getLogger("hermes_cli.auth")

ANON_AUTH_METHOD = "anonymous"

ANON_CLIENT_ID = "nas-anonymous"

ANON_ACCOUNT_TIER = "anonymous"

GUEST_MODEL = "nous/welcome"

ANON_SECRET_HEADER = "x-anonymous-api-secret"

ANON_SECRET_ENV = "HERMES_ANON_API_SECRET"

GUEST_ONBOARDING_ENV = "HERMES_GUEST_ONBOARDING"

GUEST_MINT_TIMEOUT_SECONDS = 5.0

FREE_TIER_LABEL = "Nous · free tier"


UPGRADE_HINT = (
    "Run `hermes auth upgrade` to sign in with a Nous account, or /login inside a chat."
)

FREE_TIER_NOT_SIGNED_IN = (
    "You're not signed in. Free inference and connectors are always on. "
    "Run `hermes auth` to sign in with a Nous account."
)


class AnonCredentialDead(AuthError):
    """NAS no longer knows this ``anon_`` credential (reaped, or claimed into a real account).

    The one client rule for reap AND claim: mark dead, re-mint on the next need.
    """


def _anon_err(
    message: str, code: str, *, retry_after: Optional[float] = None
) -> AuthError:
    """An ``AuthError`` for a free-tier failure *code*; terminal-ness is the code's (``ANON_TERMINAL_CODES``)."""
    return AuthError(
        message,
        code=code,
        retry_after=retry_after,
        retryable=code not in ANON_TERMINAL_CODES,
    )


ANON_GATE_CLOSED = "anon_gate_closed"

ANON_GATE_PAUSED = "anon_gate_paused"

ANON_RATE_LIMITED = "anon_rate_limited"

ANON_POW_REQUIRED = "anon_pow_required"

ANON_ACCOUNT_LOCKED = "anon_account_locked"

ANON_CREDENTIAL_DEAD = "anon_credential_dead"

ANON_UNREACHABLE = "anon_unreachable"

ANON_SERVER_ERROR = "anon_server_error"

ANON_TERMINAL_CODES = frozenset({
    ANON_GATE_CLOSED,
    ANON_POW_REQUIRED,
    ANON_ACCOUNT_LOCKED,
})

ANON_UNREACHABLE_CODES = frozenset({ANON_UNREACHABLE, ANON_SERVER_ERROR})

_SIGNIN_IS_FREE = "Signing in is free."

ANON_FAILURE_COPY = {
    ANON_GATE_CLOSED: f"This version can't be used without a Nous account. {_SIGNIN_IS_FREE}",
    ANON_GATE_PAUSED: f"Using Hermes without signing in is paused for a moment. {_SIGNIN_IS_FREE}",
    ANON_RATE_LIMITED: "Lots of people are getting started right now. Try again in {wait}. "
    "Signing in is free and skips the wait.",
    ANON_POW_REQUIRED: "The Nous server asked for a proof of work, but that isn't implemented in your "
    "Agent yet. Sign in with a Nous account to continue.",
    ANON_ACCOUNT_LOCKED: f"This session can't continue without signing in. {_SIGNIN_IS_FREE}",
    ANON_CREDENTIAL_DEAD: "Your session ended. A new one starts on its own.",
    ANON_UNREACHABLE: "The Nous service couldn't be reached. Check your internet connection and try again.",
    ANON_SERVER_ERROR: "The Nous service had a hiccup. Try again in a moment.",
}


def friendly_wait(seconds: Any) -> str:
    """A rounded, spoken duration for user copy: "a few seconds", "about a minute", "about 5 minutes",
    "about an hour". Never a raw second count."""
    try:
        s = max(0.0, float(seconds or 0))
    except (TypeError, ValueError):
        s = 0.0
    if s <= 15:
        return "a few seconds"
    if s < 90:
        return "about a minute"
    if s < 3600:
        return f"about {int(round(s / 60))} minutes"
    hours = int(round(s / 3600))
    return "about an hour" if hours <= 1 else f"about {hours} hours"


def anon_failure_copy(code: str, *, retry_after: Any = None) -> str:
    """The surface-agnostic sentence for a free-tier failure *code* (``ANON_FAILURE_COPY``)."""
    template = ANON_FAILURE_COPY.get(code) or ANON_FAILURE_COPY[ANON_SERVER_ERROR]
    return template.format(wait=friendly_wait(retry_after if retry_after else 60))


def guest_enabled(*, environment) -> bool:
    """The free tier is on for this process: the launch gate is set AND ``nous.guest`` (default
    True) has not switched it off. The only place either is read."""
    environment.require_current_scope()
    if (os.environ.get(GUEST_ONBOARDING_ENV) or "").strip() != "1":
        return False
    try:
        pass
        nous_cfg = environment.read_config().get("nous")
    except (
        Exception
    ) as exc:  # config unreadable: keep today's behaviour (no guest) rather than mint
        logger.debug("guest: config unreadable, treating nous.guest as false: %s", exc)
        return False
    if not isinstance(nous_cfg, dict):
        return True
    return bool(nous_cfg.get("guest", True))


def is_guest_state(state: Any) -> bool:
    return isinstance(state, dict) and state.get("auth_method") == ANON_AUTH_METHOD


def is_anonymous_request(provider: Any, api_key: Any) -> bool:
    """Select anonymous error UX from the credential actually sent, never the saved profile or URL.

    This is display/recovery metadata, not token verification; the gateway authenticates the JWT.
    Named free accounts and opaque API keys must retain normal provider errors.
    """
    from auth.token_validation import _decode_jwt_claims

    return (
        provider == "nous"
        and _decode_jwt_claims(api_key).get("account_tier") == ANON_ACCOUNT_TIER
    )


def is_anonymous_agent(agent: Any) -> bool:
    """:func:`is_anonymous_request` for a live agent: read at call time, since the credential rotates."""
    return is_anonymous_request(
        getattr(agent, "provider", ""), getattr(agent, "api_key", None)
    )


def current_nous_state() -> Optional[Dict[str, Any]]:
    """The profile's ``providers.nous`` state without locking or network (status/picker reads)."""
    from auth.store import _load_auth_store
    from auth.provider_state import _load_provider_state

    try:
        return _load_provider_state(_load_auth_store(), "nous")
    except Exception as exc:
        logger.debug("guest: auth store unreadable: %s", exc)
        return None


def has_guest() -> bool:
    return is_guest_state(current_nous_state())


def guest_carries_inference(*, environment) -> bool:
    """True when the profile's Nous identity is the free tier and the free tier is on.

    Profile-level: use for status, picker and notice surfaces. Routing decisions (which model a
    request may carry) must use :func:`route_is_welcome_host` on the SELECTED runtime instead: a
    credential-pool entry can pick a paid Nous key while the profile singleton is still a guest.
    """
    environment.require_current_scope()
    return guest_enabled(environment=environment) and has_guest()


WELCOME_HOSTS = frozenset({"welcome-api.nousresearch.com"})

EXTRA_WELCOME_HOSTS_ENV = "HERMES_EXTRA_WELCOME_HOSTS"


def welcome_hosts() -> frozenset[str]:
    """``WELCOME_HOSTS`` plus any ``HERMES_EXTRA_WELCOME_HOSTS`` entries (lowercased hostnames)."""
    raw = os.environ.get(EXTRA_WELCOME_HOSTS_ENV) or ""
    extra = {part.strip().lower() for part in raw.split(",") if part.strip()}
    return WELCOME_HOSTS | frozenset(extra) if extra else WELCOME_HOSTS


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
    return host in welcome_hosts()


def anon_secret() -> str:
    return (os.environ.get(ANON_SECRET_ENV) or "").strip()


def _anon_headers() -> Dict[str, str]:
    headers = {"content-type": "application/json"}
    if secret := anon_secret():
        headers[ANON_SECRET_HEADER] = secret
    return headers


_NAS_REFUSALS: Dict[tuple, tuple] = {
    (404, "unknown_token"): (AnonCredentialDead, ANON_CREDENTIAL_DEAD),
    (404, None): (
        AuthError,
        ANON_GATE_CLOSED,
    ),  # uniform with a nonexistent route, on purpose
    (401, "invalid_shared_secret"): (
        AuthError,
        ANON_GATE_CLOSED,
    ),  # pre-launch NAS builds only
    (401, None): (AnonCredentialDead, ANON_CREDENTIAL_DEAD),
    (403, "account_locked"): (AnonCredentialDead, ANON_ACCOUNT_LOCKED),
    (403, "anonymous_accounts_disabled"): (
        AuthError,
        ANON_GATE_PAUSED,
    ),  # pre-launch names
    (403, "circuit_open"): (AuthError, ANON_GATE_PAUSED),
    (428, None): (AuthError, ANON_POW_REQUIRED),
    (429, None): (AuthError, ANON_RATE_LIMITED),
    (503, "temporarily_disabled"): (AuthError, ANON_GATE_PAUSED),
}


def _raise_for_anon_status(response: httpx.Response, *, action: str) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    error = str(payload.get("error") or "")
    status = response.status_code
    if status in (200, 201):
        return payload
    if error.startswith("pow_"):
        error = "pow_"  # pow_required / pow_invalid / pow_replayed are one verdict
    cls, code = (
        _NAS_REFUSALS.get((status, error))
        or _NAS_REFUSALS.get((status, None))
        or (
            (AuthError, ANON_POW_REQUIRED)
            if error == "pow_"
            else (AuthError, ANON_SERVER_ERROR)
        )
    )
    if code == ANON_SERVER_ERROR:
        logger.info(
            "Nous free tier %s failed (%s%s)",
            action,
            status,
            f": {error}" if error else "",
        )
    retry_after = parse_retry_after_seconds(response.headers)
    raise cls(
        anon_failure_copy(code, retry_after=retry_after),
        code=code,
        retry_after=retry_after,
        retryable=code not in ANON_TERMINAL_CODES,
    )


def mint_guest(client: httpx.Client, portal_base_url: str) -> Dict[str, Any]:
    """``POST /api/anonymous/create`` -> ``{user_id, org_id, token, idle_ttl_days}``. Token shown once."""
    response = client.post(
        f"{portal_base_url.rstrip('/')}/api/anonymous/create",
        headers=_anon_headers(),
        json={},
    )
    payload = _raise_for_anon_status(response, action="sign-up")
    token = payload.get("token")
    if not isinstance(token, str) or not token.startswith("anon_"):
        logger.info("Nous free tier sign-up returned no credential")
        raise _anon_err(ANON_FAILURE_COPY[ANON_SERVER_ERROR], ANON_SERVER_ERROR)
    return payload


def exchange_anon_jwt(
    client: httpx.Client, portal_base_url: str, anon_token: str
) -> Dict[str, Any]:
    """``POST /api/anonymous/token {token}`` -> ``{access_token, expires_in, inference_base_url, ...}``.

    Raises :class:`AnonCredentialDead` on 404 ``unknown_token`` / 401 (reaped or claimed).
    """
    response = client.post(
        f"{portal_base_url.rstrip('/')}/api/anonymous/token",
        headers=_anon_headers(),
        json={"token": anon_token},
    )
    payload = _raise_for_anon_status(response, action="token exchange")
    if not isinstance(payload.get("access_token"), str) or not payload["access_token"]:
        logger.info("Nous free tier token exchange returned no token")
        raise _anon_err(ANON_FAILURE_COPY[ANON_SERVER_ERROR], ANON_SERVER_ERROR)
    return payload


def apply_exchange_to_state(state: Dict[str, Any], exchanged: Dict[str, Any]) -> None:
    """Write a fresh exchange result into a guest state in place (token, expiry, routing)."""
    from auth.providers.nous import _validate_nous_inference_url_from_network

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
    inference_url = (
        _validate_nous_inference_url_from_network(exchanged.get("inference_base_url"))
        or DEFAULT_NOUS_WELCOME_URL
    )
    scope = claims.get("scope") or claims.get("scp") or state.get("scope")
    if isinstance(scope, (list, tuple)):
        scope = " ".join(str(s) for s in scope)
    state.update(
        access_token=access_token,
        token_type="Bearer",
        scope=scope,
        obtained_at=now.isoformat(),
        expires_at=expires_at.isoformat(),
        expires_in=max(0, int((expires_at - now).total_seconds())),
        account_tier=str(claims.get("account_tier") or ANON_ACCOUNT_TIER),
    )
    state["inference_base_url"] = inference_url
    for key in ("user_id", "org_id"):
        if exchanged.get(key):
            state[key] = exchanged[key]
    state.pop("refresh_token", None)


def _portal_base_url() -> str:
    from auth.providers.nous import _nous_portal_env_override

    return (_nous_portal_env_override() or DEFAULT_NOUS_PORTAL_URL).rstrip("/")


def _shared_identity_key(state: Any) -> Optional[str]:
    """Stable identity of a Nous credential: the anon_ token for a guest, the refresh token for an
    account. Used to decide whether two stores hold the SAME identity."""
    if not isinstance(state, dict):
        return None
    return (
        state.get("anon_token") if is_guest_state(state) else state.get("refresh_token")
    )


def _mint_locked(
    client: httpx.Client,
    portal: str,
    auth_store: Dict[str, Any],
    *,
    carries_inference: bool = True,
) -> Dict[str, Any]:
    """Mint under the caller's locks. The identity is persisted as soon as ``create`` succeeds, BEFORE
    the exchange: a 429 or timeout on the exchange must not lose a credential NAS still honours (the
    next attempt exchanges the stored one instead of minting again).

    ``carries_inference`` decides whether the new identity also becomes ``active_provider``. The
    bootstrap passes False when its inventory found another usable provider: the identity exists for
    connectors, the user's own provider keeps carrying inference (NS-845 Q1.3)."""
    from auth.provider_state import _store_provider_state
    from auth.store import _save_auth_store
    from auth.providers.nous_store import _write_shared_nous_state

    minted = mint_guest(client, portal)
    state: Dict[str, Any] = {
        "auth_method": ANON_AUTH_METHOD,
        "account_tier": ANON_ACCOUNT_TIER,
        "anon_token": minted["token"],
        "client_id": ANON_CLIENT_ID,
        "portal_base_url": portal.rstrip("/"),
        "user_id": minted.get("user_id"),
        "org_id": minted.get("org_id"),
        "idle_ttl_days": minted.get("idle_ttl_days"),
    }
    _store_provider_state(auth_store, "nous", state, set_active=carries_inference)
    _save_auth_store(auth_store)
    _write_shared_nous_state(state)
    logger.info("Nous free tier ready (identity minted)")
    return state


_MINT_RETRY_LADDER = (15.0, 60.0, 300.0)

_MINT_PAUSED_MIN_WAIT = 60.0


@dataclass
class MintFailure:
    """Why the last mint for a profile failed and when the next one may run."""

    code: str
    message: str
    retryable: bool
    retry_after: (
        float  # the wait this failure asked for, in seconds (0 for a terminal code)
    )
    not_before: float  # ``time.monotonic()`` before which ``ensure_portal_identity`` stays quiet
    attempts: int = 1

    def remaining(self) -> float:
        # Rounded to the millisecond: ``(now + wait) - now`` is not exactly ``wait`` in floating point,
        # and the ceil below turned that dust into an extra whole second ("retry in 61s").
        return (
            0.0
            if not self.retryable
            else max(0.0, round(self.not_before - time.monotonic(), 3))
        )

    def as_payload(self) -> Dict[str, Any]:
        """The wire shape every status RPC carries: ``{error_code, error, retryable, retry_after}``
        with ``retry_after`` the seconds still to wait (whole, rounded up)."""
        return {
            "error_code": self.code,
            "error": self.message,
            "retryable": self.retryable,
            "retry_after": int(math.ceil(self.remaining())) if self.retryable else 0,
        }


_mint_failures: Dict[str, MintFailure] = {}


def _mint_memo_key() -> str:
    from hermes_constants import get_hermes_home_override, hermes_home_key

    return "" if get_hermes_home_override() is None else hermes_home_key()


def _mint_failure_for_profile() -> Optional[MintFailure]:
    return _mint_failures.get(_mint_memo_key())


def _clear_mint_failure() -> None:
    _mint_failures.pop(_mint_memo_key(), None)


def reset_mint_memo_for_tests() -> None:
    _mint_failures.clear()


def last_mint_failure() -> Optional[Dict[str, Any]]:
    """The most recent mint failure for this profile as a wire payload, or None (never failed, or
    cleared by a later success / retirement)."""
    failure = _mint_failure_for_profile()
    return failure.as_payload() if failure else None


def classify_mint_exception(exc: BaseException) -> AuthError:
    """Every mint failure as one ``AuthError`` with a code: the portal's own refusals already are;
    the wire's (timeout, DNS, refused connection) and anything else get a code here. Pure: *exc* is
    never mutated (an uncoded ``AuthError`` is re-raised as a server-error twin)."""
    if isinstance(exc, AuthError) and exc.code:
        return exc
    transport = (TimeoutError, ConnectionError, OSError)
    try:
        transport = transport + (httpx.TimeoutException, httpx.TransportError)
    except Exception:  # httpx unavailable (lazy proxy): the stdlib set stands
        pass
    code = ANON_UNREACHABLE if isinstance(exc, transport) else ANON_SERVER_ERROR
    wrapped = _anon_err(
        ANON_FAILURE_COPY[code], code, retry_after=getattr(exc, "retry_after", None)
    )
    wrapped.__cause__ = exc
    return wrapped


def _note_mint_failure(err: AuthError) -> MintFailure:
    code = str(err.code or ANON_SERVER_ERROR)
    previous = _mint_failure_for_profile()
    attempts = previous.attempts + 1 if previous and previous.code == code else 1
    retryable = code not in ANON_TERMINAL_CODES
    if not retryable:
        wait, not_before = 0.0, float("inf")
    else:
        hinted = err.retry_after
        wait = (
            float(hinted)
            if hinted
            else _MINT_RETRY_LADDER[min(attempts, len(_MINT_RETRY_LADDER)) - 1]
        )
        if code == ANON_GATE_PAUSED:
            wait = max(wait, _MINT_PAUSED_MIN_WAIT)
        wait = max(1.0, wait)
        not_before = time.monotonic() + wait
    failure = MintFailure(
        code=code,
        message=str(err),
        retryable=retryable,
        retry_after=wait,
        not_before=not_before,
        attempts=attempts,
    )
    _mint_failures[_mint_memo_key()] = failure
    return failure


def _reconcile_and_provision(
    *, timeout_seconds: float, carries_inference: bool = True
) -> Optional[Dict[str, Any]]:
    """The lifecycle body, run under profile lock THEN shared lock (the documented order).

    1. The shared store is the identity of record for this Hermes root. If it holds an identity
       that differs from the profile's, the profile adopts it (a stale guest never outlives a
       sibling profile's sign-in, and never overwrites it). An adopted free-tier identity claims
       ``active_provider`` under the same rule as a mint; an adopted ACCOUNT always does (the user
       signed in somewhere on this machine).
    2. Otherwise the profile's own identity stands.
    3. Nothing anywhere: mint, persisting the credential before exchanging it.
    """
    from auth.store import _auth_store_lock, _load_auth_store, _save_auth_store
    from auth.provider_state import _load_provider_state, _store_provider_state
    from auth.oauth import _resolve_verify
    from auth.providers.nous import _nous_http_client
    from auth.providers.nous_store import (
        _nous_shared_store_lock,
        _read_shared_nous_state,
        _write_shared_nous_state,
    )

    portal = _portal_base_url()
    with _auth_store_lock():
        auth_store = _load_auth_store()
        profile_state = _load_provider_state(auth_store, "nous")
        with _nous_shared_store_lock(timeout_seconds=max(timeout_seconds, 5.0)):
            shared = _read_shared_nous_state()
            if shared and _shared_identity_key(shared) != _shared_identity_key(
                profile_state
            ):
                state = dict(shared)
                _store_provider_state(
                    auth_store,
                    "nous",
                    state,
                    set_active=carries_inference or not is_guest_state(state),
                )
                _save_auth_store(auth_store)
                logger.debug("Nous identity adopted from the shared store")
                return state
            if profile_state:
                if not shared:
                    _write_shared_nous_state(profile_state)
                return profile_state
            verify = _resolve_verify(insecure=None, ca_bundle=None, auth_state=None)
            with _nous_http_client(timeout_seconds, verify) as client:
                return _mint_locked(
                    client, portal, auth_store, carries_inference=carries_inference
                )


def ensure_portal_identity(
    *,
    explicit: bool,
    timeout_seconds: float = GUEST_MINT_TIMEOUT_SECONDS,
    carries_inference: bool = True,
    force: bool = False,
    environment,
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

    A failed mint is memoised with a cooldown (``MintFailure``): until it passes, and for a
    terminal code forever, this returns None without touching the portal. ``force=True`` is the
    user's own retry (the desktop's ``free_tier.provision`` button): it makes exactly one attempt
    regardless of the cooldown. Every failure raises an ``AuthError`` whose ``code`` is one of the
    ``ANON_*`` codes and whose ``retry_after`` / ``retryable`` say what a caller may do next.
    """
    environment.require_current_scope()
    if not explicit:
        raise ValueError(
            "ensure_portal_identity: only explicit creators may call this (explicit=True)"
        )
    if not guest_enabled(environment=environment):
        return None
    failure = _mint_failure_for_profile()
    if (
        failure
        and not force
        and not current_nous_state()
        and time.monotonic() < failure.not_before
    ):
        return (
            None  # in cooldown (or terminal) for this profile; do not hammer the portal
        )
    try:
        state = _reconcile_and_provision(
            timeout_seconds=timeout_seconds, carries_inference=carries_inference
        )
    except Exception as exc:
        err = classify_mint_exception(exc)
        noted = _note_mint_failure(err)
        logger.info(
            "Nous free tier not set up (%s, attempt %d%s)",
            noted.code,
            noted.attempts,
            f", next try in {noted.retry_after:.0f}s"
            if noted.retryable
            else ", not retried",
        )
        if err is exc:
            raise
        raise err from exc
    _clear_mint_failure()
    return state


def refresh_guest_state(state: Dict[str, Any], client: httpx.Client) -> None:
    """Token-acquisition seam for a guest: re-exchange the ``anon_`` credential in place.

    The portal URL is the resolver's canonical one (env override, else the validated stored URL,
    else the default), never a raw stored value on its own.
    Raises :class:`AnonCredentialDead` when NAS no longer knows the credential; the caller owns
    re-minting (:func:`ensure_portal_identity` after :func:`clear_dead_guest`).
    """
    anon_token = state.get("anon_token")
    if not isinstance(anon_token, str) or not anon_token:
        raise AnonCredentialDead(
            ANON_FAILURE_COPY[ANON_CREDENTIAL_DEAD], code=ANON_CREDENTIAL_DEAD
        )
    from auth.providers.nous import _nous_portal_base_url

    apply_exchange_to_state(
        state, exchange_anon_jwt(client, _nous_portal_base_url(state), anon_token)
    )


def clear_dead_guest(reason: str, *, dead_token: Optional[str] = None) -> None:
    """Drop a dead guest so the next need re-mints.

    Only the identity that actually failed is removed: a stale profile whose credential NAS rejected
    must not erase a sibling profile's newer sign-in or replacement guest from the shared store. When
    *dead_token* is None the profile's current guest is treated as the failed one.
    """
    from auth.store import (
        _auth_store_lock,
        _load_auth_store,
        _save_auth_store,
        _store_section,
    )
    from auth.provider_state import _load_provider_state
    from auth.providers.nous_store import (
        _clear_shared_nous_state,
        _nous_shared_store_lock,
        _read_shared_nous_state,
    )

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
    _clear_mint_failure()
    logger.info(
        "Nous free-tier identity retired (%s); a new one is set up on next use", reason
    )
