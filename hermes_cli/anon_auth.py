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
from hermes_cli.config_credentials import credential_pool_environment as _phase6_auth_environment

from auth.providers.nous_guest import ANON_FAILURE_COPY, ANON_SERVER_ERROR, FREE_TIER_LABEL, GUEST_MODEL, _anon_err, _anon_headers, _raise_for_anon_status, current_nous_state, friendly_wait, is_guest_state, logger, route_is_welcome_host


import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from agent.retry_utils import parse_retry_after_seconds
from auth.errors import AuthError
from auth.constants import DEFAULT_NOUS_WELCOME_URL, httpx
from auth.token_validation import _decode_jwt_claims
from auth.store_migrations import DEFAULT_NOUS_PORTAL_URL


# The shared secret gates the anonymous surface during its integration phase. It is a deployment
# secret (Sid's), read from the environment only.

# Launch gate for the whole free tier while it is pre-GA: exactly "1" turns it on for this process
# (CLI, gateway, serve backend alike); anything else leaves every surface behaving as if the free
# tier did not exist. ``guest_enabled`` is the only reader. Not a user preference: never written to
# config.yaml or .env, never shown in setup. Deleted at GA together with this comment.


# Copy shared by every surface that names the free tier (R-USR-1): never guest / anonymous / account.


# --- Free-tier failure codes ------------------------------------------------------------------------
#
# Every way the account service (NAS) or the wire can refuse the free tier, as one ``AuthError.code``
# each. Surfaces key their copy on the code; the message on the error is the surface-agnostic
# fallback (no ``/login``, no ``hermes`` verb, no guest / anonymous / credential). ``retryable``
# says whether a later attempt can succeed at all; ``retry_after`` is the wait the server named.
#
# What NAS actually sends (nous-account-service ``api/anonymous/gate.ts`` and the routes behind it):
#   404 ``not_found``             the surface is not enabled on this deployment (terminal)
#   503 ``temporarily_disabled``  the ops breaker is tripped (transient, no hint)
#   429 ``temporarily_unavailable`` + Retry-After   per-address / per-credential limits
#   428 ``pow_required`` / ``pow_invalid`` / ``pow_replayed``   proof-of-work enforced (not implemented here)
#   404 ``unknown_token``         the credential was reaped or claimed (re-mint)
#   403 ``account_locked``        the account is locked (dead; never re-mint from it)
#   401                           an outstanding JWT whose account is gone (re-mint)
          # not enabled here: sign in, or another provider
          # ops breaker: keep checking in the background
        # too many sign-ups / exchanges: wait Retry-After
        # proof of work requested: deferred, sign in instead
    # dead, and no replacement is minted from it
  # reaped or claimed: replaced silently, once
          # timeout, DNS, refused connection
        # 5xx, non-JSON, malformed success body
# Codes a later attempt cannot fix (for this process / this version).

# Codes that mean the account service itself is not answering: a sign-in (which goes through the
# same service) cannot help either, so surfaces offer "try again" / "another provider" only.


# Copy per code: what happened, then the one honest way forward. The free MODEL is never "off":
# what is unavailable is using Hermes without signing in, and signing in is free.


# Dev-only: extra hostnames that count as the welcome host, comma-separated (for example
# ``127.0.0.1`` while ``NOUS_INFERENCE_BASE_URL`` points at a local stand-in). Read from the
# environment, which the user controls, so it sits at the same trust level as the URL override
# itself; it never widens the NETWORK-side allowlist in ``auth_nous``.


# (status, NAS ``error``) -> (exception class, code). ``None`` matches any error string for that
# status; an exact pair wins over the wildcard. Anything unlisted is a server error.


# Per-process mint memo: a failed mint is not retried until its cooldown has passed (a closed gate
# is never retried; a 429 waits out ``Retry-After``; an unreachable portal climbs a short ladder),
# so a boot-time blip cannot hammer the portal AND cannot disable the free tier for the whole
# process the way a plain "tried once" flag did. ``clear_dead_guest`` resets it because a retired
# credential is a reason to mint again; an explicit user retry (``force=True``) bypasses it.
# Keyed per profile home ("" for the unscoped launch profile): profile A's 429 must not stop
# profile B from ever getting an identity.
      # unreachable / server error, attempt 1, 2, 3+
                  # ops breaker: never poll it faster than this


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
    # Only reachable when the route heal (``turn_recovery._recover_welcome_tier``) could not move
    # the session: the one cause left is a user-set NOUS_INFERENCE_BASE_URL naming the paid host.
    "anon_on_paid_host": "This install is set to use a different Nous server (NOUS_INFERENCE_BASE_URL). "
                         "Unset it to use the free model, or sign in. {signin}",
    "named_on_welcome_host": "This Nous account needs to reconnect. {model_hint}",
    "tier_disabled": "Using Hermes without signing in is switched off right now. "
                     "Sign in to keep chatting, it's free. {signin}",
}
# The sign-in door, phrased for a chat surface (slash command) and for a terminal.
_SIGNIN_CHAT = "To sign in: /login."
_SIGNIN_TERMINAL = "To sign in: `hermes auth upgrade`."
_MODEL_HINT_CHAT = "Run /model and pick the Nous row again."
_MODEL_HINT_TERMINAL = "Run `hermes model` and pick the Nous row again."
# Terminal copy for a free-model outage once the retries are spent (5xx, transport failure).
FREE_TIER_OUTAGE_COPY = ("The free model is having trouble responding right now. "
                         "Try sending your message again in a minute.")


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


def welcome_refusal_copy(refusal: Dict[str, Any], *, model: str = "", in_chat: bool = True, door: bool = True) -> str:
    """User copy for a structured welcome-tier refusal: what happened and the one way forward.

    Never guest / anonymous / claim; ``in_chat`` picks ``/login`` over the terminal verb.
    ``door=False`` leaves the "To sign in: …" tail off, for a surface that renders the sign-in as
    a button beside the sentence (the desktop's error card)."""
    signin = (_SIGNIN_CHAT if in_chat else _SIGNIN_TERMINAL) if door else ""
    reason = str(refusal.get("reason") or "")
    alternates = refusal.get("alternates") or []
    serves = alternates[0] if alternates else GUEST_MODEL
    retry = int(refusal.get("retry_after") or 0)
    wait = friendly_wait(retry) if retry > 0 else "a little while"
    if reason == "model_not_free":
        what = f"{model} isn't" if model else "That model isn't"
        return (f"{what} available without signing in, so Hermes uses {serves} for now. "
                f"Sign in for more models. {signin}").rstrip()
    if reason == "feature_not_free":
        return f"That isn't available without signing in. Sign in to use it, it's free. {signin}".rstrip()
    if reason == "at_capacity":
        return ("Chatting without signing in is really busy right now. Sign in to skip the queue, "
                f"it's free, or try again in {wait}. {signin}").rstrip()
    if reason == "admission_closed":
        return ("Chatting without signing in is full right now. Sign in to keep going, "
                f"it's free, or try again in {wait}. {signin}").rstrip()
    if reason == "rate_limited":
        return (f"You've used up the allowance for chatting without signing in. It refreshes in {wait}. "
                f"Sign in for a bigger allowance, it's free. {signin}").rstrip()
    return f"Hermes couldn't send that without signing in. Signing in is free. {signin}".rstrip()


def welcome_route_refusal(status: Any, message: Any, base_url: Any = None) -> Optional[str]:
    """Which host cross-refusal a gateway 400/403 is; None for any other error.

    ``"anon_on_paid_host"``: a free-tier JWT reached the paid host. ``"named_on_welcome_host"``: an
    account or API key reached the free tier's host. ``"tier_disabled"``: the tier is dark
    (``WELCOME_MODE=off``). Each is deterministic for the request: retrying cannot help.

    The dark-tier 403 is keyed on the ROUTE, not the message: the gateway's permission error
    carries only its generic sentence (the detail stays in its logs), so any 403 answered by the
    welcome host means the tier refused this install. The message needles remain for gateways
    that do spell it out, and for the two wrong-host 400s."""
    if status not in (400, 403):
        return None
    text = str(message or "").lower()
    kind = next((kind for needle, kind in _WELCOME_ROUTE_REFUSALS if needle in text), None)
    if kind is None and status == 403 and route_is_welcome_host(base_url):
        return "tier_disabled"
    return kind


def welcome_route_refusal_copy(kind: str, *, in_chat: bool = True, door: bool = True) -> str:
    template = _WELCOME_ROUTE_COPY.get(kind) or "Hermes couldn't reach the free model on this route."
    return template.format(
        host=DEFAULT_NOUS_WELCOME_URL, signin=(_SIGNIN_CHAT if in_chat else _SIGNIN_TERMINAL) if door else "",
        model_hint=_MODEL_HINT_CHAT if in_chat else _MODEL_HINT_TERMINAL).rstrip()


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
    from auth.store import _auth_file_path, _load_auth_store, _same_path, _save_auth_store, _store_section
    from auth.provider_state import _provider_state_transaction
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
        logger.info("Nous free tier sign-in returned no transfer code")
        raise _anon_err(ANON_FAILURE_COPY[ANON_SERVER_ERROR], ANON_SERVER_ERROR)
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
    from hermes_cli.auth import PROVIDER_REGISTRY
    from auth.oauth import _coerce_ttl_seconds, _optional_base_url, _tls_state_from_verify
    from auth.providers.nous import _NOUS_EMPTY_AGENT_KEY_FIELDS, _iso_after, refresh_nous_oauth_from_state
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
    from auth.oauth import _poll_for_token as poll
    return poll(*args, **kwargs)


def persist_nous_credentials(*args, **kwargs):
    """Keep the existing auth_nous persistence seam behind the sign-in entry point."""
    from auth.providers.nous import persist_nous_credentials as persist
    return persist(*args, **kwargs, environment=_phase6_auth_environment())


# Public sign-in imports remain here for existing callers and module-attribute patches.
# The flow imports this module only inside calls, so either module can be imported first.
from hermes_cli.anon_sign_in import (  # noqa: E402
    AlreadySignedIn as AlreadySignedIn,
    Code as Code,
    Completed as Completed,
    Declined as Declined,
    FREE_TIER_RATE_LIMIT_CARD as FREE_TIER_RATE_LIMIT_CARD,
    FREE_TIER_RATE_LIMIT_CHAT as FREE_TIER_RATE_LIMIT_CHAT,
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
