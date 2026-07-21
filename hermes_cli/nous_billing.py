"""Nous Portal Remote Spending HTTP client (Phase 2b).

Thin, fail-loud client for the four ``/api/billing/*`` endpoints the terminal
billing screens drive. Companion to ``hermes_cli/nous_account.py`` (which owns
read-only entitlement/balance) — this module owns the *write* side: buy credits,
poll a charge, configure auto-reload.

Design rules:

- **Money is decimal, never float.** The server emits decimal STRINGS
  (``"142.5"`` — not fixed 2dp). We parse with :class:`decimal.Decimal` and never
  round-trip through float.
- **This client raises typed exceptions; it does NOT fail open.** Fail-open is the
  *caller's* job (the ``agent/billing_view.py`` builders) so each surface can
  decide how to degrade. A raw network/HTTP error here surfaces as
  :class:`BillingError` (or a subclass) carrying the parsed server ``error`` code,
  HTTP status, ``portalUrl`` deep-link, and ``retry_after``.
- **Auth** = the OAuth bearer JWT Hermes already holds for inference
  (``get_provider_auth_state("nous")["access_token"]``). No API-key auth on these.
- **Portal base URL** resolves with the same precedence as the device-flow login
  (``auth.py``): ``HERMES_PORTAL_BASE_URL`` → ``NOUS_PORTAL_BASE_URL`` → the
  stored auth-state ``portal_base_url`` → the registry default. This is how the
  E2E run points the client at a preview deployment with zero code change.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

DEFAULT_PORTAL_BASE_URL = "https://portal.nousresearch.com"

# Default HTTP timeout (seconds). Charge/poll calls are quick; keep this tight so
# a hung portal doesn't freeze the TUI.
DEFAULT_TIMEOUT = 15.0

# Scope the privileged billing endpoints require. Mirrored from
# hermes_cli.auth.NOUS_BILLING_MANAGE_SCOPE (kept here too so this module has no
# import-time dependency on the much heavier auth module).
BILLING_MANAGE_SCOPE = "billing:manage"


# =============================================================================
# Typed errors
# =============================================================================


class BillingError(Exception):
    """A billing HTTP call failed.

    Carries everything a surface needs to render the right message + affordance:
    the server ``error`` code, HTTP ``status``, an optional human ``message``, the
    ``portalUrl`` deep-link (present on every gate denial), and ``retry_after``
    seconds (429/503). ``payload`` is the full parsed JSON body when available.
    """

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        error: Optional[str] = None,
        portal_url: Optional[str] = None,
        retry_after: Optional[int] = None,
        payload: Optional[dict[str, Any]] = None,
        actor: Optional[str] = None,
        code: Optional[str] = None,
        recovery: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error = error
        self.portal_url = portal_url
        self.retry_after = retry_after
        self.payload = payload or {}
        # Remote-Spending contract extras (NAS PR #481): `actor` (self|admin) on a
        # revoke, `code` (the new machine code dual-emitted alongside `error`), and
        # `recovery` (reconnect|login|enable_account_toggle). Additive — absent on
        # older NAS / unrelated errors.
        self.actor = actor
        self.code = code
        self.recovery = recovery


class BillingScopeRequired(BillingError):
    """``403 insufficient_scope`` — the held token lacks ``billing:manage``.

    The lazy step-up trigger: catching this kicks off a fresh device-connect that
    requests ``billing:manage`` (and tells the user an ADMIN must select "Allow
    Remote Spending"). Also fires mid-session if the scope is stripped on refresh
    after the user loses ADMIN.
    """


class BillingAuthError(BillingError):
    """``401`` — missing/invalid bearer token (not logged in / expired)."""


class BillingRemoteSpendingRevoked(BillingError):
    """``403 remote_spending_revoked`` — THIS terminal's spending was revoked.

    Distinct from ``insufficient_scope`` (never had the grant) and from
    ``session_revoked`` (full logout). The terminal stays logged in; only the
    money path is cut. ``actor`` is ``"admin"`` or ``"self"`` (absent → treat as
    ``"self"``); recovery is **reconnect** (re-consent device-auth). The terminal
    MUST disable charge/auto-reload immediately, without waiting for the next
    token refresh (the current token still claims the scope for ~15 min).
    """


class BillingSessionRevoked(BillingAuthError):
    """``401 session_revoked`` — the whole session was logged out.

    Stronger than a spend-revoke: recovery is **re-login** (full device-auth),
    not just reconnect. Subclass of :class:`BillingAuthError` so existing 401
    handling still treats it as not-logged-in, but the typed code lets the
    surface route to re-login with the right copy.
    """


class BillingTransient(BillingError):
    """A deterministic non-charge outcome: the request definitely did NOT
    reach/complete at Stripe, so it's always safe to retry after backoff —
    never the "maybe charged" ambiguity of a real 5xx/timeout. Covers
    429 rate limiting, 503 gate-unavailable, Stripe being down, and the
    daily upgrade cap — distinct failure modes that share this one
    contract property. Catch this (not the old ad-hoc subclass hierarchy)
    wherever the intent is "any transient, definitely-not-charged billing
    failure, back off and retry/poll".
    """


class BillingRateLimited(BillingTransient):
    """``429 rate_limited`` or ``503 temporarily_unavailable``.

    NOT a payment failure. Carries ``retry_after`` (seconds) — back off and tell
    the user "try again in N min"; never auto-retry-spam (the limiter is
    5/org/hr + 5/token/hr and easy to dig deeper into). A 503 is the gate backend
    failing closed — back off, do NOT treat as revoked.
    """


class BillingStripeUnavailable(BillingTransient):
    """``503 stripe_unavailable`` — Stripe itself is down.

    TRANSIENT: back off and retry using Retry-After; this is NOT the same as
    being throttled by our own rate limiter, so surfaces must not render "rate
    limited" copy for it — they should read ``.error`` to tell the two apart.
    A BillingTransient sibling of BillingRateLimited (not a subclass) — surfaces
    must not render "rate limited" copy for it; read ``.error`` to distinguish it.
    """


class BillingUpgradeCapExceeded(BillingTransient):
    """``429 upgrade_cap_exceeded`` — the org hit its 5-upgrades/day cap.

    Distinct from the hourly ``rate_limited`` charge cap (same HTTP status,
    different meaning + no useful short-Retry-After backoff). A BillingTransient
    sibling of BillingRateLimited (not a subclass) — surfaces must read ``.error``
    to distinguish the failure mode.
    """


# =============================================================================
# Base-URL + auth resolution
# =============================================================================


def resolve_portal_base_url(state: Optional[dict[str, Any]] = None) -> str:
    """Resolve the portal base URL with login-time precedence.

    ``HERMES_PORTAL_BASE_URL`` → ``NOUS_PORTAL_BASE_URL`` → stored auth-state
    ``portal_base_url`` → registry default. Trailing slash stripped.
    """
    env = os.getenv("HERMES_PORTAL_BASE_URL") or os.getenv("NOUS_PORTAL_BASE_URL")
    if env and env.strip():
        return env.strip().rstrip("/")
    if state:
        stored = state.get("portal_base_url")
        if isinstance(stored, str) and stored.strip():
            return stored.strip().rstrip("/")
    return DEFAULT_PORTAL_BASE_URL


def _absolutize_portal_url(portal_url: Optional[str]) -> Optional[str]:
    """Resolve a (possibly relative) server portalUrl to an absolute URL.

    The server emits ``portalUrl`` relative by design (e.g. ``/billing?topup=open``)
    — it doesn't know which deployment the client points at. Resolve it against the
    client's portal base (preview / staging / prod) so deep-links are clickable.
    Idempotent: an already-absolute URL is returned unchanged (urljoin keeps it).
    """
    if not (isinstance(portal_url, str) and portal_url.strip()):
        return portal_url
    base = resolve_portal_base_url()
    # urljoin needs a trailing slash on the base to treat it as a directory and
    # join an absolute path like "/billing?..." against the host. An already-
    # absolute portal_url (with its own scheme/host) is returned as-is.
    return urllib.parse.urljoin(base.rstrip("/") + "/", portal_url)


# Short-lived cache for the resolved (token, base). `resolve_nous_access_token`
# acquires two cross-process file locks + reads two files on every call (even on
# its fast path), which is wasteful when the 2s/5-min charge poll loop calls a
# billing endpoint ~150x per purchase. Cache the result briefly: the resolver
# only ever returns a token with >=120s of life (its refresh skew), so a 30s
# cache can never hand back an about-to-expire token. A 401 still surfaces
# normally (the cache holds a valid token, not the HTTP outcome).
_TOKEN_CACHE_TTL_SECONDS = 30.0
_token_cache: tuple[float, str, str] | None = None  # (cached_at, token, base)


def invalidate_cached_token() -> None:
    """Bust the 30s token cache so post-step-up replays use the freshly-scoped token.

    ``_request`` only self-busts the cache on a 401 (an expired/invalid
    token), not on a 403 scope denial — so after a step-up grant, the
    cache would otherwise still hold the pre-grant unscoped token and
    the immediate replay would 403 again. Callers outside this module
    (e.g. the CLI's scope step-up flow) call this instead of poking
    the private ``_token_cache`` global directly.
    """
    global _token_cache
    _token_cache = None


def _billing_not_logged_in(exc: Optional[BaseException] = None) -> "BillingAuthError":
    """Build the canonical 'not logged in' BillingAuthError (single source)."""
    err = BillingAuthError(
        "Not logged into Nous Portal — run `hermes portal` to log in.",
        status=401,
        error="invalid_token",
    )
    if exc is not None:
        err.__cause__ = exc
    return err


def _resolve_token_and_base(*, use_cache: bool = True) -> tuple[str, str]:
    """Return ``(access_token, portal_base_url)`` for billing calls.

    Uses the same refresh-aware resolver the inference path uses
    (``resolve_nous_access_token``), so a short-lived (~15 min) access token that
    has expired is transparently refreshed via the stored ``refresh_token``
    instead of failing as "not logged in". Raises :class:`BillingAuthError` only
    when there is no usable Nous session at all.

    The result is cached for ``_TOKEN_CACHE_TTL_SECONDS`` to keep the charge poll
    loop from re-locking + re-reading the auth store on every 2s tick. Pass
    ``use_cache=False`` to force a fresh resolution (e.g. after a 401).
    """
    global _token_cache
    import time as _time

    if use_cache and _token_cache is not None:
        cached_at, token, base = _token_cache
        if (_time.time() - cached_at) < _TOKEN_CACHE_TTL_SECONDS:
            return token, base

    try:
        from hermes_cli.auth import get_provider_auth_state

        state = get_provider_auth_state("nous") or {}
    except Exception:
        state = {}

    base = resolve_portal_base_url(state)

    try:
        from hermes_cli.auth import AuthError, resolve_nous_access_token
    except ImportError:
        # auth module unavailable — fall back to the raw stored token.
        token = state.get("access_token")
        if isinstance(token, str) and token.strip():
            resolved = (token.strip(), base)
            _token_cache = (_time.time(), *resolved)
            return resolved
        raise _billing_not_logged_in()

    try:
        token = resolve_nous_access_token()
    except AuthError as exc:
        raise _billing_not_logged_in(exc) from exc
    resolved = (token.strip(), base)
    _token_cache = (_time.time(), *resolved)
    return resolved


# =============================================================================
# HTTP plumbing
# =============================================================================


def _retry_after_seconds(headers: Any) -> Optional[int]:
    """Parse a ``Retry-After`` header (integer seconds) — None if absent/bad."""
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except Exception:
        raw = None
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _raise_for_error(
    status: int, payload: dict[str, Any], headers: Any = None
) -> None:
    """Map an HTTP error response to the right typed :class:`BillingError`.

    Recognizes the Remote-Spending gate contract (NAS PR #481):
    403 ``remote_spending_revoked`` (this terminal's spend revoked → reconnect),
    401 ``session_revoked`` (full logout → re-login), 503 ``temporarily_unavailable``
    (gate fail-closed → back off, NOT revoked). The business-denial codes
    (``cli_billing_disabled`` + dual ``code:remote_spending_disabled``,
    ``role_required``, ``idempotency_conflict``, …) flow through as a generic
    BillingError carrying ``error``/``code``/``recovery`` for the surface to map.
    """
    error = payload.get("error") if isinstance(payload, dict) else None
    message = payload.get("message") if isinstance(payload, dict) else None
    code = payload.get("code") if isinstance(payload, dict) else None
    actor = payload.get("actor") if isinstance(payload, dict) else None
    recovery = payload.get("recovery") if isinstance(payload, dict) else None
    portal_url = _absolutize_portal_url(
        payload.get("portalUrl") if isinstance(payload, dict) else None
    )
    retry_after = _retry_after_seconds(headers)

    common = {
        "status": status,
        "error": error,
        "portal_url": portal_url,
        "retry_after": retry_after,
        "payload": payload if isinstance(payload, dict) else None,
        "actor": actor,
        "code": code,
        "recovery": recovery,
    }

    if error == "stripe_unavailable":
        raise BillingStripeUnavailable(
            message or "Stripe is temporarily unavailable — try again shortly.", **common
        )
    if error == "upgrade_cap_exceeded":
        raise BillingUpgradeCapExceeded(
            message or "Daily plan-change limit reached — try again tomorrow.", **common
        )

    if status == 401:
        # session_revoked is a full logout (→ re-login), stronger than a 401
        # expired-token. Both stay BillingAuthError-compatible for legacy callers.
        if error == "session_revoked":
            raise BillingSessionRevoked(
                message or "Your session was logged out — log in again.", **common
            )
        raise BillingAuthError(message or "Authentication required.", **common)
    if status == 403:
        # Remote spending was stopped for this terminal (NOT the same as never
        # having the scope). Disable spend UI immediately; recovery is reconnect.
        if error == "remote_spending_revoked":
            raise BillingRemoteSpendingRevoked(
                message or "Remote spending was stopped for this terminal.", **common
            )
        if error == "insufficient_scope":
            raise BillingScopeRequired(
                message or "This action needs the billing:manage scope.", **common
            )
        # Business 403s (cli_billing_disabled / role_required / no_payment_method /
        # monthly_cap_exceeded / …) → generic BillingError with code/recovery.
        raise BillingError(message or error or "Billing request denied.", **common)
    if status in (429, 503):
        raise BillingRateLimited(
            message or "Rate limited — try again shortly.", **common
        )
    raise BillingError(message or error or f"Billing request failed ({status}).", **common)


def _request(
    method: str,
    path: str,
    *,
    body: Optional[dict[str, Any]] = None,
    extra_headers: Optional[dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    _retried_auth: bool = False,
) -> dict[str, Any]:
    """Make an authenticated billing request; return the parsed JSON dict.

    Raises a typed :class:`BillingError` on any non-2xx response (or transport
    failure). 2xx with an empty body returns ``{}``. A 401 triggers exactly one
    retry with a freshly-resolved token (bypassing the short token cache) so a
    cached-but-just-expired token self-heals instead of failing the call.
    """
    token, base = _resolve_token_and_base(use_cache=not _retried_auth)
    url = f"{base}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    if extra_headers:
        headers.update(extra_headers)

    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                # A 2xx with a non-JSON body means the endpoint isn't actually
                # serving the billing API here — e.g. a reverse-proxy / SPA
                # fallback HTML page when the route isn't deployed on this
                # deployment. Surface it as a typed, non-auth error so callers
                # degrade gracefully ("unavailable") instead of crashing with a
                # raw JSONDecodeError that reads as "not logged in".
                raise BillingError(
                    "Billing endpoint returned a non-JSON response "
                    "(it may not be available on this deployment).",
                    error="endpoint_unavailable",
                    status=getattr(resp, "status", None),
                ) from exc
    except urllib.error.HTTPError as exc:
        # A 401 on a cached token → drop the cache and retry once with a fresh
        # (refresh-aware) resolve before surfacing the auth error.
        if exc.code == 401 and not _retried_auth:
            global _token_cache
            _token_cache = None
            return _request(
                method,
                path,
                body=body,
                extra_headers=extra_headers,
                timeout=timeout,
                _retried_auth=True,
            )
        raw = ""
        try:
            raw = exc.read().decode("utf-8")
        except Exception:
            raw = ""
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            payload = {}
        _raise_for_error(exc.code, payload, getattr(exc, "headers", None))
        raise  # unreachable; _raise_for_error always raises
    except urllib.error.URLError as exc:
        raise BillingError(
            f"Could not reach Nous Portal: {exc.reason}", error="network_error"
        ) from exc
    except TimeoutError as exc:
        # urlopen() wraps CONNECT-phase timeouts in URLError, but a timeout
        # during resp.read() surfaces as a bare TimeoutError — normalize it so
        # transport failures always honor the typed-BillingError contract.
        raise BillingError(
            "Could not reach Nous Portal: timed out", error="network_error"
        ) from exc


# =============================================================================
# The four endpoints
# =============================================================================


def get_billing_state(*, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``GET /api/billing/state`` — role-tiered overview (no scope required)."""
    return _request("GET", "/api/billing/state", timeout=timeout)


def patch_auto_top_up(
    *,
    enabled: bool,
    threshold: float | str,
    top_up_amount: float | str,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """``PATCH /api/billing/auto-top-up`` — configure auto-reload (scope required).

    Body is strict server-side: extra keys (``maxMonthlySpend``, a payment method)
    are rejected with 400. Numbers are sent as JSON numbers per the contract.
    """
    return _request(
        "PATCH",
        "/api/billing/auto-top-up",
        body={
            "enabled": bool(enabled),
            "threshold": float(threshold),
            "topUpAmount": float(top_up_amount),
        },
        timeout=timeout,
    )


def post_charge(
    *,
    amount_usd: float | str,
    idempotency_key: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """``POST /api/billing/charge`` — buy credits (scope required).

    ``Idempotency-Key`` header is MANDATORY (a missing header is a server 400, not
    a default): generate a UUID per user-confirmed purchase and reuse it on retry.
    Returns ``202 {chargeId}`` — money is NOT confirmed yet; poll with
    :func:`get_charge_status`.
    """
    if not (isinstance(idempotency_key, str) and idempotency_key.strip()):
        raise BillingError(
            "Idempotency-Key is required for a charge.",
            error="idempotency_key_required",
        )
    return _request(
        "POST",
        "/api/billing/charge",
        body={"amountUsd": float(amount_usd)},
        extra_headers={"Idempotency-Key": idempotency_key.strip()},
        timeout=timeout,
    )


def get_charge_status(
    charge_id: str, *, timeout: float = DEFAULT_TIMEOUT
) -> dict[str, Any]:
    """``GET /api/billing/charge/{id}`` — poll a charge (scope required).

    Returns ``{status: "pending"|"settled"|"failed", ...}``. An unknown or foreign
    id returns ``{status:"pending"}`` (never 404, never another org's data) — so a
    ``pending`` that never resolves past the 5-min cap is a *timeout*, not an error.
    """
    if not (isinstance(charge_id, str) and charge_id.strip()):
        raise BillingError("A charge id is required.", error="invalid_charge_id")
    # urllib does not need manual quoting for the opaque ids the server mints, but
    # guard against a stray slash that would change the path shape.
    safe_id = urllib.parse.quote(charge_id.strip(), safe="")
    return _request("GET", f"/api/billing/charge/{safe_id}", timeout=timeout)


def get_subscription_state(*, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``GET /api/billing/subscription`` — current plan, tiers, usage (no scope).

    Returns the raw JSON dict from NAS (WS1 Phase A). Read-only — no
    ``billing:manage`` scope required. Raises :class:`BillingAuthError`
    on 401 and :class:`BillingError` on other non-2xx.
    """
    return _request("GET", "/api/billing/subscription", timeout=timeout)


# =============================================================================
# Subscription change (V3) — preview + the pending-change resource + upgrade
# =============================================================================
#
# Mutating the plan splits into a chargeless lane and the single money route:
#   - preview  → a quote (no mutation, no charge) of what a change would do.
#   - PUT/DELETE pending-change → schedule / clear a downgrade or cancellation
#     (chargeless; takes effect at period end).
#   - POST upgrade → the ONE route that charges (prorate + charge the card on the
#     subscription + flip the plan, in one Stripe op).
# All require the ``billing:manage`` scope (a 403 insufficient_scope raises
# :class:`BillingScopeRequired`, driving the device step-up) — including preview,
# which issues live Stripe calls and reveals charge amounts.


def post_subscription_preview(
    *, subscription_type_id: str, timeout: float = DEFAULT_TIMEOUT
) -> dict[str, Any]:
    """``POST /api/billing/subscription/preview`` — a chargeless effect quote.

    Quotes a change to ``subscription_type_id`` without mutating anything:
    ``effect`` is ``charge_now`` (an upgrade → ``amountDueNowCents`` is the prorated
    upfront charge), ``scheduled`` (a downgrade → ``effectiveAt`` is period end),
    ``no_op`` (already on the tier), or ``blocked`` (``reason`` says why the commit
    would be refused). Also returns the current + target tier and the monthly-credit
    delta. ``amountDueNowCents`` is ``None`` when not a charge or when the proration
    quote is unavailable. Requires ``billing:manage`` (live Stripe calls + amounts).
    """
    return _request(
        "POST",
        "/api/billing/subscription/preview",
        body={"subscriptionTypeId": subscription_type_id},
        timeout=timeout,
    )


def put_subscription_pending_change(
    *,
    subscription_type_id: str | None = None,
    cancel: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """``PUT /api/billing/subscription/pending-change`` — set the end-of-period intent.

    A subscription has at most one pending disposition. Pass ``cancel=True`` to
    schedule a cancellation, or a ``subscription_type_id`` to schedule a downgrade /
    same-price change. UPGRADES are rejected here (they charge immediately — use
    :func:`post_subscription_upgrade`). Chargeless; requires ``billing:manage``.
    Returns ``{rail, changeType, targetTierName, message}`` for a tier change, or
    ``{rail, cancelAtPeriodEnd, message}`` for a cancellation.
    """
    if cancel:
        body: dict[str, Any] = {"type": "cancellation"}
    else:
        if not (
            isinstance(subscription_type_id, str) and subscription_type_id.strip()
        ):
            raise BillingError(
                "A subscription tier is required to schedule a plan change.",
                error="invalid_subscription_type",
            )
        body = {
            "type": "tier_change",
            "subscriptionTypeId": subscription_type_id.strip(),
        }
    return _request(
        "PUT",
        "/api/billing/subscription/pending-change",
        body=body,
        timeout=timeout,
    )


def delete_subscription_pending_change(
    *, timeout: float = DEFAULT_TIMEOUT
) -> dict[str, Any]:
    """``DELETE /api/billing/subscription/pending-change`` — clear it (resume / undo).

    Removes a scheduled downgrade OR cancellation in one call, restoring the live
    active tier and recurring renewal. Chargeless, but it re-enables recurring
    spend, so it requires ``billing:manage`` and is honored by the org kill-switch.
    Returns ``{rail, cancelAtPeriodEnd: false, message}``.
    """
    return _request(
        "DELETE",
        "/api/billing/subscription/pending-change",
        timeout=timeout,
    )


def post_subscription_upgrade(
    *,
    subscription_type_id: str,
    idempotency_key: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """``POST /api/billing/subscription/upgrade`` — immediate paid upgrade.

    The SINGLE money route: one Stripe op prorates, charges the card already on the
    subscription, and flips the plan. ``Idempotency-Key`` is MANDATORY (a missing
    header is a server 400, not a default) — reuse the same key on retry so a replay
    cannot double-charge. Returns ``{status:"upgraded"|"already_on_tier", ...}`` on
    success, or ``{status:"requires_action"|"payment_failed", reason, recoveryUrl}``
    when the charge needs 3DS / was declined and must be finished in the portal at
    ``recoveryUrl``. Requires ``billing:manage``.
    """
    if not (isinstance(idempotency_key, str) and idempotency_key.strip()):
        raise BillingError(
            "Idempotency-Key is required for an upgrade.",
            error="idempotency_key_required",
        )
    return _request(
        "POST",
        "/api/billing/subscription/upgrade",
        body={"subscriptionTypeId": subscription_type_id},
        extra_headers={"Idempotency-Key": idempotency_key.strip()},
        timeout=timeout,
    )
