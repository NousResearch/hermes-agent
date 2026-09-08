"""Nous Portal Remote Spending HTTP client.

Thin, fail-loud client for the ``/api/billing/*`` endpoints the terminal billing screens drive.
``nous_account.py`` owns read-only entitlement/balance; this module owns the *write* side: buy
credits, poll a charge, configure auto-reload, change plan. Money is decimal, never float: the
server emits decimal STRINGS (``"142.5"``), parsed with :class:`decimal.Decimal` by callers.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

DEFAULT_PORTAL_BASE_URL = "https://portal.nousresearch.com"

DEFAULT_TIMEOUT = 15.0  # tight so a hung portal doesn't freeze the TUI (charge/poll calls are quick)


# --- Typed errors ---


class BillingError(Exception):
    """A billing HTTP call failed; carries server ``error`` code, HTTP ``status``, the ``portalUrl``
    deep-link (present on every gate denial), ``retry_after`` seconds (429/503) and the parsed ``payload``.
    """

    def __init__(
        self, message: str, *, status: Optional[int] = None, error: Optional[str] = None,
        portal_url: Optional[str] = None, retry_after: Optional[int] = None,
        payload: Optional[dict[str, Any]] = None, actor: Optional[str] = None,
        code: Optional[str] = None, recovery: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status, self.error, self.portal_url, self.retry_after = status, error, portal_url, retry_after
        self.payload = payload or {}
        # Remote-Spending contract extras: `actor` (self|admin) on a revoke, `code` (machine code
        # dual-emitted alongside `error`), `recovery` (reconnect|login|enable_account_toggle).
        # Additive — absent on older NAS / unrelated errors.
        self.actor, self.code, self.recovery = actor, code, recovery


class BillingScopeRequired(BillingError):
    """``403 insufficient_scope`` — the held token lacks ``billing:manage``; the lazy step-up trigger
    (catching it kicks off a device-connect requesting the scope — an ADMIN must select "Allow Remote
    Spending"). Also fires mid-session if the scope is stripped on refresh after the user loses ADMIN.
    """


class BillingAuthError(BillingError):
    """``401`` — missing/invalid bearer token (not logged in / expired)."""


class BillingRemoteSpendingRevoked(BillingError):
    """``403 remote_spending_revoked`` — THIS terminal's spending was revoked (still logged in; only
    the money path is cut, unlike ``insufficient_scope``/``session_revoked``). ``actor`` is
    ``"admin"``/``"self"`` (absent → ``"self"``); recovery is **reconnect** (re-consent device-auth).
    """


class BillingSessionRevoked(BillingAuthError):
    """``401 session_revoked`` — the whole session was logged out; recovery is **re-login**. A
    :class:`BillingAuthError` so 401 handling still treats it as not-logged-in, with typed copy.
    """


class BillingTransient(BillingError):
    """Deterministic non-charge outcome: the request definitely did NOT complete at Stripe, so a
    retry after backoff is always safe — never the "maybe charged" ambiguity of a real 5xx/timeout.
    Covers 429 rate limiting, 503 gate-unavailable, Stripe down, and the daily upgrade cap.
    """


class BillingRateLimited(BillingTransient):
    """``429 rate_limited`` or ``503 temporarily_unavailable`` — NOT a payment failure. Carries
    ``retry_after``; never auto-retry-spam (limiter is 5/org/hr + 5/token/hr). A 503 is the gate
    failing closed — back off, do NOT treat as revoked.
    """


class BillingStripeUnavailable(BillingTransient):
    """``503 stripe_unavailable`` — Stripe itself is down; retry using Retry-After. Not our rate
    limiter: surfaces must read ``.error`` and not render "rate limited" copy.
    """


class BillingUpgradeCapExceeded(BillingTransient):
    """``429 upgrade_cap_exceeded`` — the org hit its 5-upgrades/day cap. Same status as the hourly
    ``rate_limited`` cap but no useful short backoff; a sibling (not subclass) of BillingRateLimited.
    """


# --- Base-URL + auth resolution ---


def resolve_portal_base_url(state: Optional[dict[str, Any]] = None) -> str:
    """Resolve the portal base URL with login-time precedence: env, stored state, default."""
    env = os.getenv("HERMES_PORTAL_BASE_URL") or os.getenv("NOUS_PORTAL_BASE_URL")
    for candidate in (env, state.get("portal_base_url") if state else None):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().rstrip("/")
    return DEFAULT_PORTAL_BASE_URL


def _absolutize_portal_url(portal_url: Optional[str]) -> Optional[str]:
    """Resolve a (possibly relative) server portalUrl against the client's portal base.

    The server emits ``portalUrl`` relative by design; absolute URLs pass through unchanged. urljoin
    needs the trailing slash on the base to join an absolute path like "/billing?..." to the host.
    """
    if not (isinstance(portal_url, str) and portal_url.strip()):
        return portal_url
    return urllib.parse.urljoin(resolve_portal_base_url().rstrip("/") + "/", portal_url)


# Short-lived cache for the resolved (token, base): `resolve_nous_access_token` takes two
# cross-process file locks + reads two files per call, wasteful for the 2s charge poll loop
# (~150 calls per purchase). The resolver only returns tokens with >=120s of life (its refresh
# skew), so a 30s cache can never hand back an about-to-expire token; a 401 still surfaces.
_TOKEN_CACHE_TTL_SECONDS = 30.0
_token_cache: tuple[float, str, str] | None = None  # (cached_at, token, base)


def invalidate_cached_token() -> None:
    """Bust the token cache so post-step-up replays use the freshly-scoped token.

    ``_request`` only self-busts on a 401, not on a 403 scope denial — after a step-up grant the
    cache would otherwise still hold the pre-grant unscoped token and the replay would 403 again.
    """
    global _token_cache
    _token_cache = None


def _billing_not_logged_in(exc: Optional[BaseException] = None) -> "BillingAuthError":
    """Build the canonical 'not logged in' BillingAuthError (single source)."""
    err = BillingAuthError("Not logged into Nous Portal — run `hermes portal` to log in.", status=401, error="invalid_token")
    if exc is not None:
        err.__cause__ = exc
    return err


def _resolve_token_and_base(*, use_cache: bool = True) -> tuple[str, str]:
    """``(access_token, portal_base_url)``, cached for ``_TOKEN_CACHE_TTL_SECONDS`` unless ``use_cache=False``."""
    global _token_cache
    if use_cache and _token_cache is not None:
        cached_at, token, base = _token_cache
        if (time.time() - cached_at) < _TOKEN_CACHE_TTL_SECONDS:
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
        if not (isinstance(token, str) and token.strip()):
            raise _billing_not_logged_in()
    else:
        try:
            token = resolve_nous_access_token()
        except AuthError as exc:
            raise _billing_not_logged_in(exc) from exc
    resolved = (token.strip(), base)
    _token_cache = (time.time(), *resolved)
    return resolved


# --- HTTP plumbing ---


def _retry_after_seconds(headers: Any) -> Optional[int]:
    """Parse a ``Retry-After`` header (integer seconds) — None if absent/bad."""
    from agent.retry_utils import parse_retry_after_seconds

    seconds = parse_retry_after_seconds(headers)
    return None if seconds is None else int(seconds)


# Error routing for _raise_for_error: server ``error`` code alone, then (status, error), then
# status alone, then the generic fallback. Values: (exception class, fallback message when the
# server sent no ``message``). Business 403s (cli_billing_disabled / role_required /
# no_payment_method / monthly_cap_exceeded / …) fall through to a generic BillingError carrying
# code/recovery, using the raw error code as the message.
_ERRORS_BY_CODE: dict[str, tuple[type[BillingError], str]] = {
    "stripe_unavailable": (BillingStripeUnavailable, "Stripe is temporarily unavailable — try again shortly."),
    "upgrade_cap_exceeded": (BillingUpgradeCapExceeded, "Daily plan-change limit reached — try again tomorrow."),
}
_ERRORS_BY_STATUS_CODE: dict[tuple[int, str], tuple[type[BillingError], str]] = {
    (401, "session_revoked"): (BillingSessionRevoked, "Your session was logged out — log in again."),
    (403, "remote_spending_revoked"): (BillingRemoteSpendingRevoked, "Remote spending was stopped for this terminal."),
    (403, "insufficient_scope"): (BillingScopeRequired, "This action needs the billing:manage scope."),
}
_ERRORS_BY_STATUS: dict[int, tuple[type[BillingError], str]] = {
    401: (BillingAuthError, "Authentication required."),
    403: (BillingError, "Billing request denied."),
    429: (BillingRateLimited, "Rate limited — try again shortly."),
    503: (BillingRateLimited, "Rate limited — try again shortly."),
}


def _raise_for_error(status: int, payload: dict[str, Any], headers: Any = None) -> None:
    """Map an HTTP error response to the right typed :class:`BillingError` (see tables above).

    Recognizes the Remote-Spending gate contract (NAS PR #481): 403 ``remote_spending_revoked`` (this
    terminal's spend revoked → reconnect), 401 ``session_revoked`` (full logout → re-login), 503
    ``temporarily_unavailable`` (gate fail-closed → back off, NOT revoked). The business-denial codes
    (``cli_billing_disabled`` + dual ``code:remote_spending_disabled``, ``role_required``,
    ``idempotency_conflict``, …) flow through as a generic BillingError carrying
    ``error``/``code``/``recovery`` for the surface to map.
    """
    p = payload if isinstance(payload, dict) else {}
    error = p.get("error")
    common = {
        "status": status, "error": error, "portal_url": _absolutize_portal_url(p.get("portalUrl")),
        "retry_after": _retry_after_seconds(headers), "payload": p,
        "actor": p.get("actor"), "code": p.get("code"), "recovery": p.get("recovery"),
    }
    key = error if isinstance(error, str) else None
    cls, fallback = (
        _ERRORS_BY_CODE.get(key)
        or _ERRORS_BY_STATUS_CODE.get((status, key))
        or _ERRORS_BY_STATUS.get(status)
        or (BillingError, f"Billing request failed ({status}).")
    )
    raise cls(p.get("message") or (error if cls is BillingError else None) or fallback, **common)


def _request(
    method: str, path: str, *, body: Optional[dict[str, Any]] = None,
    extra_headers: Optional[dict[str, str]] = None, timeout: float = DEFAULT_TIMEOUT, _retried_auth: bool = False,
) -> dict[str, Any]:
    """Authenticated billing request -> parsed JSON dict (``{}`` for an empty 2xx body).

    Raises a typed :class:`BillingError` on any non-2xx or transport failure. A 401 triggers exactly
    one retry with a freshly-resolved token so a cached-but-just-expired token self-heals.
    """
    token, base = _resolve_token_and_base(use_cache=not _retried_auth)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    headers.update(extra_headers or {})
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{base}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                # A 2xx non-JSON body (SPA/reverse-proxy fallback HTML when the route isn't
                # deployed) is a typed non-auth error so callers degrade to "unavailable".
                raise BillingError(
                    "Billing endpoint returned a non-JSON response (it may not be available on this deployment).",
                    error="endpoint_unavailable", status=getattr(resp, "status", None),
                ) from exc
    except urllib.error.HTTPError as exc:
        # 401 on a cached token → drop the cache and retry once with a fresh (refresh-aware) resolve.
        if exc.code == 401 and not _retried_auth:
            invalidate_cached_token()
            return _request(method, path, body=body, extra_headers=extra_headers, timeout=timeout, _retried_auth=True)
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
        raise BillingError(f"Could not reach Nous Portal: {exc.reason}", error="network_error") from exc
    except TimeoutError as exc:
        # urlopen() wraps CONNECT-phase timeouts in URLError, but a timeout during resp.read()
        # surfaces as a bare TimeoutError — normalize to the typed-BillingError contract.
        raise BillingError("Could not reach Nous Portal: timed out", error="network_error") from exc


# --- Endpoints ---


def _require_str(value: Any, message: str, error: str) -> str:
    """Return ``value.strip()`` or raise a typed BillingError when it is not a non-blank str."""
    if not (isinstance(value, str) and value.strip()):
        raise BillingError(message, error=error)
    return value.strip()


def _post_idempotent(path: str, body: dict[str, Any], idempotency_key: str, what: str, timeout: float) -> dict[str, Any]:
    """POST with a mandatory ``Idempotency-Key`` header (missing header is a server 400)."""
    key = _require_str(idempotency_key, f"Idempotency-Key is required for {what}.", "idempotency_key_required")
    return _request("POST", path, body=body, extra_headers={"Idempotency-Key": key}, timeout=timeout)


def get_billing_state(*, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``GET /api/billing/state`` — role-tiered overview (no scope required)."""
    return _request("GET", "/api/billing/state", timeout=timeout)


def patch_auto_top_up(
    *, enabled: bool, threshold: float | str, top_up_amount: float | str, timeout: float = DEFAULT_TIMEOUT
) -> dict[str, Any]:
    """``PATCH /api/billing/auto-top-up`` — configure auto-reload (scope required; strict body, JSON numbers)."""
    body = {"enabled": bool(enabled), "threshold": float(threshold), "topUpAmount": float(top_up_amount)}
    return _request("PATCH", "/api/billing/auto-top-up", body=body, timeout=timeout)


def post_charge(*, amount_usd: float | str, idempotency_key: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``POST /api/billing/charge`` — buy credits (scope required). Reuse the UUID ``idempotency_key`` on retry.

    Returns ``202 {chargeId}`` — money is NOT confirmed yet; poll with :func:`get_charge_status`.
    """
    return _post_idempotent("/api/billing/charge", {"amountUsd": float(amount_usd)}, idempotency_key, "a charge", timeout)


def get_charge_status(charge_id: str, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``GET /api/billing/charge/{id}`` — poll a charge (scope required).

    Returns ``{status: "pending"|"settled"|"failed", ...}``. An unknown or foreign id returns
    ``{status:"pending"}`` (never 404) — a ``pending`` past the 5-min cap is a *timeout*, not an error.
    """
    charge_id = _require_str(charge_id, "A charge id is required.", "invalid_charge_id")
    safe_id = urllib.parse.quote(charge_id, safe="")  # a stray slash must not change the path shape
    return _request("GET", f"/api/billing/charge/{safe_id}", timeout=timeout)


def get_subscription_state(*, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``GET /api/billing/subscription`` — current plan, tiers, usage (raw JSON; no scope)."""
    return _request("GET", "/api/billing/subscription", timeout=timeout)


# --- Subscription change — preview + the pending-change resource + upgrade ---
# Chargeless lane: preview (quote only) and PUT/DELETE pending-change (schedule/clear a downgrade
# or cancellation, effective at period end). The ONE money route: POST upgrade (prorate + charge
# + flip the plan in one Stripe op). All require ``billing:manage`` (403 insufficient_scope ->
# BillingScopeRequired, driving the device step-up) — including preview, which reveals amounts.


def post_subscription_preview(*, subscription_type_id: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``POST /api/billing/subscription/preview`` — a chargeless effect quote.

    ``effect`` is ``charge_now`` (upgrade; ``amountDueNowCents`` prorated), ``scheduled``
    (downgrade; ``effectiveAt`` period end), ``no_op``, or ``blocked`` (``reason`` says why).
    """
    return _request("POST", "/api/billing/subscription/preview", body={"subscriptionTypeId": subscription_type_id}, timeout=timeout)


def put_subscription_pending_change(
    *, subscription_type_id: str | None = None, cancel: bool = False, timeout: float = DEFAULT_TIMEOUT
) -> dict[str, Any]:
    """``PUT /api/billing/subscription/pending-change`` — set the single end-of-period intent.

    ``cancel=True`` schedules a cancellation; ``subscription_type_id`` a downgrade / same-price change.
    UPGRADES are rejected here (they charge now — use :func:`post_subscription_upgrade`).
    """
    if cancel:
        body: dict[str, Any] = {"type": "cancellation"}
    else:
        tier = _require_str(
            subscription_type_id, "A subscription tier is required to schedule a plan change.", "invalid_subscription_type"
        )
        body = {"type": "tier_change", "subscriptionTypeId": tier}
    return _request("PUT", "/api/billing/subscription/pending-change", body=body, timeout=timeout)


def delete_subscription_pending_change(*, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """``DELETE /api/billing/subscription/pending-change`` — clear a scheduled downgrade/cancellation.

    Chargeless, but re-enables recurring spend: needs ``billing:manage`` and honors the org kill-switch.
    """
    return _request("DELETE", "/api/billing/subscription/pending-change", timeout=timeout)


def post_subscription_upgrade(
    *, subscription_type_id: str, idempotency_key: str, timeout: float = DEFAULT_TIMEOUT
) -> dict[str, Any]:
    """``POST /api/billing/subscription/upgrade`` — immediate paid upgrade, the SINGLE money route.

    One Stripe op prorates, charges the card on file, flips the plan. Reuse ``idempotency_key`` on retry.
    """
    return _post_idempotent(
        "/api/billing/subscription/upgrade", {"subscriptionTypeId": subscription_type_id}, idempotency_key, "an upgrade", timeout
    )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

BILLING_MANAGE_SCOPE = "billing:manage"
# ---- END PLUGIN-COMPAT ----
