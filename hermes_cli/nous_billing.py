"""Nous Portal terminal-billing HTTP client (Phase 2b).

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
_BILLING_RESPONSE_BODY_MAX_BYTES = 1024 * 1024
_BILLING_ERROR_BODY_MAX_BYTES = 64 * 1024


class _BillingResponseTooLarge(ValueError):
    pass


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
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error = error
        self.portal_url = portal_url
        self.retry_after = retry_after
        self.payload = payload or {}


class BillingScopeRequired(BillingError):
    """``403 insufficient_scope`` — the held token lacks ``billing:manage``.

    The lazy step-up trigger: catching this kicks off a fresh device-connect that
    requests ``billing:manage`` (and tells the user an ADMIN must tick "Allow
    terminal billing"). Also fires mid-session if the scope is stripped on refresh
    after the user loses ADMIN.
    """


class BillingRateLimited(BillingError):
    """``429 rate_limited`` or ``503 temporarily_unavailable``.

    NOT a payment failure. Carries ``retry_after`` (seconds) — back off and tell
    the user "try again in N min"; never auto-retry-spam (the limiter is
    5/org/hr + 5/token/hr and easy to dig deeper into).
    """


class BillingAuthError(BillingError):
    """``401`` — missing/invalid bearer token (not logged in / expired)."""


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
    """Map an HTTP error response to the right typed :class:`BillingError`."""
    error = payload.get("error") if isinstance(payload, dict) else None
    message = payload.get("message") if isinstance(payload, dict) else None
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
    }

    if status == 401:
        raise BillingAuthError(message or "Authentication required.", **common)
    if status == 403 and error == "insufficient_scope":
        raise BillingScopeRequired(
            message or "This action needs the billing:manage scope.", **common
        )
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
            raw = _read_billing_response_text(
                resp,
                max_bytes=_BILLING_RESPONSE_BODY_MAX_BYTES,
            )
            return json.loads(raw) if raw.strip() else {}
    except _BillingResponseTooLarge as exc:
        raise BillingError(str(exc), error="response_too_large") from exc
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
            raw = _read_billing_response_text(
                exc,
                max_bytes=_BILLING_ERROR_BODY_MAX_BYTES,
            )
        except _BillingResponseTooLarge as read_exc:
            payload = {
                "error": "response_too_large",
                "message": str(read_exc),
            }
            _raise_for_error(exc.code, payload, getattr(exc, "headers", None))
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


def _read_billing_response_text(resp: Any, *, max_bytes: int) -> str:
    raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise _BillingResponseTooLarge(
            f"Billing response exceeded {max_bytes} bytes"
        )
    return raw.decode("utf-8")


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
