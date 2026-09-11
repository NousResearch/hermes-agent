"""Normalized Nous Portal account entitlement helpers."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Literal, Optional


NousAccountInfoSource = Literal["jwt", "account_api", "inference_key", "none", "error"]

# Free tool-pool coverage categories, byte-aligned with the Portal's TOOL_COVERAGE_CATEGORIES
# (minted into `tool_access.coverage` on the JWT and /api/oauth/account). `fal-video` is
# intentionally excluded from the pool.
TOOL_COVERAGE_CATEGORIES = ("firecrawl", "fal", "fal-video", "openai-audio", "browser-use", "modal")

_ACCOUNT_INFO_CACHE_TTL = 60
_account_info_cache: tuple[str, float, "NousPortalAccountInfo"] | None = None
_ACCOUNT_INFO_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class NousPortalSubscriptionInfo:
    plan: Optional[str] = None
    tier: Optional[int] = None
    monthly_charge: Optional[float] = None
    monthly_credits: Optional[float] = None
    current_period_end: Optional[str] = None
    credits_remaining: Optional[float] = None
    rollover_credits: Optional[float] = None


@dataclass(frozen=True)
class NousPaidServiceAccessInfo:
    allowed: Optional[bool] = None
    paid_access: Optional[bool] = None
    reason: Optional[str] = None
    organisation_id: Optional[str] = None
    effective_at_ms: Optional[int] = None
    has_active_subscription: Optional[bool] = None
    active_subscription_is_paid: Optional[bool] = None
    subscription_tier: Optional[int] = None
    subscription_monthly_charge: Optional[float] = None
    subscription_credits_remaining: Optional[float] = None
    purchased_credits_remaining: Optional[float] = None
    total_usable_credits: Optional[float] = None
    member_spend_cap_exceeded: Optional[bool] = None
    member_spend_cap_usd: Optional[float] = None
    member_spend_usd: Optional[float] = None
    member_spend_cap_remaining_usd: Optional[float] = None


@dataclass(frozen=True)
class NousToolAccessInfo:
    """Free tool-pool entitlement (Portal ``tool_access``), decoupled from paid/billing access.

    ``enabled``: a positive pool balance is live and not gated off; ``coverage``: tool category ->
    whether the pool funds it (FAL video is excluded).
    """

    enabled: bool = False
    coverage: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class NousPortalAccountInfo:
    logged_in: bool
    source: NousAccountInfoSource
    fresh: bool
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    org_slug: Optional[str] = None
    org_name: Optional[str] = None
    client_id: Optional[str] = None
    product_id: Optional[str] = None
    nous_client: Optional[str] = None
    portal_base_url: Optional[str] = None
    inference_base_url: Optional[str] = None
    inference_credential_present: bool = False
    credential_source: Optional[str] = None
    expires_at: Optional[datetime] = None
    email: Optional[str] = None
    privy_did: Optional[str] = None
    subscription: Optional[NousPortalSubscriptionInfo] = None
    paid_service_access: Optional[bool] = None
    paid_service_access_info: Optional[NousPaidServiceAccessInfo] = None
    tool_access: Optional[NousToolAccessInfo] = None
    raw_claims: Optional[dict[str, Any]] = None
    raw_account: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def is_paid(self) -> bool:
        return self.paid_service_access is True

    @property
    def is_free_tier(self) -> bool:
        return self.paid_service_access is False

    @property
    def tool_gateway_entitled(self) -> bool:
        """Paid access OR a live free tool pool; use ``tool_gateway_entitled_for`` per category."""
        return self.paid_service_access is True or bool(self.tool_access and self.tool_access.enabled)

    def tool_gateway_entitled_for(self, category: str) -> bool:
        """Paid users are entitled everywhere; pool users only where ``coverage[category]`` is true."""
        ta = self.tool_access
        return self.paid_service_access is True or bool(ta and ta.enabled and ta.coverage.get(category) is True)


def nous_portal_billing_url(account_info: Optional[NousPortalAccountInfo] = None) -> str:
    """Return the billing URL for a normalized Nous account snapshot."""
    try:
        from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL
    except Exception:
        DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com"

    base = account_info.portal_base_url if account_info is not None else None
    if not _nonblank(base):
        base = DEFAULT_NOUS_PORTAL_URL
    return f"{base.rstrip('/')}/billing"


def nous_portal_topup_url(account_info: Optional[NousPortalAccountInfo] = None) -> str:
    """Portal top-up URL (``?topup=open`` auto-opens the top-up modal).

    Prefers the org-pinned ``{base}/orgs/{slug}/billing`` (skips the legacy shim's multi-org
    re-resolution); falls back to ``{base}/billing`` when ``org_slug`` is null — never
    ``/orgs/None/billing``.
    """
    base = nous_portal_billing_url(account_info)[: -len("/billing")]
    slug = getattr(account_info, "org_slug", None) if account_info is not None else None
    if _nonblank(slug):
        from urllib.parse import quote

        return f"{base}/orgs/{quote(slug.strip(), safe='')}/billing?topup=open"
    return f"{base}/billing?topup=open"


def format_nous_portal_entitlement_message(
    account_info: Optional[NousPortalAccountInfo], *, capability: str = "this feature",
    include_refresh_hint: bool = True, coverage_category: Optional[str] = None,
) -> Optional[str]:
    """User-facing guidance for a missing Nous tool-gateway entitlement; ``None`` when entitled.

    Entitled = paid access OR a live free pool that covers it (normalized fields, not price:
    purchased credits without a subscription count as paid, an exhausted paid subscription does
    not). ``coverage_category`` scopes the check to one category; an otherwise-entitled user whose
    access doesn't fund it gets a neutral billing nudge, never an "exhausted" message. The
    pool-vs-paid distinction is never surfaced.
    """
    billing_url = nous_portal_billing_url(account_info)

    if account_info is not None:
        if coverage_category is not None:
            if account_info.tool_gateway_entitled_for(coverage_category):
                return None
            if account_info.tool_gateway_entitled:
                return (
                    f"{capability} isn't included with your current Nous Portal access. "
                    f"Add credits or a subscription to enable it at {billing_url}."
                )
        elif account_info.tool_gateway_entitled:
            return None
    if account_info is None:
        return (
            f"Hermes could not verify your Nous Portal entitlement, so {capability} is unavailable. "
            f"Run `hermes model` to refresh your login, or check billing at {billing_url}."
        )
    if not account_info.logged_in:
        if account_info.inference_credential_present:
            return (
                f"Nous inference credentials are configured, but Hermes cannot verify your Nous Portal "
                f"paid access for {capability}. Log in with `hermes model` to enable Portal-managed "
                f"features. Billing and credits are managed at {billing_url}."
            )
        return (
            f"Log in to Nous Portal to use {capability}: run `hermes model`. "
            f"Billing and credits are managed at {billing_url}."
        )
    if account_info.paid_service_access is None:
        detail = f"Hermes could not verify your Nous Portal paid access, so {capability} is unavailable."
        if account_info.error:
            detail += f" Account lookup failed: {account_info.error}."
        if include_refresh_hint:
            detail += " Run `hermes model` to refresh your session."
        return detail + f" Check billing at {billing_url}."
    access = account_info.paid_service_access_info
    reason = access.reason if access else None
    if reason == "account_missing":
        return (
            f"Hermes could not find a Nous Portal account or organisation for this login, so {capability} "
            f"is unavailable. Run `hermes model` to authenticate again; if the problem persists, contact Nous support."
        )
    if reason == "no_usable_credits" or account_info.paid_service_access is False:
        message = _no_paid_access_message(account_info, capability, billing_url)
        if include_refresh_hint and not account_info.fresh:
            message += " If you recently bought credits, run `hermes model` to refresh Hermes."
        return message
    return (
        f"Your Nous Portal account does not currently have paid service access, "
        f"so {capability} is unavailable. Add credits or update billing at {billing_url}."
    )


def _no_paid_access_message(account_info: NousPortalAccountInfo, capability: str, billing_url: str) -> str:
    access = account_info.paid_service_access_info or NousPaidServiceAccessInfo()
    active, paid = access.has_active_subscription, access.active_subscription_is_paid
    labelled = (
        ("usable", access.total_usable_credits),
        ("subscription", access.subscription_credits_remaining),
        ("purchased", access.purchased_credits_remaining),
    )
    parts = [f"{label} ${amount:.2f}" for label, amount in labelled if amount is not None]
    credit_detail = f" ({', '.join(parts)})" if parts else ""
    if access.member_spend_cap_exceeded:
        cap, spent = access.member_spend_cap_usd, access.member_spend_usd
        cap_detail = ""
        if cap is not None and spent is not None:
            cap_detail = f" Your organisation's per-member spend cap is ${cap:.2f} and you've spent ${spent:.2f} of it."
        elif cap is not None:
            cap_detail = f" Your organisation's per-member spend cap is ${cap:.2f}."
        return (
            f"Your Nous Portal access is paused because you've exceeded the per-member spend cap set by "
            f"your organisation.{cap_detail}{credit_detail} Ask your organisation admin to raise the "
            f"member spend cap at {billing_url}, then run `hermes model` to refresh."
        )
    if active and paid:
        return (
            f"Your Nous Portal credits are exhausted{credit_detail}, so {capability} is unavailable. "
            f"Top up or renew credits at {billing_url}."
        )
    if active and paid is False:
        return (
            f"Your current Nous Portal plan does not include paid service access, "
            f"so {capability} is unavailable. Upgrade or add credits at {billing_url}."
        )
    if active is False:
        return (
            f"Your Nous Portal account has no active subscription or usable credits{credit_detail}, "
            f"so {capability} is unavailable. Subscribe or add credits at {billing_url}."
        )
    return (
        f"Your Nous Portal account has no usable paid credits{credit_detail}, so "
        f"{capability} is unavailable. Add credits or update billing at {billing_url}."
    )


def reset_nous_portal_account_info_cache() -> None:
    """Clear the short-lived account-info cache used by tests."""
    global _account_info_cache
    _account_info_cache = None


def get_nous_portal_account_info(*, force_fresh: bool = False, min_jwt_ttl_seconds: int = 60) -> NousPortalAccountInfo:
    """Normalized Nous Portal account entitlement.

    A valid unexpired OAuth JWT serves as a local snapshot (UX gating only; the server stays
    authoritative). ``force_fresh=True`` always calls ``/api/oauth/account`` and bypasses the cache.
    """
    try:
        from hermes_cli.auth import get_provider_auth_state

        state = get_provider_auth_state("nous") or {}
    except Exception as exc:
        return _error_info(error=exc, logged_in=False)

    access_token = state.get("access_token")
    portal_base_url = _portal_base_url(state)
    if not _nonblank(access_token):
        return (
            _info_from_oauth_pool(force_fresh, min_jwt_ttl_seconds, portal_base_url)
            or _info_from_inference_key_pool(portal_base_url)
            or NousPortalAccountInfo(logged_in=False, source="none", fresh=False, portal_base_url=portal_base_url)
        )
    if not force_fresh:
        jwt_info = _info_from_valid_jwt(access_token, state, portal_base_url, min_jwt_ttl_seconds)
        if jwt_info is not None:
            return jwt_info
    return _fresh_account_info(state, force_fresh, portal_base_url)


def nous_policy_present() -> Optional[bool]:
    """Whether the caller's org carries a restrictive model/provider policy.

    ``None`` is unknown (older mint / unreadable claim) and must not be reported as "no policy".
    """
    try:
        from hermes_cli.auth import get_provider_auth_state, _decode_jwt_claims

        access_token = (get_provider_auth_state("nous") or {}).get("access_token")
        if not _nonblank(access_token):
            return None
        claims = _decode_jwt_claims(access_token)
        return _coerce_bool(claims.get("policy_present")) if claims else None
    except Exception:
        return None


def nous_policy_notice(*, removed: bool) -> str:
    """One-line notice for a list the org's policy narrowed, else ``""``.

    Blocked models are omitted, which reads as "unsupported"; this says which it is without
    enumerating them. ``removed`` must reflect whether the filter actually dropped anything — the
    catalog read fails open, so the claim alone would label a full list as filtered.
    """
    if not removed or nous_policy_present() is not True:
        return ""
    return (
        "Your organization restricts which models are available — "
        "models outside its policy are not listed."
    )


def _fresh_account_info(state: dict[str, Any], force_fresh: bool, portal_base_url: Optional[str]) -> NousPortalAccountInfo:
    global _account_info_cache

    try:
        from hermes_cli.auth import get_provider_auth_state, resolve_nous_access_token

        access_token = resolve_nous_access_token()
        refreshed_state = get_provider_auth_state("nous") or state
        portal_base_url = _portal_base_url(refreshed_state) or portal_base_url
        digest = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
        cache_key = f"{portal_base_url or ''}:{digest}"

        with _ACCOUNT_INFO_CACHE_LOCK:
            if not force_fresh and _account_info_cache is not None:
                cached_key, cached_at, cached_info = _account_info_cache
                if cached_key == cache_key and (time.monotonic() - cached_at) < _ACCOUNT_INFO_CACHE_TTL:
                    return cached_info

        info = _info_from_fetched_account(access_token, refreshed_state, portal_base_url)
        if info.source != "error":
            with _ACCOUNT_INFO_CACHE_LOCK:
                _account_info_cache = (cache_key, time.monotonic(), info)
        return info
    except Exception as exc:
        return _error_info(error=exc, logged_in=bool(state.get("access_token")), portal_base_url=portal_base_url)


def _info_from_inference_key_pool(portal_base_url: Optional[str]) -> Optional[NousPortalAccountInfo]:
    """Return an explicit unknown-entitlement snapshot for opaque Nous keys."""
    try:
        entry = _select_nous_pool_entry()
        if entry is None:
            return None
        if not _nonblank(getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")):
            return None
        return NousPortalAccountInfo(
            logged_in=False, source="inference_key", fresh=False,
            portal_base_url=getattr(entry, "portal_base_url", None) or portal_base_url,
            inference_base_url=_pool_entry_inference_url(entry),
            inference_credential_present=True,
            credential_source=f"pool:{getattr(entry, 'label', 'unknown')}",
            error="portal_oauth_missing",
        )
    except Exception:
        return None


def _info_from_oauth_pool(
    force_fresh: bool, min_jwt_ttl_seconds: int, portal_base_url: Optional[str]
) -> Optional[NousPortalAccountInfo]:
    try:
        entry = _select_nous_pool_entry()
    except Exception:
        return None
    if entry is None or not _pool_entry_is_portal_oauth(entry):
        return None
    access_token = entry.access_token  # non-blank str: checked by _pool_entry_is_portal_oauth
    entry_portal_url = getattr(entry, "portal_base_url", None) or portal_base_url
    state = {
        "access_token": access_token, "client_id": getattr(entry, "client_id", None),
        "inference_base_url": _pool_entry_inference_url(entry), "agent_key": getattr(entry, "agent_key", None),
        "credential_source": f"pool:{getattr(entry, 'label', 'unknown')}",
    }

    if not force_fresh:
        jwt_info = _info_from_valid_jwt(access_token, state, entry_portal_url, min_jwt_ttl_seconds)
        if jwt_info is not None:
            return jwt_info
    try:
        return _info_from_fetched_account(access_token, state, entry_portal_url)
    except Exception as exc:
        return _error_info(error=exc, logged_in=True, portal_base_url=entry_portal_url)


def _info_from_fetched_account(
    access_token: str, state: dict[str, Any], portal_base_url: Optional[str]
) -> NousPortalAccountInfo:
    """Call ``/api/oauth/account`` and normalize; empty or ``error`` payloads become error infos."""
    payload = _fetch_nous_account_info(access_token, portal_base_url)
    if not payload:
        return _error_info(error="empty_account_response", logged_in=True, portal_base_url=portal_base_url)
    if isinstance(payload.get("error"), str):
        return _error_info(
            error=payload["error"] or "account_response_error", logged_in=True,
            portal_base_url=portal_base_url, raw_account=payload,
        )
    return _info_from_account_payload(payload, state=state, portal_base_url=portal_base_url)


def _pool_entry_inference_url(entry: Any) -> Optional[str]:
    return getattr(entry, "inference_base_url", None) or getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None)


def _select_nous_pool_entry() -> Optional[Any]:
    """Pool entry with the latest agent-key expiry, then access expiry, then lowest priority."""
    from agent.credential_pool import load_pool

    pool = load_pool("nous")
    if not pool or not pool.has_credentials():
        return None
    entries = list(pool.entries())
    if not entries:
        return None

    def _entry_sort_key(entry: Any) -> tuple[float, float, int]:
        agent_exp = _parse_iso_timestamp(getattr(entry, "agent_key_expires_at", None)) or 0.0
        access_exp = _parse_iso_timestamp(getattr(entry, "expires_at", None)) or 0.0
        return (agent_exp, access_exp, -int(getattr(entry, "priority", 0) or 0))

    return max(entries, key=_entry_sort_key)


def _pool_entry_is_portal_oauth(entry: Any) -> bool:
    if not _nonblank(getattr(entry, "access_token", None)):
        return False
    auth_type = str(getattr(entry, "auth_type", "") or "").strip().lower()
    return auth_type.startswith("oauth") or bool(getattr(entry, "refresh_token", None))


def _fetch_nous_account_info(access_token: str, portal_base_url: Optional[str] = None) -> dict[str, Any]:
    base = (portal_base_url or "https://portal.nousresearch.com").rstrip("/")
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    req = urllib.request.Request(f"{base}/api/oauth/account", headers=headers)
    with urllib.request.urlopen(req, timeout=8) as resp:
        payload = json.loads(resp.read().decode())
    return payload if isinstance(payload, dict) else {}


def _info_from_valid_jwt(
    token: str, state: dict[str, Any], portal_base_url: Optional[str], min_jwt_ttl_seconds: int
) -> Optional[NousPortalAccountInfo]:
    try:
        from hermes_cli.auth import _decode_jwt_claims
    except Exception:
        return None
    claims = _decode_jwt_claims(token)
    if not claims:
        return None
    exp = _coerce_num(claims.get("exp"), float)
    if exp is None or exp <= time.time() + max(0, int(min_jwt_ttl_seconds)):
        return None
    paid_access = _coerce_bool(claims.get("paid_access"))
    access_info = NousPaidServiceAccessInfo(
        allowed=paid_access, paid_access=paid_access, organisation_id=_coerce_str(claims.get("org_id")),
        subscription_tier=_coerce_num(claims.get("subscription_tier"), int),
    )
    return NousPortalAccountInfo(
        logged_in=True, source="jwt", fresh=False,
        user_id=_coerce_str(claims.get("sub")),
        org_id=_coerce_str(claims.get("org_id")),
        client_id=_coerce_str(claims.get("client_id") or state.get("client_id")),
        product_id=_coerce_str(claims.get("product_id")),
        nous_client=_coerce_str(claims.get("nous_client")),
        portal_base_url=portal_base_url,
        inference_base_url=_coerce_str(state.get("inference_base_url")),
        inference_credential_present=True,
        credential_source=_coerce_str(state.get("credential_source")) or "auth_store",
        expires_at=datetime.fromtimestamp(exp, tz=timezone.utc),
        paid_service_access=paid_access, paid_service_access_info=access_info,
        tool_access=_tool_access_from_value(claims.get("tool_access")),
        raw_claims=dict(claims),
    )


def _info_from_account_payload(
    payload: dict[str, Any], *, state: dict[str, Any], portal_base_url: Optional[str]
) -> NousPortalAccountInfo:
    user = _dict_or_empty(payload.get("user"))
    organisation = _dict_or_empty(payload.get("organisation"))
    access = _coerced_dataclass(NousPaidServiceAccessInfo, payload.get("paid_service_access"))
    paid_access = None
    if access is not None:
        paid_access = access.allowed if access.allowed is not None else access.paid_access
    return NousPortalAccountInfo(
        logged_in=True, source="account_api", fresh=True,
        org_id=_coerce_str(organisation.get("id")) or (access.organisation_id if access else None),
        org_slug=_coerce_str(organisation.get("slug")),
        org_name=_coerce_str(organisation.get("name")),
        client_id=_coerce_str(state.get("client_id")),
        portal_base_url=portal_base_url,
        inference_base_url=_coerce_str(state.get("inference_base_url")),
        inference_credential_present=bool(state.get("access_token") or state.get("agent_key")),
        credential_source=_coerce_str(state.get("credential_source")) or "auth_store",
        email=_coerce_str(user.get("email")),
        privy_did=_coerce_str(user.get("privy_did")),
        subscription=_subscription_from_payload(payload.get("subscription")),
        paid_service_access=paid_access, paid_service_access_info=access,
        tool_access=_tool_access_from_value(payload.get("tool_access")),
        raw_account=dict(payload),
    )


def _tool_access_from_value(value: Any) -> Optional[NousToolAccessInfo]:
    """Parse a Portal ``tool_access`` object (JWT claim or account API).

    Fails closed: a non-object yields ``None``; only literal ``true`` counts for ``enabled`` and
    each coverage entry.
    """
    if not isinstance(value, dict):
        return None
    coverage = {k: v is True for k, v in _dict_or_empty(value.get("coverage")).items() if isinstance(k, str)}
    return NousToolAccessInfo(enabled=value.get("enabled") is True, coverage=coverage)


def _coerced_dataclass(cls, value: Any):
    """Build ``cls`` from a payload dict (field names = payload keys), coercing by declared type."""
    return cls(**{f.name: _COERCERS[f.type](value.get(f.name)) for f in fields(cls)}) if isinstance(value, dict) else None


def _subscription_from_payload(value: Any) -> Optional[NousPortalSubscriptionInfo]:
    return _coerced_dataclass(NousPortalSubscriptionInfo, value)


def _error_info(
    *, error: object, logged_in: bool, portal_base_url: Optional[str] = None, raw_account: Optional[dict[str, Any]] = None
) -> NousPortalAccountInfo:
    return NousPortalAccountInfo(
        logged_in=logged_in, source="error", fresh=False, portal_base_url=portal_base_url,
        raw_account=raw_account, error=str(error),
    )


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _portal_base_url(state: dict[str, Any]) -> Optional[str]:
    value = state.get("portal_base_url")
    return value.strip().rstrip("/") if _nonblank(value) else None


def _parse_iso_timestamp(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _coerce_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _coerce_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _coerce_num(value: Any, cast):
    """``cast(value)`` or None; bools and None are rejected, not coerced."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


# Annotations are strings (``from __future__ import annotations``).
_COERCERS = {
    "Optional[str]": _coerce_str,
    "Optional[bool]": _coerce_bool,
    "Optional[int]": lambda v: _coerce_num(v, int),
    "Optional[float]": lambda v: _coerce_num(v, float),
}
