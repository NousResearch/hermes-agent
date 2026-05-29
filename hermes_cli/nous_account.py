"""Normalized Nous Portal account entitlement helpers."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional


NousAccountInfoSource = Literal["jwt", "account_api", "inference_key", "none", "error"]

_ACCOUNT_INFO_CACHE_TTL = 60
_account_info_cache: tuple[str, float, "NousPortalAccountInfo"] | None = None


@dataclass(frozen=True)
class NousPortalSubscriptionInfo:
    plan: Optional[str] = None
    tier: Optional[int] = None
    monthly_charge: Optional[float] = None
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


@dataclass(frozen=True)
class NousPortalAccountInfo:
    logged_in: bool
    source: NousAccountInfoSource
    fresh: bool
    user_id: Optional[str] = None
    org_id: Optional[str] = None
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
        return self.paid_service_access is True


def nous_portal_billing_url(account_info: Optional[NousPortalAccountInfo] = None) -> str:
    """Return the billing URL for a normalized Nous account snapshot."""
    try:
        from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL
    except Exception:
        DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com"

    base = None
    if account_info is not None:
        base = account_info.portal_base_url
    if not isinstance(base, str) or not base.strip():
        base = DEFAULT_NOUS_PORTAL_URL
    return f"{base.rstrip('/')}/billing"


def format_nous_portal_entitlement_message(
    account_info: Optional[NousPortalAccountInfo],
    *,
    capability: str = "this feature",
    include_refresh_hint: bool = True,
) -> Optional[str]:
    """Return user-facing guidance for a missing Nous paid entitlement.

    ``None`` means the account is known to have paid service access.  The
    message intentionally works from normalized entitlement fields rather than
    subscription price alone: purchased credits without a subscription still
    count as paid access, while a paid subscription with exhausted usable
    credits does not.
    """
    billing_url = nous_portal_billing_url(account_info)

    if account_info is not None and account_info.paid_service_access is True:
        return None

    if account_info is None:
        return (
            f"Hermes could not verify your Nous Portal entitlement, so {capability} "
            f"is unavailable. Run `hermes model` to refresh your login, or check "
            f"billing at {billing_url}."
        )

    if not account_info.logged_in:
        if account_info.inference_credential_present:
            return (
                f"Nous inference credentials are configured, but Hermes cannot verify "
                f"your Nous Portal paid access for {capability}. Log in with "
                f"`hermes model` to enable Portal-managed features. Billing and "
                f"credits are managed at {billing_url}."
            )
        return (
            f"Log in to Nous Portal to use {capability}: run `hermes model`. "
            f"Billing and credits are managed at {billing_url}."
        )

    if account_info.paid_service_access is None:
        detail = (
            f"Hermes could not verify your Nous Portal paid access, so {capability} "
            f"is unavailable."
        )
        if account_info.error:
            detail += f" Account lookup failed: {account_info.error}."
        if include_refresh_hint:
            detail += " Run `hermes model` to refresh your session."
        detail += f" Check billing at {billing_url}."
        return detail

    access = account_info.paid_service_access_info
    reason = access.reason if access else None
    if reason == "account_missing":
        return (
            f"Hermes could not find a Nous Portal account or organisation for this "
            f"login, so {capability} is unavailable. Run `hermes model` to "
            f"authenticate again; if the problem persists, contact Nous support."
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


def _no_paid_access_message(
    account_info: NousPortalAccountInfo,
    capability: str,
    billing_url: str,
) -> str:
    access = account_info.paid_service_access_info
    has_active_subscription = access.has_active_subscription if access else None
    active_subscription_is_paid = access.active_subscription_is_paid if access else None
    total_usable = access.total_usable_credits if access else None
    subscription_credits = access.subscription_credits_remaining if access else None
    purchased_credits = access.purchased_credits_remaining if access else None

    if has_active_subscription and active_subscription_is_paid:
        credit_detail = _credit_detail(total_usable, subscription_credits, purchased_credits)
        return (
            f"Your Nous Portal credits are exhausted{credit_detail}, so {capability} "
            f"is unavailable. Top up or renew credits at {billing_url}."
        )

    if has_active_subscription and active_subscription_is_paid is False:
        return (
            f"Your current Nous Portal plan does not include paid service access, "
            f"so {capability} is unavailable. Upgrade or add credits at {billing_url}."
        )

    if has_active_subscription is False:
        credit_detail = _credit_detail(total_usable, subscription_credits, purchased_credits)
        return (
            f"Your Nous Portal account has no active subscription or usable credits"
            f"{credit_detail}, so {capability} is unavailable. Subscribe or add credits "
            f"at {billing_url}."
        )

    credit_detail = _credit_detail(total_usable, subscription_credits, purchased_credits)
    return (
        f"Your Nous Portal account has no usable paid credits{credit_detail}, so "
        f"{capability} is unavailable. Add credits or update billing at {billing_url}."
    )


def _credit_detail(
    total_usable: Optional[float],
    subscription_credits: Optional[float],
    purchased_credits: Optional[float],
) -> str:
    parts: list[str] = []
    if total_usable is not None:
        parts.append(f"usable ${total_usable:.2f}")
    if subscription_credits is not None:
        parts.append(f"subscription ${subscription_credits:.2f}")
    if purchased_credits is not None:
        parts.append(f"purchased ${purchased_credits:.2f}")
    if not parts:
        return ""
    return f" ({', '.join(parts)})"


def reset_nous_portal_account_info_cache() -> None:
    """Clear the short-lived account-info cache used by tests."""
    global _account_info_cache
    _account_info_cache = None


def get_nous_portal_account_info(
    *,
    force_fresh: bool = False,
    min_jwt_ttl_seconds: int = 60,
) -> NousPortalAccountInfo:
    """Return normalized Nous Portal account entitlement information.

    By default, a valid unexpired OAuth access JWT is used as a low-latency
    local account snapshot. ``force_fresh=True`` always calls
    ``/api/oauth/account`` and bypasses the short-lived cache. JWT claims are
    decoded locally for UX gating only; server APIs remain authoritative.
    """
    try:
        from hermes_cli.auth import get_provider_auth_state

        state = get_provider_auth_state("nous") or {}
    except Exception as exc:
        return _error_info(error=exc, logged_in=False)

    access_token = state.get("access_token")
    portal_base_url = _portal_base_url(state)
    if not isinstance(access_token, str) or not access_token.strip():
        pool_oauth_info = _info_from_oauth_pool(
            force_fresh=force_fresh,
            min_jwt_ttl_seconds=min_jwt_ttl_seconds,
            portal_base_url=portal_base_url,
        )
        if pool_oauth_info is not None:
            return pool_oauth_info
        pool_info = _info_from_inference_key_pool(portal_base_url)
        if pool_info is not None:
            return pool_info
        return NousPortalAccountInfo(
            logged_in=False,
            source="none",
            fresh=False,
            portal_base_url=portal_base_url,
        )

    if not force_fresh:
        jwt_info = _info_from_valid_jwt(
            access_token,
            state=state,
            portal_base_url=portal_base_url,
            min_jwt_ttl_seconds=min_jwt_ttl_seconds,
        )
        if jwt_info is not None:
            return jwt_info

    return _fresh_account_info(
        state=state,
        force_fresh=force_fresh,
        portal_base_url=portal_base_url,
    )


def _fresh_account_info(
    *,
    state: dict[str, Any],
    force_fresh: bool,
    portal_base_url: Optional[str],
) -> NousPortalAccountInfo:
    global _account_info_cache

    try:
        from hermes_cli.auth import get_provider_auth_state, resolve_nous_access_token

        access_token = resolve_nous_access_token()
        refreshed_state = get_provider_auth_state("nous") or state
        portal_base_url = _portal_base_url(refreshed_state) or portal_base_url
        cache_key = _cache_key(access_token, portal_base_url)

        if not force_fresh and _account_info_cache is not None:
            cached_key, cached_at, cached_info = _account_info_cache
            if cached_key == cache_key and (time.monotonic() - cached_at) < _ACCOUNT_INFO_CACHE_TTL:
                return cached_info

        payload = _fetch_nous_account_info(access_token, portal_base_url)
        if not payload:
            return _error_info(
                error="empty_account_response",
                logged_in=True,
                portal_base_url=portal_base_url,
            )
        if isinstance(payload.get("error"), str):
            return _error_info(
                error=payload.get("error") or "account_response_error",
                logged_in=True,
                portal_base_url=portal_base_url,
                raw_account=payload,
            )

        info = _info_from_account_payload(
            payload,
            state=refreshed_state,
            portal_base_url=portal_base_url,
        )
        _account_info_cache = (cache_key, time.monotonic(), info)
        return info
    except Exception as exc:
        return _error_info(
            error=exc,
            logged_in=bool(state.get("access_token")),
            portal_base_url=portal_base_url,
        )


def _info_from_inference_key_pool(
    portal_base_url: Optional[str],
) -> Optional[NousPortalAccountInfo]:
    """Return an explicit unknown-entitlement snapshot for opaque Nous keys."""
    try:
        entry = _select_nous_pool_entry()
        if entry is None:
            return None
        runtime_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        if not isinstance(runtime_key, str) or not runtime_key.strip():
            return None

        return NousPortalAccountInfo(
            logged_in=False,
            source="inference_key",
            fresh=False,
            portal_base_url=(
                getattr(entry, "portal_base_url", None)
                or portal_base_url
            ),
            inference_base_url=(
                getattr(entry, "inference_base_url", None)
                or getattr(entry, "runtime_base_url", None)
                or getattr(entry, "base_url", None)
            ),
            inference_credential_present=True,
            credential_source=f"pool:{getattr(entry, 'label', 'unknown')}",
            error="portal_oauth_missing",
        )
    except Exception:
        return None


def _info_from_oauth_pool(
    *,
    force_fresh: bool,
    min_jwt_ttl_seconds: int,
    portal_base_url: Optional[str],
) -> Optional[NousPortalAccountInfo]:
    try:
        entry = _select_nous_pool_entry()
    except Exception:
        return None
    if entry is None or not _pool_entry_is_portal_oauth(entry):
        return None

    access_token = getattr(entry, "access_token", None)
    if not isinstance(access_token, str) or not access_token.strip():
        return None

    entry_portal_url = (
        getattr(entry, "portal_base_url", None)
        or portal_base_url
    )
    state = {
        "access_token": access_token,
        "client_id": getattr(entry, "client_id", None),
        "inference_base_url": (
            getattr(entry, "inference_base_url", None)
            or getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", None)
        ),
        "agent_key": getattr(entry, "agent_key", None),
        "credential_source": f"pool:{getattr(entry, 'label', 'unknown')}",
    }

    if not force_fresh:
        jwt_info = _info_from_valid_jwt(
            access_token,
            state=state,
            portal_base_url=entry_portal_url,
            min_jwt_ttl_seconds=min_jwt_ttl_seconds,
        )
        if jwt_info is not None:
            return jwt_info

    try:
        payload = _fetch_nous_account_info(access_token, entry_portal_url)
    except Exception as exc:
        return _error_info(
            error=exc,
            logged_in=True,
            portal_base_url=entry_portal_url,
        )
    if not payload:
        return _error_info(
            error="empty_account_response",
            logged_in=True,
            portal_base_url=entry_portal_url,
        )
    if isinstance(payload.get("error"), str):
        return _error_info(
            error=payload.get("error") or "account_response_error",
            logged_in=True,
            portal_base_url=entry_portal_url,
            raw_account=payload,
        )
    return _info_from_account_payload(
        payload,
        state=state,
        portal_base_url=entry_portal_url,
    )


def _select_nous_pool_entry() -> Optional[Any]:
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
        priority = int(getattr(entry, "priority", 0) or 0)
        return (agent_exp, access_exp, -priority)

    return max(entries, key=_entry_sort_key)


def _pool_entry_is_portal_oauth(entry: Any) -> bool:
    access_token = getattr(entry, "access_token", None)
    if not isinstance(access_token, str) or not access_token.strip():
        return False
    auth_type = str(getattr(entry, "auth_type", "") or "").strip().lower()
    refresh_token = getattr(entry, "refresh_token", None)
    return auth_type.startswith("oauth") or bool(refresh_token)


def _fetch_nous_account_info(
    access_token: str,
    portal_base_url: Optional[str] = None,
) -> dict[str, Any]:
    base = (portal_base_url or "https://portal.nousresearch.com").rstrip("/")
    url = f"{base}/api/oauth/account"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=8) as resp:
        payload = json.loads(resp.read().decode())
    return payload if isinstance(payload, dict) else {}


def _info_from_valid_jwt(
    token: str,
    *,
    state: dict[str, Any],
    portal_base_url: Optional[str],
    min_jwt_ttl_seconds: int,
) -> Optional[NousPortalAccountInfo]:
    try:
        from hermes_cli.auth import _decode_jwt_claims
    except Exception:
        return None

    claims = _decode_jwt_claims(token)
    if not claims:
        return None

    exp = _coerce_float(claims.get("exp"))
    if exp is None or exp <= time.time() + max(0, int(min_jwt_ttl_seconds)):
        return None

    paid_access = _coerce_bool(claims.get("paid_access"))
    subscription_tier = _coerce_int(claims.get("subscription_tier"))
    access_info = NousPaidServiceAccessInfo(
        allowed=paid_access,
        paid_access=paid_access,
        organisation_id=_coerce_str(claims.get("org_id")),
        subscription_tier=subscription_tier,
    )

    return NousPortalAccountInfo(
        logged_in=True,
        source="jwt",
        fresh=False,
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
        paid_service_access=paid_access,
        paid_service_access_info=access_info,
        raw_claims=dict(claims),
    )


def _info_from_account_payload(
    payload: dict[str, Any],
    *,
    state: dict[str, Any],
    portal_base_url: Optional[str],
) -> NousPortalAccountInfo:
    user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
    organisation = (
        payload.get("organisation")
        if isinstance(payload.get("organisation"), dict)
        else {}
    )
    subscription = _subscription_from_payload(payload.get("subscription"))
    access = _paid_service_access_from_payload(payload.get("paid_service_access"))
    paid_access = access.allowed if access else None
    if paid_access is None and access is not None:
        paid_access = access.paid_access

    return NousPortalAccountInfo(
        logged_in=True,
        source="account_api",
        fresh=True,
        org_id=_coerce_str(organisation.get("id")) or (access.organisation_id if access else None),
        client_id=_coerce_str(state.get("client_id")),
        portal_base_url=portal_base_url,
        inference_base_url=_coerce_str(state.get("inference_base_url")),
        inference_credential_present=bool(state.get("access_token") or state.get("agent_key")),
        credential_source=_coerce_str(state.get("credential_source")) or "auth_store",
        email=_coerce_str(user.get("email")),
        privy_did=_coerce_str(user.get("privy_did")),
        subscription=subscription,
        paid_service_access=paid_access,
        paid_service_access_info=access,
        raw_account=dict(payload),
    )


def _subscription_from_payload(value: Any) -> Optional[NousPortalSubscriptionInfo]:
    if not isinstance(value, dict):
        return None
    return NousPortalSubscriptionInfo(
        plan=_coerce_str(value.get("plan")),
        tier=_coerce_int(value.get("tier")),
        monthly_charge=_coerce_float(value.get("monthly_charge")),
        current_period_end=_coerce_str(value.get("current_period_end")),
        credits_remaining=_coerce_float(value.get("credits_remaining")),
        rollover_credits=_coerce_float(value.get("rollover_credits")),
    )


def _paid_service_access_from_payload(value: Any) -> Optional[NousPaidServiceAccessInfo]:
    if not isinstance(value, dict):
        return None
    allowed = _coerce_bool(value.get("allowed"))
    paid_access = _coerce_bool(value.get("paid_access"))
    return NousPaidServiceAccessInfo(
        allowed=allowed,
        paid_access=paid_access,
        reason=_coerce_str(value.get("reason")),
        organisation_id=_coerce_str(value.get("organisation_id")),
        effective_at_ms=_coerce_int(value.get("effective_at_ms")),
        has_active_subscription=_coerce_bool(value.get("has_active_subscription")),
        active_subscription_is_paid=_coerce_bool(value.get("active_subscription_is_paid")),
        subscription_tier=_coerce_int(value.get("subscription_tier")),
        subscription_monthly_charge=_coerce_float(value.get("subscription_monthly_charge")),
        subscription_credits_remaining=_coerce_float(value.get("subscription_credits_remaining")),
        purchased_credits_remaining=_coerce_float(value.get("purchased_credits_remaining")),
        total_usable_credits=_coerce_float(value.get("total_usable_credits")),
    )


def _error_info(
    *,
    error: object,
    logged_in: bool,
    portal_base_url: Optional[str] = None,
    raw_account: Optional[dict[str, Any]] = None,
) -> NousPortalAccountInfo:
    return NousPortalAccountInfo(
        logged_in=logged_in,
        source="error",
        fresh=False,
        portal_base_url=portal_base_url,
        raw_account=raw_account,
        error=str(error),
    )


def _portal_base_url(state: dict[str, Any]) -> Optional[str]:
    value = state.get("portal_base_url")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().rstrip("/")


def _cache_key(access_token: str, portal_base_url: Optional[str]) -> str:
    digest = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
    return f"{portal_base_url or ''}:{digest}"


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
    if isinstance(value, str) and value:
        return value
    return None


def _coerce_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
