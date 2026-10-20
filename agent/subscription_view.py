"""Surface-agnostic core for the ``/subscription`` TUI screen.

Companion to :mod:`agent.billing_view` — same fail-open philosophy (``logged_in=False``
when not logged in / portal unreachable; never crash) and decimal money end-to-end.
The TUI drives plan changes in-terminal (preview, then schedule a downgrade/cancel/resume
or apply an upgrade); the portal deep-link (``portal_url`` + ``org_id``) is the fallback
for an upgrade that needs 3DS / was declined. A 404 from NAS takes the fail-open path.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from agent.billing_view import OrgRoleCapability, fetch_portal_state, format_money, parse_money, parse_org_fields

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CurrentSubscription:
    """Active subscription (``None``, not this object, = no plan). NAS guarantees ``tier_id`` /
    ``tier_name`` / ``monthly_credits`` / ``cycle_ends_at`` are set; the rest are optional."""

    tier_id: Optional[str] = None
    tier_name: Optional[str] = None
    monthly_credits: Optional[Decimal] = None
    credits_remaining: Optional[Decimal] = None
    cycle_ends_at: Optional[str] = None  # ISO
    pending_downgrade_tier_name: Optional[str] = None
    pending_downgrade_at: Optional[str] = None  # ISO
    cancel_at_period_end: bool = False
    cancellation_effective_at: Optional[str] = None  # ISO


@dataclass(frozen=True)
class SubscriptionTier:
    """Tier-picker row (NAS ``SubscriptionTierOption``). ``is_current`` = active plan (shown, not
    selectable); ``is_enabled=False`` = grandfathered, no longer selectable; ``tier_order`` sorts
    the picker and drives the upgrade-vs-downgrade hint."""

    tier_id: str
    name: str
    tier_order: int = 0
    dollars_per_month: Optional[Decimal] = None
    monthly_credits: Optional[Decimal] = None
    is_current: bool = False
    is_enabled: bool = True


@dataclass(frozen=True)
class SubscriptionChangePreview:
    """Parsed ``POST /api/billing/subscription/preview``. ``effect``: ``charge_now`` (upgrade; prorated
    ``amount_due_now_cents``) · ``scheduled`` (downgrade / same-price change at ``effective_at``) ·
    ``no_op`` (already on target) · ``blocked`` (commit refused; ``reason`` says why)."""

    effect: str
    reason: Optional[str] = None
    current_tier_id: Optional[str] = None
    current_tier_name: Optional[str] = None
    target_tier_id: Optional[str] = None
    target_tier_name: Optional[str] = None
    monthly_credits_delta: Optional[Decimal] = None
    amount_due_now_cents: Optional[int] = None
    effective_at: Optional[str] = None  # ISO


@dataclass(frozen=True)
class SubscriptionState(OrgRoleCapability):
    """Parsed ``GET /api/billing/subscription``. Fail-open: ``logged_in=False``
    (empty fields) when not logged in or the portal is unreachable."""

    logged_in: bool
    org_name: Optional[str] = None
    org_id: Optional[str] = None
    role: Optional[str] = None  # "OWNER" | "ADMIN" | "FINANCE_ADMIN" | "SECURITY_ADMIN" | "MEMBER"
    can_change_plan_raw: Optional[bool] = None
    context: str = "personal"  # "personal" | "team"
    current: Optional[CurrentSubscription] = None
    tiers: tuple[SubscriptionTier, ...] = ()  # selectable catalog (picker)
    portal_url: Optional[str] = None
    error: Optional[str] = None  # set when the fetch failed (vs cleanly not-logged-in)


# ── Payload parsing ──────────────────────────────────────────────────────────


def _tier_id(raw: Any) -> Optional[str]:
    """Real tier id of a NAS dict, else None ("no plan" is ``current: null``; junk is skipped)."""
    return (raw.get("tierId") or raw.get("id") or None) if isinstance(raw, dict) else None


def _parse_current(raw: Any) -> Optional[CurrentSubscription]:
    tier_id = _tier_id(raw)
    if not tier_id:
        return None
    return CurrentSubscription(
        tier_id=tier_id,
        tier_name=raw.get("tierName") or raw.get("name"),
        monthly_credits=parse_money(raw.get("monthlyCredits")),
        credits_remaining=parse_money(raw.get("creditsRemaining")),
        cycle_ends_at=raw.get("cycleEndsAt"),
        pending_downgrade_tier_name=raw.get("pendingDowngradeTierName"),
        pending_downgrade_at=raw.get("pendingDowngradeAt"),
        cancel_at_period_end=bool(raw.get("cancelAtPeriodEnd")),
        cancellation_effective_at=raw.get("cancellationEffectiveAt") or None,
    )


def _coalesce(*vals: Any) -> Any:
    """First non-``None`` value (NAS sends ``0`` for the free tier, which ``x or default`` would drop)."""
    return next((v for v in vals if v is not None), None)


def _parse_tier(raw: Any) -> Optional[SubscriptionTier]:
    tier_id = _tier_id(raw)
    if not tier_id:
        return None
    return SubscriptionTier(
        tier_id=tier_id,
        name=raw.get("name") or "",
        tier_order=int(_coalesce(raw.get("tierOrder"), 0)),
        dollars_per_month=parse_money(raw.get("dollarsPerMonthDisplay")),
        monthly_credits=parse_money(raw.get("monthlyCredits")),
        is_current=bool(raw.get("isCurrent")),
        is_enabled=bool(_coalesce(raw.get("isEnabled"), True)),
    )


def subscription_change_preview_from_payload(payload: dict[str, Any]) -> SubscriptionChangePreview:
    """Map a raw ``/subscription/preview`` JSON dict into :class:`SubscriptionChangePreview`."""
    effect = payload.get("effect")
    cents = payload.get("amountDueNowCents")
    return SubscriptionChangePreview(
        # Unrecognized/missing effect → ``blocked``: fail safe, never charge on a malformed quote.
        effect=effect if isinstance(effect, str) else "blocked",
        reason=payload.get("reason") or None,
        current_tier_id=payload.get("currentTierId"),
        current_tier_name=payload.get("currentTierName"),
        target_tier_id=payload.get("targetTierId"),
        target_tier_name=payload.get("targetTierName"),
        monthly_credits_delta=parse_money(payload.get("monthlyCreditsDelta")),
        amount_due_now_cents=int(cents) if isinstance(cents, (int, float)) else None,
        effective_at=payload.get("effectiveAt") or None,
    )


def subscription_state_from_payload(payload: dict[str, Any], *, portal_url: Optional[str] = None) -> SubscriptionState:
    """Map a raw ``/api/billing/subscription`` JSON dict into :class:`SubscriptionState`."""
    org, can_change_plan_raw = parse_org_fields(payload)
    raw_context, raw_tiers = payload.get("context"), payload.get("tiers")
    return SubscriptionState(
        logged_in=True,
        org_name=org.get("name"),
        org_id=org.get("id") or None,
        role=org.get("role"),
        can_change_plan_raw=can_change_plan_raw,
        context=raw_context if raw_context in ("personal", "team") else "personal",
        current=_parse_current(payload.get("current")),
        tiers=tuple(filter(None, map(_parse_tier, raw_tiers))) if isinstance(raw_tiers, list) else (),
        portal_url=portal_url,
    )


# ── Fail-open builders (the surface front doors) ─────────────────────────────


def build_subscription_state(*, timeout: float = 15.0) -> SubscriptionState:
    """Fetch + parse ``GET /api/billing/subscription``; fail-open like ``fetch_portal_state``.
    ``HERMES_DEV_SUBSCRIPTION_FIXTURE`` short-circuits to a fixture so every state is testable offline."""
    fixture = dev_fixture_subscription_state()
    if fixture is not None:
        return fixture
    return fetch_portal_state(
        "get_subscription_state", "subscription",
        failed=lambda **kw: SubscriptionState(logged_in=False, **kw),
        parse=lambda payload, portal_url: subscription_state_from_payload(payload, portal_url=portal_url),
        portal_fallback=lambda base: base, timeout=timeout, log=logger,
    )


def subscription_manage_url(state: SubscriptionState, tier_id: Optional[str] = None) -> Optional[str]:
    """Build ``{portal_origin}/manage-subscription?org_id=<id>[&plan=<tier_id>]`` (None if unresolvable).

    Mirrors the TUI's ``buildManageUrl``: the target is NAS's OWN ``/manage-subscription`` page
    (NOT the Stripe Billing Portal). ``org_id`` pins the account in multi-org situations; ``tier_id``
    (the stable ``tiers[]`` id, never a name/slug) preselects the plan — the portal ignores an
    unknown tier, so it's appended unconditionally.
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    try:
        parts = urlsplit(state.portal_url or "")
    except Exception:
        return None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None

    # Preserve unrelated portal query params; org_id / plan are contract-owned
    # (org_id before plan — insertion order is the emitted query order).
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    params.pop("org_id", None)
    params.pop("plan", None)
    if state.org_id:
        params["org_id"] = state.org_id
    if tier_id:
        params["plan"] = tier_id
    return urlunsplit((parts.scheme, parts.netloc, "/manage-subscription", urlencode(params), ""))


# ── Shared plan-catalog helpers (CLI Free catalog + paid picker) ─────────────


def selectable_tiers(state: SubscriptionState) -> list[SubscriptionTier]:
    """Enabled paid tiers other than the current plan, cheapest first (dropping to free is a cancellation)."""
    return sorted(
        (t for t in (state.tiers or ()) if t.is_enabled and not t.is_current and (t.tier_order or 0) > 0),
        key=lambda t: t.tier_order or 0,
    )


def format_tier_row(tier: SubscriptionTier) -> str:
    """``name · $X/mo[ · $Y credits/mo]`` (grouped money, like the TUI); credits suffix only when > 0."""
    row = f"{tier.name} · {format_money(tier.dollars_per_month, grouped=True)}/mo"
    mc = tier.monthly_credits
    if mc is not None and mc > 0:
        row += f" · {format_money(mc, grouped=True)} credits/mo"
    return row


def is_upgrade(state: SubscriptionState, tier_id: str) -> bool:
    """True when ``tier_id`` ranks above the current plan by ``tier_order``. Prefers the
    active subscription's tier; falls back to the ``tiers[]`` ``is_current`` marker, else 0."""
    tiers = state.tiers or ()
    orders = {t.tier_id: (t.tier_order or 0) for t in tiers}
    cur_id = state.current.tier_id if state.current else None
    cur_order = orders[cur_id] if cur_id in orders else next((t.tier_order or 0 for t in tiers if t.is_current), 0)
    return orders.get(tier_id, 0) > cur_order


# ── Dev fixtures (env-var driven, no live portal) ────────────────────────────

_DEV_FIXTURE_PORTAL = "https://portal.nousresearch.com/billing"
_DEV_TIER_SPECS = (("free", "Free", 0, "0", "0"), ("plus", "Plus", 1, "20", "1000"),
                   ("super", "Super", 2, "40", "3000"), ("ultra", "Ultra", 3, "80", "7000"))
_DEV_FIXTURE_ALIASES = {"logged_out": "logged-out", "loggedout": "logged-out", "mid-tier": "mid",
                        "top-tier": "top", "member": "not-admin"}


def _dev_tiers(current_id: Optional[str]) -> tuple[SubscriptionTier, ...]:
    """Sample plan catalog for fixtures (marks ``current_id`` as the active tier)."""
    return tuple(
        SubscriptionTier(
            tier_id=tid, name=name, tier_order=order, dollars_per_month=parse_money(dpm),
            monthly_credits=parse_money(mc), is_current=(tid == current_id), is_enabled=True,
        )
        for tid, name, order, dpm, mc in _DEV_TIER_SPECS
    )


def _dev_plan(tier_id: str, remaining: str, **over: Any) -> dict[str, Any]:
    """``current`` + ``tiers`` fixture fields for being on ``tier_id`` with ``remaining`` credits left."""
    tid, name, _order, _dpm, mc = next(spec for spec in _DEV_TIER_SPECS if spec[0] == tier_id)
    current = CurrentSubscription(
        tier_id=tid, tier_name=name, monthly_credits=Decimal(mc), credits_remaining=Decimal(remaining),
        cycle_ends_at="2026-07-01", **over,
    )
    return dict(current=current, tiers=_dev_tiers(tid))


def dev_fixture_subscription_state() -> Optional[SubscriptionState]:
    """``HERMES_DEV_SUBSCRIPTION_FIXTURE`` (``free | mid | top | not-admin | downgrade | cancel | team |
    logged-out``) -> fixture state; None when unset; unknown name → logged-out with ``error`` set."""
    name = (os.getenv("HERMES_DEV_SUBSCRIPTION_FIXTURE") or "").strip().lower()
    if not name:
        return None
    name = _DEV_FIXTURE_ALIASES.get(name, name)
    if name == "logged-out":
        return SubscriptionState(logged_in=False)

    common = dict(logged_in=True, org_name="Acme Inc", org_id="org_acme", role="OWNER", portal_url=_DEV_FIXTURE_PORTAL)
    states: dict[str, dict[str, Any]] = {
        "free": dict(current=None, tiers=_dev_tiers(None)),
        "mid": _dev_plan("plus", "420"),
        "top": _dev_plan("ultra", "5000"),
        "not-admin": {**_dev_plan("plus", "420"), "role": "MEMBER"},
        "downgrade": _dev_plan("super", "1500", pending_downgrade_tier_name="Plus", pending_downgrade_at="2026-07-15"),
        "cancel": _dev_plan("plus", "420", cancel_at_period_end=True, cancellation_effective_at="2026-07-01"),
        "team": dict(context="team", current=None, org_name="Acme Engineering", org_id="org_eng"),
    }
    if name not in states:
        return SubscriptionState(logged_in=False, error=f"unknown HERMES_DEV_SUBSCRIPTION_FIXTURE: {name}")
    return SubscriptionState(**{**common, **states[name]})
