"""Surface-agnostic core for the ``/subscription`` TUI screen.

Companion to :mod:`agent.billing_view` — same fail-open philosophy: when not
logged in or the portal is unreachable, return a struct with ``logged_in=False``
and let the surface degrade gracefully (never crash). Money is decimal end-to-end
(server emits decimal strings); we only format for display.

The TUI ``SubscriptionOverlay`` drives the plan change in-terminal (V3): it
previews the effect, then schedules a downgrade / cancellation / resume
(chargeless) or applies an upgrade (charges the card on the subscription). The
portal deep-link (built locally from ``portal_url`` + ``org_id``) remains the
fallback for an upgrade that needs 3DS / was declined.

WS1 dependency: ``GET /api/billing/subscription`` is a NAS endpoint (WS1 Phase A).
Until it ships, the fail-open contract handles 404s — the builder returns
``logged_in=False`` and the surface degrades gracefully.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from agent.billing_view import parse_money

logger = logging.getLogger(__name__)


# =============================================================================
# Parsed sub-structures
# =============================================================================


@dataclass(frozen=True)
class CurrentSubscription:
    """The user's active subscription. ``None`` (not this object) = no plan.

    When present, ``tier_id`` / ``tier_name`` / ``monthly_credits`` /
    ``cycle_ends_at`` are always set (NAS guarantees a present ``current`` is a
    fully-populated plan). Only ``credits_remaining`` and the cancel/downgrade
    fields are optional.
    """

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
    """A selectable plan in the catalog — one row of the in-terminal tier picker.

    Mirrors NAS's ``SubscriptionTierOption``. ``is_current`` marks the active plan
    (shown but not selectable); ``is_enabled=False`` is a grandfathered tier the
    user is on but that can no longer be selected. ``tier_order`` sorts the picker
    and drives the upgrade-vs-downgrade direction hint.
    """

    tier_id: str
    name: str
    tier_order: int = 0
    dollars_per_month: Optional[Decimal] = None
    monthly_credits: Optional[Decimal] = None
    is_current: bool = False
    is_enabled: bool = True


@dataclass(frozen=True)
class SubscriptionChangePreview:
    """Parsed ``POST /api/billing/subscription/preview`` — what a change would do.

    ``effect`` is the disposition the commit would take:
      - ``charge_now`` → an upgrade; ``amount_due_now_cents`` is the prorated charge.
      - ``scheduled``  → a downgrade / same-price change at ``effective_at`` (period end).
      - ``no_op``      → already on the target tier.
      - ``blocked``    → the commit would be refused; ``reason`` says why.
    """

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
class SubscriptionState:
    """Parsed ``GET /api/billing/subscription`` — the overview screen's data.

    Fail-open: ``logged_in=False`` (and empty fields) when not logged in or the
    portal is unreachable.
    """

    logged_in: bool
    org_name: Optional[str] = None
    org_id: Optional[str] = None  # org.id from the NAS response
    role: Optional[str] = None  # "OWNER" | "ADMIN" | "FINANCE_ADMIN" | "SECURITY_ADMIN" | "MEMBER"
    can_change_plan_raw: Optional[bool] = None
    context: str = "personal"  # "personal" | "team"
    current: Optional[CurrentSubscription] = None
    tiers: tuple[SubscriptionTier, ...] = ()  # selectable catalog (picker)
    portal_url: Optional[str] = None
    # When the fetch failed (vs cleanly not-logged-in), the message for the surface.
    error: Optional[str] = None

    @property
    def is_admin(self) -> bool:
        """Deprecated/display only — a legacy OWNER/ADMIN check.

        NOT a capability check; use :attr:`can_change_plan` for gating billing
        plan-change actions.
        """
        return (self.role or "").upper() in ("OWNER", "ADMIN")

    @property
    def can_change_plan(self) -> bool:
        """Server capability when supplied; otherwise the legacy role fallback."""
        if self.can_change_plan_raw is not None:
            return self.can_change_plan_raw
        return self.is_admin


# =============================================================================
# Payload parsing
# =============================================================================


def _parse_current(raw: Any) -> Optional[CurrentSubscription]:
    # "No plan" is wire-represented as current:null (free personal OR team) —
    # the old all-null-object shape is gone. A present current is a real plan,
    # so guard on a real tier id and return None otherwise.
    if not isinstance(raw, dict):
        return None
    tier_id = raw.get("tierId") or raw.get("id")
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
    """First non-``None`` value (preserves a legit ``0``/``0.0``, unlike ``or``).

    NAS sends ``0`` for the free tier's ``tierOrder`` / ``dollarsPerMonth``; a plain
    ``x or default`` would drop those, so coalesce on ``None`` specifically.
    """
    for v in vals:
        if v is not None:
            return v
    return None


def _parse_tier(raw: Any) -> Optional[SubscriptionTier]:
    """Map one NAS ``SubscriptionTierOption`` dict into a :class:`SubscriptionTier`."""
    if not isinstance(raw, dict):
        return None
    tier_id = raw.get("tierId") or raw.get("id")
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


def subscription_change_preview_from_payload(
    payload: dict[str, Any],
) -> SubscriptionChangePreview:
    """Map a raw ``/subscription/preview`` JSON dict into :class:`SubscriptionChangePreview`."""
    effect = payload.get("effect")
    cents = payload.get("amountDueNowCents")
    return SubscriptionChangePreview(
        # An unrecognized/missing effect is treated as ``blocked`` — fail safe, never
        # charge on a malformed quote.
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


def subscription_state_from_payload(
    payload: dict[str, Any], *, portal_url: Optional[str] = None
) -> SubscriptionState:
    """Map a raw ``/api/billing/subscription`` JSON dict into :class:`SubscriptionState`."""
    raw_org = payload.get("org")
    org: dict[str, Any] = raw_org if isinstance(raw_org, dict) else {}

    raw_context = payload.get("context")
    context = raw_context if raw_context in ("personal", "team") else "personal"

    raw_tiers = payload.get("tiers")
    tiers = (
        tuple(t for t in (_parse_tier(x) for x in raw_tiers) if t is not None)
        if isinstance(raw_tiers, list)
        else ()
    )

    return SubscriptionState(
        logged_in=True,
        org_name=org.get("name"),
        org_id=org.get("id") or None,
        role=org.get("role"),
        can_change_plan_raw=(
            payload.get("canChangePlan")
            if isinstance(payload.get("canChangePlan"), bool)
            else None
        ),
        context=context,
        current=_parse_current(payload.get("current")),
        tiers=tiers,
        portal_url=portal_url,
    )


# =============================================================================
# Fail-open builders (the surface front doors)
# =============================================================================


def build_subscription_state(*, timeout: float = 15.0) -> SubscriptionState:
    """Fetch + parse ``GET /api/billing/subscription``. Fail-open.

    Returns ``SubscriptionState(logged_in=False)`` when not logged in. On a
    portal/HTTP failure, returns ``logged_in=False`` with ``error`` set so the
    surface can show a clear message rather than crashing.

    Dev override: when ``HERMES_DEV_SUBSCRIPTION_FIXTURE`` names a fixture state,
    ``/subscription`` renders from that fixture instead of the real portal — so
    every plan/cancel/downgrade/team/not-admin state is testable on both
    the CLI and TUI without a live account. Throwaway scaffolding; see
    :func:`dev_fixture_subscription_state`.
    """
    fixture = dev_fixture_subscription_state()
    if fixture is not None:
        return fixture

    try:
        from hermes_cli.nous_billing import (
            BillingAuthError,
            BillingError,
            _absolutize_portal_url,
            get_subscription_state,
            resolve_portal_base_url,
        )
    except Exception:
        return SubscriptionState(logged_in=False, error="billing client unavailable")

    try:
        payload = get_subscription_state(timeout=timeout)
    except BillingAuthError:
        return SubscriptionState(logged_in=False)
    except BillingError as exc:
        logger.debug("subscription ▸ /state fetch failed (fail-open)", exc_info=True)
        return SubscriptionState(logged_in=False, error=str(exc))
    except Exception:
        logger.debug("subscription ▸ /state unexpected error (fail-open)", exc_info=True)
        return SubscriptionState(logged_in=False, error="could not load subscription state")

    raw_portal = payload.get("portalUrl") if isinstance(payload, dict) else None
    portal_url = _absolutize_portal_url(raw_portal) if raw_portal else None
    if not portal_url:
        try:
            portal_url = resolve_portal_base_url()
        except Exception:
            portal_url = None

    return subscription_state_from_payload(payload, portal_url=portal_url)


def subscription_manage_url(
    state: SubscriptionState, tier_id: Optional[str] = None
) -> Optional[str]:
    """Build ``{portal_origin}/manage-subscription?org_id=<id>[&plan=<tier_id>]``.

    Mirrors the TUI's ``buildManageUrl`` (``subscription.ts``): the deep-link
    target is NAS's OWN ``/manage-subscription`` page (NOT the Stripe Billing
    Portal — decided Jun 23), which routes upgrade→Checkout / downgrade→scheduled
    internally. ``org_id`` pins the page to the right account in multi-org
    situations. Returns ``None`` when no portal URL is resolvable.

    ``tier_id`` (the stable ``tiers[]`` id, never a name/slug) is appended as
    ``plan=`` so the portal preselects the picked plan — only for a NEW
    subscription / upgrade the user chose. The portal validates it and simply
    ignores an unknown tier, so the CLI appends unconditionally when a tier was
    picked (parity with the TUI's ``?plan=``).
    """
    from urllib.parse import urlencode, urlsplit, urlunsplit

    if not state.portal_url:
        return None

    try:
        parts = urlsplit(state.portal_url)
    except Exception:
        return None

    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None

    from urllib.parse import parse_qsl

    # Preserve unrelated portal query params; org_id / plan are contract-owned
    # (org_id before plan — insertion order is the emitted query order).
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    params.pop("org_id", None)
    params.pop("plan", None)
    if state.org_id:
        params["org_id"] = state.org_id
    if tier_id:
        params["plan"] = tier_id
    query = urlencode(params)
    return urlunsplit((parts.scheme, parts.netloc, "/manage-subscription", query, ""))


# =============================================================================
# Shared plan-catalog helpers (consumed by the CLI Free catalog + paid picker)
# =============================================================================


def _format_dollars_grouped(value: Optional[Decimal]) -> str:
    """``$1,000`` / ``$1,234.50`` — the whole-vs-fractional rule of
    ``billing_view.format_money`` but thousands-grouped, matching the TUI's
    ``toLocaleString('en-US')``.

    The shared ``format_money`` is intentionally ungrouped (and asserted so across
    other surfaces), so plan-catalog rows group locally to mirror the TUI.
    """
    if value is None:
        return "—"
    if value == value.to_integral_value():
        return f"${format(value.to_integral_value(), ',f')}"
    return f"${format(value.quantize(Decimal('0.01')), ',f')}"


def selectable_tiers(state: SubscriptionState) -> list[SubscriptionTier]:
    """Enabled paid tiers other than the current plan, cheapest first.

    One derivation shared by the CLI Free catalog and the paid change picker:
    ``is_enabled and not is_current and tier_order > 0`` (free / no-sub excluded —
    dropping to free is a cancellation), sorted by ``tier_order``.
    """
    return sorted(
        (
            t
            for t in (state.tiers or ())
            if t.is_enabled and not t.is_current and (t.tier_order or 0) > 0
        ),
        key=lambda t: t.tier_order or 0,
    )


def format_tier_row(tier: SubscriptionTier) -> str:
    """``name · $X/mo[ · $Y credits/mo]`` — the shared plan-catalog row.

    Mirrors the TUI Free rows (``subscriptionOverlay.tsx``): thousands-grouped
    money, and the ``$Y credits/mo`` suffix ONLY when monthly credits are present
    and > 0 (a ``None`` / zero-credits tier hides it — never ``· — credits/mo`` or
    ``· $0 credits/mo``).
    """
    row = f"{tier.name} · {_format_dollars_grouped(tier.dollars_per_month)}/mo"
    mc = tier.monthly_credits
    if mc is not None and mc > 0:
        row += f" · {_format_dollars_grouped(mc)} credits/mo"
    return row


def is_upgrade(state: SubscriptionState, tier_id: str) -> bool:
    """True when ``tier_id`` ranks above the current plan by ``tier_order``.

    Prefers the active subscription's tier; falls back to the ``tiers[]``
    ``is_current`` marker (what the picker derives from), else 0 (free).
    """
    orders = {t.tier_id: (t.tier_order or 0) for t in (state.tiers or ())}
    cur_id = state.current.tier_id if state.current else None
    if cur_id is not None and cur_id in orders:
        cur_order = orders[cur_id]
    else:
        cur_order = next((t.tier_order or 0 for t in (state.tiers or ()) if t.is_current), 0)
    return orders.get(tier_id, 0) > cur_order


# =============================================================================
# Dev fixtures (throwaway scaffolding — env-var driven, no live portal)
# =============================================================================

_DEV_FIXTURE_PORTAL = "https://portal.nousresearch.com/billing"


def _dev_current(**over: Any) -> CurrentSubscription:
    base: dict[str, Any] = dict(
        tier_id="plus",
        tier_name="Plus",
        monthly_credits=Decimal("1000"),
        credits_remaining=Decimal("420"),
        cycle_ends_at="2026-07-01",
    )
    base.update(over)
    return CurrentSubscription(**base)


def _dev_tiers(current_id: Optional[str]) -> tuple[SubscriptionTier, ...]:
    """A sample plan catalog for fixtures (marks ``current_id`` as the active tier)."""
    specs = (
        ("free", "Free", 0, "0", "0"),
        ("plus", "Plus", 1, "20", "1000"),
        ("super", "Super", 2, "40", "3000"),
        ("ultra", "Ultra", 3, "80", "7000"),
    )
    return tuple(
        SubscriptionTier(
            tier_id=tid,
            name=name,
            tier_order=order,
            dollars_per_month=parse_money(dpm),
            monthly_credits=parse_money(mc),
            is_current=(tid == current_id),
            is_enabled=True,
        )
        for tid, name, order, dpm, mc in specs
    )


def dev_fixture_subscription_state() -> Optional[SubscriptionState]:
    """Return a fixture :class:`SubscriptionState` for ``HERMES_DEV_SUBSCRIPTION_FIXTURE``.

    Lets every CLI/TUI subscription state be exercised without a live portal:

        free | mid | top | not-admin | downgrade | cancel | team |
        logged-out

    Returns ``None`` when the env var is unset/empty (the real portal path runs).
    Throwaway scaffolding — mirrors ``HERMES_DEV_CREDITS_FIXTURE``.
    """
    name = (os.getenv("HERMES_DEV_SUBSCRIPTION_FIXTURE") or "").strip().lower()
    if not name:
        return None

    common = dict(org_name="Acme Inc", org_id="org_acme", role="OWNER", portal_url=_DEV_FIXTURE_PORTAL)

    if name in ("logged-out", "logged_out", "loggedout"):
        return SubscriptionState(logged_in=False)
    if name == "free":
        return SubscriptionState(logged_in=True, current=None, tiers=_dev_tiers(None), **common)
    if name in ("mid", "mid-tier"):
        return SubscriptionState(logged_in=True, current=_dev_current(), tiers=_dev_tiers("plus"), **common)
    if name in ("top", "top-tier"):
        return SubscriptionState(
            logged_in=True,
            current=_dev_current(tier_id="ultra", tier_name="Ultra", monthly_credits=Decimal("7000"), credits_remaining=Decimal("5000")),
            tiers=_dev_tiers("ultra"),
            **common,
        )
    if name in ("not-admin", "member"):
        return SubscriptionState(logged_in=True, current=_dev_current(), tiers=_dev_tiers("plus"), **{**common, "role": "MEMBER"})
    if name == "downgrade":
        return SubscriptionState(
            logged_in=True,
            current=_dev_current(tier_id="super", tier_name="Super", monthly_credits=Decimal("3000"), credits_remaining=Decimal("1500"), pending_downgrade_tier_name="Plus", pending_downgrade_at="2026-07-15"),
            tiers=_dev_tiers("super"),
            **common,
        )
    if name == "cancel":
        return SubscriptionState(
            logged_in=True,
            current=_dev_current(cancel_at_period_end=True, cancellation_effective_at="2026-07-01"),
            tiers=_dev_tiers("plus"),
            **common,
        )
    if name == "team":
        return SubscriptionState(logged_in=True, context="team", current=None, org_name="Acme Engineering", org_id="org_eng", role="OWNER", portal_url=_DEV_FIXTURE_PORTAL)

    # Unknown name → behave as logged-out so the misconfiguration is visible.
    return SubscriptionState(logged_in=False, error=f"unknown HERMES_DEV_SUBSCRIPTION_FIXTURE: {name}")

