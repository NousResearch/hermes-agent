"""Surface-agnostic core for the Phase 2b terminal-billing screens.

One fetch/parse per concern, consumed identically by the CLI handler
(``cli.py::_show_billing``), the TUI JSON-RPC methods
(``tui_gateway/server.py``), and any other surface. Mirrors the proven
``agent/account_usage.py::build_credits_view`` pattern: parse the server payload
into a frozen dataclass; **fail open** — when not logged in or the portal is
unreachable, return a struct with ``logged_in=False`` and let the surface degrade
gracefully (never crash).

Money discipline: the server emits decimal STRINGS (``"142.5"``, not fixed 2dp).
We keep them as :class:`decimal.Decimal` end-to-end and only format for display.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Decimal money helpers
# =============================================================================


def parse_money(value: Any) -> Optional[Decimal]:
    """Parse a server money value (decimal string) into :class:`Decimal`.

    Returns None for missing/invalid input. Never raises. Accepts str/int (and,
    defensively, float — though the server always sends strings).
    """
    if value is None:
        return None
    try:
        # Decimal(str(...)) avoids binary-float artifacts if a float ever sneaks in.
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        return None


def format_money(value: Optional[Decimal]) -> str:
    """Format a Decimal as ``$X`` / ``$X.YY`` for display.

    Whole dollars show no decimals; any fractional amount shows exactly 2dp:
    ``Decimal("142.5")`` → ``"$142.50"``, ``Decimal("100")`` → ``"$100"``,
    ``Decimal("0.01")`` → ``"$0.01"``.
    """
    if value is None:
        return "—"
    if value == value.to_integral_value():
        # Whole dollars — no decimal point. format(..., "f") avoids 1E+3 for 1000.
        return f"${format(value.to_integral_value(), 'f')}"
    # Fractional — always show 2dp.
    return f"${format(value.quantize(Decimal('0.01')), 'f')}"


# =============================================================================
# Parsed sub-structures
# =============================================================================


# resolvedVia → the human answer to "why THIS card?". Keys are the server's card
# resolution rungs (NAS card-on-file ladder); absent/unknown rungs render no label
# so the display degrades cleanly on servers that don't send resolvedVia yet.
_CARD_PROVENANCE_LABELS = {
    "subPin": "the card on your subscription",
    "customerDefault": "your default card saved on the portal",
    "autoRefill": "your auto-reload card",
}


@dataclass(frozen=True)
class CardInfo:
    brand: str
    last4: str
    # NAS card-on-file field (post card-resolver): which ladder rung found the
    # card. Defaults off so pre-resolver payloads parse unchanged.
    resolved_via: Optional[str] = None

    @property
    def masked(self) -> str:
        # A Link payment method has no card number (last4 = "") — render the
        # brand alone, not "Link ····".
        if not self.last4:
            return self.brand
        return f"{self.brand} ····{self.last4}"

    @property
    def provenance(self) -> Optional[str]:
        """Human label for why this card was picked, or None (unknown rung /
        server too old to say)."""
        if self.resolved_via is None:
            return None
        return _CARD_PROVENANCE_LABELS.get(self.resolved_via)

    @property
    def display(self) -> str:
        """The one-line card display: ``Visa ····4242 — the card on your
        subscription`` (or just the masked card when provenance is unknown)."""
        label = self.provenance
        return f"{self.masked} — {label}" if label else self.masked


@dataclass(frozen=True)
class MonthlyCap:
    limit_usd: Optional[Decimal] = None
    spent_this_month_usd: Optional[Decimal] = None
    is_default_ceiling: bool = False


@dataclass(frozen=True)
class AutoReloadCard:
    kind: str  # "canonical" | "distinct" | "none"
    payment_method_id: Optional[str] = None
    brand: Optional[str] = None
    last4: Optional[str] = None


@dataclass(frozen=True)
class AutoReload:
    enabled: bool = False
    threshold_usd: Optional[Decimal] = None
    reload_to_usd: Optional[Decimal] = None
    card: Optional[AutoReloadCard] = None


@dataclass(frozen=True)
class BillingState:
    """Parsed ``GET /api/billing/state`` — the overview screen's data.

    Fail-open: ``logged_in=False`` (and empty fields) when not logged in or the
    portal is unreachable.
    """

    logged_in: bool
    org_id: Optional[str] = None
    org_slug: Optional[str] = None
    org_name: Optional[str] = None
    role: Optional[str] = None  # "OWNER" | "ADMIN" | "FINANCE_ADMIN" | "SECURITY_ADMIN" | "MEMBER"
    can_change_plan_raw: Optional[bool] = None
    balance_usd: Optional[Decimal] = None
    cli_billing_enabled: bool = False
    charge_presets: tuple[Decimal, ...] = ()
    min_usd: Optional[Decimal] = None
    max_usd: Optional[Decimal] = None
    card: Optional[CardInfo] = None
    monthly_cap: Optional[MonthlyCap] = None
    auto_reload: Optional[AutoReload] = None
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

    @property
    def can_charge(self) -> bool:
        """True when the UI should offer charge/auto-reload actions.

        Uses the server-granted plan-change capability (``can_change_plan``,
        which itself falls back to the legacy OWNER/ADMIN role check when the
        server omits ``canChangePlan``) AND the per-org kill-switch. This lets
        the server grant charge capability to non-OWNER/ADMIN roles (e.g.
        FINANCE_ADMIN) via ``canChangePlan``, instead of hard-coding the
        deprecated 3-role admin check. (The server still enforces; this is
        just for graying out actions the user can't take.)
        """
        return self.can_change_plan and self.cli_billing_enabled


def _parse_card(raw: Any) -> Optional[CardInfo]:
    if not isinstance(raw, dict):
        return None
    brand = raw.get("brand")
    last4 = raw.get("last4")
    if not (isinstance(brand, str) and isinstance(last4, str)):
        return None
    # Post-resolver fields — all optional so both payload generations parse.
    resolved_via = raw.get("resolvedVia")
    if not isinstance(resolved_via, str):
        resolved_via = None
    return CardInfo(brand=brand, last4=last4, resolved_via=resolved_via)


def _parse_monthly_cap(raw: Any) -> Optional[MonthlyCap]:
    if not isinstance(raw, dict):
        return None
    return MonthlyCap(
        limit_usd=parse_money(raw.get("limitUsd")),
        spent_this_month_usd=parse_money(raw.get("spentThisMonthUsd")),
        is_default_ceiling=bool(raw.get("isDefaultCeiling")),
    )


def _parse_auto_reload(raw: Any) -> Optional[AutoReload]:
    if not isinstance(raw, dict):
        return None
    return AutoReload(
        enabled=bool(raw.get("enabled")),
        threshold_usd=parse_money(raw.get("thresholdUsd")),
        reload_to_usd=parse_money(raw.get("reloadToUsd")),
        card=_parse_auto_reload_card(raw.get("card")),
    )


def _parse_auto_reload_card(raw: Any) -> Optional[AutoReloadCard]:
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if kind not in ("canonical", "distinct", "none"):
        return None
    if kind in ("canonical", "none"):
        return AutoReloadCard(kind=kind)

    payment_method_id = raw.get("paymentMethodId")
    brand = raw.get("brand")
    last4 = raw.get("last4")
    return AutoReloadCard(
        kind=kind,
        payment_method_id=payment_method_id if isinstance(payment_method_id, str) else None,
        brand=brand if isinstance(brand, str) else None,
        last4=last4 if isinstance(last4, str) else None,
    )


def billing_state_from_payload(
    payload: dict[str, Any], *, portal_url: Optional[str] = None
) -> BillingState:
    """Map a raw ``/api/billing/state`` JSON dict into :class:`BillingState`."""
    raw_org = payload.get("org")
    org: dict[str, Any] = raw_org if isinstance(raw_org, dict) else {}
    raw_bounds = payload.get("bounds")
    bounds: dict[str, Any] = raw_bounds if isinstance(raw_bounds, dict) else {}

    presets: list[Decimal] = []
    for item in payload.get("chargePresets") or ():
        parsed = parse_money(item)
        if parsed is not None:
            presets.append(parsed)

    return BillingState(
        logged_in=True,
        org_id=org.get("id"),
        org_slug=org.get("slug"),
        org_name=org.get("name"),
        role=org.get("role"),
        can_change_plan_raw=(
            payload.get("canChangePlan")
            if isinstance(payload.get("canChangePlan"), bool)
            else None
        ),
        balance_usd=parse_money(payload.get("balanceUsd")),
        cli_billing_enabled=bool(payload.get("cliBillingEnabled")),
        charge_presets=tuple(presets),
        min_usd=parse_money(bounds.get("minUsd")),
        max_usd=parse_money(bounds.get("maxUsd")),
        card=_parse_card(payload.get("card")),
        monthly_cap=_parse_monthly_cap(payload.get("monthlyCap")),
        auto_reload=_parse_auto_reload(payload.get("autoReload")),
        portal_url=portal_url,
    )


# =============================================================================
# Fail-open builders (the surface front doors)
# =============================================================================


def build_billing_state(*, timeout: float = 15.0) -> BillingState:
    """Fetch + parse ``/api/billing/state``. Fail-open.

    Returns ``BillingState(logged_in=False)`` when not logged in. On a portal/HTTP
    failure, returns ``logged_in=False`` with ``error`` set so the surface can show
    a clear message rather than crashing.

    Dev override: ``HERMES_DEV_BILLING_FIXTURE`` short-circuits to a fixture so the
    card-on-file / admin / scope states are testable offline (mirrors
    ``HERMES_DEV_CREDITS_FIXTURE`` for the usage model).
    """
    fixture = _dev_fixture_billing_state()
    if fixture is not None:
        return fixture

    try:
        from hermes_cli.nous_billing import (
            BillingAuthError,
            BillingError,
            _absolutize_portal_url,
            get_billing_state,
            resolve_portal_base_url,
        )
    except Exception:
        return BillingState(logged_in=False, error="billing client unavailable")

    try:
        payload = get_billing_state(timeout=timeout)
    except BillingAuthError:
        return BillingState(logged_in=False)
    except BillingError as exc:
        logger.debug("billing ▸ /state fetch failed (fail-open)", exc_info=True)
        return BillingState(logged_in=False, error=str(exc))
    except Exception:
        logger.debug("billing ▸ /state unexpected error (fail-open)", exc_info=True)
        return BillingState(logged_in=False, error="could not load billing state")

    # Prefer a server-supplied portalUrl if present (resolved to absolute in case
    # it's relative); else build the standard one.
    raw_portal = payload.get("portalUrl") if isinstance(payload, dict) else None
    portal_url = _absolutize_portal_url(raw_portal) if raw_portal else None
    if not portal_url:
        try:
            portal_url = _fallback_portal_url(resolve_portal_base_url())
        except Exception:
            portal_url = None

    return billing_state_from_payload(payload, portal_url=portal_url)


def _fallback_portal_url(base: str) -> str:
    """Standard billing deep-link when the server omits ``portalUrl``."""
    return f"{base.rstrip('/')}/billing?topup=open"


# =============================================================================
# Dev fixtures (throwaway scaffolding — env-var driven, no live portal)
# =============================================================================


def _dev_fixture_billing_state() -> Optional[BillingState]:
    """Map ``HERMES_DEV_BILLING_FIXTURE`` to a :class:`BillingState` for offline UX.

    Recognized names::

        nocard           logged in · billing on · admin · NO card on file
        card             card on file · auto-reload off
        card-autoreload  card on file · auto-reload on
        notadmin         logged in · MEMBER role (billing actions disabled)
        billing-off      logged in · admin · per-org kill-switch OFF
        logged-out       not logged in

    Returns ``None`` when the env var is unset (the real portal path runs).
    Mirrors ``HERMES_DEV_CREDITS_FIXTURE``; the usage *bar* still comes from
    ``HERMES_DEV_CREDITS_FIXTURE`` (set both to pair a bar with a billing state).
    """
    name = (os.getenv("HERMES_DEV_BILLING_FIXTURE") or "").strip().lower()
    if not name:
        return None

    # Shared fixture portal host (matches subscription_view._DEV_FIXTURE_PORTAL —
    # prod host, not staging; the ?topup=open suffix is the /topup deep-link).
    portal = "https://portal.nousresearch.com/billing?topup=open"
    common: dict[str, Any] = dict(
        org_id="org_acme",
        org_slug="acme",
        org_name="Acme Inc",
        role="OWNER",
        balance_usd=Decimal("3.40"),
        cli_billing_enabled=True,
        charge_presets=(Decimal("10"), Decimal("25"), Decimal("50")),
        min_usd=Decimal("5"),
        max_usd=Decimal("500"),
        portal_url=portal,
    )
    card = CardInfo(brand="Visa", last4="4242")
    autoreload_on = AutoReload(enabled=True, threshold_usd=Decimal("5"), reload_to_usd=Decimal("25"))

    if name in ("logged-out", "logged_out", "loggedout"):
        return BillingState(logged_in=False)
    if name == "nocard":
        return BillingState(logged_in=True, card=None, **common)
    if name == "card":
        return BillingState(logged_in=True, card=card, **common)
    if name in ("card-sub", "card_sub"):
        # Post-resolver: the card came from the subscription (provenance label).
        _sub_card = CardInfo(brand="Visa", last4="4242", resolved_via="subPin")
        return BillingState(logged_in=True, card=_sub_card, **common)
    if name in ("card-autoreload", "card_autoreload", "autoreload"):
        return BillingState(logged_in=True, card=card, auto_reload=autoreload_on, **common)
    if name in ("notadmin", "not-admin", "member"):
        opts = {**common, "role": "MEMBER"}
        return BillingState(logged_in=True, card=card, **opts)
    if name in ("billing-off", "billing_off", "off"):
        opts = {**common, "cli_billing_enabled": False}
        return BillingState(logged_in=True, card=None, **opts)

    # Unknown name → logged-out so the misconfiguration is visible.
    return BillingState(logged_in=False, error=f"unknown HERMES_DEV_BILLING_FIXTURE: {name}")


# =============================================================================
# Idempotency
# =============================================================================


def new_idempotency_key() -> str:
    """Fresh UUID for a user-confirmed purchase (reuse on retry of the SAME buy).

    The ``Idempotency-Key`` header is mandatory on ``POST /charge``; generate one
    per confirmed purchase and reuse it across retries so a double-submit collapses
    to a single charge. Never reuse a key across different amounts (the server
    returns 409 idempotency_conflict).
    """
    return str(uuid.uuid4())


# =============================================================================
# Amount validation (Screen 3 custom input)
# =============================================================================


@dataclass(frozen=True)
class AmountValidation:
    ok: bool
    amount: Optional[Decimal] = None
    error: Optional[str] = None


def validate_charge_amount(
    raw: str, *, min_usd: Optional[Decimal], max_usd: Optional[Decimal]
) -> AmountValidation:
    """Validate a custom charge amount against bounds + 2dp (multipleOf 0.01).

    Mirrors the server's accept/reject so the UI can give instant feedback rather
    than round-tripping a sure-to-fail charge. The server is still authoritative.
    """
    cleaned = (raw or "").strip().lstrip("$").strip()
    amount = parse_money(cleaned)
    if amount is None:
        return AmountValidation(ok=False, error="Enter a dollar amount, e.g. 100")
    if amount <= 0:
        return AmountValidation(ok=False, error="Amount must be greater than $0")
    # multipleOf 0.01 — reject sub-cent precision.
    if amount != amount.quantize(Decimal("0.01")):
        return AmountValidation(ok=False, error="Amount can't be smaller than a cent")
    if min_usd is not None and amount < min_usd:
        return AmountValidation(ok=False, error=f"Minimum is {format_money(min_usd)}")
    if max_usd is not None and amount > max_usd:
        return AmountValidation(ok=False, error=f"Maximum is {format_money(max_usd)}")
    return AmountValidation(ok=True, amount=amount)
