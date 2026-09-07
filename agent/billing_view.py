"""Surface-agnostic core for the Remote Spending screens (CLI ``_show_billing``, TUI JSON-RPC).

One fetch/parse per concern; the server payload is parsed into frozen dataclasses.
**Fail open**: when not logged in or the portal is unreachable, return a struct with
``logged_in=False`` and let the surface degrade gracefully (never crash).

Money discipline: the server emits decimal STRINGS (``"142.5"``, not fixed 2dp).
We keep them as :class:`decimal.Decimal` end-to-end and only format for display.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def parse_money(value: Any) -> Optional[Decimal]:
    """Server money value (decimal string; defensively int/float) -> Decimal, or None. Never raises."""
    try:
        # Decimal(str(...)) avoids binary-float artifacts if a float ever sneaks in.
        return Decimal(str(value).strip()) if value is not None else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def format_money(value: Optional[Decimal], *, grouped: bool = False) -> str:
    """``$X`` for whole dollars, ``$X.YY`` (exactly 2dp) otherwise; ``None`` -> ``—``.

    ``grouped=True`` adds thousands separators (mirrors the TUI's ``toLocaleString('en-US')``
    on plan-catalog rows); the default is intentionally ungrouped across the other surfaces.
    """
    if value is None:
        return "—"
    spec = ",f" if grouped else "f"
    if value == value.to_integral_value():
        # format(..., "f") avoids 1E+3 for 1000.
        return f"${format(value.to_integral_value(), spec)}"
    return f"${format(value.quantize(Decimal('0.01')), spec)}"


def _optional_str(raw: dict, key: str) -> Optional[str]:
    value = raw.get(key)
    return value if isinstance(value, str) else None


def _dict_parser(fn: Callable[[dict], Any]) -> Callable[[Any], Any]:
    """Sub-structure parsers accept anything and return None unless the payload is a dict."""
    return lambda raw: fn(raw) if isinstance(raw, dict) else None


# resolvedVia (server card-on-file ladder rung) → "why THIS card?". Unknown/absent
# rungs render no label so older servers degrade cleanly.
_CARD_PROVENANCE_LABELS = {
    "subPin": "the card on your subscription",
    "customerDefault": "your default card saved on the portal",
    "autoRefill": "your auto-reload card",
}


@dataclass(frozen=True)
class CardInfo:
    brand: str
    last4: str
    resolved_via: Optional[str] = None  # ladder rung; None on pre-resolver payloads

    @property
    def masked(self) -> str:
        # A Link payment method has no card number (last4 = "") — brand alone, not "Link ····".
        return f"{self.brand} ····{self.last4}" if self.last4 else self.brand

    @property
    def provenance(self) -> Optional[str]:
        """Human label for why this card was picked, or None (unknown rung / old server)."""
        return _CARD_PROVENANCE_LABELS.get(self.resolved_via) if self.resolved_via is not None else None

    @property
    def display(self) -> str:
        """``Visa ····4242 — the card on your subscription`` (masked only when provenance unknown)."""
        label = self.provenance
        return f"{self.masked} — {label}" if label else self.masked


@dataclass(frozen=True)
class PaymentMethodInfo:
    """Payment method on file. ``kind`` is "card" | "link" | "unknown" (settled at parse time
    so consumers only see fields that belong to the kind they are looking at)."""

    kind: str
    brand: Optional[str] = None
    last4: Optional[str] = None
    wallet: Optional[str] = None
    email: Optional[str] = None
    resolved_via: Optional[str] = None
    raw_kind: Optional[str] = None  # what the server called an unrecognised kind


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


class OrgRoleCapability:
    """``is_admin`` / ``can_change_plan`` shared by the billing and subscription states."""

    role: Optional[str]
    can_change_plan_raw: Optional[bool]

    @property
    def is_admin(self) -> bool:
        """Display only — legacy OWNER/ADMIN check; gate plan-change actions on :attr:`can_change_plan`."""
        return (self.role or "").upper() in ("OWNER", "ADMIN")

    @property
    def can_change_plan(self) -> bool:
        """Server capability when supplied; otherwise the legacy role fallback."""
        return self.can_change_plan_raw if self.can_change_plan_raw is not None else self.is_admin


@dataclass(frozen=True)
class BillingState(OrgRoleCapability):
    """Parsed ``GET /api/billing/state``; fail-open ``logged_in=False`` (empty fields) when unavailable."""

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
    payment_method: Optional[PaymentMethodInfo] = None
    monthly_cap: Optional[MonthlyCap] = None
    auto_reload: Optional[AutoReload] = None
    portal_url: Optional[str] = None
    error: Optional[str] = None  # set when the fetch failed (vs cleanly not-logged-in)

    @property
    def can_charge(self) -> bool:
        """Offer charge/auto-reload actions: ``can_change_plan`` (server-grantable, e.g. FINANCE_ADMIN)
        AND the per-org kill-switch. Display gating only — the server still enforces."""
        return self.can_change_plan and self.cli_billing_enabled


@_dict_parser
def _parse_card(raw: dict) -> Optional[CardInfo]:
    brand, last4 = raw.get("brand"), raw.get("last4")
    if not (isinstance(brand, str) and isinstance(last4, str)):
        return None
    return CardInfo(brand=brand, last4=last4, resolved_via=_optional_str(raw, "resolvedVia"))


@_dict_parser
def _parse_payment_method(raw: dict) -> Optional[PaymentMethodInfo]:
    if not isinstance(kind := raw.get("kind"), str):
        return None
    resolved_via = _optional_str(raw, "resolvedVia")
    brand = _optional_str(raw, "brand")
    last4 = _optional_str(raw, "last4")
    # Settle the kind here (like _parse_card) so nothing downstream re-checks fields.
    if kind == "card" and brand and last4:
        return PaymentMethodInfo(kind="card", brand=brand, last4=last4, wallet=_optional_str(raw, "wallet"), resolved_via=resolved_via)
    if kind == "link":
        return PaymentMethodInfo(kind="link", email=_optional_str(raw, "email"), resolved_via=resolved_via)
    return PaymentMethodInfo(kind="unknown", raw_kind=kind, resolved_via=resolved_via)


@_dict_parser
def _parse_monthly_cap(raw: dict) -> MonthlyCap:
    return MonthlyCap(limit_usd=parse_money(raw.get("limitUsd")), spent_this_month_usd=parse_money(raw.get("spentThisMonthUsd")),
                      is_default_ceiling=bool(raw.get("isDefaultCeiling")))


@_dict_parser
def _parse_auto_reload(raw: dict) -> AutoReload:
    return AutoReload(enabled=bool(raw.get("enabled")), threshold_usd=parse_money(raw.get("thresholdUsd")),
                      reload_to_usd=parse_money(raw.get("reloadToUsd")), card=_parse_auto_reload_card(raw.get("card")))


@_dict_parser
def _parse_auto_reload_card(raw: dict) -> Optional[AutoReloadCard]:
    if (kind := raw.get("kind")) not in ("canonical", "distinct", "none"):
        return None
    if kind != "distinct":
        return AutoReloadCard(kind=kind)
    return AutoReloadCard(kind=kind, payment_method_id=_optional_str(raw, "paymentMethodId"),
                          brand=_optional_str(raw, "brand"), last4=_optional_str(raw, "last4"))


def parse_org_fields(payload: dict[str, Any]) -> tuple[dict[str, Any], Optional[bool]]:
    """``(org dict or {}, canChangePlan if bool else None)`` — shared by both state parsers."""
    raw_org, ccp = payload.get("org"), payload.get("canChangePlan")
    return (raw_org if isinstance(raw_org, dict) else {}), (ccp if isinstance(ccp, bool) else None)


def billing_state_from_payload(payload: dict[str, Any], *, portal_url: Optional[str] = None) -> BillingState:
    """Map a raw ``/api/billing/state`` JSON dict into :class:`BillingState`."""
    org, can_change_plan_raw = parse_org_fields(payload)
    bounds: dict[str, Any] = payload.get("bounds") if isinstance(payload.get("bounds"), dict) else {}
    presets = [p for p in map(parse_money, payload.get("chargePresets") or ()) if p is not None]
    return BillingState(
        logged_in=True,
        org_id=org.get("id"),
        org_slug=org.get("slug"),
        org_name=org.get("name"),
        role=org.get("role"),
        can_change_plan_raw=can_change_plan_raw,
        balance_usd=parse_money(payload.get("balanceUsd")),
        cli_billing_enabled=bool(payload.get("cliBillingEnabled")),
        charge_presets=tuple(presets),
        min_usd=parse_money(bounds.get("minUsd")),
        max_usd=parse_money(bounds.get("maxUsd")),
        card=_parse_card(payload.get("card")),
        payment_method=_parse_payment_method(payload.get("paymentMethod")),
        monthly_cap=_parse_monthly_cap(payload.get("monthlyCap")),
        auto_reload=_parse_auto_reload(payload.get("autoReload")),
        portal_url=portal_url,
    )


def fetch_portal_state(
    endpoint: str, label: str, *, failed: Callable[..., Any], parse: Callable[[dict, Optional[str]], Any],
    portal_fallback: Callable[[str], str], timeout: float, log: logging.Logger,
):
    """Shared fail-open fetch+parse for the billing/subscription overview builders.

    ``failed(**kw)`` builds the ``logged_in=False`` struct: bare on auth failure, with ``error``
    set on a portal/HTTP failure. Portal URL: server ``portalUrl`` (absolutized), else
    ``portal_fallback(portal_base_url)``.
    """
    try:
        import hermes_cli.nous_billing as nb
    except Exception:
        return failed(error="billing client unavailable")
    try:
        payload = getattr(nb, endpoint)(timeout=timeout)
    except nb.BillingAuthError:
        return failed()
    except nb.BillingError as exc:
        log.debug("%s ▸ /state fetch failed (fail-open)", label, exc_info=True)
        return failed(error=str(exc))
    except Exception:
        log.debug("%s ▸ /state unexpected error (fail-open)", label, exc_info=True)
        return failed(error=f"could not load {label} state")
    raw_portal = payload.get("portalUrl") if isinstance(payload, dict) else None
    portal_url = nb._absolutize_portal_url(raw_portal) if raw_portal else None
    if not portal_url:
        try:
            portal_url = portal_fallback(nb.resolve_portal_base_url())
        except Exception:
            portal_url = None
    return parse(payload, portal_url)


def build_billing_state(*, timeout: float = 15.0) -> BillingState:
    """Fetch + parse ``/api/billing/state``; fail-open. ``HERMES_DEV_BILLING_FIXTURE`` short-circuits to a fixture."""
    fixture = _dev_fixture_billing_state()
    if fixture is not None:
        return fixture
    return fetch_portal_state(
        "get_billing_state", "billing", timeout=timeout, log=logger,
        failed=lambda **kw: BillingState(logged_in=False, **kw),
        parse=lambda payload, portal_url: billing_state_from_payload(payload, portal_url=portal_url),
        portal_fallback=lambda base: f"{base.rstrip('/')}/billing?topup=open",
    )


# ── Dev fixtures (env-var driven, no live portal) ────────────────────────────

_FIXTURE_ALIASES = {
    "logged_out": "logged-out", "loggedout": "logged-out", "card_sub": "card-sub", "card_autoreload": "card-autoreload",
    "autoreload": "card-autoreload", "not-admin": "notadmin", "member": "notadmin", "billing_off": "billing-off", "off": "billing-off",
}


def _dev_fixture_billing_state() -> Optional[BillingState]:
    """``HERMES_DEV_BILLING_FIXTURE`` -> :class:`BillingState` for offline UX; None when unset.

    Names: nocard · card · card-sub · card-autoreload · notadmin · billing-off · logged-out; an
    unknown name yields logged-out with ``error`` so the misconfiguration is visible.
    """
    name = (os.getenv("HERMES_DEV_BILLING_FIXTURE") or "").strip().lower()
    if not name:
        return None
    name = _FIXTURE_ALIASES.get(name, name)
    if name == "logged-out":
        return BillingState(logged_in=False)

    # Prod portal host (matches subscription_view._DEV_FIXTURE_PORTAL) + the /topup deep-link suffix.
    common: dict[str, Any] = dict(
        logged_in=True, org_id="org_acme", org_slug="acme", org_name="Acme Inc", role="OWNER",
        balance_usd=Decimal("3.40"), cli_billing_enabled=True, min_usd=Decimal("5"), max_usd=Decimal("500"),
        charge_presets=(Decimal("10"), Decimal("25"), Decimal("50")), portal_url="https://portal.nousresearch.com/billing?topup=open",
    )
    card = CardInfo(brand="Visa", last4="4242")
    overrides: dict[str, dict[str, Any]] = {
        "nocard": dict(card=None),
        "card": dict(card=card),
        "card-sub": dict(card=CardInfo(brand="Visa", last4="4242", resolved_via="subPin")),
        "card-autoreload": dict(card=card, auto_reload=AutoReload(enabled=True, threshold_usd=Decimal("5"), reload_to_usd=Decimal("25"))),
        "notadmin": dict(card=card, role="MEMBER"),
        "billing-off": dict(card=None, cli_billing_enabled=False),
    }
    if name not in overrides:
        return BillingState(logged_in=False, error=f"unknown HERMES_DEV_BILLING_FIXTURE: {name}")
    return BillingState(**{**common, **overrides[name]})


def new_idempotency_key() -> str:
    """Fresh ``Idempotency-Key`` for ``POST /charge``: reuse across retries of the SAME buy so a
    double-submit collapses to one charge; never across amounts (server 409 idempotency_conflict)."""
    return str(uuid.uuid4())


@dataclass(frozen=True)
class AmountValidation:
    ok: bool
    amount: Optional[Decimal] = None
    error: Optional[str] = None


def validate_charge_amount(raw: str, *, min_usd: Optional[Decimal], max_usd: Optional[Decimal]) -> AmountValidation:
    """Mirror the server's accept/reject (bounds + multipleOf 0.01) for instant UI feedback; server is authoritative."""
    amount = parse_money((raw or "").strip().lstrip("$").strip())
    if amount is None:
        error = "Enter a dollar amount, e.g. 100"
    elif amount <= 0:
        error = "Amount must be greater than $0"
    elif amount != amount.quantize(Decimal("0.01")):
        error = "Amount can't be smaller than a cent"
    elif min_usd is not None and amount < min_usd:
        error = f"Minimum is {format_money(min_usd)}"
    elif max_usd is not None and amount > max_usd:
        error = f"Maximum is {format_money(max_usd)}"
    else:
        return AmountValidation(ok=True, amount=amount)
    return AmountValidation(ok=False, error=error)
