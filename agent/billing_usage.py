"""Shared dollar-denominated usage model for the ``/usage`` and ``/subscription`` bars.

Terminal surfaces show **dollars**, never "credits"; the plan allowance and top-up
dollars stay distinctly visible as two SEPARATE bars (a three-segment bar is
unreadable at terminal widths). Source: ``NousPortalAccountInfo.paid_service_access_info``
(USD floats despite the legacy ``*_credits`` names) plus ``subscription.monthly_credits``
(plan bar denominator) and ``current_period_end``. Fail-open: missing/non-finite fields
degrade to fewer bars; logged-out / unreachable portal yields ``available=False``.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Below this TOTAL spendable ($) a paid account is flagged "low" — the alert state
# that nudges top-up/upgrade before a mid-run cutoff (product: "below $5 is an alert").
LOW_BALANCE_THRESHOLD_USD = 5.0


def _finite(value: Any) -> Optional[float]:
    """Float iff a real finite number (not bool/NaN/Inf); json.loads admits bare NaN, which would render ``$nan``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _fmt_usd(value: Optional[float]) -> str:
    """``$X,XXX.YY`` for display. ``None`` -> ``$0.00`` (callers gate on presence)."""
    return f"${(value or 0.0):,.2f}"


def nous_logged_in() -> bool:
    """Cheap local auth-state check: a Nous access token is present. Fail-closed."""
    try:
        from hermes_cli.auth import get_provider_auth_state
        tok = (get_provider_auth_state("nous") or {}).get("access_token")
        return isinstance(tok, str) and bool(tok.strip())
    except Exception:
        return False


def fetch_nous_account(timeout: float):
    """Wall-clock-bounded fresh portal account fetch. Raises on failure/timeout."""
    import concurrent.futures
    from hermes_cli.nous_account import get_nous_portal_account_info
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(get_nous_portal_account_info, force_fresh=True).result(timeout=timeout)


def format_renews(value: Optional[str]) -> Optional[str]:
    """ISO date/timestamp -> ``Jul 24, 2026``; unparseable input is returned unchanged."""
    from datetime import datetime
    text = str(value or "").strip()
    if not text:
        return None
    iso = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        try:
            dt = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return text
    # %-d isn't portable to Windows; build the day without a leading zero.
    return f"{dt.strftime('%b')} {dt.day}, {dt.year}"


@dataclass(frozen=True)
class UsageBar:
    """One bar: ``spent`` of ``total`` plus remaining. ``plan`` shows % used; ``topup`` has no
    denominator (``spent`` is 0 and ``total == remaining`` so it renders full)."""

    kind: str  # "plan" | "topup"
    remaining_usd: float
    total_usd: float
    spent_usd: float = 0.0

    @property
    def pct_used(self) -> Optional[int]:
        if self.kind != "plan" or self.total_usd <= 0:
            return None
        return max(0, min(100, round(self.spent_usd / self.total_usd * 100)))

    @property
    def fill_fraction(self) -> float:
        """Fraction of the bar that should read as 'remaining' (filled)."""
        return max(0.0, min(1.0, self.remaining_usd / self.total_usd)) if self.total_usd > 0 else 0.0


@dataclass(frozen=True)
class UsageModel:
    """Dollar usage model shared by /usage and /subscription. ``status``: ``free`` (no plan / no
    paid access), ``low`` (spendable < $5, ALERT), ``healthy`` (>= $5), ``depleted`` (paid access lost)."""

    available: bool
    status: str = "free"
    plan_name: Optional[str] = None
    renews_at: Optional[str] = None
    renews_display: Optional[str] = None
    subscription_remaining_usd: Optional[float] = None
    topup_remaining_usd: Optional[float] = None
    total_spendable_usd: Optional[float] = None
    plan_bar: Optional[UsageBar] = None
    topup_bar: Optional[UsageBar] = None

    @property
    def has_topup(self) -> bool:
        return bool(self.topup_remaining_usd and self.topup_remaining_usd > 0)


def usage_model_from_account(account_info: Any) -> UsageModel:
    """Build a :class:`UsageModel` from a ``NousPortalAccountInfo``. Never raises."""
    try:
        if account_info is None or not getattr(account_info, "logged_in", False):
            return UsageModel(available=False)
        # Sub-structs are dataclasses or None; getattr(None, ..., None) is None, so no guards needed.
        access = getattr(account_info, "paid_service_access_info", None)
        sub = getattr(account_info, "subscription", None)
        paid = getattr(account_info, "paid_service_access", None)
        sub_remaining = _finite(getattr(access, "subscription_credits_remaining", None))
        topup_remaining = _finite(getattr(access, "purchased_credits_remaining", None))
        total_usable = _finite(getattr(access, "total_usable_credits", None))
        plan_name = getattr(sub, "plan", None)
        renews_at = getattr(sub, "current_period_end", None)
        monthly = _finite(getattr(sub, "monthly_credits", None))
        has_subscription = bool(plan_name) or (monthly is not None and monthly > 0)
        has_topup = bool(topup_remaining and topup_remaining > 0)
        # Prefer the server's total; else sum the parts we have.
        parts = [v for v in (sub_remaining, topup_remaining) if v is not None]
        total_spendable = total_usable if total_usable is not None else (sum(parts) if parts else None)
        if paid is False:
            status = "depleted"
        elif not has_subscription and not has_topup:
            status = "free"  # no plan and no purchased balance -> free-models-only
        elif total_spendable is not None and total_spendable < LOW_BALANCE_THRESHOLD_USD:
            status = "low"
        else:
            status = "healthy"
        # Plan bar needs a positive allowance AND a remaining to place on it; spent is
        # clamped so a debt/over-cap balance reads fully spent, not negative.
        plan_bar = topup_bar = None
        if monthly is not None and monthly > 0 and sub_remaining is not None:
            plan_bar = UsageBar(kind="plan", remaining_usd=max(0.0, min(monthly, sub_remaining)), total_usd=monthly,
                                spent_usd=max(0.0, monthly - sub_remaining))
        # Top-up has no monthly cap, so the bar renders full = balance.
        if topup_remaining is not None and topup_remaining > 0:
            topup_bar = UsageBar(kind="topup", remaining_usd=topup_remaining, total_usd=topup_remaining, spent_usd=0.0)
        return UsageModel(
            available=True, status=status, plan_name=plan_name, renews_at=renews_at,
            renews_display=format_renews(renews_at), subscription_remaining_usd=sub_remaining,
            topup_remaining_usd=topup_remaining, total_spendable_usd=total_spendable,
            plan_bar=plan_bar, topup_bar=topup_bar,
        )
    except Exception:
        logger.debug("usage ▸ model build failed (fail-open)", exc_info=True)
        return UsageModel(available=False)


def build_usage_model(*, timeout: float = 10.0) -> UsageModel:
    """Fetch account-info and build the usage model; fail-open. ``HERMES_DEV_CREDITS_FIXTURE`` short-circuits to a fixture."""
    fixture = _dev_fixture_usage_model()
    if fixture is not None:
        return fixture
    if not nous_logged_in():
        return UsageModel(available=False)
    try:
        return usage_model_from_account(fetch_nous_account(timeout))
    except Exception:
        logger.debug("usage ▸ portal fetch failed (fail-open)", exc_info=True)
        return UsageModel(available=False)


def _plan_bar(remaining: float, spent: float) -> UsageBar:
    return UsageBar(kind="plan", remaining_usd=remaining, total_usd=20.0, spent_usd=spent)


def _dev_fixture_usage_model() -> Optional[UsageModel]:
    """``HERMES_DEV_CREDITS_FIXTURE`` -> fixture model (``free|healthy|low|topup|depleted``), else None."""
    name = (os.getenv("HERMES_DEV_CREDITS_FIXTURE") or "").strip().lower()
    name = {"mid": "healthy", "top-up": "topup"}.get(name, name)
    plus = dict(available=True, plan_name="Plus", renews_at="2026-07-01")
    specs: dict[str, dict] = {
        "free": dict(available=True, status="free", plan_name=None),
        "healthy": dict(**plus, status="healthy", subscription_remaining_usd=14.0, total_spendable_usd=14.0, plan_bar=_plan_bar(14.0, 6.0)),
        "topup": dict(
            **plus, status="healthy", subscription_remaining_usd=14.0, topup_remaining_usd=12.0,
            total_spendable_usd=26.0, plan_bar=_plan_bar(14.0, 6.0),
            topup_bar=UsageBar(kind="topup", remaining_usd=12.0, total_usd=12.0, spent_usd=0.0),
        ),
        "low": dict(**plus, status="low", subscription_remaining_usd=3.4, total_spendable_usd=3.4, plan_bar=_plan_bar(3.4, 16.6)),
        "depleted": dict(**plus, status="depleted", subscription_remaining_usd=0.0, total_spendable_usd=0.0, plan_bar=_plan_bar(0.0, 20.0)),
    }
    spec = specs.get(name)
    return UsageModel(**spec) if spec else None
