"""Shared dollar-denominated usage model for the billing/subscription surfaces.

The single source of truth behind the ``/usage`` and ``/subscription`` usage
bars (TUI + CLI). User feedback (Jun 2026): the terminal surfaces show
**dollars**, never "credits", and every usage bar must make the monthly
subscription allowance and separately-purchased top-up dollars distinctly
visible.

Data source: the NAS account-info fetch (``NousPortalAccountInfo``), whose
``paid_service_access_info`` carries the three dollar magnitudes we render
(despite the legacy ``*_credits`` field names, these are USD floats):

  - ``subscription_credits_remaining``  -> plan dollars left this month
  - ``purchased_credits_remaining``     -> top-up dollars left (rolls over)
  - ``total_usable_credits``            -> total spendable

plus ``subscription.monthly_credits`` (the plan's monthly $ allowance, the
denominator for the "% used" plan bar) and ``current_period_end`` (renewal).

Design: two SEPARATE bars (decided with the user) rather than one crammed
three-segment bar — at terminal widths three same-glyph density segments are
unreadable. The plan bar is "spent vs allowance this month" (carries % used);
the top-up bar is "money you bought, doesn't expire". Each gets full
resolution and a single fill glyph, so the bar is never ambiguous and never
relies on color.

Fail-open everywhere: any missing/non-finite field degrades to fewer bars or a
magnitudes-only view; a logged-out / unreachable portal yields
``available=False`` and the surface shows nothing.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Below this TOTAL spendable ($), a paid account is flagged "low" — the alert
# state that nudges top-up/upgrade before a mid-run cutoff. Product threshold
# (user feedback): "any amount below $5 should be an alert status."
LOW_BALANCE_THRESHOLD_USD = 5.0


def _finite(value: Any) -> Optional[float]:
    """Return value as a float iff it's a real finite number (not bool/NaN/Inf)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _fmt_usd(value: Optional[float]) -> str:
    """``$X.YY`` for display. ``None`` -> ``$0.00`` (callers gate on presence)."""
    return f"${(value or 0.0):,.2f}"


def format_renews(value: Optional[str]) -> Optional[str]:
    """Format an ISO date/timestamp as a human date, e.g. ``Jul 24, 2026``.

    Accepts ``2026-07-24``, ``2026-07-24T11:05:01.000Z``, etc. Returns the raw
    string unchanged if it can't be parsed (never raises), and ``None`` for
    empty input.
    """
    if not value:
        return None
    from datetime import datetime

    text = str(value).strip()
    if not text:
        return None
    iso = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        # Fall back to a bare date prefix (YYYY-MM-DD) if present.
        try:
            dt = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return text
    # %-d isn't portable to Windows; build the day without a leading zero.
    return f"{dt.strftime('%b')} {dt.day}, {dt.year}"


@dataclass(frozen=True)
class UsageBar:
    """One full-resolution bar: ``spent`` of ``total``, plus a remaining figure.

    ``kind`` is ``"plan"`` (monthly allowance, shows % used) or ``"topup"``
    (purchased dollars, no denominator — ``spent`` is 0 and ``total`` ==
    ``remaining`` so it renders as a full bar of available balance).
    """

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
        if self.total_usd <= 0:
            return 0.0
        return max(0.0, min(1.0, self.remaining_usd / self.total_usd))


@dataclass(frozen=True)
class UsageModel:
    """Surface-agnostic dollar usage model shared by /usage and /subscription.

    ``status`` classifies the account for copy selection:
      - ``"free"``     : no paid access / no subscription (free models only)
      - ``"low"``      : paid, but total spendable < $5 (ALERT)
      - ``"healthy"``  : paid, total spendable >= $5
      - ``"depleted"`` : paid access lost (balance exhausted)
    """

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
    """Build a :class:`UsageModel` from a ``NousPortalAccountInfo``. Fail-open.

    Returns ``UsageModel(available=False)`` when there's no usable account info
    (logged out, no entitlement block). Never raises.
    """
    try:
        if account_info is None or not getattr(account_info, "logged_in", False):
            return UsageModel(available=False)

        access = getattr(account_info, "paid_service_access_info", None)
        sub = getattr(account_info, "subscription", None)
        paid = getattr(account_info, "paid_service_access", None)

        sub_remaining = _finite(getattr(access, "subscription_credits_remaining", None)) if access else None
        topup_remaining = _finite(getattr(access, "purchased_credits_remaining", None)) if access else None
        total_usable = _finite(getattr(access, "total_usable_credits", None)) if access else None

        plan_name = getattr(sub, "plan", None) if sub is not None else None
        renews_at = getattr(sub, "current_period_end", None) if sub is not None else None
        monthly = _finite(getattr(sub, "monthly_credits", None)) if sub is not None else None

        has_subscription = bool(plan_name) or (monthly is not None and monthly > 0)

        # Total spendable: prefer the server's total; else sum the parts we have.
        if total_usable is not None:
            total_spendable = total_usable
        else:
            parts = [v for v in (sub_remaining, topup_remaining) if v is not None]
            total_spendable = sum(parts) if parts else None

        # Status classification.
        if paid is False:
            status = "depleted"
        elif not has_subscription and not (topup_remaining and topup_remaining > 0):
            # No plan and no purchased balance -> free-models-only.
            status = "free"
        elif total_spendable is not None and total_spendable < LOW_BALANCE_THRESHOLD_USD:
            status = "low"
        else:
            status = "healthy"

        # Plan bar — only with a positive monthly allowance AND a remaining we
        # can place on it. spent = cap - remaining, clamped (a debt/over-cap
        # balance reads as fully spent rather than a nonsensical negative).
        plan_bar: Optional[UsageBar] = None
        if monthly is not None and monthly > 0 and sub_remaining is not None:
            remaining = max(0.0, min(monthly, sub_remaining))
            plan_bar = UsageBar(
                kind="plan",
                remaining_usd=remaining,
                total_usd=monthly,
                spent_usd=max(0.0, monthly - sub_remaining),
            )

        # Top-up bar — only when there are purchased dollars to show. No
        # denominator (top-up has no monthly cap), so it renders full = balance.
        topup_bar: Optional[UsageBar] = None
        if topup_remaining is not None and topup_remaining > 0:
            topup_bar = UsageBar(
                kind="topup",
                remaining_usd=topup_remaining,
                total_usd=topup_remaining,
                spent_usd=0.0,
            )

        return UsageModel(
            available=True,
            status=status,
            plan_name=plan_name,
            renews_at=renews_at,
            renews_display=format_renews(renews_at),
            subscription_remaining_usd=sub_remaining,
            topup_remaining_usd=topup_remaining,
            total_spendable_usd=total_spendable,
            plan_bar=plan_bar,
            topup_bar=topup_bar,
        )
    except Exception:
        logger.debug("usage ▸ model build failed (fail-open)", exc_info=True)
        return UsageModel(available=False)


def build_usage_model(*, timeout: float = 10.0) -> UsageModel:
    """Fetch account-info and build the shared usage model. Fail-open.

    Dev override: ``HERMES_DEV_CREDITS_FIXTURE`` short-circuits to a fixture so
    every usage state is testable without a live account (mirrors the existing
    ``/usage`` credits-block fixture path).
    """
    fixture = _dev_fixture_usage_model()
    if fixture is not None:
        return fixture

    try:
        from hermes_cli.auth import get_provider_auth_state

        tok = (get_provider_auth_state("nous") or {}).get("access_token")
        if not (isinstance(tok, str) and tok.strip()):
            return UsageModel(available=False)
    except Exception:
        return UsageModel(available=False)

    try:
        import concurrent.futures

        from hermes_cli.nous_account import get_nous_portal_account_info

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            account = pool.submit(get_nous_portal_account_info, force_fresh=True).result(timeout=timeout)
        return usage_model_from_account(account)
    except Exception:
        logger.debug("usage ▸ portal fetch failed (fail-open)", exc_info=True)
        return UsageModel(available=False)


# =============================================================================
# Dev fixtures (throwaway scaffolding — env-var driven, no live portal)
# =============================================================================


def _dev_fixture_usage_model() -> Optional[UsageModel]:
    """Map ``HERMES_DEV_CREDITS_FIXTURE`` to a usage model for offline UX work.

    Recognized names: ``free | healthy | low | topup | depleted``. Returns
    ``None`` when the env var is unset (real portal path runs).
    """
    name = (os.getenv("HERMES_DEV_CREDITS_FIXTURE") or "").strip().lower()
    if not name:
        return None

    if name == "free":
        return UsageModel(available=True, status="free", plan_name=None)

    if name in ("healthy", "mid"):
        return UsageModel(
            available=True,
            status="healthy",
            plan_name="Plus",
            renews_at="2026-07-01",
            subscription_remaining_usd=14.0,
            total_spendable_usd=14.0,
            plan_bar=UsageBar(kind="plan", remaining_usd=14.0, total_usd=20.0, spent_usd=6.0),
        )

    if name in ("topup", "top-up"):
        return UsageModel(
            available=True,
            status="healthy",
            plan_name="Plus",
            renews_at="2026-07-01",
            subscription_remaining_usd=14.0,
            topup_remaining_usd=12.0,
            total_spendable_usd=26.0,
            plan_bar=UsageBar(kind="plan", remaining_usd=14.0, total_usd=20.0, spent_usd=6.0),
            topup_bar=UsageBar(kind="topup", remaining_usd=12.0, total_usd=12.0, spent_usd=0.0),
        )

    if name == "low":
        return UsageModel(
            available=True,
            status="low",
            plan_name="Plus",
            renews_at="2026-07-01",
            subscription_remaining_usd=3.4,
            total_spendable_usd=3.4,
            plan_bar=UsageBar(kind="plan", remaining_usd=3.4, total_usd=20.0, spent_usd=16.6),
        )

    if name == "depleted":
        return UsageModel(
            available=True,
            status="depleted",
            plan_name="Plus",
            renews_at="2026-07-01",
            subscription_remaining_usd=0.0,
            total_spendable_usd=0.0,
            plan_bar=UsageBar(kind="plan", remaining_usd=0.0, total_usd=20.0, spent_usd=20.0),
        )

    return None
