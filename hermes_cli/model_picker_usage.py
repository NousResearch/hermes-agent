"""Plan-usage chrome for the model pickers (CLI radiolist and the gateway ``model.options`` inventory).

On a usage-window plan every listed model is included, so the picker shows how much of each allowance
is used instead of per-token prices. Each priced model row carries the gateway's cost multiplier
against one base model (``usage_multiplier`` on ``/v1/models``, already rounded), shown as ``3×`` /
``0.4×`` (``<0.1×`` below 0.1); $0 rows say ``Free``. The base model is named beside ``data``
(``usage_basis``, shown as ``1× = <name> price``). Hermes
only formats what the gateway sends: it never derives a multiplier or a tier from prices.

Every other account (credit-grant, pay-as-you-go, free tier) gets nothing from here.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

BAR_WIDTH = 10
BAR_FILL = "|"
BAR_EMPTY = " "

GaugeStyle = Literal["green", "yellow", "red"]

# Remaining-fraction thresholds (allowance left). Green under half used, red at 80%+ used.
_YELLOW_BELOW = 0.5
_RED_BELOW = 0.2

# Words and limit lines come from ``plan_usage.*`` in locales/ (``usage_chip_label``, ``basis_line``,
# ``_t``); the `hermes model` setup picker passes ``lang="en"`` (its own text is English, as on main).
# The multiplier sign the chips use (a real multiplication sign, not the letter x).
MULTIPLIER_SIGN = "\u00d7"
# Window pickers split the list here; messaging pickers just order recommended first.
SECTION_RECOMMENDED = "Recommended"
SECTION_OTHER = "Other models"

_BP_FULL = 10_000

# The server's allowance ids Hermes has its own words for (agent/usage_gauges.py); any other id
# renders with the server's label.
GAUGE_TODAY = "day"
GAUGE_WEEK = "week"
GAUGE_MONTH = "month"
GAUGE_TOP_TIER = "top_tier"


@dataclass(frozen=True)
class UsageGauge:
    """One server allowance and when it refills, in the server's order. The wire carries ``left_fraction``;
    every surface draws it as USED (bar fills 0 → 100, same as the portal's "N% used"). ``scope``: which
    models a full one stops (``all`` / ``top_tier``). ``title``: the server's display fallback."""

    label: str            # the server's id
    left_fraction: float
    resets_at_ms: Optional[int] = None
    scope: str = "all"
    title: str = ""

    @property
    def used_fraction(self) -> float:
        return 1.0 - clamp_fraction(self.left_fraction)

    @property
    def style(self) -> GaugeStyle:
        return gauge_style(self.left_fraction)

    @property
    def bar(self) -> str:
        return render_bar(self.used_fraction)

    @property
    def percent_used(self) -> str:
        return format_percent_used(self.used_fraction)


@dataclass(frozen=True)
class PickerUsage:
    """What the pickers need to draw a usage-window plan, derived from the portal account snapshot.
    Window plans only: other accounts (credit-grant, pay-as-you-go, free) get no plan chrome."""

    plan: Optional[str]
    gauges: tuple[UsageGauge, ...]
    paused_reason: Optional[str] = None
    # Window only: our picks, in order; pickers list them first. Every model is usable.
    featured_model_ids: tuple[str, ...] = ()
    # Window only: model ids that draw on nothing (a ``:free`` id or a free catalog row). A full
    # allowance never blocks them. Empty until applied.
    free_ids: frozenset[str] = frozenset()
    # Window only: model ids the gateway bills to a subscription the account holds (``billing_mode:
    # subscription``). They stay in ``free_ids`` (they draw on nothing, so nothing dims them); this set only
    # keeps the word "Free" off them, since that subscription pays. Empty until applied.
    subscription_ids: frozenset[str] = frozenset()
    # Window only: past a full allowance, does the turn bill top-up credits (pay as you go)? True: "Keep
    # going on credits" is on and there is a top-up balance. False: it stops. None: unknown (older portal).
    credits_pay: Optional[bool] = None

    def gauge(self, label: str) -> Optional[UsageGauge]:
        return next((g for g in self.gauges if g.label == label), None)

    def with_pricing(self, pricing: Optional[dict[str, dict[str, Any]]]) -> PickerUsage:
        """A copy that knows which rows of *pricing* (picker pricing shape) are free or subscription-billed."""
        if not pricing:
            return self
        free = frozenset(usage_free_models(list(pricing), pricing))
        subscribed = frozenset(m for m in pricing if is_subscription_billed_row(pricing.get(m)))
        from dataclasses import replace
        return replace(self, free_ids=free, subscription_ids=subscribed)

    def exhausted_gauge(self, model_id: str) -> Optional[UsageGauge]:
        """Window only: the spent all-scope allowance that stops this model (of several, the one that frees up
        last), or None while it still has included room. A free model is never stopped. Same rule as
        agent/plan_limits.py ``blocking_gauge``. A top-tier allowance never dims a row."""
        if model_id in self.free_ids or model_id.endswith(":free"):
            return None
        spent = [g for g in self.gauges_for_model(model_id) if g.left_fraction <= 0.0]
        return max(spent, key=lambda g: g.resets_at_ms or 0) if spent else None

    def blocks(self, model_id: str) -> bool:
        """Window only: a full allowance stops this model and top-up credits won't pay for it, so a picker
        dims the row (the limit line says why, once). Same rule as apps/shared ``planBlocks``."""
        return self.credits_pay is not True and self.exhausted_gauge(model_id) is not None

    def _full_all_gauge(self) -> Optional[UsageGauge]:
        spent = [g for g in self.gauges if g.scope == "all" and g.left_fraction <= 0.0]
        return max(spent, key=lambda g: g.resets_at_ms or 0) if spent else None

    def limit_line(self, lang: Optional[str] = None) -> Optional[str]:
        """One line for the top of a window-plan picker once an allowance is full, saying what happens
        next: ``Weekly limit reached · resets in 21h · paid models now use your credits (pay as you go)``.
        None while nothing is full (and while paused: the plan header already says so)."""
        if self.paused_reason:
            return None
        full = self._full_all_gauge()
        top = None if full is not None else next(
            (g for g in self.gauges if g.scope == GAUGE_TOP_TIER and g.left_fraction <= 0.0), None)
        g = full or top
        if g is None:
            return None
        head = _capitalize(_t("limit_reached", lang, name=_limit_name(g, lang), resets=_resets(g.resets_at_ms, lang)))
        large = "" if full is not None else "_large"
        if self.credits_pay:
            return f"{head} · {_t('then_credits' + large, lang)}"
        if self.credits_pay is False:
            return f"{head} · {_t('then_wait' + large, lang)}"
        return head

    @property
    def included_summary(self) -> str:
        """Header suffix: every row is included, unless an allowance is full (then what happens next) or the
        plan is paused."""
        if self.paused_reason:
            return "included usage paused"
        # Only the `hermes model` setup picker shows this, and its own text is English.
        return self.limit_line(lang="en") or "every model below is included"

    def gauges_for_model(self, model_id: str) -> tuple[UsageGauge, ...]:
        """Every all-scope allowance (the ones that stop every paid model)."""
        return tuple(g for g in self.gauges if g.scope == "all")

    def to_payload(self) -> dict[str, Any]:
        """Wire shape for ``model.options`` (``ModelOptionProvider.plan_usage``)."""
        return {
            "kind": "usage_window",
            "plan": self.plan,
            "gauges": [
                {"label": g.label, "left_fraction": round(g.left_fraction, 4), "style": g.style,
                 "resets_at_ms": g.resets_at_ms, "scope": g.scope, "title": g.title}
                for g in self.gauges
            ],
            # Subscription-billed: still in ``usage_free_models`` (never dimmed), but
            # pickers show no "Free" chip on them (``plan-usage.ts`` ``usageChip``).
            "subscription_model_ids": sorted(self.subscription_ids),
            "featured_model_ids": list(self.featured_model_ids),
            "paused_reason": self.paused_reason,
            "credits_pay": self.credits_pay,
        }


def clamp_fraction(value: float) -> float:
    return 0.0 if value < 0.0 else min(value, 1.0)


def gauge_style(left_fraction: float) -> GaugeStyle:
    """Allowance color: green under half used, yellow from 50%, red once 80%+ is used."""
    left = clamp_fraction(left_fraction)
    if left < _RED_BELOW:
        return "red"
    if left < _YELLOW_BELOW:
        return "yellow"
    return "green"


def _filled_cells(fraction: float, width: int) -> int:
    value = clamp_fraction(fraction)
    filled = round(value * width)
    if value > 0.0 and filled == 0:
        filled = 1
    if value < 1.0 and filled == width:
        filled = width - 1
    return filled


def render_bar(fraction: float, width: int = BAR_WIDTH) -> str:
    """``[||||      ]`` — filled cells are *fraction* (usage gauges pass the USED fraction; cost bars
    their intensity). Any non-zero value shows at least one cell and a non-full one always leaves one
    empty, so "barely used" never reads as untouched and "almost out" never reads as exhausted."""
    filled = _filled_cells(fraction, width)
    return f"[{BAR_FILL * filled}{BAR_EMPTY * (width - filled)}]"


def featured_first(model_ids: list[str], featured: tuple[str, ...] | list[str]) -> list[str]:
    """The plan's featured picks first (plan order, only those listed), then the rest unchanged."""
    head, rest = split_recommended(model_ids, featured)
    return head + rest


def split_recommended(
    model_ids: list[str], featured: tuple[str, ...] | list[str],
) -> tuple[list[str], list[str]]:
    """``(recommended, other)``: the plan's featured picks that are listed (plan order), then every
    other listed model in its catalog order. No featured picks: ``([], model_ids)``."""
    listed = set(model_ids)
    head = [m for m in dict.fromkeys(featured or ()) if m in listed]
    seen = set(head)
    return head, [m for m in model_ids if m not in seen]


def usage_free_models(model_ids: list[str], pricing: Optional[dict[str, dict[str, Any]]]) -> list[str]:
    """The listed models that draw nothing from the window: a ``:free`` id or a row the catalog lists as
    free (subscription-billed rows count: that subscription pays). A full allowance never dims them."""
    from agent.plan_limits import is_free
    return [mid for mid in model_ids if is_free(mid, pricing)]


def is_subscription_billed_row(pricing_row: Any) -> bool:
    """The gateway bills this catalog row to a subscription the account holds (``billing_mode: subscription``)."""
    from hermes_cli.models import _is_subscription_billed
    return _is_subscription_billed(pricing_row)


# A multiplier above this is not a real price ratio: the row shows no chip rather than ``1e+308×``.
MAX_USAGE_MULTIPLIER = 100_000


def usage_multiplier(pricing_row: Any) -> Optional[float]:
    """The gateway's ``usage_multiplier`` on a catalog row (a finite number above 0 and at most
    :data:`MAX_USAGE_MULTIPLIER`, never a bool), or None. Only usage-window callers' catalogs carry it;
    it is already rounded, so this never re-rounds."""
    value = pricing_row.get("usage_multiplier") if isinstance(pricing_row, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # Range first: a huge JSON integer would make math.isfinite raise OverflowError (NaN fails both checks).
    return float(value) if 0 < value <= MAX_USAGE_MULTIPLIER and math.isfinite(value) else None


def format_usage_multiplier(value: float) -> str:
    """``<0.1×`` below 0.1, else the gateway's rounded value as sent (``0.4×``, ``3×``, ``15×``). Twin of
    apps/shared ``formatUsageMultiplier``."""
    if value < 0.1:
        return f"<0.1{MULTIPLIER_SIGN}"
    # One decimal at most (the gateway sends one below 1): a float artefact never prints 0.30000000000000004.
    # Half rounds up, like the TypeScript twin's Math.round (Python's round() would send 0.25 to 0.2).
    number = str(int(value)) if float(value).is_integer() else f"{math.floor(value * 10 + 0.5) / 10:g}"
    return f"{number}{MULTIPLIER_SIGN}"


def free_label(lang: Optional[str] = None) -> str:
    """The ``Free`` chip in the catalog language."""
    return _t("level_free", lang)


def usage_chip_label(model_id: str, pricing: Optional[dict[str, dict[str, Any]]], lang: Optional[str] = None) -> str:
    """The chip a window-plan picker row shows: ``Free`` (wins over any multiplier), ``0.4×`` / ``N×`` from the
    gateway's ``usage_multiplier``, or ``""``. A subscription-billed row gets none: "Free" would read as free
    of charge when that subscription pays, and the gateway sends it no multiplier. Display only."""
    row = (pricing or {}).get(model_id)
    if is_subscription_billed_row(row):
        return ""
    from agent.plan_limits import is_free
    if is_free(model_id, pricing):
        return free_label(lang)
    value = usage_multiplier(row)
    return format_usage_multiplier(value) if value is not None else ""


def usage_basis() -> Optional[dict[str, str]]:
    """``{"model_id", "name"}``: the base model the gateway priced this caller's Nous catalog against (sent
    beside ``data``, never on a row), or None. Cache-only; never raises."""
    try:
        from hermes_cli.models_pricing import nous_usage_basis
        return nous_usage_basis()
    except Exception:
        return None


def basis_line(model_ids: list[str], pricing: Optional[dict[str, dict[str, Any]]], lang: Optional[str] = None,
               basis: Optional[dict[str, str]] = None) -> str:
    """``1× = Model A price`` under a window-plan picker's section headings, named by the gateway: shown
    when the gateway sent a basis (*basis*, else :func:`usage_basis`) and at least one listed row shows a
    multiplier. The basis row itself need not be listed. ``""`` otherwise."""
    basis = basis if basis is not None else usage_basis()
    if not basis or not any(usage_chip_label(m, pricing, lang).endswith(MULTIPLIER_SIGN) for m in model_ids):
        return ""
    return _t("basis_line", lang, model=basis.get("name") or basis.get("model_id") or "")


def is_free_model(pricing_row: Optional[dict[str, Any]]) -> bool:
    """True when the catalog lists the model at $0 in and $0 out (``:free`` / natively free). A row
    with no pricing is NOT free — unknown price is not a promise."""
    from hermes_cli.models_pricing import _price_float
    if not pricing_row:
        return False
    inp = _price_float(pricing_row.get("prompt"), positive=False)
    out = _price_float(pricing_row.get("completion"), positive=False)
    return inp is not None and out is not None and inp <= 0.0 and out <= 0.0


def format_percent_used(used_fraction: float) -> str:
    """Whole-percent used, same rule as the portal (``formatUsedPercentLabel``): floor, any sliver
    reads ``<1%`` (never ``0%``), and ``100%`` only once the allowance is actually exhausted."""
    # round() first: used = 1 - left can land a hair under an exact basis point (0.00999…).
    bp = int(round(clamp_fraction(used_fraction) * _BP_FULL, 6))
    if bp <= 0:
        return "0%"
    if bp < 100:
        return "<1%"
    return f"{bp // 100}%"


def compact_duration(resets_at_ms: Optional[int], now_ms: Optional[int] = None) -> str:
    """``"5h"`` / ``"2d 3h"`` / ``"12m"`` until *resets_at_ms*; ``""`` when past, under a minute or unknown."""
    if not isinstance(resets_at_ms, int):
        return ""
    now = int(time.time() * 1000) if now_ms is None else now_ms
    seconds = max(0, (resets_at_ms - now) // 1000)
    if seconds < 60:
        return ""
    days, rem = divmod(seconds, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h" if hours else f"{days}d"
    return f"{hours}h" if hours else f"{minutes}m"


def format_resets_in(resets_at_ms: Optional[int], now_ms: Optional[int] = None) -> str:
    """``"resets in 5h"`` / ``"resets in 2d 3h"`` / ``"resets soon"`` (past or unknown → soon). English:
    ``/usage`` (untranslated, like main's credits lines); pickers use the catalog's words."""
    d = compact_duration(resets_at_ms, now_ms)
    return f"resets in {d}" if d else "resets soon"


def _t(key: str, lang: Optional[str] = None, **kwargs: Any) -> str:
    from agent.i18n import t
    return t(f"plan_usage.{key}", lang, **kwargs)


def _resets(resets_at_ms: Optional[int], lang: Optional[str] = None) -> str:
    d = compact_duration(resets_at_ms)
    return _t("resets_in", lang, duration=d) if d else _t("resets_soon", lang)


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _limit_name(g: UsageGauge, lang: Optional[str] = None) -> str:
    """``weekly`` for ``Weekly limit reached`` / ``your weekly limit``; an unknown id shows the server's title."""
    if g.label in (GAUGE_TODAY, GAUGE_WEEK, GAUGE_MONTH, GAUGE_TOP_TIER):
        return _t(f"limit_{g.label}", lang)
    return g.title or g.label


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def picker_usage_from_account(info: Any) -> Optional[PickerUsage]:
    """``NousPortalAccountInfo`` (or a dict shaped like one) → PickerUsage, or None when there is nothing to
    gauge: only a usage-window plan has plan chrome (credit-grant, pay-as-you-go and free tier get
    None, as does an unreadable snapshot). Never raises."""
    try:
        if info is None or not _get(info, "logged_in", True):
            return None
        sub = _get(info, "subscription")
        plan = _get(sub, "plan")
        plan = plan if isinstance(plan, str) and plan else None
        window = _get(info, "usage_window")
        if window is None:
            return None
        from agent.usage_gauges import all_scope, parse_gauges
        parsed = parse_gauges(_get(window, "gauges"))
        if not parsed or not all_scope(parsed):
            return None  # same requirement as the tracker seed: an allowance that stops every model
        gauges = [UsageGauge(g.id, 1.0 - g.used_fraction, g.resets_at_ms, g.scope, g.label) for g in parsed]
        featured = tuple(m for m in (_get(window, "featured_model_ids") or ()) if isinstance(m, str) and m)
        paused = _get(window, "paused_reason")
        return PickerUsage(
            plan=plan, gauges=tuple(gauges),
            paused_reason=paused if isinstance(paused, str) and paused else None,
            featured_model_ids=featured,
            credits_pay=_credits_pay(info, window),
        )
    except Exception:
        logging.getLogger(__name__).debug("Plan usage parse failed", exc_info=True)
        return None


def _credits_pay(info: Any, window: Any) -> Optional[bool]:
    """Past a full allowance, top-up credits pay (pay as you go): "Keep going on credits" is not off AND a
    top-up balance is left. Same rule as ``CreditsState.credits_pay``. None when the balance is unknown."""
    if _get(window, "keep_going_on_credits") is False:
        return False
    purchased = _get(_get(info, "paid_service_access_info"), "purchased_credits_remaining")
    if not isinstance(purchased, (int, float)) or isinstance(purchased, bool):
        return None
    return purchased > 0


def nous_picker_usage(*, warm: bool = False, any_age: bool = False) -> Optional[PickerUsage]:
    """PickerUsage from the portal account snapshot. Cache-only by default so a picker never adds a
    round trip; ``warm=True`` fetches once (fills the 60s cache) so a window plan is known
    on a cold start. ``any_age`` reads the last snapshot past the 60s TTL (plan-shape callers only:
    its percents may be stale). Never raises."""
    try:
        from hermes_cli.nous_account import cached_nous_portal_account_info, get_nous_portal_account_info
        if warm and cached_nous_portal_account_info() is None:
            # A known window plan needs a real fetch: the JWT snapshot carries no gauges and never fills the cache.
            from agent.credits_tracker import _window_seed_fields
            on_window = _window_seed_fields(_get(cached_nous_portal_account_info(any_age=True), "usage_window")) is not None
            get_nous_portal_account_info(force_fresh=on_window)
        return picker_usage_from_account(cached_nous_portal_account_info(any_age=any_age))
    except Exception:
        logging.getLogger(__name__).debug("Nous plan usage unavailable", exc_info=True)
        return None


def payload_chip_label(provider_row: Any, model_id: str, lang: Optional[str] = None) -> str:
    """:func:`usage_chip_label` from a ``model.options`` provider row (``usage_free_models`` /
    ``usage_multipliers``, which only a window plan's row carries). ``""`` for every other row."""
    plan = provider_row.get("plan_usage") if isinstance(provider_row, dict) else None
    if not isinstance(plan, dict) or plan.get("kind") != "usage_window":
        return ""
    if model_id in (plan.get("subscription_model_ids") or ()):
        return ""
    if model_id in (provider_row.get("usage_free_models") or ()):
        return free_label(lang)
    multipliers = provider_row.get("usage_multipliers")
    value = usage_multiplier({"usage_multiplier": multipliers.get(model_id)}) if isinstance(multipliers, dict) else None
    return format_usage_multiplier(value) if value is not None else ""


def window_row_labels(provider_row: Any, model_ids: list) -> list[str]:
    """CLI picker rows: ``model-b   3×`` / ``model-free:free   Free`` from the payload's chips
    (:func:`payload_chip_label`), which only a window plan's row carries. Any other row, or anything
    unexpected: the bare ids."""
    ids = [str(m) for m in model_ids]
    tags = [payload_chip_label(provider_row, m) for m in ids]
    if not any(tags):
        return ids
    width = min(max(len(m) for m in ids), 48)
    return [f"{m.ljust(width)}  {tag}".rstrip() for m, tag in zip(ids, tags)]


def picker_row_suffix(provider_row: Any, model_id: str) -> str:
    """Messaging pickers: ``3×`` / ``Free`` for one row, from the labels
    ``model_switch_providers._attach_window_model_notes`` stamps (window plans only); "" otherwise."""
    if not isinstance(provider_row, dict):
        return ""
    return str((provider_row.get("model_labels") or {}).get(model_id) or "")


def _gauge_name(g: UsageGauge) -> str:
    from agent.usage_gauges import WindowGauge, gauge_short
    return gauge_short(WindowGauge(g.label, g.scope, g.title or g.label, 0))


def plan_gauge_lines(usage: PickerUsage) -> list[str]:
    """Stacked plan summary: one line per server allowance, in the server's order, so the picker header
    stays readable at a normal terminal width."""
    from agent.usage_gauges import paused_for_payment
    paused = (" (paused: renewal unpaid)" if paused_for_payment(usage.paused_reason) else " (paused)") if usage.paused_reason else ""
    head = (usage.plan or "Hermes plan") + paused
    label_w = max((len(_gauge_name(g)) for g in usage.gauges), default=4)
    return [head, *(
        f"{_gauge_name(g):<{label_w}}  {g.bar}  {g.percent_used + ' used':<9}  {format_resets_in(g.resets_at_ms)}"
        for g in usage.gauges
    )]


def window_plan_default(model_ids: list[str]) -> str:
    """A usage-window plan's first featured pick that is listed (the picker's top Recommended row), or
    "" for every other account. Cache-only: the caller's fresh tier read just filled the snapshot. Never raises."""
    try:
        usage = nous_picker_usage(any_age=True)
        if usage is None:
            return ""
        recommended, _other = split_recommended(model_ids, usage.featured_model_ids)
        return recommended[0] if recommended else ""
    except Exception:
        logging.getLogger(__name__).debug("Window plan default unavailable", exc_info=True)
        return ""


def annotate_window_plan_rows(provider_data: dict, model_ids: list) -> tuple[list, Optional[str]]:
    """``(labels, limit_line)`` for the /model picker's model stage. On a Nous usage-window plan rows carry
    the TUI's ``3×`` / ``Free`` chips, and with a full allowance ``limit_line`` (above the hint) says it
    once, with what happens next. Every other account: the bare ids and None. Display only."""
    if str((provider_data or {}).get("slug") or "").lower() != "nous":
        return list(model_ids), None
    try:
        usage = nous_picker_usage()  # fresh snapshot only: a stale percent would mislabel the line
        return window_row_labels(provider_data, model_ids), (usage.limit_line() if usage is not None else None)
    except Exception:
        logging.getLogger(__name__).debug("Window plan row labels unavailable", exc_info=True)
        return list(model_ids), None
