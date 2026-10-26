"""Plan-usage chrome for the model pickers (CLI radiolist and the gateway ``model.options`` inventory).

On a usage-window plan every listed model is included, so the picker shows how much of each allowance
is used instead of per-token prices. Each model row carries one word, ``Small`` / ``Medium`` /
``Large`` / ``Free``, that the gateway sends per caller (``usage_level`` on ``/v1/models``).

Every other account (credit-grant, pay-as-you-go, free tier) gets nothing from here.
"""

from __future__ import annotations

import logging
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

# Per-model label on a window plan, as the gateway sends it (``usage_level``). Display words only.
USAGE_LEVELS = ("free", "low", "medium", "top_tier")
# Words, limit lines and switch notices come from ``plan_usage.*`` in locales/ (``usage_level_label``,
# ``_t``); the `hermes model` setup picker passes ``lang="en"`` (its own text is English, as on main).
# One line for that picker, shown only when a Large row is listed.
TOP_TIER_FOOTNOTE = "Large models also use your monthly large-model limit."
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
    # Window only: model ids the gateway labels Large in this caller's ``/v1/models``. Empty until applied.
    top_tier_ids: frozenset[str] = frozenset()
    # Window only: model ids that draw on nothing (the gateway's ``free`` word or a free catalog row). A full
    # allowance never blocks them. Empty until applied.
    free_ids: frozenset[str] = frozenset()
    # Window only: past a full allowance, does the turn bill top-up credits (pay as you go)? True: "Keep
    # going on credits" is on and there is a top-up balance. False: it stops. None: unknown (older portal).
    credits_pay: Optional[bool] = None

    def gauge(self, label: str) -> Optional[UsageGauge]:
        return next((g for g in self.gauges if g.label == label), None)

    def is_top_tier(self, model_id: str) -> bool:
        """The gateway's ``top_tier`` mark. Unmarked: not top tier (fail open)."""
        return model_id in self.top_tier_ids

    def with_pricing(self, pricing: Optional[dict[str, dict[str, Any]]]) -> PickerUsage:
        """A copy that reads the gateway's ``top_tier`` marks from *pricing* (picker pricing shape)."""
        if not pricing:
            return self
        levels = usage_levels(list(pricing), pricing)  # same rule as the tags and agent.plan_limits.is_top_tier
        free = frozenset(m for m, level in levels.items() if level == "free")
        marked = frozenset(m for m, level in levels.items() if level == "top_tier")
        from dataclasses import replace
        return replace(self, top_tier_ids=marked, free_ids=free)

    def exhausted_gauge(self, model_id: str) -> Optional[UsageGauge]:
        """Window only: the spent allowance that stops this model (an all-scope one, or the top-tier one for a
        top-tier model; of several, the one that frees up last), or None while it still has included room. A
        free model is never stopped. Same rule as agent/plan_limits.py ``blocking_gauge``."""
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

    def large_model_notice(self, model_id: str, *, detail: bool = True, lang: Optional[str] = None) -> str:
        """The warning shown when someone switches to a Large model (never a confirm). ``detail`` adds the
        allowance state; pass False for a stale snapshot, whose percents could mislead."""
        g = self.exhausted_gauge(model_id) if detail and not self.paused_reason else None
        if g is not None:
            if self.credits_pay:
                return _t("large_notice_credits", lang)
            key = "large_notice_limit_stops" if self.credits_pay is False else "large_notice_limit"
            return _t(key, lang, name=_limit_name(g, lang), resets=_resets(g.resets_at_ms, lang))
        top = next((g for g in self.gauges if g.scope == GAUGE_TOP_TIER), None) if detail else None
        if top is not None:
            return _t("large_notice_percent", lang, percent=top.percent_used)
        return _t("large_notice", lang)

    @property
    def included_summary(self) -> str:
        """Header suffix: every row is included, unless an allowance is full (then what happens next) or the
        plan is paused."""
        if self.paused_reason:
            return "included usage paused"
        # Only the `hermes model` setup picker shows this, and its own text is English.
        return self.limit_line(lang="en") or "every model below is included"

    def gauges_for_model(self, model_id: str) -> tuple[UsageGauge, ...]:
        """Every all-scope allowance, plus the top-tier ones when the model draws on them."""
        top = self.is_top_tier(model_id)
        return tuple(g for g in self.gauges if g.scope == "all" or (g.scope == "top_tier" and top))

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
            # The gateway's marks (what the desktop's used-up notes read).
            "top_tier_model_ids": sorted(self.top_tier_ids),
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


def usage_levels(model_ids: list[str], pricing: Optional[dict[str, dict[str, Any]]]) -> dict[str, str]:
    """``{model: "free" | "low" | "medium" | "top_tier"}`` from what the gateway stamped on this
    caller's catalog. An older gateway sends no ``usage_level``: then only what it does send is used
    (its ``top_tier`` mark, and a $0/$0 row is ``free``); every other row gets no label."""
    from agent.plan_limits import is_free
    out: dict[str, str] = {}
    for mid in model_ids:
        row = (pricing or {}).get(mid) or {}
        level = row.get("usage_level")
        if is_free(mid, pricing):
            out[mid] = "free"
        elif level in USAGE_LEVELS:
            out[mid] = level
        elif row.get("top_tier") is True:
            out[mid] = "top_tier"
    return out


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


# The levels a picker row names: Large (its own monthly limit) and Free (never runs out). Small / Medium
# only change how fast usage drains, so they carry no tag.
SHOWN_USAGE_LEVELS = ("free", "top_tier")


def usage_level_label(level: Optional[str], lang: Optional[str] = None) -> str:
    """The row tag in the catalog language (``Large`` / ``Free``); "" for every other level."""
    return _t(f"level_{level}", lang) if level in SHOWN_USAGE_LEVELS else ""


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


def _nous_catalog(*, fetch: bool = False) -> dict[str, dict[str, Any]]:
    """This profile's Nous ``/v1/models`` rows: its resident cache, one fetch only when ``fetch``. Never a
    catalog cached for another credential (another profile's plan marks different models Large)."""
    from hermes_cli.models_pricing import get_pricing_for_provider
    rows = get_pricing_for_provider("nous", cached_only=True) or {}
    if not rows and fetch:
        rows = get_pricing_for_provider("nous") or {}
    return rows


def large_model_notice(
    model_id: str, provider: str, *, fetch: bool = False, lang: Optional[str] = None,
) -> Optional[str]:
    """Warning line for a switch to a Large model, or None. Window plans only. The gateway decides "Large"
    (``usage_level: top_tier``; older gateways ``top_tier: true``) and stamps it ONLY on a usage-window
    caller's catalog, so a marked row is itself proof of a window plan. With a plan snapshot the catalog may
    be fetched once (``fetch``); without one only the resident cache is read, so any other account adds no
    network call. Never raises."""
    try:
        if str(provider or "").strip().lower() != "nous" or not model_id:
            return None
        fresh = nous_picker_usage()
        usage = fresh or nous_picker_usage(any_age=True)
        rows = _nous_catalog(fetch=fetch and usage is not None)
        row = rows.get(model_id)
        if not row:
            return None
        if usage_levels([model_id], {model_id: row}).get(model_id) != "top_tier":
            return None
        if usage is None:
            return _t("large_notice", lang)
        return usage.with_pricing({model_id: row}).large_model_notice(model_id, detail=fresh is not None, lang=lang)
    except Exception:
        logging.getLogger(__name__).debug("Large-model notice unavailable", exc_info=True)
        return None


def window_row_labels(provider_row: Any, model_ids: list) -> list[str]:
    """CLI picker rows: ``model-b   Large`` from the payload's ``usage_levels``, which only a window plan's
    row carries (the TUI's rule: Large / Free only). Any other row, or anything unexpected: the bare ids."""
    ids = [str(m) for m in model_ids]
    plan = provider_row.get("plan_usage") if isinstance(provider_row, dict) else None
    levels = provider_row.get("usage_levels") if isinstance(plan, dict) and plan.get("kind") == "usage_window" else None
    tags = [usage_level_label(levels.get(m)) for m in ids] if isinstance(levels, dict) else []
    if not any(tags):
        return ids
    width = min(max(len(m) for m in ids), 48)
    return [f"{m.ljust(width)}  {tag}".rstrip() for m, tag in zip(ids, tags)]


def picker_row_suffix(provider_row: Any, model_id: str) -> str:
    """Messaging pickers: ``Large`` / ``Free`` for one row, from the labels
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


def window_plan_default(model_ids: list[str], pricing: Optional[dict] = None) -> str:
    """A usage-window plan's first featured pick that is listed (the picker's top Recommended row), or
    "" for every other account. Cache-only: the caller's fresh tier read just filled the snapshot. Never raises."""
    try:
        usage = nous_picker_usage(any_age=True)
        if usage is None:
            return ""
        recommended, _other = split_recommended(model_ids, usage.featured_model_ids)
        rows = pricing or {}
        # A default must not spend the Large allowance: skip Large picks (a user can still choose one).
        usable = [mid for mid in recommended if not (rows.get(mid) or {}).get("top_tier")
                  and (rows.get(mid) or {}).get("usage_level") != "top_tier"]
        return usable[0] if usable else ""
    except Exception:
        logging.getLogger(__name__).debug("Window plan default unavailable", exc_info=True)
        return ""


def annotate_window_plan_rows(provider_data: dict, model_ids: list) -> tuple[list, Optional[str]]:
    """``(labels, limit_line)`` for the /model picker's model stage. On a Nous usage-window plan rows carry
    the TUI's ``Large`` / ``Free`` tags, and with a full allowance ``limit_line`` (above the hint) says it
    once, with what happens next. Every other account: the bare ids and None. Display only."""
    if str((provider_data or {}).get("slug") or "").lower() != "nous":
        return list(model_ids), None
    try:
        usage = nous_picker_usage()  # fresh snapshot only: a stale percent would mislabel the line
        return window_row_labels(provider_data, model_ids), (usage.limit_line() if usage is not None else None)
    except Exception:
        logging.getLogger(__name__).debug("Window plan row labels unavailable", exc_info=True)
        return list(model_ids), None
