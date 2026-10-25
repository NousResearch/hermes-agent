"""Nous credits: parse ``x-nous-credits-*`` / ``x-nous-tool-pool-*`` response
headers into a validated CreditsState (depletion = paid_access, subscription-cap
used_fraction, warn-once schema-version gating) and drive the notice policy.
Header contract: see ``_HEADER_FIELDS``. Money is micros ints only; ``*_usd``
strings are preserved verbatim (never re-parsed to float)."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional

from utils import is_truthy_value

from agent.usage_gauges import (
    BP_FULL,
    SCOPE_TOP_TIER,
    WindowGauge,
    all_scope,
    binding_gauge,
    gauge_limit_name,
    parse_gauges,
    paused_for_payment,
    top_tier_gauge,
)

logger = logging.getLogger(__name__)

_version_warning_emitted: bool = False  # warn-once latch (per process)
_VALID_DENOMINATOR_KINDS = frozenset({"subscription_cap", "none", "usage_window"})
_USD_RE = re.compile(r"^-?\d+\.\d{2}$")  # optional minus, digits, exactly 2 decimals
_SENTINEL = object()  # "parse failed"
# Usage-window plans: the allowances are the server's gauge list
# (agent/usage_gauges.py), percents in basis points. Which allowances exist, their
# order and which models each stops all come from the server.
_VALID_SPEND_SOURCES = frozenset({"included", "payg"})


def _safe_int(value: Any) -> Any:
    """Exact int (money-safe) or ``_SENTINEL``. ``int()`` directly, NOT ``int(float())``
    (precision loss above 2**53 corrupts money); float-shaped strings fail."""
    try:
        return _SENTINEL if value is None else int(str(value))
    except (TypeError, ValueError):
        return _SENTINEL


@dataclass
class CreditsState:
    """Credits state parsed from x-nous-credits-* response headers."""

    version: int = 0
    remaining_micros: int = 0
    remaining_usd: str = ""
    subscription_micros: int = 0  # SIGNED — the ONLY field allowed negative (debt)
    subscription_usd: str = ""
    subscription_limit_micros: Optional[int] = None  # PAIRED + OPTIONAL (only when subscription_cap)
    subscription_limit_usd: Optional[str] = None
    rollover_micros: int = 0
    purchased_micros: int = 0
    purchased_usd: str = ""
    tool_pool_micros: int = 0
    tool_pool_gated_off: bool = False
    denominator_kind: str = "none"  # "subscription_cap" | "none" | "usage_window"
    paid_access: bool = True  # depletion keys off THIS == False, NEVER remaining==0
    disabled_reason: Optional[str] = None  # header omitted entirely when null
    as_of_ms: int = 0
    captured_at: float = 0.0  # time.time() when captured
    from_header: bool = False  # True only when populated by parse_credits_headers()
    # ── usage_window only (empty / defaults otherwise): the server's allowances, in display order.
    gauges: tuple[WindowGauge, ...] = ()
    spend_source: Optional[str] = None  # "included" | "payg" — which wallet the LAST request drew on
    # An admitted reply the server did not meter (no spend source: a skipped guard). A full gauge on it is no
    # refusal, so it raises no limit stop; the next metered reply or 402 decides.
    unmetered: bool = False
    paused_reason: Optional[str] = None  # e.g. "payment_past_due": the window is withheld, not spent
    auto_top_up_enabled: bool = False  # seed-only today (the header set does not carry it)
    # The account setting "Keep going on credits". False = a used-up
    # allowance stops (402) even with a balance; None = a server that predates it (top-up is spent).
    keep_going_on_credits: Optional[bool] = None

    @property
    def credits_pay(self) -> bool:
        """Top-up credits will cover Hermes usage once the included allowance is used up."""
        return self.purchased_micros > 0 and self.keep_going_on_credits is not False

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.captured_at if self.has_data else float("inf")

    @property
    def depleted(self) -> bool:
        """``paid_access == False`` ONLY — ``remaining_micros == 0`` is a false positive
        when the balance is zero but access is live (renewal pending)."""
        return not self.paid_access

    @property
    def is_usage_window(self) -> bool:
        """A usage-window plan with at least one allowance that stops every paid model."""
        return self.denominator_kind == "usage_window" and bool(all_scope(self.gauges))

    @property
    def binding_gauge(self) -> Optional[WindowGauge]:
        """The all-scope allowance that binds (see :func:`usage_gauges.binding_gauge`). Usage window only."""
        return binding_gauge(self.gauges) if self.is_usage_window else None

    @property
    def top_tier_gauge(self) -> Optional[WindowGauge]:
        return top_tier_gauge(self.gauges) if self.is_usage_window else None

    @property
    def used_fraction(self) -> Optional[float]:
        """Fraction of the cap consumed in [0.0, 1.0]; None without a computable denominator.
        Credit-grant: guarded on the LIMIT FIELD (the real denominator), not ``denominator_kind``.
        Usage window: the binding all-scope allowance (whichever stops every model first)."""
        if self.is_usage_window:
            return self.binding_gauge.used_fraction  # type: ignore[union-attr]
        lim = self.subscription_limit_micros
        if not isinstance(lim, int) or lim <= 0:
            return None
        return max(0.0, min(1.0, (lim - self.subscription_micros) / lim))

    @property
    def window_resets_at_ms(self) -> Optional[int]:
        """Reset instant of the binding allowance."""
        g = self.binding_gauge
        return g.resets_at_ms if g is not None else None

    @property
    def top_tier_resets_at_ms(self) -> Optional[int]:
        g = self.top_tier_gauge
        return g.resets_at_ms if g is not None else None

    @property
    def top_tier_used_bp(self) -> Optional[int]:
        g = self.top_tier_gauge
        return g.used_bp if g is not None else None

    @property
    def window_exhausted(self) -> bool:
        """An allowance that stops every paid model is full. Paused (payment) is NOT exhausted: nothing was spent."""
        return self.is_usage_window and any(g.full for g in all_scope(self.gauges))

    @property
    def top_tier_exhausted(self) -> bool:
        """The top-tier allowance is full (a percent, like the others — not a blocked flag)."""
        g = self.top_tier_gauge
        return g is not None and g.full


# ── Credits policy constants. Switching notices sticky→TTL later also needs a
# paired *_TTL_MS per notice kind (AgentNotice has the field; not plumbed yet).
CREDITS_NOTICE_KIND = "sticky"      # v1: credits notices are sticky
CREDITS_RESTORED_TTL_MS = 8000     # the only TTL notice in v1 (depletion-recovery confirmation)
# Usage-gauge bands (ascending): (threshold_fraction, level, label_pct). One
# escalating line shows the HIGHEST band reached; climbing replaces it, recovery steps down.
CREDITS_USAGE_BANDS: tuple[tuple[float, str, int], ...] = ((0.50, "info", 50), (0.75, "warn", 75), (0.90, "warn", 90))
CREDITS_USAGE_KEY = "credits.usage"
# Min subscription balance counting as "grant not yet spent" for the grant_spent
# gate. 1¢: portal-seeded states (float dollars → micros) can carry sub-cent residue
# where headers report 0 — without the floor a seed opens the gate and the first
# header re-creates the at-open nag.
GRANT_UNSPENT_MIN_MICROS = 10_000


# Usage-window notice keys. ``credits.window_payg`` is the one-shot "now spending top-up credits" line;
# ``credits.depleted`` is reused for the hard stop so the TUI/desktop native-notification mapping
# (keyed on ``credits.depleted`` / ``credits.restored``) keeps working without a client change.
CREDITS_WINDOW_PAYG_KEY = "credits.window_payg"
CREDITS_DEPLETED_KEY = "credits.depleted"
# A window plan's limit line (day / week / month / top-tier used up, credits won't pay). Its own key so the
# desktop can leave it to the refused turn's error card, which says the same sentence with buttons; the
# CLI and TUI show it like any other notice. Payment-paused and depleted stay on ``credits.depleted``.
CREDITS_WINDOW_LIMIT_KEY = "credits.window_limit"
CREDITS_RESTORED_KEY = "credits.restored"
CREDITS_GRANT_SPENT_KEY = "credits.grant_spent"
_DEPLETED_TEXT = "✕ Credit access paused · run /topup to top up"
_PAUSED_TEXT = "✕ Included usage paused · your subscription payment failed · update your card, or add credits in /topup to keep going"
_PAUSED_OTHER_TEXT = "✕ Included usage paused · see Billing in the Portal, or add credits in /topup to keep going"
_TOP_TIER_LIMIT = "You've hit your monthly large-model limit"
_AUTO_TOP_UP_SUFFIX = " · auto top-up is on"


def new_credits_latch() -> dict:
    """Fresh notice latch for :func:`evaluate_credits_notices`; every producer builds it here so a new gate key lands everywhere."""
    return {"active": set(), "seen_below_90": False, "usage_band": None, "seen_grant_unspent": False, "texts": {}}


def format_window_reset(resets_at_ms: Optional[int], now_ms: Optional[int] = None) -> str:
    """Local-time reset for notices: ``HH:MM`` within 24h, ``Tue HH:MM`` within a week, ``Nov 3`` beyond (the
    monthly large-model allowance), ``shortly`` once passed (the server rolls the clock on the next request, so a stale
    stamp is never contradicted)."""
    if not isinstance(resets_at_ms, int) or resets_at_ms <= 0:
        return "the next reset"
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if resets_at_ms <= now:
        return "shortly"
    try:
        stamp = time.localtime(resets_at_ms / 1000)
    except (ValueError, OverflowError, OSError):
        return "the next reset"
    left = resets_at_ms - now
    if left >= 7 * 24 * 3600 * 1000:
        return f"{time.strftime('%b', stamp)} {stamp.tm_mday}"
    return time.strftime("%H:%M" if left < 24 * 3600 * 1000 else "%a %H:%M", stamp)


def _resets(resets_at_ms: Optional[int]) -> str:
    """`` · resets 15:56``; empty when the instant is unknown."""
    return f" · resets {format_window_reset(resets_at_ms)}" if isinstance(resets_at_ms, int) and resets_at_ms > 0 else ""


def _spent_window(state: CreditsState) -> tuple[str, Optional[int]]:
    """The binding allowance's limit name (``daily``; an unknown id: the server's label) and its reset."""
    g = state.binding_gauge
    return (gauge_limit_name(g), g.resets_at_ms) if g is not None else ("daily", None)


# A window plan's approach warning: once per allowance per reset, at this fraction (the 50 / 75 bands are
# noise on a plan whose gauges reset daily). The percent always comes from the server.
WINDOW_APPROACH_FRACTION = 0.90
CREDITS_WINDOW_APPROACH_PREFIX = "credits.window."
WINDOW_APPROACH_TTL_MS = 10_000


def window_approach_key(gauge_id: str) -> str:
    """``credits.window.<gauge id>``: one key per allowance, so a client replaces it in place."""
    return CREDITS_WINDOW_APPROACH_PREFIX + gauge_id


def _window_approach_text(g: WindowGauge) -> str:
    """``⚠ 90% of your weekly limit used · resets Thu 09:00``."""
    pct = min(99, int(g.used_fraction * 100))
    return f"⚠ {pct}% of your {gauge_limit_name(g)} limit used{_resets(g.resets_at_ms)}"


def _stop_gauge(state: CreditsState, *, model_is_top_tier: Optional[bool]) -> Optional[WindowGauge]:
    """The full allowance behind the limit line, by :func:`agent.plan_limits.blocking_gauge` (the rule
    ``limit_hit`` reads, so the line, the retry time and the card agree). A full top-tier allowance counts only on
    a refusal: the API stamps ``spend-source`` on every admitted request and never on a 402, so header-sourced
    + no source = this model was refused; an ``included`` source means the model is not top-tier and a seed
    cannot know the model. A refusal no all-scope allowance explains is the top-tier one (a cold catalog).
    ``model_is_top_tier=None`` (tier unknown) counts the top-tier allowance as one that may stop the model, so
    of several full ones the later reset wins and no retry is promised before the model can run."""
    if not state.is_usage_window:
        return None
    from agent.plan_limits import blocking_gauge
    refused = state.from_header and state.spend_source is None
    g = blocking_gauge("", state.gauges, free=False, top_tier=refused and model_is_top_tier is not False)
    if g is None and refused and state.top_tier_exhausted:
        return state.top_tier_gauge
    return g


def _window_hard_stop(state: CreditsState, *, model_is_free: bool, model_is_top_tier: Optional[bool] = False) -> Optional[str]:
    """Blocking copy for a usage-window account, or None. ``paid_access`` is the portal's word (dunning);
    then: payment paused with no credits (credits keep a paused plan going whatever the setting says; a
    card update restores included usage), the large-model allowance used with no pay path, the window used with no
    pay path. A free model keeps inferencing, so only the payment-paused line survives it.
    The credits step comes from :func:`agent.plan_limits.credits_action`, the rule every limit surface reads."""
    purchased = state.purchased_micros
    if state.paused_reason and purchased <= 0:
        return _PAUSED_TEXT if paused_for_payment(state.paused_reason) else _PAUSED_OTHER_TEXT
    if not state.paid_access:
        return None if model_is_free else _DEPLETED_TEXT
    if model_is_free or state.credits_pay:
        return None
    if state.unmetered:
        return None
    from agent.plan_limits import credits_action
    action = _CREDITS_STEP[credits_action(state.keep_going_on_credits, purchased)]
    g = _stop_gauge(state, model_is_top_tier=model_is_top_tier)
    if g is None:
        return None
    if g.scope == SCOPE_TOP_TIER:
        auto = _AUTO_TOP_UP_SUFFIX if state.auto_top_up_enabled and action == _ADD_CREDITS_ACTION else ""
        return f"✕ {_TOP_TIER_LIMIT}{_resets(g.resets_at_ms)} · {action}{auto}"
    return f"✕ You've hit your {gauge_limit_name(g)} limit{_resets(g.resets_at_ms)} · {action}"


def window_stop_resets_at_ms(state: CreditsState, *, model_is_top_tier: Optional[bool] = False) -> Optional[int]:
    """When the allowance behind :func:`_window_hard_stop`'s limit line frees up (the same gauge)."""
    if state.unmetered:
        return None
    g = _stop_gauge(state, model_is_top_tier=model_is_top_tier)
    return g.resets_at_ms if g is not None else None


# "Keep going on credits": the account setting (/topup or Portal billing), off by default on a window plan.
_ADD_CREDITS_ACTION = "/topup to add credits"
_CREDITS_STEP = {  # keyed by agent.plan_limits.credits_action
    "use_credits": "turn on Keep going in /topup",
    "add_credits": _ADD_CREDITS_ACTION,
    "add_credits_and_keep_going": "add credits and turn on Keep going in /topup",
}


def _window_payg_text(state: CreditsState) -> Optional[str]:
    """The one-shot PAYG line: the last request drew on top-up credits instead of the window. None otherwise."""
    if not (state.spend_source == "payg" and state.purchased_micros > 0):
        return None
    if state.window_exhausted:
        name, resets = _spent_window(state)
        text = f"• You've hit your {name} limit{_resets(resets)} · now using credits"
    elif state.top_tier_exhausted:
        text = f"• {_TOP_TIER_LIMIT}{_resets(state.top_tier_resets_at_ms)} · large models now use credits"
    else:
        text = "• Now using credits"
    return text + (_AUTO_TOP_UP_SUFFIX if state.auto_top_up_enabled else "")


def _reconcile_sticky(key: str, text: Optional[str], level: str, latch: dict, to_show: list, to_clear: list) -> None:
    """Show / replace / clear one sticky line by key: a text change re-emits (clear first), None clears."""
    active, texts = latch["active"], latch.setdefault("texts", {})
    if text is not None and (key not in active or texts.get(key) != text):
        if key in active:
            to_clear.append(key)
        to_show.append(_sticky_notice(text, level, key))
        active.add(key)
        texts[key] = text
    elif text is None and key in active:
        to_clear.append(key)
        active.discard(key)
        texts.pop(key, None)


def _evaluate_window_notices(state: CreditsState, latch: dict, *, model_is_free: bool,
                             model_is_top_tier: bool = False) -> tuple[list, list]:
    """Notice policy for a usage-window account (see :func:`evaluate_credits_notices` for the contract).

    Approach: each allowance warns once at 90% per reset (``credits.window.<id>``, a TTL heads-up; the
    limit line takes over when it runs out). The top-tier allowance warns only while the session is on a
    top-tier model (no other model moves it); a free model warns about nothing. grant_spent never fires (no
    grant); a payment pause / depleted account reuses ``credits.depleted`` and its ``credits.restored``."""
    to_show: list = []
    to_clear: list = []
    active = latch["active"]
    warned: dict = latch.setdefault("window_warned", {})
    if CREDITS_USAGE_KEY in active:  # the credit-grant band does not apply to a window plan
        _reconcile_sticky(CREDITS_USAGE_KEY, None, "info", latch, to_show, to_clear)
        latch["usage_band"] = None
    for g in state.gauges:
        relevant = not model_is_free and (g.scope != SCOPE_TOP_TIER or model_is_top_tier)
        if relevant and not g.full and g.used_fraction >= WINDOW_APPROACH_FRACTION \
                and warned.get(g.id) != g.resets_at_ms:
            # A heads-up, not a state: shown once (TTL) and never re-shown for the same reset.
            warned[g.id] = g.resets_at_ms
            key = window_approach_key(g.id)
            to_show.append(AgentNotice(text=_window_approach_text(g), level="warn", kind="ttl",
                                       ttl_ms=WINDOW_APPROACH_TTL_MS, key=key, id=key))
    if CREDITS_GRANT_SPENT_KEY in active:  # a plan flip mid-session: the grant line has no meaning here
        to_clear.append(CREDITS_GRANT_SPENT_KEY)
        active.discard(CREDITS_GRANT_SPENT_KEY)

    stop = _window_hard_stop(state, model_is_free=model_is_free, model_is_top_tier=model_is_top_tier)
    limit = stop is not None and stop not in (_PAUSED_TEXT, _PAUSED_OTHER_TEXT, _DEPLETED_TEXT)
    was_stopped = CREDITS_DEPLETED_KEY in active or CREDITS_WINDOW_LIMIT_KEY in active
    _reconcile_sticky(CREDITS_WINDOW_LIMIT_KEY, stop if limit else None, "error", latch, to_show, to_clear)
    _reconcile_sticky(CREDITS_DEPLETED_KEY, None if limit else stop, "error", latch, to_show, to_clear)
    if was_stopped and stop is None and state.paid_access and not model_is_free:
        # Genuine recovery (a reset, a top-up, a paid renewal) — a free-model switch is not "restored".
        to_show.append(AgentNotice(text="✓ Credit access restored", level="success", kind="ttl",
                                   ttl_ms=CREDITS_RESTORED_TTL_MS, key=CREDITS_RESTORED_KEY, id=CREDITS_RESTORED_KEY))
    _reconcile_sticky(CREDITS_WINDOW_PAYG_KEY, None if stop else _window_payg_text(state), "warn", latch, to_show, to_clear)
    return (to_show, to_clear)


@dataclass
class AgentNotice:
    """Driver-agnostic out-of-band notice (``AIAgent.notice_callback`` / ``notice_clear_callback``); each driver
    renders its own way. ``kind``/``ttl_ms`` stay expressive so a future config can switch v1's sticky notices to TTL."""

    text: str
    level: str = "info"            # info | warn | error | success
    kind: str = "sticky"           # sticky | ttl
    ttl_ms: Optional[int] = None   # honored only when kind == "ttl"
    key: Optional[str] = None      # dedupe / fired-once-latch / clear key
    id: Optional[str] = None


def _sticky_notice(text: str, level: str, key: str) -> AgentNotice:
    return AgentNotice(text=text, level=level, kind=CREDITS_NOTICE_KIND, key=key, id=key)


def _is_nous_welcome_route(base_url: str) -> bool:
    """True when *base_url* is the Nous welcome host, which serves only the free tier. Local data only;
    False wherever the free tier is not built in. The host is the evidence, not the model name: the paid
    inference host can serve ``nous/welcome`` to a named account, and that account's depletion is real."""
    try:
        from hermes_cli.anon_auth import route_is_welcome_host
    except ImportError:
        return False
    return route_is_welcome_host(base_url)


def is_free_tier_model(model: str, base_url: str = "") -> bool:
    """True when *model* is a Nous free-tier model, using ONLY local data: (1) ``:free`` suffix — canonical
    Nous free SKU marker; (2) ``stealth/`` prefix — stealth-preview SKUs are free without the suffix
    (naming-convention trust: a PAID ``stealth/`` model would wrongly suppress the banner); (3) a PEEK into
    ``hermes_cli.models``' pricing cache (filled by the model picker; a miss never fetches). Fail-open to
    False (depleted notice still shows): a wrong warning is recoverable noise; hiding it masks a real block."""
    if not model:
        return False
    if model.endswith(":free") or model.startswith("stealth/"):
        return True
    if not base_url:
        return False
    # (4) the Nous free tier: the welcome host serves only the free tier. A free-tier identity carries $0
    # by design, so the portal seed reports paid_access=False for it; that is not a depleted account, and
    # "run /topup" means nothing to it. Local data only, same as the rules above.
    if _is_nous_welcome_route(base_url):
        return True
    try:
        from hermes_cli.models import _is_model_free
        from hermes_cli.models_pricing import peek_cached_pricing

        pricing = peek_cached_pricing(base_url)  # owns the /v1-suffix and auth-state key details
        return bool(pricing) and _is_model_free(model, pricing)
    except Exception:
        return False


def evaluate_credits_notices(state: CreditsState, latch: dict, *, model_is_free: bool = False,
                             model_is_top_tier: bool = False) -> tuple[list[AgentNotice], list[str]]:
    """Reconcile credits notices against the latch (see :func:`new_credits_latch`); mutates ``latch`` IN
    PLACE. Pure — no I/O, no agent/run_agent imports. ``model_is_free`` suppresses ``credits.depleted`` (a
    depleted account on a free model keeps inferencing) WITHOUT emitting "restored" — that fires only on a
    genuine ``paid_access`` flip back to True. Returns ``(to_show, to_clear)``; caller emits to_clear FIRST.
    A usage-window state (day/week percents, no grant) takes the window policy; everything else is unchanged."""
    if state.is_usage_window:
        return _evaluate_window_notices(state, latch, model_is_free=model_is_free,
                                        model_is_top_tier=model_is_top_tier)
    to_show: list[AgentNotice] = []
    to_clear: list[str] = []
    uf, active = state.used_fraction, latch["active"]
    for window_key in (CREDITS_WINDOW_PAYG_KEY, CREDITS_WINDOW_LIMIT_KEY):
        if window_key in active:  # only ever set by the window policy; a stale line after a plan change
            to_clear.append(window_key)
            active.discard(window_key)
            latch.get("texts", {}).pop(window_key, None)
    # Crossing latch: band notices fire only once uf was observed below the LOWEST
    # band, so a session opening mid-range doesn't fire on its first observation
    # (the cold-start seed primes this when it WANTS an open-high warning).
    if uf is not None and uf < CREDITS_USAGE_BANDS[0][0]:
        latch["seen_below_90"] = True
    # Grant-spent gate: fires only after this session OBSERVED the grant unspent
    # (≥1¢). Opening at grant-spent is a steady STATE (/usage carries it), not an
    # event. Unlike seen_below_90, seeds must NOT prime this gate.
    if uf is not None and uf < 1.0 and state.subscription_micros >= GRANT_UNSPENT_MIN_MICROS:
        latch["seen_grant_unspent"] = True

    # Highest band reached (ascending → last match wins); None below all. Top-up
    # suppression: with purchased credits the cap gauge is the wrong denominator
    # ("90% used" on $50 of top-up is noise; it used to stick PERMANENTLY beside
    # grant_spent at >=100%) — grant_spent covers the cap-reached case, and a
    # mid-session top-up flips current_band → None so the clear path removes the line.
    current_band: Optional[tuple[float, str, int]] = None
    if uf is not None and state.purchased_micros <= 0:
        current_band = next((b for b in reversed(CREDITS_USAGE_BANDS) if uf >= b[0]), None)

    # ── usage gauge: highest crossed band only; replace on band change (climb or
    # step-down); clear below the lowest band or when the denominator vanishes.
    target_band = current_band[2] if (current_band and latch["seen_below_90"]) else None
    if target_band != latch.get("usage_band"):
        if CREDITS_USAGE_KEY in active:
            to_clear.append(CREDITS_USAGE_KEY)
            active.discard(CREDITS_USAGE_KEY)
        if target_band is not None:
            # Absolute dollars used (a bare "N%" is only meaningful against a Nous cap): cap − remaining,
            # clamped [0, cap]; "$?" if a producer set the limit without its *_usd. Re-emits on band change only.
            level = current_band[1]  # type: ignore[index]  (current_band set when target_band set)
            lim = state.subscription_limit_micros or 0
            used_usd = f"{max(0, min(lim, lim - state.subscription_micros)) / 1_000_000:.2f}" if lim else "?"
            text = f"{'⚠' if level == 'warn' else '•'} You've used ${used_usd} of your ${state.subscription_limit_usd or '?'} cap"
            to_show.append(_sticky_notice(text, level, CREDITS_USAGE_KEY))
            active.add(CREDITS_USAGE_KEY)
        latch["usage_band"] = target_band

    # ── grant_spent: the gate guards only the SHOW and is CONSUMED by it — one
    # announcement per crossing. A header flicker (uf → None → 1.0) clears the
    # line but cannot re-announce; only a renewal re-opening the gate (fresh ≥1¢
    # observation) arms the next. .get(): default closed for hand-built latches.
    grant_cond = (
        state.denominator_kind == "subscription_cap" and uf is not None and uf >= 1.0 and state.purchased_micros > 0
    )
    if grant_cond and "credits.grant_spent" not in active and latch.get("seen_grant_unspent", False):
        to_show.append(_sticky_notice(f"• Grant spent · ${state.purchased_usd} top-up left", "info", "credits.grant_spent"))
        active.add("credits.grant_spent")
        latch["seen_grant_unspent"] = False
    elif "credits.grant_spent" in active and not grant_cond:
        to_clear.append("credits.grant_spent")
        active.discard("credits.grant_spent")

    # ── depleted: suppressed while the model is free (inference still works).
    depleted_cond = not state.paid_access
    show_depleted = depleted_cond and not model_is_free
    if show_depleted and "credits.depleted" not in active:
        to_show.append(_sticky_notice("✕ Credit access paused · run /topup to top up", "error", "credits.depleted"))
        active.add("credits.depleted")
    elif "credits.depleted" in active and not show_depleted:
        to_clear.append("credits.depleted")
        active.discard("credits.depleted")
        if not depleted_cond:  # genuine recovery only — a free-model switch while depleted is NOT "restored"
            to_show.append(AgentNotice(
                text="✓ Credit access restored", level="success", kind="ttl",
                ttl_ms=CREDITS_RESTORED_TTL_MS, key="credits.restored", id="credits.restored",
            ))
    return (to_show, to_clear)


# Header contract: (field, kind[, default-when-absent]); a field is REQUIRED unless it has a default.
# Header name = ``x-nous-credits-<field>`` (``x-nous-<field>`` for tool_pool_*), underscores → dashes.
# micros: int >= 0 ("signed": may be negative; "bp": 0..10000); usd: the server's formatted string ^-?\d+\.\d{2}$
# (never re-parsed); bool: "true"/"false" STRING. Handled inline: subscription-limit-* (PAIRED/optional),
# denominator-kind ("subscription_cap" | "none"), disabled-reason (omitted when null).
_HEADER_FIELDS: tuple[tuple, ...] = (
    ("remaining_micros", "micros"), ("subscription_micros", "signed"), ("rollover_micros", "micros"),
    ("purchased_micros", "micros"), ("as_of_ms", "micros"), ("tool_pool_micros", "micros", 0),
    ("remaining_usd", "usd"), ("subscription_usd", "usd"), ("purchased_usd", "usd"),
    ("paid_access", "bool", True),  # absent → fail-open (assume access)
    ("tool_pool_gated_off", "bool", False),
)
# Credit-grant plan fields. Required on a credit note; a usage_window note may omit them (the server can
# retire them with legacy plans). Malformed is still a miss. The top-up balance (purchased_*) is optional on a
# window note too: a refused (402) request carries only the window headers; the capture keeps the last balance.
_GRANT_ONLY_FIELDS = frozenset({
    "remaining_micros", "remaining_usd", "subscription_micros", "subscription_usd", "rollover_micros", "as_of_ms",
})
_WINDOW_OPTIONAL_FIELDS = _GRANT_ONLY_FIELDS | {"purchased_micros", "purchased_usd"}


# Usage-window headers, read only when ``denominator-kind`` is ``usage_window``: the gauge list
# (``x-nous-credits-window-gauges``, REQUIRED: a note that names the kind but has no readable list is a
# miss), plus the optional ``-spend-source`` / ``-window-paused-reason`` / ``-keep-going-on-credits``.
_GAUGES_HEADER = "x-nous-credits-window-gauges"


def _header_name(field: str) -> str:
    return "x-nous-" + ("" if field.startswith("tool_pool_") else "credits-") + field.replace("_", "-")


def _parse_window_fields(lowered: Mapping[str, str]) -> Optional[dict[str, Any]]:
    """Usage-window fields from lowercased headers; None when the gauge list is absent or unreadable.
    ``spend_source`` outside its vocabulary reads as absent (fail-open)."""
    gauges = parse_gauges(lowered.get(_GAUGES_HEADER))
    if gauges is None or not all_scope(gauges):
        return None
    out: dict[str, Any] = {"gauges": gauges}
    source = lowered.get("x-nous-credits-spend-source")
    out["spend_source"] = source if source in _VALID_SPEND_SOURCES else None
    paused = lowered.get("x-nous-credits-window-paused-reason")
    out["paused_reason"] = paused.strip() if isinstance(paused, str) and paused.strip() else None
    keep_going = (lowered.get("x-nous-credits-keep-going-on-credits") or "").strip().lower()
    out["keep_going_on_credits"] = {"on": True, "off": False}.get(keep_going)  # anything else: unknown (older server)
    return out


def _parse_field(kind: str, raw: Optional[str], default: Any = _SENTINEL) -> Any:
    """One header value → field value; ``default`` when absent, ``_SENTINEL`` on a contract violation."""
    if raw is None:
        return default
    if kind in ("micros", "signed", "bp"):
        val = _safe_int(raw)
        if val is _SENTINEL or (kind != "signed" and val < 0):
            return _SENTINEL
        # bp: basis points of a window, 0..10000 inclusive (anything past is a violation).
        return _SENTINEL if kind == "bp" and val > BP_FULL else val
    if kind == "usd":
        return raw if _USD_RE.match(raw) else _SENTINEL
    flag = raw.strip().lower()
    return _SENTINEL if flag not in ("true", "false") else flag == "true"


def parse_credits_headers(headers: Mapping[str, str], provider: str = "") -> Optional[CreditsState]:
    """Parse x-nous-credits-* (and x-nous-tool-pool-*) headers into a CreditsState.
    None (miss) on ANY of: no version header; version != 1 (> 1 also warns once);
    a required field violating ``_HEADER_FIELDS``; unknown ``denominator_kind``;
    any unexpected exception. Fail-open on the subscription_limit pair: a
    half-pair (only -micros or only -usd) parses as both-absent (both None)."""
    global _version_warning_emitted
    try:
        # Cheap probe before the lowercase copy (header names are case-insensitive): bail when the
        # version header is absent — the hot path for non-Nous providers.
        if not any(k.lower() == "x-nous-credits-version" for k in headers):
            return None
        lowered = {k.lower(): v for k, v in headers.items()}
        version_val = _safe_int(lowered.get("x-nous-credits-version"))
        if version_val is _SENTINEL:
            return None
        if version_val != 1:
            if version_val > 1 and not _version_warning_emitted:
                _version_warning_emitted = True
                logger.warning("credits header version %d unsupported, ignoring — update Hermes", version_val)
            return None
        denominator_kind = lowered.get("x-nous-credits-denominator-kind", "none")
        window_note = denominator_kind == "usage_window"
        fields: dict[str, Any] = {}
        for name, kind, *default in _HEADER_FIELDS:
            raw = lowered.get(_header_name(name))
            if raw is None and window_note and name in _WINDOW_OPTIONAL_FIELDS:
                continue  # a window note may drop these: dataclass default
            fields[name] = _parse_field(kind, raw, *default)
        lim_micros_raw = lowered.get("x-nous-credits-subscription-limit-micros")
        lim_usd_raw = lowered.get("x-nous-credits-subscription-limit-usd")
        if lim_micros_raw is not None and lim_usd_raw is not None:
            fields["subscription_limit_micros"] = _parse_field("micros", lim_micros_raw)
            fields["subscription_limit_usd"] = _parse_field("usd", lim_usd_raw)
        if _SENTINEL in fields.values() or denominator_kind not in _VALID_DENOMINATOR_KINDS:
            return None
        if window_note:
            window = _parse_window_fields(lowered)
            if window is None:
                return None
            fields.update(window)
        disabled_reason = lowered.get("x-nous-credits-disabled-reason")  # None if absent (omitted when null)
        return CreditsState(version=version_val, denominator_kind=denominator_kind, disabled_reason=disabled_reason,
                            captured_at=time.time(), from_header=True, **fields)
    except Exception:  # fail-open → miss; the breadcrumb distinguishes a parser regression from a no-headers response
        logger.debug("credits ▸ parse_credits_headers raised (fail-open miss)", exc_info=True)
        return None


def response_is_usage_window(http_response: Any) -> bool:
    """True iff *http_response* carries complete usage-window ``x-nous-credits-*`` headers (the same
    test as ``CreditsState.is_usage_window``). Lets the window-only header captures leave every other
    account untouched. Never raises."""
    try:
        headers = getattr(http_response, "headers", None) if http_response is not None else None
        if not headers:
            return False
        state = parse_credits_headers(headers)
        return state is not None and state.is_usage_window
    except Exception:
        logger.debug("Usage-window header parse failed", exc_info=True)
        return False


# ── Dev fixtures (HERMES_DEV_CREDITS_FIXTURE): throwaway scaffolding to trigger any notice state
# without real spend. Value is a state NAME or a FILE PATH whose contents are a name (re-read every
# turn → `echo depleted > /tmp/cf` flips live). Drives per-turn notices, the cold-start seed, and /usage.
def _fixture(remaining: str, subscription: str, limit: Optional[str] = None, purchased: Optional[str] = None,
             *, paid: bool = True, reason: Optional[str] = None) -> dict:
    """Fixture spec from *_usd strings; micros derived exactly (Decimal)."""
    d: dict = {}
    for field, usd in (("remaining", remaining), ("subscription", subscription), ("subscription_limit", limit), ("purchased", purchased)):
        if usd is not None:
            d[f"{field}_micros"], d[f"{field}_usd"] = int(Decimal(usd) * 1_000_000), usd
    if limit is not None:
        d["denominator_kind"] = "subscription_cap"
    d["paid_access"] = paid
    return d if reason is None else {**d, "disabled_reason": reason}


_DEV_FIXTURES: dict[str, dict] = {
    "healthy": _fixture("30.34", "18.00", "20.00", "12.34"),  # used_fraction ~0.1, paid → no notice (recovery target)
    "sub_50pct": _fixture("10.00", "10.00", "20.00"),  # used_fraction == 0.5 → credits.usage band 50 (info)
    "sub_75pct": _fixture("5.00", "5.00", "20.00"),  # used_fraction == 0.75 → band 75 (warn)
    "sub_90pct": _fixture("2.00", "2.00", "20.00"),  # used_fraction == 0.9 → band 90 (warn)
    # uf == 1.0 + purchased>0 → SILENT at open (crossing-gated); flip healthy →
    # grant_exhausted via the fixture-file path to see credits.grant_spent
    "grant_exhausted": _fixture("12.34", "0.00", "20.00", "12.34"),
    "depleted": _fixture("0.00", "0.00", None, "0.00", paid=False, reason="out_of_credits"),  # → credits.depleted
    # subscription in debt (negative, the only signed field) → depleted
    "debt": _fixture("0.00", "-5.00", "20.00", "0.00", paid=False, reason="out_of_credits"),
}


def dev_fixture_credits_state() -> Optional[CreditsState]:
    """Fixture CreditsState for HERMES_DEV_CREDITS_FIXTURE, or None (unknown name / unset). Prod-leak guard:
    applies ONLY when HERMES_DEV_CREDITS is also on, so a stray fixture env var never surfaces fabricated balances."""
    name = os.environ.get("HERMES_DEV_CREDITS_FIXTURE", "").strip()
    if not name or not is_truthy_value(os.environ.get("HERMES_DEV_CREDITS")):
        return None
    if os.path.sep in name or "/" in name:  # looks like a path → read the name from the file
        try:
            with open(name, "r", encoding="utf-8-sig") as fh:
                name = fh.read().strip()
        except OSError:
            return None
    if not (spec := _DEV_FIXTURES.get(name.lower())):
        return None
    # Stamp what the REAL parser always guarantees so a fixture is field-identical to a
    # parse_credits_headers() result: version 1 and a valid purchased_usd (zero top-up = "0.00").
    return CreditsState(**{"version": 1, "purchased_usd": "0.00", **spec}, from_header=True, captured_at=time.time())


def _credits_state_from_account(info) -> Optional[CreditsState]:
    """Map a NousPortalAccountInfo into a header-shaped CreditsState for the seed. Float account dollars →
    micros plus a DISPLAY *_usd (formatting account floats is allowed; parsing a server *_usd is not). Fail-open → None."""
    try:
        acc = getattr(info, "paid_service_access_info", None)
        sub = getattr(info, "subscription", None)

        def _money(dollars) -> tuple[int, str]:  # (micros, display usd); (0, "") when absent
            return (round(dollars * 1_000_000), f"{dollars:.2f}") if isinstance(dollars, (int, float)) else (0, "")
        fields: dict[str, Any] = {}
        for prefix, attr in (("remaining", "total_usable_credits"), ("subscription", "subscription_credits_remaining"),
                             ("purchased", "purchased_credits_remaining")):
            fields[f"{prefix}_micros"], fields[f"{prefix}_usd"] = _money(getattr(acc, attr, None))
        monthly = getattr(sub, "monthly_credits", None)
        cap = _money(monthly) if isinstance(monthly, (int, float)) and monthly > 0 else (None, None)
        paid = getattr(info, "paid_service_access", None)
        window = _window_seed_fields(getattr(info, "usage_window", None))
        if window is not None:
            # A usage-window account: the gauge is the window, the grant fields are meaningless (kept 0).
            fields["subscription_micros"], fields["subscription_usd"] = 0, "0.00"
            cap = (None, None)
        return CreditsState(
            **fields, subscription_limit_micros=cap[0], subscription_limit_usd=cap[1], from_header=False, captured_at=time.time(),
            rollover_micros=_money(getattr(sub, "rollover_credits", None))[0], paid_access=paid if isinstance(paid, bool) else True,
            denominator_kind="usage_window" if window is not None else ("subscription_cap" if cap[0] is not None else "none"),
            **(window or {}),
        )
    except Exception:
        logger.debug("credits ▸ seed account→state mapping failed", exc_info=True)
        return None


def _window_seed_fields(uw) -> Optional[dict[str, Any]]:
    """Portal ``usage_window`` (NousUsageWindowInfo or dict) → CreditsState window fields; None unless it
    carries a readable gauge list with an all-scope allowance (same requirement as the header parser)."""
    if uw is None:
        return None
    get = uw.get if isinstance(uw, dict) else (lambda k, d=None: getattr(uw, k, d))
    gauges = parse_gauges(get("gauges"))
    if gauges is None or not all_scope(gauges):
        return None
    out: dict[str, Any] = {"gauges": gauges}
    out["auto_top_up_enabled"] = get("auto_top_up_enabled") is True
    keep_going = get("keep_going_on_credits")
    out["keep_going_on_credits"] = keep_going if isinstance(keep_going, bool) else None
    paused = get("paused_reason")
    out["paused_reason"] = paused if isinstance(paused, str) and paused else None
    return out


# Process-lifetime memory of the usage band last shown per session_id: see
# _remember_shown_band. Bounded MRU so a long-lived gateway cannot grow it without limit.
_seen_usage_bands: dict = {}
_SEEN_USAGE_BANDS_MAX = 500


def _remember_shown_band(session_id, band) -> None:
    """Remember the usage band last evaluated for ``session_id`` (process-lifetime).

    Desktop reap/resume rebuilds the AIAgent per turn, so the in-memory ``_credits_latch``
    (and its ``usage_band``) never survives across messages — every rebuild re-announces the
    current band as a fresh crossing (#101578). ``agent.session_id`` IS stable across those
    rebuilds, so this map lets ``_hydrate_seed_state`` restore the band the user already saw.
    Deliberately NOT durable: a real process restart is a real ``session open`` and the
    cold-start seed should still warn immediately. Bounded (MRU eviction) so a long-lived
    gateway juggling many sessions cannot grow it without limit."""
    if not session_id:
        return
    _seen_usage_bands.pop(session_id, None)  # re-insert at the MRU end
    _seen_usage_bands[session_id] = band
    if len(_seen_usage_bands) > _SEEN_USAGE_BANDS_MAX:
        _seen_usage_bands.pop(next(iter(_seen_usage_bands)))


# Window plans: the same per-session memory for the window notices (limit / now-using-credits / paused lines
# and the 90% heads-ups), so a rebuilt agent does not re-show a line this session already showed.
_WINDOW_NOTICE_KEYS = (CREDITS_WINDOW_PAYG_KEY, CREDITS_WINDOW_LIMIT_KEY, CREDITS_DEPLETED_KEY)
_seen_window_notices: dict = {}


def restore_window_notices(session_id, latch: dict) -> None:
    """Load what this session already showed into a fresh latch (once per latch). Never raises."""
    try:
        if not session_id or latch.get("window_restored"):
            return
        latch["window_restored"] = True
        saved = _seen_window_notices.get(session_id)
        if not saved:
            return
        texts = latch.setdefault("texts", {})
        for key, text in saved["texts"].items():
            if key not in latch["active"]:
                latch["active"].add(key)
                texts[key] = text
        latch.setdefault("window_warned", {}).update(saved["warned"])
    except Exception:
        logger.debug("window notice restore failed", exc_info=True)


def apply_keep_going_change(agent, enabled: bool) -> None:
    """The user just changed "Keep going on credits": the agent's state takes the new value now (the next
    reply's headers confirm it) and the notice policy re-runs, so a "turn on Keep going" line clears at once.
    The account snapshot the pickers read is refreshed in the background. Never raises."""
    try:
        from dataclasses import replace
        state = getattr(agent, "_credits_state", None)
        if state is None or not state.is_usage_window:
            return
        agent._credits_state = replace(state, keep_going_on_credits=enabled)
        _rerun_notice_policy(agent)
        from agent.memory_provider import spawn_context_thread
        from hermes_cli.nous_account import get_nous_portal_account_info
        spawn_context_thread(lambda: get_nous_portal_account_info(force_fresh=True), name="keep-going-refresh").start()
    except Exception:
        logger.debug("keep going ▸ local refresh failed", exc_info=True)


def forget_window_notices(session_id) -> None:
    """The session is no longer on a window plan: drop its saved lines. Never raises."""
    _seen_window_notices.pop(session_id, None)


def remember_window_notices(session_id, latch: dict) -> None:
    """Record the window notices this session shows now (process lifetime, bounded MRU). Never raises."""
    try:
        if not session_id:
            return
        texts = latch.get("texts", {})
        entry = {"texts": {k: texts[k] for k in _WINDOW_NOTICE_KEYS if k in latch["active"] and k in texts},
                 "warned": dict(latch.get("window_warned", {}))}
        _seen_window_notices.pop(session_id, None)
        _seen_window_notices[session_id] = entry
        if len(_seen_window_notices) > _SEEN_USAGE_BANDS_MAX:
            _seen_window_notices.pop(next(iter(_seen_window_notices)))
    except Exception:
        logger.debug("window notice remember failed", exc_info=True)


def _hydrate_seed_state(agent, state) -> None:
    """Install a seed CreditsState on the agent and fire the notice policy once. Primes the crossing gate:
    the cold-start snapshot IS the first observation, so a session opening in a band warns immediately."""
    agent._credits_state = state
    if getattr(agent, "_credits_session_start_micros", None) is None:
        agent._credits_session_start_micros = state.remaining_micros
    latch = getattr(agent, "_credits_latch", None)
    if isinstance(latch, dict) and state.used_fraction is not None:
        latch["seen_below_90"] = True  # ONLY this gate — priming seen_grant_unspent would revive the steady-state nag
        # Restore the band this logical session already showed (desktop reap/resume rebuilds the
        # agent object but keeps session_id) so an unchanged band is not re-announced per message.
        prior_band = _seen_usage_bands.get(getattr(agent, "session_id", None))
        if prior_band is not None:
            latch["usage_band"] = prior_band
            latch["active"].add(CREDITS_USAGE_KEY)
    _rerun_notice_policy(agent)


def _rerun_notice_policy(agent) -> None:
    if callable(emit := getattr(agent, "_emit_credits_notices", None)):
        emit()


def _warm_nous_pricing_cache() -> None:
    """Fill the in-process Nous pricing catalog so :func:`is_free_tier_model`'s peek can answer.
    The peek never fetches and nothing else warms it during session start, so a cold process reads
    an EMPTY catalog and a subscription-billed model — which spends no credits — still draws the
    depleted banner. Fail-open: a miss leaves the peek exactly as it was."""
    try:
        from hermes_cli.models_pricing import get_pricing_for_provider

        get_pricing_for_provider("nous")
    except Exception:
        logger.debug("credits ▸ nous pricing warm failed", exc_info=True)


def rewarm_pricing_before_depleted_notice(agent) -> bool:
    """Depleted account, model the peek cannot vouch for, Nous catalog cold: start a background warm
    whose completion re-runs the policy and return True so the caller defers the depleted decision
    to it — the session-start warm is one-shot but the catalog expires after
    ``_NOUS_CATALOG_TTL_SECONDS``, and a cold peek would bring the banner back for a
    subscription-billed model. False = decide now: peek warm, not on Nous, a warm in flight (the
    warm's own re-run included — fail-open, a failed fetch leaves the peek cold and the banner
    shows), or a fetch already failed and the cache is still refusing to dial. May raise; the notice
    path that calls it catches."""
    base_url = getattr(agent, "base_url", "") or ""
    if getattr(agent, "provider", "") != "nous" or not base_url:
        return False
    from hermes_cli.models_pricing import peek_cached_pricing, pricing_fetch_suppressed

    if peek_cached_pricing(base_url) or pricing_fetch_suppressed(base_url):
        return False
    inflight = getattr(agent, "_credits_pricing_warm", None)
    if inflight is not None and inflight.is_alive():
        return False

    def _warm_then_rerun() -> None:
        _warm_nous_pricing_cache()
        _rerun_notice_policy(agent)

    from agent.memory_provider import spawn_context_thread

    agent._credits_pricing_warm = spawn_context_thread(_warm_then_rerun, name="credits-pricing-warm")
    agent._credits_pricing_warm.start()
    return True


def seed_credits_at_session_start(agent) -> bool:
    """Hydrate agent._credits_state from the portal account (or a dev fixture) and fire the notice policy so
    warnings show at session OPEN (TUI/desktop "ready" and plain-CLI first-turn setup). Idempotent once a seed
    or real header populated _credits_state. Returns True iff it seeded this call. Never raises."""
    try:
        if getattr(agent, "provider", "") != "nous" or getattr(agent, "_credits_state", None) is not None:
            return False
        try:
            fixture = dev_fixture_credits_state()
        except Exception:
            fixture = None
        if fixture is not None:  # synchronous: instant, and tests rely on state + notice landing before return
            _hydrate_seed_state(agent, fixture)
            return True

        def _bg_seed() -> None:  # FIRE-AND-FORGET: a slow portal must never delay "ready"
            try:
                from hermes_cli.nous_account import get_nous_portal_account_info
                # BEFORE the policy runs (either branch below): the free-model gate only PEEKS the
                # pricing cache, and this thread is the first thing a chat session runs that can
                # afford to fill it.
                _warm_nous_pricing_cache()
                info = get_nous_portal_account_info(force_fresh=True)
                if getattr(agent, "_credits_state", None) is not None:
                    # A live inference header beat us — don't clobber it, but DO re-run the policy:
                    # it evaluated against the cold cache and may be showing a banner the warm
                    # catalog now suppresses.
                    _rerun_notice_policy(agent)
                    return
                if (state := _credits_state_from_account(info)) is not None:
                    _hydrate_seed_state(agent, state)
            except Exception:
                logger.debug("credits ▸ session-start seed (background) failed", exc_info=True)
        from agent.memory_provider import spawn_context_thread

        spawn_context_thread(_bg_seed, name="credits-seed").start()  # the warm reads the profile's auth store
        return True
    except Exception:
        logger.debug("credits ▸ session-start seed failed (fail-open)", exc_info=True)  # innermost log: diagnosable dead seed
        return False
