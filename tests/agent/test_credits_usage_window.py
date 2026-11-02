"""Usage-window plans in agent.credits_tracker & friends.

Header parsing (the server's gauge list is REQUIRED; spend-source / paused-reason optional), the
derived gauge (the binding all-scope allowance, exhausted flags by scope), the notice policy
(bands by percent + reset, the PAYG one-shot, the three hard stops, restored), the cold-start seed
from ``account.usage_window``, each notice state by name, ``/usage`` rendering, the classifier verdicts and
the 402/403 header capture. Legacy (credit-grant) behaviour is pinned unchanged alongside.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agent import credits_tracker as ct
from agent.credits_tracker import (
    CREDITS_DEPLETED_KEY,
    CREDITS_WINDOW_LIMIT_KEY,
    CREDITS_RESTORED_KEY,
    CREDITS_USAGE_KEY,
    CREDITS_WINDOW_PAYG_KEY,
    CreditsState,
    evaluate_credits_notices,
    format_window_reset,
    new_credits_latch,
    parse_credits_headers,
    window_approach_key,
)

NOW_MS = 1_800_000_000_000
DAY_RESET = NOW_MS + 6 * 3600 * 1000
WEEK_RESET = NOW_MS + 3 * 24 * 3600 * 1000
MONTH_RESET = NOW_MS + 20 * 24 * 3600 * 1000


def _gauges(day_bp=1_200, week_bp=900, *, top_tier_bp=0, month_bp=None, day_reset=DAY_RESET, week_reset=WEEK_RESET,
            top_reset=None, month_reset=MONTH_RESET) -> list[dict]:
    """The server's allowance list (the Portal account's ``usage_window.gauges`` / the API's gauges header), in display order.
    ``top_tier_bp=None`` omits the top-tier allowance; ``month_bp=None`` omits the month."""
    out = [{"id": "day", "scope": "all", "label": "Today", "used_bp": day_bp, "resets_at_ms": day_reset},
           {"id": "week", "scope": "all", "label": "This week", "used_bp": week_bp, "resets_at_ms": week_reset}]
    if month_bp is not None:
        out.append({"id": "month", "scope": "all", "label": "This month", "used_bp": month_bp, "resets_at_ms": month_reset})
    if top_tier_bp is not None:
        out.append({"id": "top_tier", "scope": "top_tier", "label": "Large models (monthly)", "used_bp": top_tier_bp,
                    "resets_at_ms": top_reset or day_reset})
    return out


def _gauge(state, gid):
    return next((g for g in state.gauges if g.id == gid), None)


def _micros(dollars: float) -> str:
    return str(round(dollars * 1_000_000))


def _window_headers(day_bp: int = 1_200, week_bp: int = 900, *, top_tier_bp: int | None = 0, purchased: float = 0.0,
                    source: str | None = "included", paid: bool = True, paused: str | None = None, **extra) -> dict:
    h = {
        "x-nous-credits-version": "1",
        "x-nous-credits-remaining-micros": _micros(purchased),
        "x-nous-credits-remaining-usd": f"{purchased:.2f}",
        "x-nous-credits-subscription-micros": "0",
        "x-nous-credits-subscription-usd": "0.00",
        "x-nous-credits-rollover-micros": "0",
        "x-nous-credits-purchased-micros": _micros(purchased),
        "x-nous-credits-purchased-usd": f"{purchased:.2f}",
        "x-nous-credits-paid-access": "true" if paid else "false",
        "x-nous-credits-as-of-ms": str(NOW_MS),
        "x-nous-credits-denominator-kind": "usage_window",
        "x-nous-credits-window-gauges": json.dumps(_gauges(day_bp, week_bp, top_tier_bp=top_tier_bp)),
    }
    if source is not None:
        h["x-nous-credits-spend-source"] = source
    if paused is not None:
        h["x-nous-credits-window-paused-reason"] = paused
    h.update(extra)
    return h


_GAUGE_KW = {"window_day_used_bp": "day_bp", "window_week_used_bp": "week_bp", "window_top_tier_used_bp": "top_tier_bp",
             "window_month_used_bp": "month_bp", "window_day_resets_at_ms": "day_reset",
             "window_week_resets_at_ms": "week_reset", "window_top_tier_resets_at_ms": "top_reset",
             "window_month_resets_at_ms": "month_reset"}


def _state(**kw) -> CreditsState:
    """A header-shaped window state (from_header=True, like a parse result). The ``window_*`` kwargs build
    the server's gauge list (see :func:`_gauges`)."""
    gauge_kw = {_GAUGE_KW[k]: kw.pop(k) for k in list(kw) if k in _GAUGE_KW}
    base = dict(version=1, denominator_kind="usage_window", paid_access=True, purchased_usd="0.00", from_header=True,
                captured_at=time.time(), spend_source="included",
                gauges=ct.parse_gauges(_gauges(**gauge_kw)) or ())
    base.update(kw)
    return CreditsState(**base)


def _run(state: CreditsState, latch: dict, **kw):
    show, clear = evaluate_credits_notices(state, latch, **kw)
    return [(n.key, n.level, n.text) for n in show], clear


def _keys(show):
    return [k for k, _lvl, _txt in show]


# ── parser ────────────────────────────────────────────────────────────────────


class TestParser:
    def test_full_window_note_parses(self):
        s = parse_credits_headers(_window_headers(7_600, 4_000, top_tier_bp=2_500, purchased=5.0, source="payg"))
        assert s is not None and s.from_header
        assert s.denominator_kind == "usage_window" and s.is_usage_window
        assert [(g.id, g.scope, g.used_bp) for g in s.gauges] == [
            ("day", "all", 7_600), ("week", "all", 4_000), ("top_tier", "top_tier", 2_500)]
        assert (_gauge(s, "day").resets_at_ms, _gauge(s, "week").resets_at_ms) == (DAY_RESET, WEEK_RESET)
        assert s.spend_source == "payg" and s.purchased_micros == 5_000_000
        assert s.paused_reason is None and s.auto_top_up_enabled is False

    def test_a_window_note_without_a_readable_gauge_list_is_a_miss(self):
        h = _window_headers()
        del h["x-nous-credits-window-gauges"]
        assert parse_credits_headers(h) is None
        for bad in ("not json", "{}", "[]", json.dumps(_gauges(top_tier_bp=None)[1:0])):
            assert parse_credits_headers({**h, "x-nous-credits-window-gauges": bad}) is None
        # Only a top-tier allowance: nothing stops every model, so not a usage window.
        top_only = [g for g in _gauges() if g["scope"] == "top_tier"]
        assert parse_credits_headers({**h, "x-nous-credits-window-gauges": json.dumps(top_only)}) is None

    def test_the_server_list_decides_which_allowances_exist(self):
        """A new allowance (here a 5-hour window) parses and binds with no Hermes release; an entry with an
        unknown scope, a repeated id or a bad percent is dropped, never misread."""
        gauges = [{"id": "five_hour", "scope": "all", "label": "Next 5 hours", "used_bp": 10_000,
                   "resets_at_ms": NOW_MS + 3600_000},
                  *_gauges(),
                  {"id": "week", "scope": "all", "label": "dup", "used_bp": 1, "resets_at_ms": 1},
                  {"id": "x", "scope": "per_model", "label": "x", "used_bp": 1, "resets_at_ms": 1},
                  {"id": "y", "scope": "all", "label": "y", "used_bp": 10_001, "resets_at_ms": 1}]
        s = parse_credits_headers(_window_headers(**{"x-nous-credits-window-gauges": json.dumps(gauges)}))
        assert s is not None and [g.id for g in s.gauges] == ["five_hour", "day", "week", "top_tier"]
        assert s.window_exhausted and s.binding_gauge.id == "five_hour" and s.window_resets_at_ms == NOW_MS + 3600_000
        assert _gauge(s, "week").label == "This week"
        show, _ = _run(_state(gauges=s.gauges, spend_source=None), new_credits_latch())
        assert show[-1][2].startswith("✕ You've hit your Next 5 hours limit")

    def test_month_and_top_tier_reset_come_from_the_list(self):
        top_reset = NOW_MS + 20 * 86_400_000
        s = parse_credits_headers(_window_headers(**{"x-nous-credits-window-gauges": json.dumps(
            _gauges(month_bp=3_300, top_reset=top_reset))}))
        assert s is not None and _gauge(s, "month").used_bp == 3_300 and s.top_tier_resets_at_ms == top_reset
        old = parse_credits_headers(_window_headers())  # no month allowance listed: no month gauge
        assert old is not None and old.is_usage_window and _gauge(old, "month") is None

    def test_a_list_without_top_tier_reads_as_not_used(self):
        s = parse_credits_headers(_window_headers(top_tier_bp=None, source=None))
        assert s is not None and s.top_tier_used_bp is None and not s.top_tier_exhausted and s.spend_source is None
        s = parse_credits_headers(_window_headers(**{"x-nous-credits-spend-source": "wallet"}))
        assert s is not None and s.spend_source is None  # outside the vocabulary → absent

    def test_paused_reason_header(self):
        s = parse_credits_headers(_window_headers(paid=False, paused="payment_past_due", source=None))
        assert s is not None and s.paused_reason == "payment_past_due" and s.paid_access is False

    def test_top_up_setting_header(self):
        """``on`` / ``off`` is the account's top-up-for-Hermes setting; absent or unrecognised = an older server."""
        on = parse_credits_headers(_window_headers(purchased=5.0, **{"x-nous-credits-keep-going-on-credits": "on"}))
        off = parse_credits_headers(_window_headers(purchased=5.0, **{"x-nous-credits-keep-going-on-credits": "OFF"}))
        old = parse_credits_headers(_window_headers(purchased=5.0))
        bad = parse_credits_headers(_window_headers(purchased=5.0, **{"x-nous-credits-keep-going-on-credits": "maybe"}))
        assert (on.keep_going_on_credits, off.keep_going_on_credits, old.keep_going_on_credits, bad.keep_going_on_credits) == (True, False, None, None)
        # A balance pays only when the setting is not off.
        assert on.credits_pay and old.credits_pay and not off.credits_pay

    def test_keep_going_off_with_a_balance_stops_and_names_the_setting(self):
        state = _state(window_day_used_bp=10_000, purchased_micros=5_000_000, purchased_usd="5.00",
                       spend_source=None, keep_going_on_credits=False)
        latch = new_credits_latch()
        latch["seen_below_90"] = True
        show, _ = _run(state, latch)
        stops = [t for k, _l, t in show if k == CREDITS_WINDOW_LIMIT_KEY]
        assert stops == [f"✕ You've hit your daily limit · resets {ct.format_window_reset(DAY_RESET)}"
                         " · turn on Keep going in /topup"]
        # The same state with the setting on (or unknown): top-up pays, so no stop line.
        for keep_going in (True, None):
            latch = new_credits_latch()
            latch["seen_below_90"] = True
            show, _ = _run(_state(window_day_used_bp=10_000, purchased_micros=5_000_000, purchased_usd="5.00",
                                  spend_source="payg", keep_going_on_credits=keep_going), latch)
            assert CREDITS_WINDOW_LIMIT_KEY not in _keys(show)

    def test_keep_going_off_top_tier_refusal_names_the_setting(self):
        state = _state(window_top_tier_used_bp=10_000, purchased_micros=5_000_000, purchased_usd="5.00",
                       spend_source=None, keep_going_on_credits=False)
        show, _ = _run(state, new_credits_latch())
        stop = next(t for k, _l, t in show if k == CREDITS_WINDOW_LIMIT_KEY)
        assert stop.startswith("✕ You've hit your monthly large-model limit") and stop.endswith(
            " · turn on Keep going in /topup")
        assert "—" not in stop and "$" not in stop

    def test_header_names_are_case_insensitive(self):
        h = {k.upper(): v for k, v in _window_headers().items()}
        assert parse_credits_headers(h) is not None

    def test_legacy_notes_ignore_window_headers(self):
        """A credit-grant note carrying stray window headers stays a credit-grant state (no gauge)."""
        h = _window_headers()
        h["x-nous-credits-denominator-kind"] = "none"
        s = parse_credits_headers(h)
        assert s is not None and not s.is_usage_window and s.gauges == () and s.spend_source is None

    def test_unknown_kind_is_still_a_miss(self):
        assert parse_credits_headers(_window_headers(**{"x-nous-credits-denominator-kind": "usage_windows"})) is None

    def test_a_window_note_may_drop_the_credit_grant_fields(self):
        """The server can retire the grant headers with legacy plans; a window note still parses, and only
        a window note: a credit note without them stays a miss."""
        grant = ["remaining-micros", "remaining-usd", "subscription-micros", "subscription-usd", "rollover-micros",
                 "as-of-ms"]
        h = _window_headers(1_200, 900, purchased=5.0)
        for name in grant:
            del h[f"x-nous-credits-{name}"]
        s = parse_credits_headers(h)
        assert s is not None and s.is_usage_window and _gauge(s, "day").used_bp == 1_200
        assert s.purchased_micros == 5_000_000 and s.remaining_micros == 0 and s.subscription_usd == ""
        assert parse_credits_headers({**h, "x-nous-credits-denominator-kind": "none"}) is None
        # A refused (402) window response carries no top-up balance; it still parses (balance defaults to 0).
        bare = {k: v for k, v in h.items() if not k.startswith("x-nous-credits-purchased-")}
        s = parse_credits_headers(bare)
        assert s is not None and s.is_usage_window and s.purchased_micros == 0
        assert parse_credits_headers({**bare, "x-nous-credits-denominator-kind": "none"}) is None
        # Present but malformed is still a contract violation.
        assert parse_credits_headers({**h, "x-nous-credits-rollover-micros": "-1"}) is None


# ── derived state ─────────────────────────────────────────────────────────────


class TestDerivedState:
    def test_used_fraction_is_the_fuller_window(self):
        assert _state(window_day_used_bp=3_000, window_week_used_bp=9_100).used_fraction == pytest.approx(0.91)
        assert _state(window_day_used_bp=7_600, window_week_used_bp=4_000).used_fraction == pytest.approx(0.76)

    def test_binding_window_and_its_reset(self):
        day = _state(window_day_used_bp=7_600, window_week_used_bp=4_000)
        assert day.binding_gauge.id == "day" and day.window_resets_at_ms == DAY_RESET
        week = _state(window_day_used_bp=3_000, window_week_used_bp=9_100)
        assert week.binding_gauge.id == "week" and week.window_resets_at_ms == WEEK_RESET
        assert _state(window_day_used_bp=5_000, window_week_used_bp=5_000).binding_gauge.id == "day"  # tie → first

    def test_exhausted_flags(self):
        assert _state(window_day_used_bp=10_000).window_exhausted
        assert _state(window_week_used_bp=10_000).window_exhausted
        assert not _state(window_day_used_bp=9_999).window_exhausted
        assert _state(window_top_tier_used_bp=10_000).top_tier_exhausted
        assert not _state(window_top_tier_used_bp=None).top_tier_exhausted  # not listed ≠ used
        assert not _state(paused_reason="payment_past_due").window_exhausted  # paused is not spent

    def test_an_implausible_reset_stamp_reads_as_unknown(self):
        """More than a year ahead is a bad stamp (the longest allowance is a month): no date decades away."""
        far = int(time.time() * 1000) + 10 * 366 * 86_400_000
        gauges = ct.parse_gauges([{"id": "day", "scope": "all", "used_bp": 100, "resets_at_ms": far},
                                  {"id": "week", "scope": "all", "used_bp": 100, "resets_at_ms": WEEK_RESET}])
        assert gauges[0].resets_at_ms is None and gauges[1].resets_at_ms == WEEK_RESET

    def test_no_gauges_is_not_a_usage_window(self):
        s = _state(gauges=())
        assert not s.is_usage_window and s.used_fraction is None and s.binding_gauge is None

    def test_depleted_still_keys_off_paid_access(self):
        assert not _state(window_day_used_bp=10_000).depleted
        assert _state(paid_access=False).depleted


class TestFormatWindowReset:
    def test_within_a_day_is_a_clock_time(self):
        assert format_window_reset(NOW_MS + 5 * 3600 * 1000, NOW_MS) == time.strftime("%H:%M", time.localtime((NOW_MS + 5 * 3600 * 1000) / 1000))

    def test_beyond_a_day_names_the_weekday(self):
        out = format_window_reset(WEEK_RESET, NOW_MS)
        assert out == time.strftime("%a %H:%M", time.localtime(WEEK_RESET / 1000))

    def test_a_week_or_more_out_names_the_date(self):
        top_reset = NOW_MS + 20 * 86_400_000
        stamp = time.localtime(top_reset / 1000)
        assert format_window_reset(top_reset, NOW_MS) == f"{time.strftime('%b', stamp)} {stamp.tm_mday}"

    def test_an_implausible_stamp_never_raises(self):
        assert format_window_reset(10**20, NOW_MS) == "the next reset"

    def test_passed_and_unknown(self):
        # Past stamp reads "shortly", never "now": a hard stop saying "used up · resets now" contradicts itself.
        assert format_window_reset(NOW_MS - 1, NOW_MS) == "shortly"
        assert format_window_reset(None) == "the next reset"
        assert format_window_reset(0) == "the next reset"


# ── notice policy ─────────────────────────────────────────────────────────────


class TestApproachWarnings:
    """A heads-up per allowance at 90%, once per reset (``credits.window.<id>``, TTL). No 50 / 75 bands."""

    def test_healthy_and_75_are_silent(self):
        latch = new_credits_latch()
        assert _run(_state(), latch) == ([], [])
        assert _run(_state(window_day_used_bp=7_600, window_week_used_bp=8_900), latch) == ([], [])

    def test_each_allowance_warns_at_90_with_its_own_key_and_reset(self):
        latch = new_credits_latch()
        show, _ = _run(_state(window_day_used_bp=9_100, window_week_used_bp=9_300), latch)
        assert _keys(show) == [window_approach_key("day"), window_approach_key("week")]
        day, week = show[0][2], show[1][2]
        assert day == f"⚠ 91% of your daily limit used · resets {format_window_reset(DAY_RESET)}"
        assert week == f"⚠ 93% of your weekly limit used · resets {format_window_reset(WEEK_RESET)}"
        assert all(lvl == "warn" for _k, lvl, _t in show) and "$" not in day

    def test_once_per_reset_not_per_request(self):
        latch = new_credits_latch()
        _run(_state(window_day_used_bp=9_100), latch)
        assert _run(_state(window_day_used_bp=9_600), latch) == ([], [])
        # A new day: the same allowance may warn again.
        show, _ = _run(_state(window_day_used_bp=9_100, window_day_resets_at_ms=DAY_RESET + 86_400_000), latch)
        assert _keys(show) == [window_approach_key("day")]

    def test_a_ttl_heads_up_never_needs_a_clear(self):
        latch = new_credits_latch()
        notices, _ = evaluate_credits_notices(_state(window_day_used_bp=9_100), latch)
        assert notices[0].kind == "ttl" and notices[0].ttl_ms
        # Running out: the limit line takes over; nothing to clear for the heads-up.
        show, clear = _run(_state(window_day_used_bp=10_000, spend_source=None), latch)
        assert clear == [] and _keys(show) == [CREDITS_WINDOW_LIMIT_KEY]

    def test_top_tier_warns_only_on_a_top_tier_model(self):
        s = _state(window_top_tier_used_bp=9_200)
        assert _run(s, new_credits_latch()) == ([], [])
        show, _ = _run(s, new_credits_latch(), model_is_top_tier=True)
        assert show[0][0] == window_approach_key("top_tier")
        assert show[0][2].startswith("⚠ 92% of your monthly large-model limit used")

    def test_a_free_model_warns_about_nothing(self):
        assert _run(_state(window_day_used_bp=9_500), new_credits_latch(), model_is_free=True) == ([], [])

    def test_a_new_allowance_warns_with_the_server_label(self):
        five = {"id": "five_hour", "scope": "all", "label": "5-hour", "used_bp": 9_400, "resets_at_ms": DAY_RESET}
        s = _state(gauges=ct.parse_gauges([five, *_gauges()]) or ())
        show, _ = _run(s, new_credits_latch())
        assert show[0][0] == window_approach_key("five_hour") and "of your 5-hour limit used" in show[0][2]

    def test_shows_even_with_top_up_credits(self):
        """Top-up credits do not silence the heads-up: the included allowance is still the gauge."""
        show, _ = _run(_state(window_day_used_bp=9_000, purchased_micros=5_000_000, purchased_usd="5.00"),
                       new_credits_latch())
        assert _keys(show) == [window_approach_key("day")]

    def test_a_stale_dollar_band_is_cleared_on_a_window_plan(self):
        latch = new_credits_latch()
        latch["active"].add(CREDITS_USAGE_KEY)
        latch["texts"][CREDITS_USAGE_KEY] = "x"
        _show, clear = _run(_state(), latch)
        assert clear == [CREDITS_USAGE_KEY]

    def test_grant_spent_never_fires_on_a_window(self):
        latch = new_credits_latch()
        latch["seen_grant_unspent"] = True
        show, _ = _run(_state(window_day_used_bp=10_000, purchased_micros=5_000_000, spend_source="payg"), latch)
        assert "credits.grant_spent" not in _keys(show)


class TestPaygOneShot:
    def test_window_exhausted_with_top_up(self):
        latch = new_credits_latch()
        _run(_state(), latch)
        show, _ = _run(_state(window_day_used_bp=10_000, purchased_micros=5_000_000, purchased_usd="5.00", spend_source="payg"), latch)
        keys = _keys(show)
        assert CREDITS_WINDOW_PAYG_KEY in keys and CREDITS_WINDOW_LIMIT_KEY not in keys
        text = next(t for k, _l, t in show if k == CREDITS_WINDOW_PAYG_KEY)
        assert text == f"• You've hit your daily limit · resets {format_window_reset(DAY_RESET)} · now using credits"
        assert "$" not in text

    def test_auto_top_up_is_appended_when_on(self):
        latch = new_credits_latch()
        show, _ = _run(_state(window_day_used_bp=10_000, purchased_micros=5_000_000, spend_source="payg", auto_top_up_enabled=True), latch)
        text = next(t for k, _l, t in show if k == CREDITS_WINDOW_PAYG_KEY)
        assert text.endswith(" · now using credits · auto top-up is on")

    def test_top_tier_slice_has_its_own_copy(self):
        latch = new_credits_latch()
        top_reset = NOW_MS + 20 * 86_400_000
        show, _ = _run(_state(window_day_used_bp=4_000, window_top_tier_used_bp=10_000, window_top_tier_resets_at_ms=top_reset,
                              purchased_micros=5_000_000, spend_source="payg"), latch)
        text = next(t for k, _l, t in show if k == CREDITS_WINDOW_PAYG_KEY)
        assert text == (f"• You've hit your monthly large-model limit · resets {format_window_reset(top_reset)}"
                        " · large models now use credits")

    def test_one_shot_does_not_repeat_and_clears_when_included_again(self):
        latch = new_credits_latch()
        payg = _state(window_day_used_bp=10_000, purchased_micros=5_000_000, spend_source="payg")
        _run(payg, latch)
        show, clear = _run(payg, latch)
        assert show == [] and clear == []  # same text: nothing re-emitted
        show, clear = _run(_state(window_day_used_bp=500, window_week_used_bp=500, purchased_micros=5_000_000, spend_source="included"), latch)
        assert CREDITS_WINDOW_PAYG_KEY in clear and CREDITS_WINDOW_PAYG_KEY not in _keys(show)

    def test_payg_without_top_up_is_not_payg(self):
        """spend_source=payg with $0 purchased: the line needs a balance to draw on."""
        latch = new_credits_latch()
        show, _ = _run(_state(window_day_used_bp=10_000, spend_source="payg"), latch)
        assert CREDITS_WINDOW_PAYG_KEY not in _keys(show)


class TestHardStops:
    def test_exhausted_with_empty_wallets(self):
        latch = new_credits_latch()
        show, _ = _run(_state(window_day_used_bp=10_000, window_week_used_bp=6_000, spend_source="included"), latch)
        assert _keys(show)[-1] == CREDITS_WINDOW_LIMIT_KEY
        _key, level, text = show[-1]
        assert level == "error"
        assert text == f"✕ You've hit your daily limit · resets {format_window_reset(DAY_RESET)} · /topup to add credits"
        assert CREDITS_WINDOW_PAYG_KEY not in _keys(show)

    def test_exhausted_names_the_month_and_its_reset_when_it_binds(self):
        s = _state(window_day_used_bp=10_000, window_week_used_bp=6_000, window_month_used_bp=10_000,
                   window_month_resets_at_ms=MONTH_RESET, spend_source=None)
        assert s.binding_gauge.id == "month" and s.window_resets_at_ms == MONTH_RESET
        show, _ = _run(s, new_credits_latch())
        assert show[-1][2].startswith(f"✕ You've hit your monthly limit · resets {format_window_reset(MONTH_RESET)} · ")
        # A month with room never binds over a fuller day.
        assert _state(window_day_used_bp=10_000, window_month_used_bp=4_000).binding_gauge.id == "day"

    def test_exhausted_names_the_week_when_it_binds(self):
        show, _ = _run(_state(window_day_used_bp=6_000, window_week_used_bp=10_000, spend_source=None), new_credits_latch())
        assert show[-1][2].startswith(f"✕ You've hit your weekly limit · resets {format_window_reset(WEEK_RESET)} · ")

    def test_day_and_week_both_full_names_the_one_that_frees_up_last(self):
        """The day's reset alone lets nothing through while the week is also full."""
        show, _ = _run(_state(window_day_used_bp=10_000, window_week_used_bp=10_000, spend_source=None), new_credits_latch())
        assert show[-1][2].startswith(f"✕ You've hit your weekly limit · resets {format_window_reset(WEEK_RESET)} · ")

    def test_payment_paused_offers_the_card_and_credits(self):
        """A paused plan bills credits whatever the setting says, so with none left the stop names both
        ways on: fix the card (restores included usage) or add credits (keeps going now)."""
        show, _ = _run(_state(paid_access=False, paused_reason="payment_past_due", spend_source=None), new_credits_latch())
        assert [k for k, _l, _t in show] == [CREDITS_DEPLETED_KEY]
        text = show[0][2]
        assert "payment failed" in text and "update your card" in text and "add credits in /topup" in text
        assert "Keep going on credits" not in text  # the setting doesn't gate a paused plan

    def test_a_pause_for_another_reason_never_claims_a_payment_failed(self):
        """The server may send any reason (\"unknown\" for a malformed one): still a pause, neutral copy."""
        from agent import account_usage
        from hermes_cli import nous_account
        show, _ = _run(_state(paid_access=False, paused_reason="unknown", spend_source=None), new_credits_latch())
        assert [k for k, _l, _t in show] == [CREDITS_DEPLETED_KEY]
        assert "payment" not in show[0][2] and "add credits in /topup" in show[0][2]
        window = SimpleNamespace(gauges=ct.parse_gauges(_gauges(100, 50)), auto_top_up_enabled=False,
                                 paused_reason="unknown", keep_going_on_credits=None)
        info = SimpleNamespace(logged_in=True, paid_service_access=True, usage_window=window, portal_base_url=None,
                               paid_service_access_info=SimpleNamespace(purchased_credits_remaining=0.0), subscription=None)
        lines = account_usage.render_account_usage_lines(account_usage.build_nous_credits_snapshot(info))
        assert account_usage._WINDOW_PAUSED_OTHER_LINE in lines and account_usage._WINDOW_PAUSED_LINE not in lines
        assert nous_account.format_usage_window_block_guidance(info).startswith("Included usage is paused, or add credits")

    def test_payment_paused_wins_over_exhausted_and_survives_a_free_model(self):
        s = _state(paid_access=False, paused_reason="payment_past_due", window_day_used_bp=10_000, spend_source=None)
        show, _ = _run(s, new_credits_latch())
        assert "payment failed" in show[0][2]
        show, _ = _run(s, new_credits_latch(), model_is_free=True)
        assert "payment failed" in show[0][2]  # a free model does not un-pause a subscription

    def test_paused_with_top_up_is_not_a_stop(self):
        """Top-up credits keep PAYG alive during dunning: the window is withheld, the wallet is not."""
        show, _ = _run(_state(paused_reason="payment_past_due", purchased_micros=5_000_000, spend_source="payg"), new_credits_latch())
        assert CREDITS_DEPLETED_KEY not in _keys(show) and CREDITS_WINDOW_PAYG_KEY in _keys(show)

    def test_top_tier_blocked_on_a_402(self):
        """The 402 carries the headers without a spend-source: this model was refused."""
        top_reset = NOW_MS + 20 * 86_400_000
        show, _ = _run(_state(window_day_used_bp=4_000, window_top_tier_used_bp=10_000, window_top_tier_resets_at_ms=top_reset,
                              spend_source=None), new_credits_latch())
        assert show[-1][0] == CREDITS_WINDOW_LIMIT_KEY
        assert show[-1][2] == f"✕ You've hit your monthly large-model limit · resets {format_window_reset(top_reset)} · /topup to add credits"
        # An older server sends no large-model reset: the reset falls back to the day's.
        old, _ = _run(_state(window_day_used_bp=4_000, window_top_tier_used_bp=10_000, spend_source=None), new_credits_latch())
        assert f"resets {format_window_reset(DAY_RESET)} ·" in old[-1][2]

    def test_top_tier_blocked_mentions_auto_top_up(self):
        show, _ = _run(_state(window_day_used_bp=4_000, window_top_tier_used_bp=10_000, spend_source=None, auto_top_up_enabled=True),
                       new_credits_latch())
        assert show[-1][2].endswith(" · /topup to add credits · auto top-up is on")

    def test_top_tier_used_but_this_model_admitted_is_not_a_stop(self):
        """An ``included`` spend-source means the current model is not top-tier — no block."""
        show, _ = _run(_state(window_day_used_bp=4_000, window_top_tier_used_bp=10_000, spend_source="included"), new_credits_latch())
        assert CREDITS_WINDOW_LIMIT_KEY not in _keys(show)

    def test_top_tier_used_on_a_seed_is_not_a_stop(self):
        """A seed cannot know which model the session will use; /usage carries the top-tier line."""
        show, _ = _run(_state(window_day_used_bp=4_000, window_top_tier_used_bp=10_000, spend_source=None, from_header=False),
                       new_credits_latch())
        assert CREDITS_WINDOW_LIMIT_KEY not in _keys(show)

    def test_portal_paid_access_false_without_a_pause_is_the_legacy_depleted_line(self):
        show, _ = _run(_state(paid_access=False, spend_source=None), new_credits_latch())
        assert show[0][2] == "✕ Credit access paused · run /topup to top up"

    def test_free_model_suppresses_the_window_stop_without_restored(self):
        latch = new_credits_latch()
        _run(_state(window_day_used_bp=10_000, spend_source=None), latch)
        show, clear = _run(_state(window_day_used_bp=10_000, spend_source=None), latch, model_is_free=True)
        assert clear == [CREDITS_WINDOW_LIMIT_KEY] and show == []

    def test_reset_restores(self):
        latch = new_credits_latch()
        _run(_state(window_day_used_bp=10_000, spend_source=None), latch)
        show, clear = _run(_state(window_day_used_bp=100, window_week_used_bp=100, spend_source="included"), latch)
        assert CREDITS_WINDOW_LIMIT_KEY in clear
        restored = [n for n in show if n[0] == CREDITS_RESTORED_KEY]
        assert len(restored) == 1 and restored[0][1] == "success"

    def test_top_up_mid_block_restores_then_payg(self):
        latch = new_credits_latch()
        _run(_state(window_day_used_bp=10_000, spend_source=None), latch)
        show, clear = _run(_state(window_day_used_bp=10_000, purchased_micros=5_000_000, spend_source="payg"), latch)
        assert CREDITS_WINDOW_LIMIT_KEY in clear
        assert [k for k, _l, _t in show] == [CREDITS_RESTORED_KEY, CREDITS_WINDOW_PAYG_KEY]

    def test_stop_text_change_re_emits(self):
        latch = new_credits_latch()
        _run(_state(window_day_used_bp=10_000, spend_source=None), latch)
        show, clear = _run(_state(paid_access=False, paused_reason="payment_past_due", window_day_used_bp=10_000, spend_source=None), latch)
        assert clear == [CREDITS_WINDOW_LIMIT_KEY]
        assert show[0][0] == CREDITS_DEPLETED_KEY and "payment failed" in show[0][2]
        assert not any(k == CREDITS_RESTORED_KEY for k, _l, _t in show)


class TestPlanFlipHygiene:
    def test_window_policy_clears_a_stale_grant_line(self):
        latch = new_credits_latch()
        latch["active"].add("credits.grant_spent")
        _show, clear = _run(_state(), latch)
        assert "credits.grant_spent" in clear and "credits.grant_spent" not in latch["active"]

    def test_legacy_policy_clears_a_stale_payg_line(self):
        latch = new_credits_latch()
        latch["active"].add(CREDITS_WINDOW_PAYG_KEY)
        legacy = CreditsState(version=1, denominator_kind="none", paid_access=True, from_header=True, captured_at=time.time())
        _show, clear = evaluate_credits_notices(legacy, latch)
        assert CREDITS_WINDOW_PAYG_KEY in clear

    def test_legacy_path_is_unchanged_for_credit_grant_states(self):
        """Pin: a subscription_cap state at 90% with no top-up still yields today's dollar copy."""
        latch = new_credits_latch()
        latch["seen_below_90"] = True
        s = CreditsState(version=1, denominator_kind="subscription_cap", subscription_micros=2_000_000, subscription_usd="2.00",
                         subscription_limit_micros=20_000_000, subscription_limit_usd="20.00", paid_access=True, from_header=True,
                         captured_at=time.time())
        show, _ = evaluate_credits_notices(s, latch)
        assert [n.key for n in show] == [CREDITS_USAGE_KEY]
        assert show[0].text == "⚠ You've used $18.00 of your $20.00 cap"


# ── seed + fixtures ───────────────────────────────────────────────────────────


class TestSeed:
    def _info(self, uw, *, paid=True, purchased=5.0):
        return SimpleNamespace(
            paid_service_access=paid,
            paid_service_access_info=SimpleNamespace(total_usable_credits=purchased, subscription_credits_remaining=0.0,
                                                    purchased_credits_remaining=purchased),
            subscription=SimpleNamespace(monthly_credits=0, rollover_credits=0),
            usage_window=uw,
        )

    def test_account_usage_window_seeds_a_window_state(self):
        uw = SimpleNamespace(gauges=_gauges(7_600, 4_000, top_tier_bp=10_000), auto_top_up_enabled=True, paused_reason=None)
        s = ct._credits_state_from_account(self._info(uw))
        assert s is not None and s.is_usage_window and not s.from_header
        assert s.used_fraction == pytest.approx(0.76) and s.top_tier_used_bp == 10_000 and s.auto_top_up_enabled
        assert s.purchased_micros == 5_000_000 and s.subscription_micros == 0 and s.subscription_limit_micros is None
        assert s.spend_source is None and s.paused_reason is None
        assert s.top_tier_resets_at_ms == DAY_RESET

    def test_seed_carries_the_monthly_top_tier_reset(self):
        top_reset = NOW_MS + 20 * 86_400_000
        uw = SimpleNamespace(gauges=_gauges(0, 0, top_reset=top_reset))
        s = ct._credits_state_from_account(self._info(uw))
        assert s is not None and s.top_tier_resets_at_ms == top_reset

    def test_seed_carries_the_monthly_cap(self):
        uw = SimpleNamespace(gauges=_gauges(0, 0, month_bp=10_000))
        s = ct._credits_state_from_account(self._info(uw))
        assert s is not None and s.window_exhausted and s.window_resets_at_ms == MONTH_RESET

    def test_paused_seed(self):
        uw = {"gauges": _gauges(0, 0, top_tier_bp=None), "paused_reason": "payment_past_due"}
        s = ct._credits_state_from_account(self._info(uw, paid=False, purchased=0.0))
        assert s is not None and s.paused_reason == "payment_past_due" and s.paid_access is False
        assert s.top_tier_used_bp is None and not s.top_tier_exhausted  # not listed → not used

    def test_a_usage_window_without_a_gauge_list_falls_back_to_the_legacy_mapping(self):
        for uw in (SimpleNamespace(gauges=None), SimpleNamespace(gauges=()), {"day_used_bp": 1}):
            s = ct._credits_state_from_account(self._info(uw))
            assert s is not None and s.denominator_kind == "none" and not s.is_usage_window

    def test_no_usage_window_is_todays_mapping(self):
        info = self._info(None)
        info.subscription = SimpleNamespace(monthly_credits=20.0, rollover_credits=0)
        s = ct._credits_state_from_account(info)
        assert s is not None and s.denominator_kind == "subscription_cap" and s.subscription_limit_micros == 20_000_000

    def test_a_session_opening_past_90_gets_the_heads_up(self):
        uw = SimpleNamespace(gauges=_gauges(9_200, 4_000), auto_top_up_enabled=False, paused_reason=None)
        state = ct._credits_state_from_account(self._info(uw))
        emitted = []
        agent = SimpleNamespace(_credits_state=None, _credits_session_start_micros=None, _credits_latch=new_credits_latch())
        agent._emit_credits_notices = lambda: emitted.extend(evaluate_credits_notices(agent._credits_state, agent._credits_latch)[0])
        ct._hydrate_seed_state(agent, state)
        assert [n.key for n in emitted] == [window_approach_key("day")] and "92%" in emitted[0].text


def _plan_state(day_bp: int, week_bp: int, *, top_tier_bp: int = 0, purchased: str = "0.00", source: str | None = None,
                auto_top_up: bool = False, paused: str | None = None, paid: bool = True,
                keep_going: bool | None = None) -> CreditsState:
    """A header-shaped window state: percents in bp, resets 6h / 3d out from now, a top-up wallet, the
    last spend source, auto top-up, a pause and the account's top-up-for-Hermes setting."""
    from decimal import Decimal

    from agent.usage_gauges import SCOPE_ALL, SCOPE_TOP_TIER, WindowGauge
    now_ms = int(time.time() * 1000)
    micros = int(Decimal(purchased) * 1_000_000)
    day_reset = now_ms + 6 * 3600 * 1000
    gauges = (
        WindowGauge("day", SCOPE_ALL, "Today", day_bp, day_reset),
        WindowGauge("week", SCOPE_ALL, "This week", week_bp, now_ms + 3 * 24 * 3600 * 1000),
        WindowGauge("top_tier", SCOPE_TOP_TIER, "Large models (monthly)", top_tier_bp, day_reset),
    )
    return CreditsState(
        version=1, denominator_kind="usage_window", paid_access=paid,
        remaining_micros=micros, remaining_usd=purchased, purchased_micros=micros, purchased_usd=purchased,
        subscription_micros=0, subscription_usd="0.00", gauges=gauges,
        spend_source=source, auto_top_up_enabled=auto_top_up, paused_reason=paused,
        keep_going_on_credits=keep_going, from_header=True, captured_at=time.time(),
    )


# Every usage-window notice state, by name.
_PLAN_STATES = {
    "healthy": lambda: _plan_state(1_200, 900, source="included"),  # 12% today → no notice
    "75pct": lambda: _plan_state(7_600, 4_000, source="included"),  # 75%: quiet (no band on a window plan)
    "week_90pct": lambda: _plan_state(3_000, 9_100, source="included"),  # the week's 90% heads-up
    # window used up, $5.00 top-up but top-up for Hermes turned off → the stop names the setting
    "keep_going_off": lambda: _plan_state(10_000, 6_000, purchased="5.00", keep_going=False),
    "keep_going_off_no_balance": lambda: _plan_state(10_000, 6_000, keep_going=False),
    # window used up, $5.00 top-up → PAYG one-shot
    "payg": lambda: _plan_state(10_000, 6_000, purchased="5.00", source="payg"),
    "payg_autotopup": lambda: _plan_state(10_000, 6_000, purchased="5.00", source="payg", auto_top_up=True),
    # large-model allowance used, window still open, top-up present → the model-specific PAYG line
    "payg_top_tier": lambda: _plan_state(4_000, 2_000, top_tier_bp=10_000, purchased="5.00", source="payg"),
    # large-model allowance used, no top-up, last response blocked → hard stop naming the model, not the window
    "top_tier_blocked": lambda: _plan_state(4_000, 2_000, top_tier_bp=10_000),
    "top_tier_blocked_autotopup": lambda: _plan_state(4_000, 2_000, top_tier_bp=10_000, auto_top_up=True),
    "exhausted": lambda: _plan_state(10_000, 6_000),  # both wallets empty → hard stop + /topup + reset
    "paused": lambda: _plan_state(2_000, 1_000, paused="payment_past_due", paid=False),  # dunning, no /topup
}


class TestPlanStates:
    @pytest.mark.parametrize("name, expect_keys, must_contain", [
        ("healthy", [], ""),
        ("75pct", [], ""),
        ("week_90pct", [window_approach_key("week")], "91% of your weekly limit used"),
        ("payg", [CREDITS_WINDOW_PAYG_KEY], "now using credits"),
        ("payg_autotopup", [CREDITS_WINDOW_PAYG_KEY], "auto top-up is on"),
        ("payg_top_tier", [CREDITS_WINDOW_PAYG_KEY], "You've hit your monthly large-model limit · resets"),
        ("top_tier_blocked", [CREDITS_WINDOW_LIMIT_KEY], "You've hit your monthly large-model limit · resets"),
        ("top_tier_blocked_autotopup", [CREDITS_WINDOW_LIMIT_KEY], "auto top-up is on"),
        ("exhausted", [CREDITS_WINDOW_LIMIT_KEY], "/topup to add credits"),
        # A balance, but top-up for Hermes is off: the stop names the account setting, not /topup.
        ("keep_going_off", [CREDITS_WINDOW_LIMIT_KEY], "turn on Keep going in /topup"),
        # No balance and the setting off (the default): the gateway refuses either way, so both steps.
        ("keep_going_off_no_balance", [CREDITS_WINDOW_LIMIT_KEY], "add credits and turn on Keep going in /topup"),
        ("paused", [CREDITS_DEPLETED_KEY], "payment failed"),
    ])
    def test_every_window_state_drives_its_notice(self, name, expect_keys, must_contain):
        state = _PLAN_STATES[name]()
        assert state.is_usage_window and state.from_header and state.version == 1
        latch = new_credits_latch()
        latch["seen_below_90"] = True  # what a live capture leaves after a reply under 90%
        show, _ = evaluate_credits_notices(state, latch)
        assert [n.key for n in show] == expect_keys
        if must_contain:
            assert any(must_contain in n.text for n in show)
        assert not any("$" in n.text for n in show)

    def test_legacy_fixtures_still_resolve(self, monkeypatch):
        monkeypatch.setenv("HERMES_DEV_CREDITS", "1")
        monkeypatch.setenv("HERMES_DEV_CREDITS_FIXTURE", "depleted")
        state = ct.dev_fixture_credits_state()
        assert state is not None and not state.is_usage_window and state.paid_access is False

    def test_window_fixture_names_are_not_dev_fixtures(self, monkeypatch):
        monkeypatch.setenv("HERMES_DEV_CREDITS", "1")
        monkeypatch.setenv("HERMES_DEV_CREDITS_FIXTURE", "window_exhausted")
        assert ct.dev_fixture_credits_state() is None


# ── /usage ────────────────────────────────────────────────────────────────────


class TestUsageCommand:
    def test_portal_window_account_renders_today_week_and_top_tier(self, monkeypatch):
        from agent import account_usage
        top_reset_ms = int(time.time() * 1000) + 20 * 86_400_000
        monkeypatch.setattr("hermes_cli.nous_account.nous_portal_topup_url", lambda info: "https://portal/topup")
        info = SimpleNamespace(
            logged_in=True, paid_service_access=True,
            paid_service_access_info=SimpleNamespace(purchased_credits_remaining=5.0, subscription_credits_remaining=0.0, total_usable_credits=5.0),
            subscription=SimpleNamespace(plan="Window Plan", monthly_credits=0, credits_remaining=0, rollover_credits=0, current_period_end="2026-10-01"),
            usage_window=SimpleNamespace(gauges=_gauges(7_600, 4_000, top_tier_bp=10_000, day_reset=int(time.time() * 1000) + 3600_000,
                                                        week_reset=int(time.time() * 1000) + 3 * 86_400_000, top_reset=top_reset_ms),
                                         auto_top_up_enabled=False, paused_reason=None),
        )
        snap = account_usage.build_nous_credits_snapshot(info)
        assert snap is not None and snap.plan == "Window Plan"
        assert [w.label for w in snap.windows] == ["Today", "This week", "Large models (monthly)"]
        # The monthly row resets on the large-model allowance's own clock, and the used-up line names that date.
        top_local = datetime.fromtimestamp(top_reset_ms / 1000, tz=UTC).astimezone()
        assert snap.windows[2].reset_at == datetime.fromtimestamp(top_reset_ms / 1000, tz=UTC)
        assert (f"Monthly large-model usage is used up. Resets {top_local:%b} {top_local.day}. "
                "Past this, large models use your credits.") in snap.details
        assert [round(w.used_percent) for w in snap.windows] == [76, 40, 100]
        lines = account_usage.render_account_usage_lines(snap)
        joined = "\n".join(lines)
        assert "76% used" in joined and "[" in joined
        assert "40% used" in joined
        assert "100% used" in joined
        assert joined.split("Today")[1].split("This week")[0].count("[") == 1
        assert any(l.startswith(account_usage._WINDOW_TOP_TIER_USED_UP) for l in lines)
        assert "Top-up credits: $5.00" in lines and "Renews: Oct 1, 2026" in lines
        assert not any(l.startswith(("Subscription credits", "Total usable", "Subscription:")) for l in lines)
        assert account_usage._DEPLETED_LINE not in lines

    def test_portal_rows_follow_the_server_list(self, monkeypatch):
        from agent import account_usage
        now = int(time.time() * 1000)
        kw = dict(day_reset=now + 3600_000, week_reset=now + 3 * 86_400_000, top_reset=now + 20 * 86_400_000,
                  month_reset=now + 20 * 86_400_000)
        window = SimpleNamespace(gauges=_gauges(100, 200, **kw))
        info = SimpleNamespace(logged_in=True, paid_service_access=True, paid_service_access_info=None, subscription=None,
                               usage_window=window)
        assert [w.label for w in account_usage.build_nous_credits_snapshot(info).windows] == [
            "Today", "This week", "Large models (monthly)"]  # no month allowance listed: no month row
        window.gauges = _gauges(100, 200, month_bp=3_300, **kw)
        snap = account_usage.build_nous_credits_snapshot(info)
        assert [w.label for w in snap.windows] == ["Today", "This week", "This month", "Large models (monthly)"]
        assert round(snap.windows[2].used_percent) == 33
        # A new allowance renders with the server's label, where the server put it.
        window.gauges = [{"id": "five_hour", "scope": "all", "label": "Next 5 hours", "used_bp": 5_000,
                          "resets_at_ms": now + 3600_000}, *_gauges(100, 200, **kw)]
        assert next(w.label for w in account_usage.build_nous_credits_snapshot(info).windows) == "Next 5 hours"

    def test_an_unknown_reset_keeps_the_allowance_row(self):
        """A valid percent with no usable reset still shows as a window row (no reset phrase), never the
        dollar view: the window plan's details stay on screen."""
        from agent import account_usage
        gauges = [{"id": "day", "scope": "all", "label": "Today", "used_bp": 4_200},
                  {"id": "week", "scope": "all", "label": "This week", "used_bp": 1_000, "resets_at_ms": "soon"}]
        info = SimpleNamespace(logged_in=True, paid_service_access=True, paid_service_access_info=None, subscription=None,
                               usage_window=SimpleNamespace(gauges=gauges, paused_reason=None))
        snap = account_usage.build_nous_credits_snapshot(info)
        assert snap.usage_window and [(w.label, round(w.used_percent), w.reset_at) for w in snap.windows] == [
            ("Today", 42, None), ("This week", 10, None)]
        assert not any("resets" in line for line in account_usage.render_account_usage_lines(snap))

    def test_header_gauge_raises_stale_portal_percents(self, monkeypatch):
        from agent import account_usage
        monkeypatch.setattr("hermes_cli.nous_account.nous_portal_topup_url", lambda info: "https://portal/topup")
        info = SimpleNamespace(
            logged_in=True, paid_service_access=True,
            paid_service_access_info=SimpleNamespace(purchased_credits_remaining=5.0, subscription_credits_remaining=0.0, total_usable_credits=5.0),
            subscription=SimpleNamespace(plan="Window Plan", monthly_credits=0, credits_remaining=0, rollover_credits=0, current_period_end="2026-10-01"),
            usage_window=SimpleNamespace(gauges=_gauges(2_200, 1_100, top_tier_bp=3_400, day_reset=int(time.time() * 1000) + 3600_000,
                                                        week_reset=int(time.time() * 1000) + 3 * 86_400_000),
                                         auto_top_up_enabled=False, paused_reason=None),
        )
        snap = account_usage.build_nous_credits_snapshot(info)
        future = int(time.time() * 1000) + 3600_000
        raised = account_usage.apply_header_gauge(snap, _state(
            window_day_used_bp=4_000, window_week_used_bp=1_600, window_top_tier_used_bp=10_000,
            window_day_resets_at_ms=future, window_week_resets_at_ms=future))
        assert [round(w.used_percent) for w in raised.windows] == [40, 16, 100]
        assert sum(l.startswith(account_usage._WINDOW_TOP_TIER_USED_UP) for l in raised.details) == 1
        # A header behind the portal does not pull the bar backwards.
        behind = _state(window_day_used_bp=100, window_week_used_bp=100, window_top_tier_used_bp=100,
                        window_day_resets_at_ms=future, window_week_resets_at_ms=future)
        assert account_usage.apply_header_gauge(snap, behind) is snap
        # A header from before its gauge reset describes the old window: the portal stands.
        past = int(time.time() * 1000) - 1
        reset = _state(window_day_used_bp=10_000, window_week_used_bp=1_600, window_top_tier_used_bp=None,
                       window_day_resets_at_ms=past, window_week_resets_at_ms=past + 3 * 86_400_000)
        assert [round(w.used_percent) for w in account_usage.apply_header_gauge(snap, reset).windows][:2] == [22, 16]

    def test_portal_paused_window_account(self, monkeypatch):
        from agent import account_usage
        monkeypatch.setattr("hermes_cli.nous_account.nous_portal_topup_url", lambda info: "https://portal/topup")
        info = SimpleNamespace(
            logged_in=True, paid_service_access=False,
            paid_service_access_info=SimpleNamespace(purchased_credits_remaining=0.0),
            subscription=None,
            usage_window=SimpleNamespace(gauges=_gauges(0, 0, day_reset=int(time.time() * 1000) + 1,
                                                        week_reset=int(time.time() * 1000) + 2),
                                         auto_top_up_enabled=False, paused_reason="payment_past_due"),
        )
        lines = account_usage.render_account_usage_lines(account_usage.build_nous_credits_snapshot(info))
        assert account_usage._WINDOW_PAUSED_LINE in lines and account_usage._DEPLETED_LINE not in lines

    def test_portal_keep_going_off_says_so_and_never_says_add_credits(self, monkeypatch):
        """Keep going on credits off: /usage names the setting, the used-up line points at the other
        included usage (a balance doesn't unlock it), and the 402 guidance links billing, not top-up."""
        from agent import account_usage
        from hermes_cli import nous_account
        monkeypatch.setattr("hermes_cli.nous_account.nous_portal_topup_url", lambda info: "https://portal/topup")
        monkeypatch.setattr("hermes_cli.nous_account.nous_portal_billing_url", lambda info: "https://portal/billing")
        now = int(time.time() * 1000)
        window = SimpleNamespace(gauges=_gauges(7_600, 4_000, top_tier_bp=10_000, day_reset=now + 3600_000,
                                                week_reset=now + 3 * 86_400_000, top_reset=now + 20 * 86_400_000),
                                 auto_top_up_enabled=False, paused_reason=None, keep_going_on_credits=False)
        info = SimpleNamespace(
            logged_in=True, paid_service_access=True,
            paid_service_access_info=SimpleNamespace(purchased_credits_remaining=5.0),
            subscription=None, usage_window=window,
        )
        lines = account_usage.render_account_usage_lines(account_usage.build_nous_credits_snapshot(info))
        assert account_usage._WINDOW_KEEP_GOING_OFF_LINE in lines
        assert any(line.endswith("Your other included usage is still available.") for line in lines)
        assert not any("use your credits" in line for line in lines)
        guidance = nous_account.format_usage_window_block_guidance(info)
        # Points at the setting (billing page or /topup), never at the buy-credits modal.
        assert guidance is not None and "https://portal/billing" in guidance and "topup=open" not in guidance
        assert "/topup" in guidance
        for text in [*lines, guidance]:
            assert "\u2014" not in text
        # On (says so, and that credits = the pay-as-you-go balance), or a portal that predates the setting.
        for keep_going in (True, None):
            info.usage_window = SimpleNamespace(**{**vars(window), "keep_going_on_credits": keep_going})
            lines = account_usage.render_account_usage_lines(account_usage.build_nous_credits_snapshot(info))
            assert account_usage._WINDOW_KEEP_GOING_OFF_LINE not in lines
            assert (account_usage._WINDOW_KEEP_GOING_ON_LINE in lines) is (keep_going is True)
            assert any(line.endswith("Past this, large models use your credits.") for line in lines)
            assert "https://portal/topup" in nous_account.format_usage_window_block_guidance(info)

    def test_account_dto_carries_keep_going_on_credits(self):
        from hermes_cli.nous_account import NousUsageWindowInfo, _coerced_dataclass
        assert _coerced_dataclass(NousUsageWindowInfo, {"keep_going_on_credits": False}).keep_going_on_credits is False
        assert _coerced_dataclass(NousUsageWindowInfo, {}).keep_going_on_credits is None

    def test_partial_or_empty_usage_window_block_is_treated_as_absent(self):
        """Gate on a readable gauge list, not on the block's presence: an empty block, or one whose list
        has nothing that stops every model, keeps the dollar usage model and the credit-grant 402 guidance."""
        from agent.billing_usage import usage_model_from_account
        from hermes_cli.nous_account import NousUsageWindowInfo, format_usage_window_block_guidance
        complete = NousUsageWindowInfo(gauges=ct.parse_gauges(_gauges(100, 50)))
        top_only = NousUsageWindowInfo(gauges=ct.parse_gauges([g for g in _gauges() if g["scope"] == "top_tier"]))
        for window, is_window in ((complete, True), (NousUsageWindowInfo(), False), (top_only, False)):
            info = SimpleNamespace(
                logged_in=True, paid_service_access=True, usage_window=window, portal_base_url=None,
                paid_service_access_info=SimpleNamespace(purchased_credits_remaining=1.0, subscription_credits_remaining=18.0,
                                                         total_usable_credits=19.0),
                subscription=SimpleNamespace(plan="Credit Plan", monthly_credits=20.0, credits_remaining=18.0, rollover_credits=0,
                                             current_period_end=None),
            )
            assert usage_model_from_account(info).available is (not is_window)
            assert (format_usage_window_block_guidance(info) is not None) is is_window

    def test_portal_legacy_account_is_unchanged(self, monkeypatch):
        from agent import account_usage
        monkeypatch.setattr("hermes_cli.nous_account.nous_portal_topup_url", lambda info: "https://portal/topup")
        info = SimpleNamespace(
            logged_in=True, paid_service_access=True,
            paid_service_access_info=SimpleNamespace(purchased_credits_remaining=1.0, subscription_credits_remaining=18.0, total_usable_credits=19.0),
            subscription=SimpleNamespace(plan="Credit Plan", monthly_credits=20.0, credits_remaining=18.0, rollover_credits=0, current_period_end=None),
            usage_window=None,
        )
        snap = account_usage.build_nous_credits_snapshot(info)
        assert snap is not None and not snap.usage_window
        lines = account_usage.render_account_usage_lines(snap)
        # Byte-for-byte today's layout: emoji title, provider line, "% remaining (% used)", portal link.
        assert lines[0] == "📈 Nous credits" and lines[1] == "Provider: nous (Credit Plan)"
        assert any(l.startswith("Subscription: 90% remaining (10% used)") for l in lines)
        assert "Subscription credits: $18.00" in lines
        assert "Top up: https://portal/topup" in lines and "(or run /topup)" in lines
        assert not any("% used  " in l or "Top up via" in l for l in lines)

    def test_fixture_window_state_renders(self, monkeypatch):
        from agent import account_usage
        monkeypatch.setattr(ct, "dev_fixture_credits_state", _PLAN_STATES["top_tier_blocked"])
        lines = account_usage.nous_credits_lines()
        joined = "\n".join(lines)
        assert "40% used" in joined and "This week" in joined and "[" in joined
        assert any(l.startswith(account_usage._WINDOW_TOP_TIER_USED_UP) for l in lines) and "Top-up credits: $0.00" in lines
        assert "(dev fixture — HERMES_DEV_CREDITS_FIXTURE)" in lines


# ── classifier ────────────────────────────────────────────────────────────────


class TestClassifier:
    def _err(self, status, code, message):
        class E(Exception):
            pass
        e = E(message)
        e.status_code = status
        e.body = {"error": {"code": code, "message": message}}
        e.response = SimpleNamespace(headers={})
        return e

    def test_window_402_is_billing_and_not_retryable(self):
        from agent.error_classifier import FailoverReason, classify_api_error
        c = classify_api_error(self._err(402, "usage_window_and_credits_exhausted",
                                         "Your included Hermes usage for today is used up and you have no top-up credits. It resets at 18:00."),
                               provider="nous", model="example/model-c")
        assert c.reason == FailoverReason.billing and not c.retryable
        assert c.error_context.get("usage_window") is True

    def test_other_402_billing_codes_keep_main_text_rules(self):
        """Only the Nous window code is decisive on a 402; any other provider's billing code
        with transient wording stays a rate limit, as on main."""
        from agent.error_classifier import FailoverReason, classify_api_error
        c = classify_api_error(self._err(402, "insufficient_quota",
                                         "quota exceeded, please retry after the window resets"),
                               provider="example", model="example/model-a")
        assert c.reason == FailoverReason.rate_limit
        assert not c.error_context.get("usage_window")

    def test_window_402_carries_the_server_wording_when_sent(self):
        from agent.error_classifier import classify_api_error
        err = self._err(402, "usage_window_and_credits_exhausted", "Your included Hermes usage for today is used up.")
        assert "display_message" not in classify_api_error(err, provider="nous", model="m").error_context
        err.body = {"status": 402, "code": "usage_window_and_credits_exhausted", "message": "m",
                    "display_message": "Daily limit hit. Back at 4pm."}
        c = classify_api_error(err, provider="nous", model="m")
        assert c.error_context == {"usage_window": True, "display_message": "Daily limit hit. Back at 4pm."}
        # Never on any other 402.
        other = self._err(402, "insufficient_credit", "no credits")
        other.body["display_message"] = "x"
        assert "display_message" not in classify_api_error(other, provider="nous", model="m").error_context

    def test_window_402_never_falls_back_to_another_model(self):
        """A plan limit is the user's call (the limit card offers a switch): never the fallback chain."""
        from agent.error_classifier import classify_api_error
        c = classify_api_error(self._err(402, "usage_window_and_credits_exhausted", "used up"), provider="nous", model="m")
        assert c.should_fallback is False

    def test_window_402_code_from_another_upstream_keeps_main_rules(self):
        from agent.error_classifier import FailoverReason, classify_api_error
        c = classify_api_error(self._err(402, "usage_window_and_credits_exhausted", "quota exceeded, retry after the window resets"),
                               provider="example", model="m", base_url="https://example.invalid/v1")
        assert c.reason == FailoverReason.rate_limit and not c.error_context.get("usage_window")

    def test_window_code_on_a_429_from_another_upstream_keeps_main_rules(self):
        """The code is honoured only through the gated 402 path: on any other status (a 429 from another
        provider or a custom endpoint) it is not a billing code, so the turn backs off and retries as on main."""
        from agent.error_classifier import FailoverReason, classify_api_error
        for provider, base_url in (("example", "https://example.invalid/v1"), ("custom", "https://custom.example.invalid/v1")):
            c = classify_api_error(self._err(429, "usage_window_and_credits_exhausted", "Too many requests"),
                                   provider=provider, model="m", base_url=base_url)
            assert c.reason == FailoverReason.rate_limit and c.retryable
            assert not c.error_context.get("usage_window")

    def test_a_plain_403_still_classifies_as_before(self):
        from agent.error_classifier import FailoverReason, classify_api_error
        c = classify_api_error(self._err(403, "forbidden", "Forbidden"), provider="nous", model="m")
        assert c.reason != FailoverReason.model_not_found


# ── 402 / 403 header capture ──────────────────────────────────────────────────


class TestBlockedResponseCapture:
    def test_a_nous_402_refreshes_the_credits_state(self, monkeypatch):
        from agent.rate_limit_credits import RateLimitCreditsMixin

        class Agent(RateLimitCreditsMixin):
            provider = "nous"
            model = "example/model-c"
            base_url = "https://inference-api.nousresearch.com/v1"
            _credits_state = None
            _credits_session_start_micros = None
            _credits_latch = new_credits_latch()
            notice_callback = None
            notice_clear_callback = None

            def __init__(self):
                self.emitted = []

            def _emit_notice(self, n):
                self.emitted.append(n)

            def _emit_notice_clear(self, k):
                pass

            def _credits_notices_enabled(self):
                return True

        agent = Agent()
        agent.notice_callback = lambda n: None
        monkeypatch.delenv("HERMES_DEV_CREDITS_FIXTURE", raising=False)
        # A warm catalog (a cold one defers the stop to a background re-warm, like a depleted account).
        monkeypatch.setattr(ct, "rewarm_pricing_before_depleted_notice", lambda agent: False)
        err = SimpleNamespace(status_code=402, response=SimpleNamespace(
            headers=_window_headers(4_000, 2_000, top_tier_bp=10_000, source=None)))
        # Mirror the turn_api_error hook: a 402 from Nous feeds the same capture path as a stream open.
        agent._capture_credits(err.response, refused=True)
        assert agent._credits_state is not None and agent._credits_state.top_tier_exhausted
        assert [n.key for n in agent.emitted] == [CREDITS_WINDOW_LIMIT_KEY]
        assert "You've hit your monthly large-model limit" in agent.emitted[0].text

    def test_non_streaming_success_feeds_the_header_capture(self):
        """A non-streaming 200 must observe x-nous-credits-* like a stream open does
        (spend_source / window gauges otherwise go stale until the next stream or 402)."""
        from agent.chat_completion_headers import _chat_completion_capturing_headers

        seen = []
        agent = SimpleNamespace(
            provider="nous", _credits_state=_state(),
            _capture_rate_limits=lambda r: seen.append(("rl", r)),
            _capture_credits=lambda r: seen.append(("credits", r)),
            _capture_nous_model_switch=lambda r: None,
            _check_openrouter_cache_status=lambda r: None,
        )
        parsed = SimpleNamespace(choices=[])
        raw = SimpleNamespace(headers=_window_headers(), parse=lambda: parsed)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            with_raw_response=SimpleNamespace(create=lambda **kw: raw),
            create=lambda **kw: pytest.fail("plain create must not be used when raw is available"))))

        assert _chat_completion_capturing_headers(agent, client, {"model": "m"}) is parsed
        assert [k for k, _ in seen] == ["rl", "credits"] and all(r is raw for _, r in seen)

        # A capture that raises never fails the turn; the others still run.
        seen.clear()
        def _boom(_r):
            raise ValueError("bad header")
        agent._capture_rate_limits = _boom
        assert _chat_completion_capturing_headers(agent, client, {"model": "m"}) is parsed
        assert [k for k, _ in seen] == ["credits"]

        # Other providers keep the plain call they had before (no raw response).
        plain = SimpleNamespace(choices=["plain"])
        other = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            with_raw_response=SimpleNamespace(create=lambda **kw: pytest.fail("raw must not be used off Nous")),
            create=lambda **kw: plain)))
        assert _chat_completion_capturing_headers(
            SimpleNamespace(provider="example"), other, {"model": "m"}) is plain

        # A client without the raw surface keeps the plain call and captures nothing.
        seen.clear()
        agent._capture_rate_limits = lambda r: seen.append(("rl", r))
        bare = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: plain)))
        assert _chat_completion_capturing_headers(agent, bare, {"model": "m"}) is plain
        assert seen == []

    def test_non_streaming_non_window_keeps_mains_behaviour(self):
        """Not a usage-window response: nothing is captured. A session not known to be on a window
        plan (no state yet, or a credit-grant state) keeps main's plain call."""
        from agent.chat_completion_headers import _chat_completion_capturing_headers

        seen = []
        agent = SimpleNamespace(
            provider="nous", _credits_state=_state(),
            _capture_rate_limits=lambda r: seen.append("rl"), _capture_credits=lambda r: seen.append("credits"),
            _capture_nous_model_switch=lambda r: seen.append("switch"),
            _check_openrouter_cache_status=lambda r: seen.append("or"),
        )
        parsed = SimpleNamespace(choices=[])
        grant = {**_window_headers(), "x-nous-credits-denominator-kind": "subscription_cap"}
        for headers in ({}, grant, {"x-nous-credits-remaining-micros": "1"}):
            raw = SimpleNamespace(headers=headers, parse=lambda: parsed)
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=lambda **kw: raw), create=lambda **kw: None)))
            assert _chat_completion_capturing_headers(agent, client, {"model": "m"}) is parsed
        assert seen == []

        plain = SimpleNamespace(choices=["plain"])
        known = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            with_raw_response=SimpleNamespace(create=lambda **kw: pytest.fail("non-window: plain call")),
            create=lambda **kw: plain)))
        for state in (None, _state(denominator_kind="subscription_cap")):
            agent._credits_state = state
            assert _chat_completion_capturing_headers(agent, known, {"model": "m"}) is plain
        assert seen == []

    def test_response_is_usage_window(self):
        assert ct.response_is_usage_window(SimpleNamespace(headers=_window_headers()))
        grant = {**_window_headers(), "x-nous-credits-denominator-kind": "subscription_cap"}
        partial = {k: v for k, v in _window_headers().items() if k != "x-nous-credits-window-gauges"}
        for headers in ({}, grant, partial):
            assert not ct.response_is_usage_window(SimpleNamespace(headers=headers))
        assert not ct.response_is_usage_window(None)

    @pytest.mark.parametrize("provider, status, headers, captured", [
        ("nous", 402, "window", True),
        ("nous", 403, "window", True),
        ("openrouter", 402, "window", False),  # window headers from any other provider: main's behaviour
        ("nous", 402, "grant", False),         # a credit-grant account: main's behaviour
        ("nous", 500, "window", False),
    ])
    def test_handle_api_error_captures_only_a_nous_window_refusal(self, provider, status, headers, captured):
        """A refused (402/403) Nous request's usage-window headers reach the credits capture, marked refused;
        nothing else does. The turn is stopped right after the capture (interpreter shutdown) so only the
        hook runs."""
        from unittest.mock import patch

        from agent.turn_api_error import handle_api_error
        from run_agent import AIAgent
        with patch("model_tools.get_tool_definitions", return_value=[]), \
                patch("model_tools.check_toolset_requirements", return_value={}), \
                patch("agent.process_bootstrap.OpenAI"):
            agent = AIAgent(api_key="test-key-1234567890", base_url="https://inference-api.nousresearch.com/v1",
                            quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent.provider = provider
        h = _window_headers(10_000, source=None)
        if headers == "grant":
            h = {**h, "x-nous-credits-denominator-kind": "subscription_cap"}
        error = type("_Err", (Exception,), {"status_code": status})("refused")
        error.response = SimpleNamespace(headers=h, status_code=status)
        seen = []
        with patch.object(agent, "_capture_credits", lambda r, **kw: seen.append((r, kw))), \
                patch("tools.interpreter_shutdown.interpreter_shutting_down", return_value=True):
            verdict = handle_api_error(
                agent, api_error=error, _retry=None, thinking_spinner=None, messages=[], api_messages=[],
                api_kwargs={}, system_message="", active_system_prompt="", conversation_history=[],
                approx_tokens=0, retry_count=0, max_retries=1, compression_attempts=0, max_compression_attempts=1,
                api_call_count=1, api_request_id="r", api_start_time=0.0, effective_task_id="t", turn_id="u")
        assert verdict.action == "return"
        assert seen == ([(error.response, {"refused": True})] if captured else [])


# ── picker ────────────────────────────────────────────────────────────────────


class TestPicker:
    def test_custom_entry_is_always_offered(self, monkeypatch):
        # Window plans no longer restrict models, so every picker keeps "Enter custom model name".
        from hermes_cli import auth_model_picker as picker
        captured = {}

        def fake_radiolist(title, choices, **kw):
            captured["choices"] = choices
            return -1

        monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", fake_radiolist)
        picker._prompt_model_selection(["a/m1", "b/m2"])
        assert picker._CUSTOM_LABEL in captured["choices"]

    def test_account_payload_usage_window_is_parsed(self):
        from hermes_cli import nous_account as na
        payload = {
            "user": {"email": "x@y.z"}, "organisation": {"id": "o1"},
            "paid_service_access": {"allowed": True},
            "usage_window": {"gauges": _gauges(1200, 900), "auto_top_up_enabled": False,
                             "paused_reason": "payment_past_due",
                             # The fixed per-window fields older clients read: Hermes ignores them.
                             "day_used_bp": 9999, "week_used_bp": 9999},
        }
        info = na._info_from_account_payload(payload, state={}, portal_base_url=None)
        assert info.usage_window is not None and info.usage_window.paused_reason == "payment_past_due"
        assert [(g.id, g.used_bp) for g in info.usage_window.gauges] == [("day", 1200), ("week", 900), ("top_tier", 0)]
        assert info.usage_window.gauges[1].resets_at_ms == WEEK_RESET
        legacy = na._info_from_account_payload({"user": {}, "paid_service_access": {"allowed": True}}, state={}, portal_base_url=None)
        assert legacy.usage_window is None


class TestRefusalReadsAsTheStopLine:
    """A window 402's chat text and error card say what the stop line says, nothing generic, plus the
    structured limit (agent/plan_limits.py) the surfaces turn into buttons."""

    @pytest.fixture(autouse=True)
    def _no_plan_caches(self, monkeypatch):
        from agent import plan_limits as pl
        monkeypatch.setattr(pl, "_plan_usage", lambda: None)
        monkeypatch.setattr(pl, "_portal_free_models", list)
        monkeypatch.setattr(pl, "cached_pricing", lambda base_url: {})

    def _agent(self, **state_kw):
        state = _state(window_day_used_bp=10_000, spend_source=None, **state_kw)
        return SimpleNamespace(_credits_state=state, _credits_latch=new_credits_latch(), model="example/model-b",
                               base_url="https://inference-api.nousresearch.com/v1")

    def test_the_refusal_text_is_the_stop_line(self):
        from agent.conversation_loop import _usage_window_refusal
        window = SimpleNamespace(error_context={"usage_window": True})
        agent = self._agent(purchased_micros=5_000_000, keep_going_on_credits=False)
        text, resets_at, limit = _usage_window_refusal(agent, window)
        assert text == f"You've hit your daily limit · resets {format_window_reset(DAY_RESET)} · turn on Keep going in /topup"
        assert resets_at == DAY_RESET / 1000
        assert limit["gauge_id"] == "day" and limit["scope"] == "all" and limit["message"] == text
        assert limit["credits"] == {"action": "use_credits"} and limit["current_model"] == "example/model-b"

    def test_only_a_billing_refusal_with_the_window_mark_benches_one_model(self):
        """The model-scoped bench applies to the usage-window 402 alone, and wins over the unverified-billing
        sizing; any other reason keeps the pool's own handling."""
        from agent.credential_pool import FAILURE_REASON_BILLING_UNVERIFIED
        from agent.credential_pool_model_cooldowns import FAILURE_REASON_USAGE_WINDOW, pool_failure_reason
        from agent.error_classifier import FailoverReason
        window = {"usage_window": True}
        assert pool_failure_reason(FailoverReason.billing, window, True) == FAILURE_REASON_USAGE_WINDOW
        assert pool_failure_reason(FailoverReason.rate_limit, window, False) == FailoverReason.rate_limit.value
        assert pool_failure_reason(FailoverReason.billing, {}, True) == FAILURE_REASON_BILLING_UNVERIFIED
        assert pool_failure_reason(FailoverReason.billing, None, False) == FailoverReason.billing.value

    def test_the_pool_bench_lasts_until_the_refused_allowance_resets(self):
        """The 402's own headers are captured before recovery, so the pool gets the window's reset (not its
        default 60 s / 1 h bench); a state naming no reset leaves the default."""
        from agent.turn_recovery import _usage_window_reset_context
        assert _usage_window_reset_context(self._agent()) == {"reset_at": DAY_RESET / 1000}
        assert _usage_window_reset_context(SimpleNamespace(_credits_state=None)) == {}

    def test_a_regular_model_refused_by_its_day_names_the_day_everywhere(self, monkeypatch):
        """Day and Large-model allowances both full on a 402 for a model that is not top tier: the line, the retry
        time, the pool bench and the card all name the day (what stops this model), not the month-away reset."""
        from agent import plan_limits as pl
        from agent.conversation_loop import _usage_window_refusal
        from agent.turn_recovery import _usage_window_reset_context
        monkeypatch.setattr(pl, "cached_pricing", lambda base_url: {
            "example/model-b": {"prompt": "0.0000004", "completion": "0.000004", "usage_level": "medium"}})
        agent = self._agent(window_top_tier_used_bp=10_000, window_top_tier_resets_at_ms=MONTH_RESET)
        text, resets_at, limit = _usage_window_refusal(agent, SimpleNamespace(error_context={"usage_window": True}))
        assert text.startswith(f"You've hit your daily limit · resets {format_window_reset(DAY_RESET)}")
        assert resets_at == DAY_RESET / 1000 and limit["gauge_id"] == "day"
        assert _usage_window_reset_context(agent) == {"reset_at": DAY_RESET / 1000}

    def test_a_model_of_unknown_tier_is_never_promised_the_earlier_reset(self):
        """Day and Large-model allowances both full on a 402 while the catalog has no row for the model (cold, or
        its 300 s cache lapsed): it may be Large, so the line, retry time, pool bench and card name the later
        Large-model reset, never a daily one it would still be refused after."""
        from agent.conversation_loop import _usage_window_refusal
        from agent.turn_recovery import _usage_window_reset_context
        agent = self._agent(window_top_tier_used_bp=10_000, window_top_tier_resets_at_ms=MONTH_RESET)
        text, resets_at, limit = _usage_window_refusal(agent, SimpleNamespace(error_context={"usage_window": True}))
        assert text.startswith(f"You've hit your monthly large-model limit · resets {format_window_reset(MONTH_RESET)}")
        assert resets_at == MONTH_RESET / 1000 and limit["gauge_id"] == "top_tier"
        assert _usage_window_reset_context(agent) == {"reset_at": MONTH_RESET / 1000}

    def test_keep_going_off_with_no_balance_says_both_steps_everywhere(self):
        """A new subscriber's first limit (setting off by default, no top-up): the stop line, the next-steps
        payload and the 402 guidance all name both steps, since either one alone is still refused."""
        from agent.conversation_loop import _usage_window_refusal
        from hermes_cli import nous_account
        agent = self._agent(purchased_micros=0, keep_going_on_credits=False)
        text, _resets_at, limit = _usage_window_refusal(agent, SimpleNamespace(error_context={"usage_window": True}))
        assert text.endswith(" · add credits and turn on Keep going in /topup")
        assert limit["credits"] == {"action": "add_credits_and_keep_going"}
        window = SimpleNamespace(gauges=ct.parse_gauges(_gauges(100, 50)), auto_top_up_enabled=False,
                                 paused_reason=None, keep_going_on_credits=False)
        for purchased in (0.0, None):
            info = SimpleNamespace(logged_in=True, paid_service_access=True, usage_window=window, portal_base_url=None,
                                   paid_service_access_info=SimpleNamespace(purchased_credits_remaining=purchased))
            guidance = nous_account.format_usage_window_block_guidance(info)
            assert guidance.startswith("Add credits and turn on Keep going in /topup")
        # Any other billing failure keeps its own copy and carries no limit.
        assert _usage_window_refusal(agent, SimpleNamespace(error_context={})) == ("", None, None)

    def test_the_server_wording_replaces_the_stop_line_when_sent(self):
        """``display_message`` on the 402: shown verbatim (a reword ships with the API); the retry clock
        still comes from the gauges. Absent: Hermes's own stop line, as above."""
        from agent.conversation_loop import _usage_window_refusal
        agent = self._agent()
        server = SimpleNamespace(error_context={"usage_window": True, "display_message": "  Daily limit hit. Back at 4pm.  "})
        text, resets_at, limit = _usage_window_refusal(agent, server)
        assert (text, resets_at) == ("Daily limit hit. Back at 4pm.", DAY_RESET / 1000)
        assert limit["message"] == "Daily limit hit. Back at 4pm."
        blank = SimpleNamespace(error_context={"usage_window": True, "display_message": "   "})
        assert _usage_window_refusal(agent, blank)[0].startswith("You've hit your daily limit")

    def test_the_server_sentence_keeps_it_a_window_refusal_when_local_state_is_quiet(self):
        """Local gauges show nothing spent (stale snapshot / thinks credits pay) but the API refused with
        the window code: still a window refusal in the server's words, never the generic billing wall
        (which on desktop meant an "Out of credits" card + toast + banner)."""
        from agent.conversation_loop import _usage_window_refusal
        quiet = SimpleNamespace(_credits_state=_state(), _credits_latch=new_credits_latch(), model="example/model-b",
                                base_url="https://inference-api.nousresearch.com/v1")
        refused = SimpleNamespace(error_context={"usage_window": True},
                                  message="Your included Hermes usage for this week is used up.")
        assert _usage_window_refusal(quiet, refused)[0] == "Your included Hermes usage for this week is used up."
        assert _usage_window_refusal(quiet, SimpleNamespace(error_context={"usage_window": True}))[0] == ""

    def test_the_refusal_names_the_plans_recommended_model(self, monkeypatch):
        from agent import plan_limits as pl
        from agent.conversation_loop import _usage_window_refusal
        monkeypatch.setattr(pl, "_plan_usage", lambda: SimpleNamespace(featured_model_ids=("example/model-b", "example/model-free:free")))
        monkeypatch.setattr(pl, "cached_pricing", lambda base_url: {
            "example/model-b": {"prompt": "0.0000004", "completion": "0.000004", "name": "Model B"},
            "example/model-free:free": {"prompt": "0", "completion": "0", "name": "Model Free"},
        })
        _text, _at, limit = _usage_window_refusal(self._agent(), SimpleNamespace(error_context={"usage_window": True}))
        # The day is spent: only the free pick still works; never the current model.
        assert limit["suggestions"] == [{"model": "example/model-free:free", "label": "Model Free", "note": "free", "provider": "nous"}]
        assert limit["credits"] == {"action": "add_credits"}

    def test_the_text_never_carries_a_command_hint(self):
        """The old stop-line hint (", or /model x") is gone: suggestions travel structured."""
        from agent.conversation_loop import _usage_window_refusal
        text = _usage_window_refusal(self._agent(), SimpleNamespace(error_context={"usage_window": True}))[0]
        assert "/model" not in text

    def test_result_and_error_surface_carry_the_sentence(self, monkeypatch):
        from agent import conversation_loop as cl
        from agent.error_surface import build_error_surface_from_result
        monkeypatch.setattr(cl, "_billing_or_entitlement_message", lambda **kw: "Add credits to keep going now: https://x")
        classified = SimpleNamespace(billing_unverified=False, retryable=False, reason=SimpleNamespace(value="billing"))
        kw = dict(classified=classified, summary="HTTP 402", messages=[], api_call_count=1,
                  provider="nous", base_url="https://inference-api.nousresearch.com/v1", model="m")
        sentence = "You've hit your daily limit · resets 15:56 · /topup to add credits"
        limit = {"gauge_id": "day", "scope": "all", "message": sentence, "credits": {"action": "add_credits"},
                 "suggestions": [{"model": "a/b:free", "label": "A", "note": "free", "provider": "nous"}]}
        result = cl._billing_failure_result(**kw, usage_window=(sentence, 1_800_000_000.0, limit))
        assert result["final_response"] == sentence and result["failure_reason"] == "billing"
        assert result["usage_limit"] == limit
        surface = build_error_surface_from_result(result, provider="nous", model="m")
        assert surface["layer"] == "billing" and surface["code"] == "usage_window" and surface["message"] == sentence
        assert surface["usage_limit"] == limit
        # An older 2-item tuple still reads (no structured limit).
        two = cl._billing_failure_result(**kw, usage_window=(sentence, 1_800_000_000.0))
        assert "usage_limit" not in two and "usage_limit" not in build_error_surface_from_result(two, provider="nous", model="m")
        # Retry at the reset is offered: the card's "Retry at 15:56".
        assert surface["retryable"] is True and surface["resets_at"] == 1_800_000_000.0
        # Without it (every other account): the generic billing result and card, unchanged.
        plain = cl._billing_failure_result(**kw)
        assert "usage_window_message" not in plain and plain["final_response"].startswith("Billing or credits exhausted")
        assert build_error_surface_from_result(plain, provider="nous", model="m")["code"] == "billing"


class TestRebuildMemory:
    """A rebuilt agent (fresh latch, same session) must not re-show a window line it already showed."""

    def _emit(self, session_id, latch, state):
        ct.restore_window_notices(session_id, latch)
        show, clear = _run(state, latch)
        ct.remember_window_notices(session_id, latch)
        return show, clear

    def test_same_line_not_reshown_after_rebuild(self, monkeypatch):
        monkeypatch.setattr(ct, "_seen_window_notices", {})
        spent = _state(window_day_used_bp=10_000, purchased_micros=5_000_000, spend_source="payg")
        show, _ = self._emit("s1", new_credits_latch(), spent)
        assert CREDITS_WINDOW_PAYG_KEY in _keys(show)
        show, clear = self._emit("s1", new_credits_latch(), spent)
        assert show == [] and clear == []

    def test_changed_line_still_shown_after_rebuild(self, monkeypatch):
        monkeypatch.setattr(ct, "_seen_window_notices", {})
        self._emit("s2", new_credits_latch(),
                   _state(window_day_used_bp=10_000, purchased_micros=5_000_000, spend_source="payg"))
        show, _ = self._emit("s2", new_credits_latch(),
                             _state(window_day_used_bp=10_000, purchased_micros=0, spend_source="payg"))
        assert CREDITS_WINDOW_LIMIT_KEY in _keys(show)

    def test_recovery_clears_after_rebuild(self, monkeypatch):
        monkeypatch.setattr(ct, "_seen_window_notices", {})
        self._emit("s3", new_credits_latch(),
                   _state(window_day_used_bp=10_000, purchased_micros=5_000_000, spend_source="payg"))
        _show, clear = self._emit("s3", new_credits_latch(), _state(window_day_used_bp=500))
        assert CREDITS_WINDOW_PAYG_KEY in clear

    def test_approach_warning_not_reshown_after_rebuild(self, monkeypatch):
        monkeypatch.setattr(ct, "_seen_window_notices", {})
        near = _state(window_day_used_bp=9_200)
        show, _ = self._emit("s4", new_credits_latch(), near)
        assert any(k.startswith("credits.window.") for k in _keys(show))
        show, _ = self._emit("s4", new_credits_latch(), near)
        assert show == []

    def test_other_session_unaffected(self, monkeypatch):
        monkeypatch.setattr(ct, "_seen_window_notices", {})
        spent = _state(window_day_used_bp=10_000, purchased_micros=5_000_000, spend_source="payg")
        self._emit("s5", new_credits_latch(), spent)
        show, _ = self._emit("s6", new_credits_latch(), spent)
        assert CREDITS_WINDOW_PAYG_KEY in _keys(show)


def test_a_refused_request_keeps_the_last_known_top_up_balance():
    """The gateway's 402 sends only the window headers. The capture must not read that as "no credits"."""
    from types import SimpleNamespace as NS
    from agent.rate_limit_credits import RateLimitCreditsMixin

    class _Agent(RateLimitCreditsMixin):
        provider = "nous"
        notice_callback = notice_clear_callback = None
        _credits_session_start_micros = None

    agent = _Agent()
    agent._credits_state = parse_credits_headers(_window_headers(9_800, 500, purchased=8.3, source="included"))
    refused = {k: v for k, v in _window_headers(10_000, 500, source=None).items()
               if not k.startswith("x-nous-credits-purchased-")}
    agent._capture_credits(NS(headers=refused))
    assert agent._credits_state.window_exhausted
    assert agent._credits_state.purchased_micros == 8_300_000 and agent._credits_state.purchased_usd == "8.30"


def test_leaving_a_window_plan_clears_its_limit_line():
    latch = new_credits_latch()
    _run(_state(window_day_used_bp=10_000), latch)
    assert CREDITS_WINDOW_LIMIT_KEY in latch["active"]
    credit_plan = CreditsState(version=1, denominator_kind="subscription_cap", paid_access=True, from_header=True,
                               captured_at=time.time(), purchased_usd="0.00", remaining_micros=5_000_000,
                               subscription_micros=5_000_000, subscription_limit_micros=10_000_000,
                               subscription_limit_usd="10.00")
    _show, clear = _run(credit_plan, latch)
    assert CREDITS_WINDOW_LIMIT_KEY in clear and CREDITS_WINDOW_LIMIT_KEY not in latch["active"]


def test_window_headers_from_another_upstream_are_ignored():
    """Only Nous replies can put a session on a window plan (no window UI or Keep going button from a proxy)."""
    from types import SimpleNamespace as NS
    from agent.rate_limit_credits import RateLimitCreditsMixin

    class _Agent(RateLimitCreditsMixin):
        provider = "example"
        base_url = "https://example.invalid/v1"
        notice_callback = notice_clear_callback = None
        _credits_session_start_micros = None
        _credits_state = None

    agent = _Agent()
    agent._capture_credits(NS(headers=_window_headers(10_000, 500)))
    assert agent._credits_state is None


def test_an_unmetered_reply_raises_no_limit_stop():
    """An admitted reply without a spend source (the server skipped metering) is no refusal: no stop line."""
    from types import SimpleNamespace as NS
    from agent.rate_limit_credits import RateLimitCreditsMixin

    class _Agent(RateLimitCreditsMixin):
        provider = "nous"
        notice_callback = notice_clear_callback = None
        _credits_session_start_micros = None
        _credits_state = None

    agent = _Agent()
    agent._capture_credits(NS(headers=_window_headers(4_000, 2_000, top_tier_bp=10_000, source=None)))
    assert agent._credits_state.unmetered and ct._window_hard_stop(agent._credits_state, model_is_free=False) is None
    agent._capture_credits(NS(headers=_window_headers(4_000, 2_000, top_tier_bp=10_000, source=None)), refused=True)
    assert "large-model limit" in ct._window_hard_stop(agent._credits_state, model_is_free=False)
