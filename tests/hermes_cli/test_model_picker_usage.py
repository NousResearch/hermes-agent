"""Window vs. credit-grant chrome for the model pickers."""

from dataclasses import replace
from types import SimpleNamespace

from hermes_cli.auth_model_picker import _ModelPickerRows
from tests.usage_window_fixtures import FAR, usage_window
from hermes_cli.model_picker_usage import (
    GAUGE_MONTH,
    GAUGE_TODAY,
    GAUGE_WEEK,
    TOP_TIER_FOOTNOTE,
    format_percent_used,
    picker_usage_from_account,
    plan_gauge_lines,
    render_bar,
    usage_levels,
)


def test_render_bar_fills_with_the_fraction_given():
    assert render_bar(1.0) == "[||||||||||]"
    assert render_bar(0.0) == "[          ]"
    assert "|" in render_bar(0.01) and render_bar(0.01) != "[||||||||||]"
    assert render_bar(0.999) != "[||||||||||]"


def test_percent_used_matches_the_portal_rule():
    assert format_percent_used(0.0) == "0%"
    assert format_percent_used(0.0001) == "<1%"
    assert format_percent_used(0.0047) == "<1%"
    assert format_percent_used(0.01) == "1%"
    assert format_percent_used(0.996) == "99%"
    assert format_percent_used(0.9999) == "99%"
    assert format_percent_used(1.0) == "100%"


def test_window_gauges_read_as_used_and_fill_up():
    # 47 bp of the day used → "<1% used", one cell filled (never "99% left").
    usage = _window_usage(day_used_bp=47, week_used_bp=0, top_tier_used_bp=10_000)
    lines = plan_gauge_lines(usage)
    today = next(l for l in lines if l.startswith("today"))
    week = next(l for l in lines if l.startswith("week"))
    top = next(l for l in lines if l.startswith("large models"))
    assert "[|         ]" in today and "<1% used" in today
    assert "[          ]" in week and "0% used" in week
    assert "[||||||||||]" in top and "100% used" in top
    assert usage.gauge(GAUGE_TODAY).style == "green" and usage.gauge("top_tier").style == "red"


def test_usage_level_is_the_gateway_word_never_computed_here():
    pricing = {
        "small": {"prompt": "0.00000014", "completion": "0.00000028", "usage_level": "low"},
        "medium": {"prompt": "0.000001", "completion": "0.0000032", "usage_level": "medium"},
        "large": {"prompt": "0.000005", "completion": "0.000025", "usage_level": "top_tier", "top_tier": True},
        "gift:free": {"prompt": "0", "completion": "0", "usage_level": "free"},
    }
    assert usage_levels(list(pricing), pricing) == {
        "small": "low", "medium": "medium", "large": "top_tier", "gift:free": "free"}
    # Same prices with no gateway word: Hermes does not invent low / medium.
    bare = {m: {k: v for k, v in p.items() if k != "usage_level"} for m, p in pricing.items()}
    assert usage_levels(list(bare), bare) == {"large": "top_tier", "gift:free": "free"}
    # Unknown price is not a promise of free; an unknown word is ignored.
    assert usage_levels(["u", "x"], {"u": {}, "x": {"usage_level": "huge"}}) == {}
    assert usage_levels(["large"], pricing) == {"large": "top_tier"}


def test_window_picker_shows_the_usage_word_not_prices_or_bars():
    usage = picker_usage_from_account(SimpleNamespace(
        logged_in=True,
        subscription=SimpleNamespace(plan="Window Plan"),
        usage_window=SimpleNamespace(**usage_window(
            day_used_bp=2000, week_used_bp=1000, top_tier_used_bp=0,
            day_resets_at_ms=1, week_resets_at_ms=2, paused_reason=None,
        )),
    ))
    assert usage is not None
    rows = _ModelPickerRows(
        ["cheap/m", "pricey/m"],
        {
            "cheap/m": {"prompt": "0.00000006", "completion": "0.00000012", "usage_level": "low"},
            "pricey/m": {"prompt": "0.000003", "completion": "0.000015", "usage_level": "top_tier",
                         "top_tier": True},
        },
        current_model="", sale_chrome=True, usage=usage,
    )
    cheap, pricey = rows.label("cheap/m"), rows.label("pricey/m")
    assert "$" not in cheap and "$" not in pricey and "[" not in cheap and "[" not in pricey
    # Only Large / Free carry a tag; a Small / Medium row is the bare name.
    assert cheap.strip() == "cheap/m" and pricey.rstrip().endswith("Large")
    assert not pricey.startswith("★")
    assert len(cheap.rstrip()) <= len(pricey)  # one aligned column
    header = "\n".join(rows.description_lines())
    assert "Usage" in header and "bar" not in header and "★" not in header
    assert rows.footnote() == TOP_TIER_FOOTNOTE
    desc = rows.description_lines()
    today_line = next(line for line in desc if "today" in line.lower())
    week_line = next(line for line in desc if "week" in line.lower())
    top_line = next(line for line in desc if "large models" in line.lower())
    assert today_line != week_line != top_line
    assert "Window Plan" in desc[0] and "included" in desc[0]


def test_footnote_only_when_a_top_tier_row_is_listed():
    rows = _ModelPickerRows(["cheap/m"], {"cheap/m": {"prompt": "0.0000001", "completion": "0.0000002",
                                                       "usage_level": "low"}},
                            current_model="", sale_chrome=True, usage=_window_usage())
    assert rows.footnote() == ""
    # Credit-grant accounts never get it.
    assert _ModelPickerRows(["x"], {"x": {"top_tier": True}}, current_model="", sale_chrome=True).footnote() == ""


def test_credit_grant_account_gets_no_plan_line():
    """Credit-grant accounts get no plan chrome: pickers stay exactly as on main (no gauge, no badge)."""
    assert picker_usage_from_account({
        "logged_in": True,
        "subscription": {"plan": "Credit Plan", "monthly_credits": 20, "credits_remaining": 12.4},
        "usage_window": None,
    }) is None


def test_free_or_payg_account_has_no_picker_usage():
    assert picker_usage_from_account(SimpleNamespace(logged_in=True, subscription=None, usage_window=None)) is None
    assert picker_usage_from_account(None) is None


def test_window_plan_gauges_stack_on_separate_lines():
    usage = picker_usage_from_account(SimpleNamespace(
        logged_in=True, subscription=SimpleNamespace(plan="Window Plan"),
        usage_window=SimpleNamespace(**usage_window(
            day_used_bp=0, week_used_bp=0, top_tier_used_bp=0,
            day_resets_at_ms=FAR, week_resets_at_ms=FAR, paused_reason=None,
        )),
    ))
    stacked = plan_gauge_lines(usage)
    assert stacked[0] == "Window Plan"
    assert any(line.startswith("today") for line in stacked)
    assert any(line.startswith("week") for line in stacked)
    assert any(line.startswith("large models") for line in stacked)
    assert len(stacked) == 4


def test_window_gauges_for_model_omit_top_tier_on_regular_ids():
    usage = picker_usage_from_account(SimpleNamespace(
        logged_in=True, subscription=SimpleNamespace(plan="Window Plan"),
        usage_window=SimpleNamespace(**usage_window(
            day_used_bp=0, week_used_bp=0, top_tier_used_bp=5000, day_resets_at_ms=1, week_resets_at_ms=2,
        )),
    )).with_pricing({"example/model-b": {"prompt": "0.000005", "completion": "0.000025", "top_tier": True}})
    assert [g.label for g in usage.gauges_for_model("example/model-a")] == [GAUGE_TODAY, GAUGE_WEEK]
    assert [g.label for g in usage.gauges_for_model("example/model-b")][-1] == "top_tier"


def _window_usage(**window):
    """A window plan where the gateway marked Model B top tier."""
    base = dict(day_used_bp=0, week_used_bp=0, top_tier_used_bp=0,
                day_resets_at_ms=FAR, week_resets_at_ms=FAR, paused_reason=None)
    usage = picker_usage_from_account(SimpleNamespace(
        logged_in=True, subscription=SimpleNamespace(plan="Window Plan"),
        usage_window=SimpleNamespace(**usage_window(**{**base, **window})),
    ))
    return replace(usage, top_tier_ids=frozenset({"example/model-b"})) if usage is not None else None


def test_month_row_only_when_the_server_lists_it():
    assert _window_usage().gauge(GAUGE_MONTH) is None  # no month allowance listed: no month row
    usage = _window_usage(month_used_bp=10_000, month_resets_at_ms=333, top_tier_used_bp=2_000)
    assert [g.label for g in usage.gauges] == [GAUGE_TODAY, GAUGE_WEEK, GAUGE_MONTH, "top_tier"]
    assert usage.gauge(GAUGE_MONTH).resets_at_ms == 333
    assert [line.split()[0] for line in plan_gauge_lines(usage)[1:]] == ["today", "week", "month", "large"]
    # A used-up month stops every non-free model, like a used-up day or week.
    assert usage.included_summary.startswith("Monthly limit reached") and usage.blocks("any/model")


def test_picker_draws_whatever_allowances_the_server_lists():
    """A new allowance (a 5-hour window) shows with the server's label and stops every model when full,
    with no Hermes release; the order is the server's."""
    usage = picker_usage_from_account(SimpleNamespace(
        logged_in=True, subscription=SimpleNamespace(plan="Window Plan"),
        usage_window=SimpleNamespace(gauges=[
            {"id": "five_hour", "scope": "all", "label": "Next 5 hours", "used_bp": 10_000, "resets_at_ms": FAR},
            *usage_window()["gauges"],
        ], featured_model_ids=()),
    ))
    assert next(g.label for g in usage.gauges) == "five_hour"
    assert plan_gauge_lines(usage)[1].startswith("Next 5 hours")
    assert usage.included_summary.startswith("Next 5 hours limit reached")
    assert usage.blocks("any/model")


def test_top_tier_gauge_is_labelled_monthly_and_resets_on_its_own_clock():
    usage = _window_usage(top_tier_used_bp=2_000, day_resets_at_ms=111, top_tier_resets_at_ms=222)
    assert usage.gauge("top_tier").resets_at_ms == 222
    assert any(line.startswith("large models (monthly)  ") for line in plan_gauge_lines(usage))
    # No large-model reset listed: the day's reset.
    assert _window_usage(top_tier_used_bp=2_000, day_resets_at_ms=111).gauge("top_tier").resets_at_ms == 111


def test_blocks_only_models_whose_allowance_is_empty():
    usage = _window_usage(top_tier_used_bp=10_000)
    assert usage.blocks("example/model-b")
    assert not usage.blocks("example/model-a")
    assert usage.included_summary.startswith("Monthly large-model limit reached")
    assert _window_usage().included_summary == "every model below is included"


def test_blocks_every_paid_model_when_the_day_is_spent():
    usage = _window_usage(day_used_bp=10_000)
    assert usage.blocks("example/model-a")
    assert usage.included_summary.startswith("Daily limit reached")


def test_pay_as_you_go_wording_when_top_up_credits_pay():
    from dataclasses import replace as _replace
    usage = _replace(_window_usage(day_used_bp=10_000), credits_pay=True)
    # Credits pay: the row stays usable; the limit line says so once.
    assert not usage.blocks("example/model-a")
    assert usage.limit_line().startswith("Daily limit reached · resets ")
    assert usage.limit_line().endswith(" · paid models now use your credits (pay as you go)")
    stops = _replace(usage, credits_pay=False)
    assert stops.blocks("example/model-a")
    assert stops.limit_line().endswith(" · only Free models until then")


def test_large_model_limit_blocks_only_large_rows():
    from dataclasses import replace as _replace
    usage = _replace(_window_usage(top_tier_used_bp=10_000), credits_pay=True)
    assert not usage.blocks("example/model-b") and not usage.blocks("example/model-a")
    stops = _replace(usage, credits_pay=False)
    assert stops.blocks("example/model-b") and not stops.blocks("example/model-a")
    assert usage.limit_line().startswith("Monthly large-model limit reached")
    assert usage.limit_line().endswith(" · Large models now use your credits (pay as you go)")
    assert _replace(usage, credits_pay=False).limit_line().endswith(" · Large models paused until then")


def test_credits_pay_reads_keep_going_and_the_top_up_balance():
    from hermes_cli.model_picker_usage import _credits_pay
    window = SimpleNamespace(keep_going_on_credits=True)
    with_balance = SimpleNamespace(paid_service_access_info=SimpleNamespace(purchased_credits_remaining=12.5))
    empty = SimpleNamespace(paid_service_access_info=SimpleNamespace(purchased_credits_remaining=0.0))
    unknown = SimpleNamespace(paid_service_access_info=None)
    assert _credits_pay(with_balance, window) is True
    assert _credits_pay(empty, window) is False
    assert _credits_pay(unknown, window) is None
    assert _credits_pay(with_balance, SimpleNamespace(keep_going_on_credits=False)) is False


def test_a_free_model_is_never_used_up():
    """Same rule as agent/plan_limits.py: a $0 row or a ``:free`` id draws on no allowance, so a spent day
    (or large-model allowance) never marks it."""
    pricing = {
        "gift/m:free": {"prompt": "0", "completion": "0"},
        "zero/m": {"prompt": "0", "completion": "0"},
        "paid/m": {"prompt": "0.000001", "completion": "0.000002"},
    }
    usage = _window_usage(day_used_bp=10_000, top_tier_used_bp=10_000).with_pricing(pricing)
    assert not usage.blocks("gift/m:free")
    assert not usage.blocks("zero/m")
    assert usage.blocks("paid/m")
    # Without a catalog, a ``:free`` id is still known free.
    assert not _window_usage(day_used_bp=10_000).blocks("other/x:free")


def test_the_note_names_the_allowance_that_frees_up_last():
    """Day and week both spent: the day's reset alone lets nothing through, so the note waits for the week."""
    usage = _window_usage(day_used_bp=10_000, week_used_bp=10_000, day_resets_at_ms=1_000, week_resets_at_ms=FAR)
    assert usage.exhausted_gauge("paid/m").label == GAUGE_WEEK


def test_window_picker_labels_free_models_from_the_catalog_price():
    usage = _window_usage()
    pricing = {
        "cheap/m": {"prompt": "0.00000006", "completion": "0.00000012", "usage_level": "low"},
        "gift/m:free": {"prompt": "0", "completion": "0"},
        "unpriced/m": {},
    }
    models = list(pricing)
    rows = _ModelPickerRows(models, pricing, current_model="", sale_chrome=True, usage=usage)
    free = rows.label("gift/m:free")
    assert free.split("gift/m:free", 1)[1].strip() == "Free"
    # Unknown price is not a promise of free: no word at all.
    assert rows.label("unpriced/m").split("unpriced/m", 1)[1].strip() == ""
    assert len(free) == len(rows.label("cheap/m"))  # column stays aligned


def test_credit_picker_free_rows_are_unchanged():
    rows = _ModelPickerRows(["gift/m:free"], {"gift/m:free": {"prompt": "0", "completion": "0"}},
                            current_model="", sale_chrome=True)
    assert not rows.window_mode and "free" in rows.label("gift/m:free")


# ── gateway top-tier marks + featured ordering ──────────────────────────────────

# Picker pricing shape; the gateway marks top-tier rows for window callers only.
_PRICES = {
    "example/model-a": {"prompt": "0.0000000108", "completion": "0.00000128"},
    "example/model-b": {"prompt": "0.00000041", "completion": "0.00000399"},
    "example/model-c": {"prompt": "0.000005", "completion": "0.000025", "top_tier": True},
}


def test_top_tier_comes_from_the_gateway_marks():
    usage = _window_usage(featured_model_ids=("example/model-a", "example/model-b"))
    marked = usage.with_pricing(_PRICES)
    assert marked.is_top_tier("example/model-c")
    # The catalog's marks replace any earlier ones: the medium model is unmarked here, so not top tier.
    assert not marked.is_top_tier("example/model-b")
    payload = marked.to_payload()
    assert payload["top_tier_model_ids"] == ["example/model-c"]
    assert payload["featured_model_ids"] == ["example/model-a", "example/model-b"]


def test_no_marks_means_nothing_is_top_tier():
    """Fail open: an older gateway, an unpriced model or no catalog marks nothing top tier."""
    unmarked = {k: {kk: vv for kk, vv in v.items() if kk != "top_tier"} for k, v in _PRICES.items()}
    usage = _window_usage().with_pricing(unmarked)
    assert not usage.is_top_tier("example/model-b")
    assert not usage.is_top_tier("example/model-c")
    bare = picker_usage_from_account(SimpleNamespace(
        logged_in=True, subscription=SimpleNamespace(plan="Window Plan"),
        usage_window=SimpleNamespace(**usage_window(top_tier_used_bp=10_000)),
    ))
    assert bare is not None and not bare.is_top_tier("example/model-b") and not bare.blocks("example/model-b")
    assert bare.to_payload()["top_tier_model_ids"] == []


def test_featured_first_puts_plan_picks_first_and_keeps_the_rest_in_order():
    from hermes_cli.model_picker_usage import featured_first, split_recommended
    listed = ["a/x", "example/model-b", "b/y", "example/model-a"]
    featured = ("example/model-a", "example/model-b", "not/listed")
    assert featured_first(listed, featured) == [
        "example/model-a", "example/model-b", "a/x", "b/y"]
    assert featured_first(listed, ()) == listed
    assert split_recommended(listed, featured) == (
        ["example/model-a", "example/model-b"], ["a/x", "b/y"])
    assert split_recommended(listed, ()) == ([], listed)


def test_cli_rows_mark_top_tier_from_the_gateway_word():
    prices = {**_PRICES, "example/model-c": {**_PRICES["example/model-c"],
                                                        "usage_level": "top_tier"}}
    rows = _ModelPickerRows(list(prices), prices, current_model="", sale_chrome=True, usage=_window_usage())
    assert rows.has_top_tier
    assert rows.label("example/model-c").endswith("Large")
    assert rows.usage.is_top_tier("example/model-c")
    assert not rows.usage.is_top_tier("example/model-b")


def test_catalog_parse_keeps_the_gateway_usage_level_for_nous_only(monkeypatch):
    from hermes_cli import models_pricing as mp
    payload = {"data": [
        {"id": "a/m", "pricing": {"prompt": "0.000001", "completion": "0.000002"}, "usage_level": "medium"},
        {"id": "b/m", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
    ]}
    monkeypatch.setattr(mp, "_get_json", lambda *a, **k: payload)
    monkeypatch.setattr(mp, "_seed_reasoning_caps", lambda *a, **k: None)
    nous = mp.fetch_models_with_pricing("k", "https://nous.example", force_refresh=True,
                                        include_sale_original=True)
    assert nous["a/m"]["usage_level"] == "medium" and "usage_level" not in nous["b/m"]
    other = mp.fetch_models_with_pricing("k", "https://other.example", force_refresh=True)
    assert "usage_level" not in other["a/m"]


def test_catalog_parse_keeps_the_top_tier_mark_for_nous_only(monkeypatch):
    from hermes_cli import models_pricing as mp
    payload = {"data": [
        {"id": "a/m", "pricing": {"prompt": "0.000001", "completion": "0.000002"}, "top_tier": True},
        {"id": "b/m", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
    ]}
    monkeypatch.setattr(mp, "_get_json", lambda *a, **k: payload)
    monkeypatch.setattr(mp, "_seed_reasoning_caps", lambda *a, **k: None)
    nous = mp.fetch_models_with_pricing("k", "https://nous.example", force_refresh=True,
                                        include_sale_original=True)
    assert nous["a/m"]["top_tier"] is True and "top_tier" not in nous["b/m"]
    other = mp.fetch_models_with_pricing("k", "https://other.example", force_refresh=True)
    assert "top_tier" not in other["a/m"]


def test_account_payload_parses_featured_ids():
    from hermes_cli import nous_account as na
    info = na._coerced_dataclass(na.NousUsageWindowInfo, usage_window(featured_model_ids=["a/b", "c/d"]))
    assert info.featured_model_ids == ("a/b", "c/d")


def _stub_account_cache(monkeypatch, stale, fetched):
    """Fresh cache empty, last snapshot ``stale``; a fetch records its force_fresh and fills the cache."""
    from hermes_cli import nous_account as na
    calls, fresh = [], {}

    def cached(*, any_age=False):
        return fresh.get("info") or (stale if any_age else None)

    def get(*, force_fresh=False, **_):
        calls.append(force_fresh)
        if force_fresh:
            fresh["info"] = fetched
        return fetched

    monkeypatch.setattr(na, "cached_nous_portal_account_info", cached)
    monkeypatch.setattr(na, "get_nous_portal_account_info", get)
    return calls


def test_warm_picker_refetches_a_known_window_plan_past_the_cache_ttl(monkeypatch):
    from hermes_cli.model_picker_usage import nous_picker_usage
    window = SimpleNamespace(logged_in=True, subscription=SimpleNamespace(plan="Window Plan"),
                             usage_window=SimpleNamespace(**usage_window(day_used_bp=100, week_used_bp=200)))
    calls = _stub_account_cache(monkeypatch, stale=window, fetched=window)
    usage = nous_picker_usage(warm=True)
    assert calls == [True]
    assert usage is not None and usage.gauge(GAUGE_TODAY) is not None


def test_warm_picker_never_forces_a_fetch_for_a_non_window_account(monkeypatch):
    from hermes_cli.model_picker_usage import nous_picker_usage
    credit = SimpleNamespace(logged_in=True, subscription=SimpleNamespace(plan="Credit Plan"), usage_window=None)
    for stale in (credit, None):
        calls = _stub_account_cache(monkeypatch, stale=stale, fetched=credit)
        assert nous_picker_usage(warm=True) is None
        assert calls == [False]


# ── where the "used up" note reaches outside the pickers above ───────────────────

def test_the_in_chat_cli_picker_says_the_limit_once_on_nous_only(monkeypatch):
    from hermes_cli import model_picker_usage as mpu

    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: _window_usage(day_used_bp=10_000))
    rows, line = mpu.annotate_window_plan_rows({"slug": "nous"}, ["paid/m", "gift/m:free"])
    # Rows stay the bare ids; the limit is said once, above the list.
    assert rows == ["paid/m", "gift/m:free"] and line.startswith("Daily limit reached")
    # Any other provider, or no fresh snapshot: the bare ids and no line.
    assert mpu.annotate_window_plan_rows({"slug": "example"}, ["paid/m"]) == (["paid/m"], None)
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: None)
    assert mpu.annotate_window_plan_rows({"slug": "nous"}, ["paid/m"]) == (["paid/m"], None)


def test_the_in_chat_cli_picker_tags_large_and_free_rows_on_window_plans_only(monkeypatch):
    from hermes_cli import model_picker_usage as mpu

    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: None)
    window = {"slug": "nous", "plan_usage": {"kind": "usage_window"},
              "usage_levels": {"example/model-b": "top_tier", "example/model-a": "low", "example/model-free:free": "free"}}
    ids = ["example/model-b", "example/model-a", "example/model-free:free"]
    rows, _line = mpu.annotate_window_plan_rows(window, ids)
    assert [r.split() for r in rows] == [["example/model-b", "Large"], ["example/model-a"], ["example/model-free:free", "Free"]]
    # Same payload without a window plan (or a stray levels map): the bare ids.
    assert mpu.annotate_window_plan_rows({**window, "plan_usage": None}, ids)[0] == ids


_LEVELS = {"example/model-b": {"usage_level": "top_tier", "top_tier": True},
           "example/model-a": {"usage_level": "low"}}


def test_messaging_pickers_get_large_free_labels_and_the_limit_line(monkeypatch):
    from hermes_cli import model_picker_usage as mpu
    from hermes_cli import model_switch_providers as msp
    from agent import plan_limits as pl

    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: _window_usage(top_tier_used_bp=10_000))
    monkeypatch.setattr(pl, "cached_pricing", lambda base_url="": dict(_LEVELS))
    providers = [{"slug": "nous", "api_url": "https://x/v1", "models": ["example/model-b", "example/model-a"]},
                 {"slug": "example", "models": ["example/model-b"]}]
    msp._attach_window_model_notes(providers)
    nous = providers[0]
    # Only Large / Free are tagged; no per-row limit notes (the plan line says it once).
    assert nous["model_labels"] == {"example/model-b": "Large"}
    assert nous["plan_line"].startswith("Monthly large-model limit reached")
    assert "model_notes" not in nous
    assert mpu.picker_row_suffix(nous, "example/model-b") == "Large"
    assert mpu.picker_row_suffix(nous, "example/model-a") == ""
    assert not {"model_labels", "plan_line"} & set(providers[1])
    # Nothing spent: labels only. Not a window plan: no field at all (pickers unchanged).
    quiet = [{"slug": "nous", "api_url": "https://x/v1", "models": ["example/model-b"]}]
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: _window_usage())
    msp._attach_window_model_notes(quiet)
    assert quiet[0].get("model_labels") == {"example/model-b": "Large"}
    assert "plan_line" not in quiet[0]
    plain = [{"slug": "nous", "api_url": "https://x/v1", "models": ["example/model-b"]}]
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: None)
    msp._attach_window_model_notes(plain)
    assert plain == [{"slug": "nous", "api_url": "https://x/v1", "models": ["example/model-b"]}]
    assert mpu.picker_row_suffix(plain[0], "example/model-b") == ""


def test_large_model_notice_window_plans_only(monkeypatch):
    from dataclasses import replace as _replace
    from hermes_cli import model_picker_usage as mpu
    monkeypatch.setattr(mpu, "_nous_catalog", lambda fetch=False: dict(_LEVELS))
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: _window_usage())
    note = mpu.large_model_notice("example/model-b", "nous")
    assert note.startswith("Large model: uses more of your plan's usage per message.") and "Monthly large-model limit:" in note
    assert mpu.large_model_notice("example/model-a", "nous") is None  # not Large
    assert mpu.large_model_notice("example/model-b", "example") is None  # not Nous
    # Pay as you go past the large-model limit: the credits wording.
    monkeypatch.setattr(mpu, "nous_picker_usage",
                        lambda **kw: _replace(_window_usage(top_tier_used_bp=10_000), credits_pay=True))
    assert mpu.large_model_notice("example/model-b", "nous") == "Large model: uses more credits per message (pay as you go)."
    # No plan snapshot yet (cold CLI): a Large mark in the resident catalog alone proves a window plan.
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: None)
    assert mpu.large_model_notice("example/model-b", "nous") == "Large model: uses more of your plan's usage per message."
    # Any other account: its catalog carries no marks (None), and it is never fetched.
    seen = []
    monkeypatch.setattr(mpu, "_nous_catalog",
                        lambda fetch=False: seen.append(fetch) or {"example/model-b": {"prompt": "0.000001"}})
    assert mpu.large_model_notice("example/model-b", "nous", fetch=True) is None
    assert seen == [False]


def test_catalog_reads_never_answer_from_another_profiles_cache(monkeypatch):
    """One process, several profiles: tags and the Large notice read only this profile's catalog, never the
    newest catalog cached for any credential (another plan can mark different models Large)."""
    import hermes_cli.models_pricing as mp
    from agent import plan_limits as pl
    from hermes_cli import model_picker_usage as mpu
    monkeypatch.setattr(mp, "peek_cached_pricing", lambda base_url: dict(_LEVELS))  # another profile's catalog
    monkeypatch.setattr(mp, "get_pricing_for_provider", lambda provider, **kw: {})  # this profile: cold
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: None)  # this profile: no window plan
    assert pl.cached_pricing("https://inference-api.nousresearch.com/v1") == {}
    assert mpu.large_model_notice("example/model-b", "nous") is None


def test_picker_words_follow_the_display_language(monkeypatch):
    """Pickers main translates read the catalog: English stays byte-identical, other languages translate."""
    from hermes_cli import model_picker_usage as mpu
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    assert mpu.usage_level_label("top_tier") == "Large"
    assert mpu.usage_level_label("bogus") == ""
    spent = _replace_credits(_window_usage(day_used_bp=10_000), True)
    assert spent.limit_line().startswith("Daily limit reached · resets ")
    monkeypatch.setenv("HERMES_LANGUAGE", "de")
    assert mpu.usage_level_label("top_tier") == "Groß"
    assert spent.limit_line().startswith("Tageslimit erreicht · Zurücksetzung in ")
    assert not spent.blocks("example/model-a")
    # Surfaces whose own text is English (setup picker, ACP) ask for English explicitly.
    assert spent.limit_line(lang="en").startswith("Daily limit reached")
    assert spent.included_summary.startswith("Daily limit reached")


def _replace_credits(usage, credits_pay):
    from dataclasses import replace as _replace
    return _replace(usage, credits_pay=credits_pay)


def test_window_plan_feeds_the_existing_usage_chip(monkeypatch):
    """A window plan fills main's ``usage.windows`` (the existing usage chip) from the same snapshot; a
    paused plan or any other account adds nothing, and a provider usage row is never overwritten."""
    from hermes_cli import inventory
    from hermes_cli import model_picker_usage as mpu
    usage = _window_usage(day_used_bp=8_500)
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: usage)
    row: dict = {"slug": "nous"}
    inventory._attach_nous_plan_usage(row, warm=False)
    windows = row["usage"]["windows"]
    assert windows[0] == {**windows[0], "label": "Today", "used_percent": 85.0, "scope": "account"}
    assert all(w["resets_at"] for w in windows)
    # A full allowance: the picker's limit line says it, so no chip repeats it.
    full: dict = {"slug": "nous"}
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: _window_usage(day_used_bp=10_000))
    inventory._attach_nous_plan_usage(full, warm=False)
    assert "plan_usage" in full and "usage" not in full
    paused: dict = {"slug": "nous"}
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: replace(usage, paused_reason="renewal_unpaid"))
    inventory._attach_nous_plan_usage(paused, warm=False)
    assert "plan_usage" in paused and "usage" not in paused
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: None)
    other: dict = {"slug": "nous"}
    inventory._attach_nous_plan_usage(other, warm=False)
    assert other == {"slug": "nous"}
