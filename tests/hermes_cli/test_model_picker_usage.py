"""Window vs. credit-grant chrome for the model pickers."""

from dataclasses import replace
from types import SimpleNamespace

from hermes_cli.auth_model_picker import _ModelPickerRows
from tests.usage_window_fixtures import FAR, usage_window
from hermes_cli.model_picker_usage import (
    GAUGE_MONTH,
    GAUGE_TODAY,
    GAUGE_WEEK,
    basis_line,
    format_percent_used,
    format_usage_multiplier,
    picker_usage_from_account,
    plan_gauge_lines,
    render_bar,
    usage_chip_label,
    usage_free_models,
    usage_multiplier,
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


def test_multiplier_format_is_the_gateway_value_as_sent():
    assert format_usage_multiplier(0.05) == "<0.1×"
    assert format_usage_multiplier(0.01) == "<0.1×"
    assert format_usage_multiplier(0.1) == "0.1×"
    assert format_usage_multiplier(0.4) == "0.4×"
    assert format_usage_multiplier(0.8) == "0.8×"
    # One decimal at most, half up (the TypeScript twin's Math.round), never a float artefact.
    assert format_usage_multiplier(0.25) == "0.3×"
    assert format_usage_multiplier(0.1 + 0.2) == "0.3×"
    assert format_usage_multiplier(1) == "1×"
    assert format_usage_multiplier(3.0) == "3×"
    assert format_usage_multiplier(15) == "15×"
    # A real multiplication sign, never the letter x.
    assert "x" not in format_usage_multiplier(2) and "\u00d7" in format_usage_multiplier(2)


def test_usage_multiplier_reads_only_a_finite_positive_number():
    assert usage_multiplier({"usage_multiplier": 3}) == 3.0
    assert usage_multiplier({"usage_multiplier": 0.5}) == 0.5
    assert usage_multiplier({"usage_multiplier": 100_000}) == 100_000.0
    for bad in (True, False, 0, -1, float("nan"), float("inf"), 100_001, 1e308, "3", None):
        assert usage_multiplier({"usage_multiplier": bad}) is None, bad
    assert usage_multiplier({}) is None and usage_multiplier(None) is None


def test_chip_is_the_gateway_multiplier_never_computed_from_prices():
    pricing = {
        "cheap/m": {"prompt": "0.00000014", "completion": "0.00000028", "usage_multiplier": 0.5},
        "base/m": {"prompt": "0.0000003", "completion": "0.0000006", "usage_multiplier": 1},
        "big/m": {"prompt": "0.000005", "completion": "0.000025", "usage_multiplier": 40},
        "gift:free": {"prompt": "0", "completion": "0"},
    }
    assert {m: usage_chip_label(m, pricing) for m in pricing} == {
        "cheap/m": "0.5×", "base/m": "1×", "big/m": "40×", "gift:free": "Free"}
    # Same prices with no gateway multiplier: Hermes derives nothing (only Free stays).
    bare = {m: {k: v for k, v in p.items() if k != "usage_multiplier"} for m, p in pricing.items()}
    assert {m: usage_chip_label(m, bare) for m in bare} == {"cheap/m": "", "base/m": "", "big/m": "", "gift:free": "Free"}
    # Free wins over any multiplier; unknown price is not a promise of free.
    assert usage_chip_label("odd:free", {"odd:free": {"usage_multiplier": 3}}) == "Free"
    assert usage_chip_label("u", {"u": {}}) == ""


def test_an_older_gateways_tier_words_show_nothing():
    old = {"large": {"prompt": "0.000005", "completion": "0.000025", "usage_level": "top_tier", "top_tier": True},
           "small": {"prompt": "0.0000001", "completion": "0.0000002", "usage_level": "low"}}
    assert usage_chip_label("large", old) == "" and usage_chip_label("small", old) == ""
    assert basis_line(list(old), old, basis=_BASIS) == ""


_BASIS = {"model_id": "base/m", "name": "Base Model"}


def test_basis_line_is_the_gateways_name_whenever_a_multiplier_shows():
    pricing = {
        "base/m": {"prompt": "0.0000003", "completion": "0.0000006", "usage_multiplier": 1},
        "big/m": {"prompt": "0.000005", "completion": "0.000025", "usage_multiplier": 15},
    }
    assert basis_line(list(pricing), pricing, basis=_BASIS) == "1× = Base Model price"
    # The basis row need not be listed (an org allowlist can hide it): still named.
    assert basis_line(["big/m"], pricing, basis=_BASIS) == "1× = Base Model price"
    # No basis from the gateway, or no row shows a multiplier: no line.
    assert basis_line(list(pricing), pricing, basis={}) == ""
    assert basis_line(["x"], {"x": {}}, basis=_BASIS) == ""


def test_usage_basis_reads_the_top_level_field_kept_with_the_nous_catalog(monkeypatch):
    from hermes_cli import models_pricing as mp
    priced = {"prompt": "0.000001", "completion": "0.000002"}
    payload = {"data": [{"id": "a/m", "pricing": priced, "usage_multiplier": 3}],
               "usage_basis": {"model_id": " base/m ", "name": "Base Model"}}
    monkeypatch.setattr(mp, "_get_json", lambda *a, **k: payload)
    monkeypatch.setattr(mp, "_seed_reasoning_caps", lambda *a, **k: None)
    key = "https://nous.example" + mp._pricing_auth_fingerprint("k")
    mp.fetch_models_with_pricing("k", "https://nous.example", force_refresh=True, include_sale_original=True)
    assert mp._usage_basis_by_catalog[key] == {"model_id": "base/m", "name": "Base Model"}
    # No name: the id. Absent or malformed: dropped, never a stale basis.
    payload["usage_basis"] = {"model_id": "base/m"}
    mp.fetch_models_with_pricing("k", "https://nous.example", force_refresh=True, include_sale_original=True)
    assert mp._usage_basis_by_catalog[key] == {"model_id": "base/m", "name": "base/m"}
    for bad in (None, "base/m", {"model_id": ""}, {"model_id": 7}):
        payload["usage_basis"] = bad
        mp.fetch_models_with_pricing("k", "https://nous.example", force_refresh=True, include_sale_original=True)
        assert key not in mp._usage_basis_by_catalog
    # Other catalogs never keep one.
    payload["usage_basis"] = {"model_id": "base/m"}
    mp.fetch_models_with_pricing("k", "https://other.example", force_refresh=True)
    assert "https://other.example" + mp._pricing_auth_fingerprint("k") not in mp._usage_basis_by_catalog


def test_portal_sent_names_are_sanitized_before_any_terminal_prints_them(monkeypatch):
    """The basis name/id and the catalog row name are printed verbatim by the CLI/TUI pickers and limit lines:
    escape sequences, control and invisible tag characters are stripped, long text is capped, and a value
    with nothing printable left is dropped (the id stands in for the name)."""
    from hermes_cli import models_pricing as mp
    priced = {"prompt": "0.000001", "completion": "0.000002"}
    payload = {"data": [{"id": "a/m", "pricing": priced, "usage_multiplier": 3,
                         "name": "Model\x1b[2J A\x07"}],
               "usage_basis": {"model_id": "base/m\x1b]0;pwned\x07", "name": "\x1b[31mBase\r\nModel\U000E0041"}}
    monkeypatch.setattr(mp, "_get_json", lambda *a, **k: payload)
    monkeypatch.setattr(mp, "_seed_reasoning_caps", lambda *a, **k: None)
    key = "https://nous.example" + mp._pricing_auth_fingerprint("k")
    rows = mp.fetch_models_with_pricing("k", "https://nous.example", force_refresh=True, include_sale_original=True)
    assert mp._usage_basis_by_catalog[key] == {"model_id": "base/m", "name": "Base Model"}
    assert rows["a/m"]["name"] == "Model A"
    # Capped at 64 characters with an ellipsis.
    payload["usage_basis"] = {"model_id": "base/m", "name": "N" * 500}
    mp.fetch_models_with_pricing("k", "https://nous.example", force_refresh=True, include_sale_original=True)
    name = mp._usage_basis_by_catalog[key]["name"]
    assert len(name) == 64 and name.endswith("…")
    # Nothing printable left: the name falls back to the id; an id with nothing left drops the basis.
    payload["usage_basis"] = {"model_id": "base/m", "name": "\x1b[0m\x07"}
    mp.fetch_models_with_pricing("k", "https://nous.example", force_refresh=True, include_sale_original=True)
    assert mp._usage_basis_by_catalog[key]["name"] == "base/m"
    payload["usage_basis"] = {"model_id": "\x1b[0m \x07"}
    mp.fetch_models_with_pricing("k", "https://nous.example", force_refresh=True, include_sale_original=True)
    assert key not in mp._usage_basis_by_catalog

def test_window_picker_shows_the_multiplier_not_prices_or_bars(monkeypatch):
    from hermes_cli import model_picker_usage as mpu
    monkeypatch.setattr(mpu, "usage_basis", lambda: {"model_id": "cheap/m", "name": "Cheap Model"})
    usage = picker_usage_from_account(SimpleNamespace(
        logged_in=True,
        subscription=SimpleNamespace(plan="Window Plan"),
        usage_window=SimpleNamespace(**usage_window(
            day_used_bp=2000, week_used_bp=1000,
            day_resets_at_ms=1, week_resets_at_ms=2, paused_reason=None,
        )),
    ))
    assert usage is not None
    rows = _ModelPickerRows(
        ["cheap/m", "pricey/m"],
        {
            "cheap/m": {"prompt": "0.00000006", "completion": "0.00000012", "usage_multiplier": 1},
            "pricey/m": {"prompt": "0.000003", "completion": "0.000015", "usage_multiplier": 25},
        },
        current_model="", sale_chrome=True, usage=usage,
    )
    cheap, pricey = rows.label("cheap/m"), rows.label("pricey/m")
    assert "$" not in cheap and "$" not in pricey and "[" not in cheap and "[" not in pricey
    assert cheap.split("cheap/m", 1)[1].strip() == "1×" and pricey.rstrip().endswith("25×")
    assert not pricey.startswith("★")
    assert len(cheap) == len(pricey)  # one aligned column
    header = "\n".join(rows.description_lines())
    assert "Usage" in header and "bar" not in header and "★" not in header
    # The basis line sits in the header, just above the rows.
    assert rows.description_lines()[-1].strip() == "1× = Cheap Model price"
    desc = rows.description_lines()
    today_line = next(line for line in desc if "today" in line.lower())
    week_line = next(line for line in desc if "week" in line.lower())
    assert today_line != week_line
    assert "Window Plan" in desc[0] and "included" in desc[0]


def test_no_basis_line_without_a_gateway_basis_or_off_window(monkeypatch):
    from hermes_cli import model_picker_usage as mpu
    monkeypatch.setattr(mpu, "usage_basis", lambda: None)
    rows = _ModelPickerRows(["cheap/m"], {"cheap/m": {"prompt": "0.0000001", "completion": "0.0000002",
                                                       "usage_multiplier": 2}},
                            current_model="", sale_chrome=True, usage=_window_usage())
    assert not any("1×" in line for line in rows.description_lines())
    # Credit-grant accounts never get a chip or the line.
    monkeypatch.setattr(mpu, "usage_basis", lambda: {"model_id": "x", "name": "X"})
    credit = _ModelPickerRows(["x"], {"x": {"usage_multiplier": 1}}, current_model="",
                              sale_chrome=True)
    assert not any("1×" in line for line in credit.description_lines()) and "×" not in credit.label("x")


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


def test_window_gauges_for_model_are_the_all_scope_ones():
    """A spent top-tier allowance dims no row."""
    usage = picker_usage_from_account(SimpleNamespace(
        logged_in=True, subscription=SimpleNamespace(plan="Window Plan"),
        usage_window=SimpleNamespace(**usage_window(
            day_used_bp=0, week_used_bp=0, top_tier_used_bp=5000, day_resets_at_ms=1, week_resets_at_ms=2,
        )),
    )).with_pricing({"example/model-b": {"prompt": "0.000005", "completion": "0.000025", "top_tier": True,
                                         "usage_level": "top_tier"}})
    assert [g.label for g in usage.gauges_for_model("example/model-a")] == [GAUGE_TODAY, GAUGE_WEEK]
    assert [g.label for g in usage.gauges_for_model("example/model-b")] == [GAUGE_TODAY, GAUGE_WEEK]


def _window_usage(**window):
    """A window plan from the account snapshot (default: nothing used)."""
    base = dict(day_used_bp=0, week_used_bp=0, top_tier_used_bp=0,
                day_resets_at_ms=FAR, week_resets_at_ms=FAR, paused_reason=None)
    return picker_usage_from_account(SimpleNamespace(
        logged_in=True, subscription=SimpleNamespace(plan="Window Plan"),
        usage_window=SimpleNamespace(**usage_window(**{**base, **window})),
    ))


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


def test_an_older_servers_full_top_tier_gauge_dims_no_row():
    usage = _window_usage(top_tier_used_bp=10_000)
    assert not usage.blocks("example/model-b")
    assert not usage.blocks("example/model-a")
    # The gauge still parses and the limit line still says it (generic gauge rendering).
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
        "cheap/m": {"prompt": "0.00000006", "completion": "0.00000012", "usage_multiplier": 0.5},
        "gift/m:free": {"prompt": "0", "completion": "0"},
        "unpriced/m": {},
    }
    models = list(pricing)
    rows = _ModelPickerRows(models, pricing, current_model="", sale_chrome=True, usage=usage)
    free = rows.label("gift/m:free")
    assert free.split("gift/m:free", 1)[1].strip() == "Free"
    assert rows.label("cheap/m").split("cheap/m", 1)[1].strip() == "0.5×"
    # Unknown price is not a promise of free: no chip at all.
    assert rows.label("unpriced/m").split("unpriced/m", 1)[1].strip() == ""
    assert len(free) == len(rows.label("cheap/m"))  # column stays aligned


def test_credit_picker_free_rows_are_unchanged():
    rows = _ModelPickerRows(["gift/m:free"], {"gift/m:free": {"prompt": "0", "completion": "0"}},
                            current_model="", sale_chrome=True)
    assert not rows.window_mode and "free" in rows.label("gift/m:free")


# ── payload + featured ordering ──────────────────────────────────────────────────

def test_payload_carries_no_top_tier_marks():
    """Top-tier marks on a row are not read."""
    usage = _window_usage(featured_model_ids=("example/model-a", "example/model-b"))
    payload = usage.with_pricing({"example/model-c": {"prompt": "0.000005", "completion": "0.000025",
                                                      "top_tier": True, "usage_level": "top_tier"}}).to_payload()
    assert "top_tier_model_ids" not in payload
    assert payload["featured_model_ids"] == ["example/model-a", "example/model-b"]


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


def test_catalog_parse_keeps_the_multiplier_for_nous_only(monkeypatch):
    from hermes_cli import models_pricing as mp
    priced = {"prompt": "0.000001", "completion": "0.000002"}
    payload = {"data": [
        {"id": "a/m", "pricing": priced, "usage_multiplier": 3},
        # A row-level ``usage_basis`` (never sent now) is not copied.
        {"id": "base/m", "pricing": priced, "usage_multiplier": 1, "usage_basis": True},
        {"id": "b/m", "pricing": priced},
        # Tier words and malformed values: never copied.
        {"id": "old/m", "pricing": priced, "usage_level": "top_tier", "top_tier": True},
        {"id": "bad/m", "pricing": priced, "usage_multiplier": True, "usage_basis": "true"},
        {"id": "neg/m", "pricing": priced, "usage_multiplier": -2},
    ]}
    monkeypatch.setattr(mp, "_get_json", lambda *a, **k: payload)
    monkeypatch.setattr(mp, "_seed_reasoning_caps", lambda *a, **k: None)
    nous = mp.fetch_models_with_pricing("k", "https://nous.example", force_refresh=True,
                                        include_sale_original=True)
    assert nous["a/m"]["usage_multiplier"] == 3 and "usage_basis" not in nous["a/m"]
    assert nous["base/m"]["usage_multiplier"] == 1 and "usage_basis" not in nous["base/m"]
    assert not {"usage_multiplier", "usage_basis"} & set(nous["b/m"])
    assert not {"usage_level", "top_tier", "usage_multiplier"} & set(nous["old/m"])
    assert not {"usage_multiplier", "usage_basis"} & set(nous["bad/m"])
    assert "usage_multiplier" not in nous["neg/m"]
    other = mp.fetch_models_with_pricing("k", "https://other.example", force_refresh=True)
    assert not {"usage_multiplier", "usage_basis"} & (set(other["a/m"]) | set(other["base/m"]))


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


def test_the_in_chat_cli_picker_shows_multiplier_and_free_chips_on_window_plans_only(monkeypatch):
    from hermes_cli import model_picker_usage as mpu

    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: None)
    window = {"slug": "nous", "plan_usage": {"kind": "usage_window"},
              "usage_multipliers": {"example/model-b": 15, "example/model-a": 0.5},
              "usage_basis": "example/model-c",
              "usage_free_models": ["example/model-free:free"]}
    ids = ["example/model-b", "example/model-a", "example/model-c", "example/model-free:free"]
    rows, _line = mpu.annotate_window_plan_rows(window, ids)
    assert [r.split() for r in rows] == [["example/model-b", "15×"], ["example/model-a", "0.5×"],
                                         ["example/model-c"], ["example/model-free:free", "Free"]]
    # Same payload without a window plan (or a stray multipliers map): the bare ids.
    assert mpu.annotate_window_plan_rows({**window, "plan_usage": None}, ids)[0] == ids
    # A ``usage_levels`` payload: no chips at all.
    old = {"slug": "nous", "plan_usage": {"kind": "usage_window"}, "usage_levels": {"example/model-b": "top_tier"}}
    assert mpu.annotate_window_plan_rows(old, ids)[0] == ids


_MULTIPLIERS = {"example/model-b": {"prompt": "0.000005", "completion": "0.000025", "usage_multiplier": 15},
                "example/model-a": {"prompt": "0.0000003", "completion": "0.0000006", "usage_multiplier": 1}}
_MULTIPLIER_BASIS = {"model_id": "example/model-a", "name": "example/model-a"}


def test_messaging_pickers_get_multiplier_chips_and_the_limit_line(monkeypatch):
    from hermes_cli import model_picker_usage as mpu
    from hermes_cli import model_switch_providers as msp
    from agent import plan_limits as pl

    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: _window_usage(week_used_bp=10_000))
    monkeypatch.setattr(pl, "cached_pricing", lambda base_url="": dict(_MULTIPLIERS))
    providers = [{"slug": "nous", "api_url": "https://x/v1", "models": ["example/model-b", "example/model-a"]},
                 {"slug": "example", "models": ["example/model-b"]}]
    msp._attach_window_model_notes(providers)
    nous = providers[0]
    assert nous["model_labels"] == {"example/model-b": "15×", "example/model-a": "1×"}
    assert nous["plan_line"].startswith("Weekly limit reached")
    assert "model_notes" not in nous
    assert mpu.picker_row_suffix(nous, "example/model-b") == "15×"
    assert mpu.picker_row_suffix(nous, "example/model-c") == ""
    assert not {"model_labels", "plan_line"} & set(providers[1])
    # Nothing spent: chips only. Not a window plan: no field at all (pickers unchanged).
    quiet = [{"slug": "nous", "api_url": "https://x/v1", "models": ["example/model-b"]}]
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: _window_usage())
    msp._attach_window_model_notes(quiet)
    assert quiet[0].get("model_labels") == {"example/model-b": "15×"}
    assert "plan_line" not in quiet[0]
    plain = [{"slug": "nous", "api_url": "https://x/v1", "models": ["example/model-b"]}]
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: None)
    msp._attach_window_model_notes(plain)
    assert plain == [{"slug": "nous", "api_url": "https://x/v1", "models": ["example/model-b"]}]
    assert mpu.picker_row_suffix(plain[0], "example/model-b") == ""


def test_catalog_reads_never_answer_from_another_profiles_cache(monkeypatch):
    """One process, several profiles: chips read only this profile's catalog, never the newest catalog
    cached for any credential (another plan's catalog carries its own multipliers)."""
    import hermes_cli.models_pricing as mp
    from agent import plan_limits as pl
    from hermes_cli import model_picker_usage as mpu
    from hermes_cli import model_switch_providers as msp
    monkeypatch.setattr(mp, "peek_cached_pricing", lambda base_url: dict(_MULTIPLIERS))  # another profile's catalog
    monkeypatch.setattr(mp, "get_pricing_for_provider", lambda provider, **kw: {})  # this profile: cold
    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: _window_usage())
    assert pl.cached_pricing("https://inference-api.nousresearch.com/v1") == {}
    providers = [{"slug": "nous", "api_url": "https://x/v1", "models": ["example/model-b"]}]
    msp._attach_window_model_notes(providers)
    assert "model_labels" not in providers[0]


def test_picker_words_follow_the_display_language(monkeypatch):
    """Pickers main translates read the catalog: English stays byte-identical, other languages translate."""
    from hermes_cli import model_picker_usage as mpu
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    assert mpu.free_label() == "Free"
    assert basis_line(list(_MULTIPLIERS), _MULTIPLIERS, basis=_MULTIPLIER_BASIS) == "1× = example/model-a price"
    spent = _replace_credits(_window_usage(day_used_bp=10_000), True)
    assert spent.limit_line().startswith("Daily limit reached · resets ")
    monkeypatch.setenv("HERMES_LANGUAGE", "de")
    assert mpu.free_label() == "Kostenlos"
    assert basis_line(list(_MULTIPLIERS), _MULTIPLIERS, basis=_MULTIPLIER_BASIS) == "1× = Preis von example/model-a"
    # The multiplier itself is a number, the same in every language.
    assert usage_chip_label("example/model-b", _MULTIPLIERS) == "15×"
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


# ── subscription-billed rows ────────────────────────────────

# Picker pricing shape. The gateway marks a subscription-billed row ``billing_mode: subscription`` and gives it no
# ``usage_multiplier``; its listed price never applies.
_SUBSCRIPTION_BILLED = {
    "example/model-sub": {"prompt": "0.000005", "completion": "0.00004", "billing_mode": "subscription"},
    "gift/m:free": {"prompt": "0", "completion": "0"},
    "example/model-b": {"prompt": "0.000005", "completion": "0.000025", "usage_multiplier": 15},
}


def test_subscription_billed_rows_draw_nothing_but_carry_no_free_chip():
    # Still free to the window: nothing dims it.
    assert usage_free_models(list(_SUBSCRIPTION_BILLED), _SUBSCRIPTION_BILLED) == ["example/model-sub", "gift/m:free"]
    # No chip on it; real $0 rows and multiplier rows keep theirs.
    assert {m: usage_chip_label(m, _SUBSCRIPTION_BILLED) for m in _SUBSCRIPTION_BILLED} == {
        "example/model-sub": "", "gift/m:free": "Free", "example/model-b": "15×"}
    usage = _window_usage(day_used_bp=10_000).with_pricing(_SUBSCRIPTION_BILLED)
    assert not usage.blocks("example/model-sub")
    assert not usage.blocks("gift/m:free")
    assert usage.blocks("example/model-b")
    assert usage.to_payload()["subscription_model_ids"] == ["example/model-sub"]
    # No catalog: nothing marked.
    assert _window_usage().to_payload()["subscription_model_ids"] == []


def test_window_picker_rows_leave_subscription_billed_rows_unchipped():
    rows = _ModelPickerRows(list(_SUBSCRIPTION_BILLED), _SUBSCRIPTION_BILLED, current_model="", sale_chrome=True,
                            usage=_window_usage(day_used_bp=10_000))
    assert rows.label("example/model-sub").split("example/model-sub", 1)[1].strip() == ""
    assert rows.label("gift/m:free").split("gift/m:free", 1)[1].strip() == "Free"
    assert rows.label("example/model-b").split("example/model-b", 1)[1].strip() == "15×"
    # Dimming is unchanged: the full day stops the paid row only.
    assert not rows.usage.blocks("example/model-sub") and rows.usage.blocks("example/model-b")


def test_the_in_chat_cli_picker_leaves_subscription_billed_rows_unchipped(monkeypatch):
    from hermes_cli import model_picker_usage as mpu

    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: None)
    row = {"slug": "nous",
           "plan_usage": {"kind": "usage_window", "subscription_model_ids": ["example/model-sub"]},
           "usage_free_models": ["example/model-sub", "gift/m:free"]}
    rows, _line = mpu.annotate_window_plan_rows(row, ["example/model-sub", "gift/m:free"])
    assert [r.split() for r in rows] == [["example/model-sub"], ["gift/m:free", "Free"]]


def test_messaging_pickers_leave_subscription_billed_rows_untagged(monkeypatch):
    from hermes_cli import model_picker_usage as mpu
    from hermes_cli import model_switch_providers as msp
    from agent import plan_limits as pl

    monkeypatch.setattr(mpu, "nous_picker_usage", lambda **kw: _window_usage())
    monkeypatch.setattr(pl, "cached_pricing", lambda base_url="": dict(_SUBSCRIPTION_BILLED))
    providers = [{"slug": "nous", "api_url": "https://x/v1", "models": list(_SUBSCRIPTION_BILLED)}]
    msp._attach_window_model_notes(providers)
    assert providers[0]["model_labels"] == {"gift/m:free": "Free", "example/model-b": "15×"}
    assert mpu.picker_row_suffix(providers[0], "example/model-sub") == ""
