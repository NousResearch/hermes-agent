"""DeepSeek picker pricing: the models.dev registry table plus DeepSeek's own peak/off-peak rule.

The pricing page itself is client-rendered (its HTML carries no numbers), so the fetcher reads the
registry Hermes already caches and doubles the off-peak list rate while the peak window is open.
"""

from datetime import UTC, datetime

import pytest

import agent.models_dev
from hermes_cli import models_pricing as mp


DEEPSEEK_BLOCK = {
    "deepseek-flash": {"cost": {"input": 0.15, "output": 0.6, "cache_read": 0.003}},
    "deepseek-v4-pro": {"cost": {"input": 0.66, "output": 1.98, "cache_read": 0.022}},
}


@pytest.fixture(autouse=True)
def _clean_pricing_state():
    """The module caches per endpoint across calls and tests."""
    mp._pricing_cache.clear()
    mp._pricing_cache_retry_after.clear()
    mp._pricing_provider_cache_keys.clear()
    yield
    mp._pricing_cache.clear()
    mp._pricing_cache_retry_after.clear()
    mp._pricing_provider_cache_keys.clear()


@pytest.fixture
def registry(monkeypatch):
    """Fake models.dev registry; returns the call counter."""
    calls = []

    def fake(provider):
        calls.append(provider)
        return DEEPSEEK_BLOCK if provider == "deepseek" else {}

    monkeypatch.setattr(agent.models_dev, "_get_provider_models", fake)
    return calls


@pytest.fixture(autouse=True)
def _no_holiday_network(monkeypatch):
    """The holiday calendar is the only other network call; keep tests offline and deterministic."""
    mp._cn_holiday_cache.clear()

    def fake_get_json(*_args, **_kwargs):
        return {"days": []}

    monkeypatch.setattr(mp, "_get_json", fake_get_json)
    yield
    mp._cn_holiday_cache.clear()


@pytest.mark.parametrize(
    "moment,peak",
    [
        (datetime(2026, 10, 22, 0, 59, tzinfo=UTC), False),  # Thu, before the window
        (datetime(2026, 10, 22, 1, 0, tzinfo=UTC), True),  # Thu, window start
        (datetime(2026, 10, 22, 3, 59, tzinfo=UTC), True),  # Thu, still peak
        (datetime(2026, 10, 22, 4, 0, tzinfo=UTC), False),  # Thu, 04:00-06:00 gap
        (datetime(2026, 10, 22, 6, 0, tzinfo=UTC), True),  # Thu, second window
        (datetime(2026, 10, 23, 9, 59, tzinfo=UTC), True),  # Fri, still peak
        (datetime(2026, 10, 23, 10, 0, tzinfo=UTC), False),  # Fri, peak over
        (datetime(2026, 10, 24, 7, 0, tzinfo=UTC), False),  # Sat, weekend is off-peak
        (datetime(2026, 10, 25, 2, 0, tzinfo=UTC), False),  # Sun, weekend is off-peak
        (datetime(2026, 10, 26, 6, 30, tzinfo=UTC), True),  # Mon, back to peak
    ],
)
def test_peak_window_boundaries(moment, peak):
    assert mp.deepseek_peak_active(moment) is peak


def test_off_peak_prices_are_the_registry_list_rates(registry, monkeypatch):
    monkeypatch.setattr(mp, "deepseek_peak_active", lambda now=None: False)

    table = mp._fetch_deepseek_pricing()

    flash = table["deepseek-flash"]
    assert mp._format_price_per_mtok(flash["prompt"]) == "$0.15"
    assert mp._format_price_per_mtok(flash["completion"]) == "$0.60"
    assert mp._format_price_per_mtok(flash["input_cache_read"]) == "$0.003"

    pro = table["deepseek-v4-pro"]
    assert mp._format_price_per_mtok(pro["prompt"]) == "$0.66"
    assert mp._format_price_per_mtok(pro["completion"]) == "$1.98"
    assert registry == ["deepseek"]


def test_peak_doubles_the_off_peak_rate(registry, monkeypatch):
    monkeypatch.setattr(mp, "deepseek_peak_active", lambda now=None: True)

    table = mp._fetch_deepseek_pricing()

    flash = table["deepseek-flash"]
    assert mp._format_price_per_mtok(flash["prompt"]) == "$0.30"
    assert mp._format_price_per_mtok(flash["completion"]) == "$1.20"
    assert mp._format_price_per_mtok(flash["input_cache_read"]) == "$0.006"


def test_table_is_cached_and_force_refresh_bypasses_it(registry, monkeypatch):
    monkeypatch.setattr(mp, "deepseek_peak_active", lambda now=None: False)

    first = mp._fetch_deepseek_pricing()
    second = mp._fetch_deepseek_pricing()

    assert first == second
    assert registry == ["deepseek"], (
        "a second read inside the TTL must not re-read the registry"
    )

    mp._fetch_deepseek_pricing(force_refresh=True)
    assert registry == ["deepseek", "deepseek"]


def test_provider_registration_routes_to_the_fetcher(registry, monkeypatch):
    monkeypatch.setattr(mp, "deepseek_peak_active", lambda now=None: False)

    assert "deepseek" in mp._PRICING_FETCHERS

    table = mp.get_pricing_for_provider("deepseek")
    assert mp._format_price_per_mtok(table["deepseek-flash"]["prompt"]) == "$0.15"


def test_cached_only_read_serves_the_warmed_table(registry, monkeypatch):
    monkeypatch.setattr(mp, "deepseek_peak_active", lambda now=None: False)

    mp._fetch_deepseek_pricing()  # warms the cache the picker's cached_only path reads
    calls_before = list(registry)

    table = mp.get_pricing_for_provider("deepseek", cached_only=True)

    assert registry == calls_before, "cached_only must not touch the registry"
    assert mp._format_price_per_mtok(table["deepseek-v4-pro"]["completion"]) == "$1.98"


def test_registry_failure_yields_no_pricing(monkeypatch):
    def boom(provider):
        raise RuntimeError("registry down")

    monkeypatch.setattr(agent.models_dev, "_get_provider_models", boom)

    assert mp._fetch_deepseek_pricing() == {}


def _holiday_calendar(*days):
    """A calendar payload shaped like the published one: `isOffDay` marks a holiday."""
    return {"days": [{"date": date, "isOffDay": off} for date, off in days]}


def test_weekday_public_holiday_is_off_peak(monkeypatch):
    monkeypatch.setattr(
        mp,
        "_get_json",
        lambda *a, **k: _holiday_calendar(("2026-02-17", True), ("2026-02-10", False)),
    )

    # Spring Festival: a Tuesday inside the window, but a holiday → off-peak all day.
    assert mp.deepseek_peak_active(datetime(2026, 2, 17, 7, 0, tzinfo=UTC)) is False
    # The same weekday/hour a week earlier is not a holiday → peak.
    assert mp.deepseek_peak_active(datetime(2026, 2, 10, 7, 0, tzinfo=UTC)) is True


def test_holiday_calendar_is_read_once_per_ttl(monkeypatch):
    calls = []

    def counting(url, *_args, **_kwargs):
        calls.append(url)
        return _holiday_calendar(("2026-02-17", True))

    monkeypatch.setattr(mp, "_get_json", counting)

    assert mp.deepseek_peak_active(datetime(2026, 2, 17, 7, 0, tzinfo=UTC)) is False
    assert mp.deepseek_peak_active(datetime(2026, 2, 17, 8, 0, tzinfo=UTC)) is False

    assert len(calls) == 1, "one read per year fills the cache for the TTL"
    assert "2026.json" in calls[0]


def test_calendar_is_not_read_outside_a_peak_window(monkeypatch):
    calls = []

    def counting(url, *_args, **_kwargs):
        calls.append(url)
        return {"days": []}

    monkeypatch.setattr(mp, "_get_json", counting)

    # Off-peak by hour, and a weekend inside the window: neither needs the calendar.
    assert mp.deepseek_peak_active(datetime(2026, 10, 22, 12, 0, tzinfo=UTC)) is False
    assert mp.deepseek_peak_active(datetime(2026, 10, 24, 7, 0, tzinfo=UTC)) is False

    assert calls == []


def test_unreachable_calendar_keeps_the_peak_rate(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("mirror down")

    monkeypatch.setattr(mp, "_get_json", boom)

    assert mp._cn_holiday_days(2026) == frozenset()
    # A failed calendar means "no known holiday": peak stands, so the estimate never under-quotes.
    assert mp.deepseek_peak_active(datetime(2026, 2, 17, 7, 0, tzinfo=UTC)) is True
