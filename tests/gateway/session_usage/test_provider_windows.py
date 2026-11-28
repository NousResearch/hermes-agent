"""The status bar's plan gauge for providers that report their own usage windows.

Nous Portal sends a usage-window plan down the reply headers, so ``plan_gauge`` reads it off the
agent. ``opencode-go`` has no headers but does have ``GET /zen/go/v1/usage`` (already behind
``/usage`` via its provider profile): the same gauge comes from that snapshot. Go has no daily
window, so the gauge carries its shortest one — the 5-hour rolling window, the only figure that
moves while a turn runs — and yields to a window that is nearly exhausted.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

# The provider's own refresh cadence for the status bar (the cache default is two minutes).
STATUS_BAR_REFRESH_S = 60.0


def _snapshot(*windows: AccountUsageWindow, minutes_old: float = 0.0) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider="opencode-go",
        source="test",
        fetched_at=datetime.now(UTC) - timedelta(minutes=minutes_old),
        windows=tuple(windows),
    )


def _window(label: str, used: float, *, resets_in_s: float = 3600) -> AccountUsageWindow:
    return AccountUsageWindow(
        label=label, used_percent=used, reset_at=datetime.now(UTC) + timedelta(seconds=resets_in_s)
    )


@pytest.fixture
def cache(monkeypatch):
    """Stand in for the account-usage cache: a snapshot to read, the refreshes asked for."""
    from agent import account_usage_cache

    state = {"snapshot": None, "refreshes": []}

    def _refresh(providers, **kwargs):
        state["refreshes"].append({"providers": list(providers), **kwargs})

    monkeypatch.setattr(
        account_usage_cache, "cached_account_usage", lambda provider, **kw: state["snapshot"]
    )
    monkeypatch.setattr(account_usage_cache, "refresh_account_usage_async", _refresh)
    return state


def _agent(provider: str = "opencode-go"):
    return SimpleNamespace(provider=provider)


def test_the_five_hour_window_rides_the_status_bar(cache):
    """Go's shortest window is the live one: monthly being 70% used must not hide the 5-hour figure."""
    from gateway.session_usage import plan_gauge
    from tui_gateway.contracts.common import PlanGauge

    cache["snapshot"] = _snapshot(
        _window("Rolling window", 9), _window("Weekly", 29), _window("Monthly", 70)
    )
    gauge = plan_gauge(_agent())

    assert gauge == {
        "id": "rolling",
        "label": "5h",
        "used_bp": 900,
        "resets_at_ms": int(cache["snapshot"].windows[0].reset_at.timestamp() * 1000),
    }
    PlanGauge.model_validate(gauge)  # the wire shape the Desktop reads


def test_a_near_limit_window_outranks_the_five_hour_one(cache):
    """A window that is about to block is the news, even when a shorter one still has room."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(
        _window("Rolling window", 5), _window("Weekly", 12), _window("Monthly", 85)
    )
    gauge = plan_gauge(_agent())

    assert (gauge["id"], gauge["used_bp"]) == ("month", 8_500)


def test_the_alert_threshold_is_inclusive(cache):
    """80% used is already "about to block" — the boundary belongs to the alerting window."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(_window("Rolling window", 5), _window("Weekly", 80))
    assert plan_gauge(_agent())["id"] == "week"


def test_window_order_comes_from_the_gauge_table_not_the_snapshot(cache):
    """A snapshot that lists its windows in another order still shows the 5-hour one."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(
        _window("Monthly", 70), _window("Weekly", 29), _window("Rolling window", 9)
    )
    assert plan_gauge(_agent())["id"] == "rolling"


def test_the_refresh_is_throttled_at_a_minute(cache):
    """The status bar asks its provider once a minute — often enough to see a window burn, not spam."""
    from gateway.session_usage import plan_gauge

    plan_gauge(_agent())

    assert cache["refreshes"] == [{"providers": ["opencode-go"], "interval": STATUS_BAR_REFRESH_S}]


@pytest.mark.parametrize(
    ("label", "expected_id", "expected_label"),
    [("Rolling window", "rolling", "5h"), ("Weekly", "week", "Weekly"), ("Monthly", "month", "Monthly")],
)
def test_each_go_window_maps_onto_a_gauge(cache, label, expected_id, expected_label):
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(_window(label, 42))
    gauge = plan_gauge(_agent())

    assert (gauge["id"], gauge["label"], gauge["used_bp"]) == (expected_id, expected_label, 4_200)


def test_no_snapshot_yet_asks_for_a_refresh_and_shows_nothing(cache):
    """The first usage frame of a session has no snapshot: ask for one, never block the frame."""
    from gateway.session_usage import plan_gauge

    assert plan_gauge(_agent()) is None
    assert cache["refreshes"][0]["providers"] == ["opencode-go"]


def test_a_stale_snapshot_is_not_rendered(cache):
    """A quarter-hour-old reading is not a live gauge (same rule the model picker applies)."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(_window("Monthly", 70), minutes_old=16)
    assert plan_gauge(_agent()) is None
    assert cache["refreshes"][0]["providers"] == ["opencode-go"]


def test_a_past_reset_reads_as_empty_not_as_spent(cache):
    """Go keeps the last percent until a window refills, so a passed reset means 0% — not a reason to
    jump to a coarser window and drop the live one from the bar."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(
        _window("Rolling window", 100, resets_in_s=-60), _window("Monthly", 40)
    )
    gauge = plan_gauge(_agent())

    assert (gauge["id"], gauge["used_bp"], gauge["resets_at_ms"]) == ("rolling", 0, None)


def test_the_most_used_alerting_window_wins(cache):
    """Two windows over the alert line: the one that blocks first is the one named."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(
        _window("Rolling window", 10), _window("Weekly", 85), _window("Monthly", 95)
    )
    gauge = plan_gauge(_agent())

    assert (gauge["id"], gauge["used_bp"]) == ("month", 9_500)


def test_a_window_is_identified_by_its_key_not_its_copy(cache):
    """The wire key decides the gauge: a reworded label ("5-hour rolling") cannot switch the feature off."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(
        AccountUsageWindow(label="5-hour rolling", used_percent=12, key="rolling"),
    )
    gauge = plan_gauge(_agent())

    assert (gauge["id"], gauge["label"], gauge["used_bp"]) == ("rolling", "5h", 1_200)


def test_a_window_without_a_key_falls_back_to_its_label(cache):
    """Providers that never send a key keep working off the label's first word."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(AccountUsageWindow(label="Weekly", used_percent=29))
    assert plan_gauge(_agent())["id"] == "week"


def test_the_go_profile_sends_keys_the_table_knows(monkeypatch):
    """The contract with the real provider profile: its windows arrive keyed, and the table uses them."""
    import httpx

    from gateway.session_usage import _PROVIDER_WINDOW_GAUGES
    from providers import get_provider_profile

    payload = {"usage": {
        "rolling": {"status": "ok", "percent": 4, "resetsAt": "2026-11-19T14:52:00.000Z"},
        "weekly": {"status": "ok", "percent": 32, "resetsAt": "2026-11-23T00:00:00.000Z"},
        "monthly": {"status": "ok", "percent": 71, "resetsAt": "2026-12-05T07:53:29.000Z"},
    }}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **kw):
            class _Response:
                def raise_for_status(self):
                    return None

                def json(self):
                    return payload

            return _Response()

    from hermes_cli import runtime_provider

    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda **kw: {"api_key": "test-key"})

    snapshot = get_provider_profile("opencode-go").fetch_account_usage()

    assert [(w.key, w.label) for w in snapshot.windows] == [
        ("rolling", "Rolling window"), ("weekly", "Weekly"), ("monthly", "Monthly"),
    ]
    assert {w.key for w in snapshot.windows} == set(_PROVIDER_WINDOW_GAUGES["opencode-go"])


def test_a_window_without_a_reset_still_renders(cache):
    """The fetcher may leave the reset out: the percentage still counts, the tooltip just loses its clock."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(AccountUsageWindow(label="Weekly", used_percent=29))
    gauge = plan_gauge(_agent())

    assert (gauge["id"], gauge["used_bp"], gauge["resets_at_ms"]) == ("week", 2_900, None)


def test_a_window_without_a_percentage_is_skipped(cache):
    """A window the provider reports as bare status (no ``percent``) cannot become a gauge."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(AccountUsageWindow(label="Monthly", used_percent=None))
    assert plan_gauge(_agent()) is None


def test_a_failing_provider_probe_cannot_break_usage_reporting(cache, monkeypatch):
    """The status bar is a readout: a broken fetcher loses the gauge, never the usage frame."""
    from gateway import session_usage

    def _boom(agent):
        raise RuntimeError("provider probe exploded")

    monkeypatch.setattr(session_usage, "_provider_plan_gauge", _boom)

    assert session_usage.plan_gauge(_agent()) is None


def _pool(*tokens: str):
    """A REAL ``CredentialPool``: ``entries`` there is a method, not an attribute — a fake with a list
    attribute is exactly how "attribute, not method" shipped and blanked the gauge in the gateway."""
    from agent.credential_pool import CredentialPool, PooledCredential

    return CredentialPool(
        "opencode-go",
        [PooledCredential.from_dict("opencode-go", {"id": f"c{i}", "access_token": t}) for i, t in enumerate(tokens)],
    )


def test_a_pooled_provider_shows_no_gauge(cache):
    """Two credentials for one provider: the provider-wide figure could be the other account's."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(_window("Rolling window", 37))
    agent = _agent()
    agent._credential_pool = _pool("key-a", "key-b")

    assert plan_gauge(agent) is None
    assert cache["refreshes"] == []


@pytest.mark.parametrize("pooled", [None, "single"], ids=["no-pool", "one-entry"])
def test_one_account_still_shows_the_gauge(cache, pooled):
    """No pool, or a single-entry one, is one account — the exact figure, not a fabricated average."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(_window("Rolling window", 37))
    agent = _agent()
    if pooled:
        agent._credential_pool = _pool("key-a")

    assert plan_gauge(agent)["id"] == "rolling"


def test_other_providers_get_no_gauge(cache):
    """Only providers with their own window report are touched; everyone else is unchanged."""
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(_window("Monthly", 70))
    assert plan_gauge(_agent("deepseek")) is None
    assert cache["refreshes"] == []


def test_a_portal_plan_state_does_not_survive_a_switch_to_another_provider(cache):
    """``_credits_state`` is adopted from Portal headers and outlives an in-place model switch, so a Go
    turn after a Nous one must read Go's windows — otherwise the stale allowance hides them for good."""
    from agent.credits_tracker import CreditsState
    from agent.usage_gauges import WindowGauge
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(_window("Rolling window", 37), _window("Monthly", 71))
    day = WindowGauge(id="day", scope="all", label="today", used_bp=4_200, resets_at_ms=4_000_000_000_000)
    agent = _agent()  # provider=opencode-go, the switch already landed
    agent._credits_state = CreditsState(denominator_kind="usage_window", gauges=(day,))

    assert plan_gauge(agent)["id"] == "rolling"


@pytest.mark.parametrize(
    ("primary", "expected"),
    [({"provider": "nous", "base_url": "https://inference-api.nousresearch.com/v1"}, "day"),
     ({"provider": "nous-api", "base_url": "https://api.nousresearch.com/v1"}, "day")],
)
def test_a_portal_usage_window_plan_still_wins_on_a_nous_route(cache, primary, expected):
    """On a Nous route the Portal's own gauges come first; a Go snapshot never overrides them."""
    from agent.credits_tracker import CreditsState
    from agent.usage_gauges import WindowGauge
    from gateway.session_usage import plan_gauge

    cache["snapshot"] = _snapshot(_window("Monthly", 70))
    day = WindowGauge(id="day", scope="all", label="today", used_bp=4_200, resets_at_ms=4_000_000_000_000)
    agent = _agent()
    agent._primary_runtime = primary
    agent._credits_state = CreditsState(denominator_kind="usage_window", gauges=(day,))

    assert plan_gauge(agent)["id"] == expected
    assert cache["refreshes"] == []


def test_a_full_portal_window_outranks_today(cache):
    """A used-up weekly allowance blocks the account, so it is what the bar must name — not today's 10%."""
    from agent.credits_tracker import CreditsState
    from agent.usage_gauges import WindowGauge
    from gateway.session_usage import plan_gauge

    future = 4_000_000_000_000
    day = WindowGauge(id="day", scope="all", label="today", used_bp=1_000, resets_at_ms=future)
    week = WindowGauge(id="week", scope="all", label="weekly", used_bp=10_000, resets_at_ms=future)
    agent = _agent()
    agent._primary_runtime = {"provider": "nous", "base_url": "https://inference-api.nousresearch.com/v1"}
    agent._credits_state = CreditsState(denominator_kind="usage_window", gauges=(day, week))

    assert plan_gauge(agent)["id"] == "week"


def test_a_portal_window_past_its_reset_is_stale(cache):
    """The Portal's own rule: a window whose reset has passed is no longer current."""
    from agent.credits_tracker import CreditsState
    from agent.usage_gauges import WindowGauge
    from gateway.session_usage import plan_gauge

    day = WindowGauge(id="day", scope="all", label="today", used_bp=4_200, resets_at_ms=1_000)
    agent = _agent()
    agent._primary_runtime = {"provider": "nous", "base_url": "https://inference-api.nousresearch.com/v1"}
    agent._credits_state = CreditsState(denominator_kind="usage_window", gauges=(day,))

    assert plan_gauge(agent) is None


def test_a_credits_state_that_is_not_a_usage_window_yields_no_gauge(cache):
    """A subscription-cap account has credits, not allowances: the bar has nothing to name."""
    from agent.credits_tracker import CreditsState
    from gateway.session_usage import plan_gauge

    agent = _agent("nous")
    agent._credits_state = CreditsState(denominator_kind="subscription_cap")

    assert plan_gauge(agent) is None


def test_a_broken_portal_route_cannot_break_usage_reporting(cache, monkeypatch):
    """Same readout rule as the provider probe: a failure loses the gauge, never the usage frame."""
    from gateway import session_usage

    def _boom(agent):
        raise RuntimeError("route probe exploded")

    monkeypatch.setattr(session_usage, "_portal_route", _boom)

    assert session_usage.plan_gauge(_agent("nous")) is None


def test_a_live_fallback_away_from_nous_keeps_its_plan(cache):
    """The primary route is what the next turn restores: a fallback provider mid-turn must not hide the
    plan the session is actually spending."""
    from agent.credits_tracker import CreditsState
    from agent.usage_gauges import WindowGauge
    from gateway.session_usage import plan_gauge

    day = WindowGauge(id="day", scope="all", label="today", used_bp=4_200, resets_at_ms=4_000_000_000_000)
    agent = _agent("deepseek")  # the fallback is serving this turn
    agent._primary_runtime = {"provider": "nous", "base_url": "https://inference-api.nousresearch.com/v1"}
    agent._credits_state = CreditsState(denominator_kind="usage_window", gauges=(day,))

    assert plan_gauge(agent)["id"] == "day"

