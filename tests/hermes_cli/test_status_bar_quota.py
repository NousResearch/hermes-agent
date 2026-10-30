"""The status-bar plan-allowance segment (⊛ 5h 87% · ⊛ wk 44%).

The indicator answers "how much of my plan is LEFT", so it renders the REMAINING
percent of each provider window, read from the shared account-usage cache (never from
the network) and only for the active provider.
"""

from datetime import UTC, datetime, timedelta

import pytest
import re

import cli as cli_mod  # imported at module scope, like test_cli_status_bar.py: the status
# bar imports ``cli`` lazily at render time, and a first import inside a test body
# would run under the real-home I/O guard.
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from hermes_cli import status_bar_quota


def _snapshot(*windows: AccountUsageWindow, provider: str = "minimax") -> AccountUsageSnapshot:
    return AccountUsageSnapshot(provider=provider, source="token_plan_api",
                                fetched_at=datetime.now(UTC), windows=windows)


def _texts(segments) -> list[str]:
    """The rendered text of each segment, one string per segment."""
    return [seg[0][1] for seg in segments]


@pytest.fixture
def cached(monkeypatch):
    """Install a snapshot in the account-usage cache for a provider."""
    def _install(provider: str, snapshot: AccountUsageSnapshot | None) -> None:
        monkeypatch.setattr("agent.account_usage_cache.cached_account_usage",
                            lambda slug, **_kw: snapshot if slug == provider else None)
    return _install


class TestRemainingSegments:
    def test_shows_what_is_left_not_what_was_spent(self, cached):
        """The wire reports remaining percent; the bar shows the same number."""
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=13.0),
            AccountUsageWindow(label="weekly limit", short_label="wk", used_percent=56.0)))

        segments = status_bar_quota.remaining_segments("minimax")

        assert _texts(segments) == ["⊛ 5h 87%", "⊛ wk 44%"]

    def test_each_window_is_its_own_segment(self, cached):
        """Separate segments keep the bar's separator between the two windows."""
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=13.0),
            AccountUsageWindow(label="weekly limit", short_label="wk", used_percent=56.0)))

        segments = status_bar_quota.remaining_segments("minimax")

        assert len(segments) == 2
        assert all(len(seg) == 1 for seg in segments)

    def test_uses_the_short_label_when_the_provider_gives_one(self, cached):
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=10.0)))

        segments = status_bar_quota.remaining_segments("minimax")

        assert segments[0][0][1] == "⊛ 5h 90%"

    def test_falls_back_to_the_full_label(self, cached):
        """A provider with no short label still renders, just less compactly."""
        cached("minimax", _snapshot(
            AccountUsageWindow(label="Current session", used_percent=25.0)))

        segments = status_bar_quota.remaining_segments("minimax")

        assert segments[0][0][1] == "⊛ Current session 75%"

    def test_no_snapshot_means_no_segment(self, cached):
        cached("minimax", None)

        assert status_bar_quota.remaining_segments("minimax") == []

    def test_unknown_provider_means_no_segment(self):
        assert status_bar_quota.remaining_segments("") == []
        assert status_bar_quota.remaining_segments(None) == []

    def test_a_window_without_a_percent_is_skipped(self, cached):
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=None),
            AccountUsageWindow(label="weekly limit", short_label="wk", used_percent=50.0)))

        segments = status_bar_quota.remaining_segments("minimax")

        assert _texts(segments) == ["⊛ wk 50%"]

    def test_a_stale_snapshot_is_not_shown_as_a_live_number(self, monkeypatch):
        """An old snapshot would mislabel an account that has since recovered."""
        stale = _snapshot(AccountUsageWindow(label="5h limit", short_label="5h", used_percent=90.0))
        monkeypatch.setattr("agent.account_usage_cache.cached_account_usage", lambda slug, **_kw: stale)
        monkeypatch.setattr("agent.account_usage_cache.snapshot_is_stale", lambda snapshot: True)

        assert status_bar_quota.remaining_segments("minimax") == []

    def test_a_full_window_reads_as_full(self, cached):
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=0.0)))

        segments = status_bar_quota.remaining_segments("minimax")

        assert segments[0][0][1] == "⊛ 5h 100%"

    def test_an_exhausted_window_reads_as_empty(self, cached):
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=100.0)))

        segments = status_bar_quota.remaining_segments("minimax")

        assert segments[0][0][1] == "⊛ 5h 0%"


class TestResetCountdown:
    def test_the_rolling_window_carries_its_reset_countdown(self, cached):
        """The first window says when the allowance comes back."""
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=18.0,
                               reset_at=datetime.now(UTC) + timedelta(hours=1, minutes=43)),
            AccountUsageWindow(label="weekly limit", short_label="wk", used_percent=68.0,
                               reset_at=datetime.now(UTC) + timedelta(days=2, hours=15))))

        segments = status_bar_quota.remaining_segments("minimax")

        # The countdown is read a moment after the window was built, so it can land on
        # either side of a minute boundary: assert the shape, not the exact tick.
        assert re.fullmatch(r"⊛ 5h 82% · 1h 4[23]m \(\d{2}\.\d{2}\)", _texts(segments)[0])

    def test_the_reset_clock_is_reported_alongside_the_segments(self, cached):
        """The bar gives up its context meter for this clock, so it has to know."""
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=18.0,
                               reset_at=datetime.now(UTC) + timedelta(hours=1, minutes=43))))

        _segments, has_clock = status_bar_quota.remaining_segments_with_clock("minimax")

        assert has_clock is True

    def test_no_clock_is_reported_when_no_window_has_one(self, cached):
        """Without a reset instant there is no clock, and the meter stays."""
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=18.0,
                               reset_at=None)))

        _segments, has_clock = status_bar_quota.remaining_segments_with_clock("minimax")

        assert has_clock is False

    def test_the_later_window_does_not_carry_a_countdown(self, cached):
        """A weekly reset is days out: it does not belong in a one-line bar."""
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=18.0,
                               reset_at=datetime.now(UTC) + timedelta(hours=1)),
            AccountUsageWindow(label="weekly limit", short_label="wk", used_percent=68.0,
                               reset_at=datetime.now(UTC) + timedelta(days=2))))

        segments = status_bar_quota.remaining_segments("minimax")

        assert _texts(segments)[1] == "⊛ wk 32%"

    def test_a_window_without_a_reset_shows_no_countdown(self, cached):
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=18.0,
                               reset_at=None)))

        segments = status_bar_quota.remaining_segments("minimax")

        assert _texts(segments)[0] == "⊛ 5h 82%"

    def test_a_reset_in_the_past_shows_no_countdown(self, cached):
        """A stale reset would read as a countdown into the negative."""
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=18.0,
                               reset_at=datetime.now(UTC) - timedelta(minutes=5))))

        segments = status_bar_quota.remaining_segments("minimax")

        assert _texts(segments)[0] == "⊛ 5h 82%"

    def test_a_reset_under_a_minute_shows_no_countdown(self, cached):
        """Under a minute the countdown would tick every second: noise, not information."""
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=18.0,
                               reset_at=datetime.now(UTC) + timedelta(seconds=30))))

        segments = status_bar_quota.remaining_segments("minimax")

        assert _texts(segments)[0] == "⊛ 5h 82%"


class TestRemainingStyle:
    def test_colours_by_what_is_left(self):
        assert status_bar_quota._remaining_style(90.0) == "class:status-bar-good"
        assert status_bar_quota._remaining_style(40.0) == "class:status-bar-warn"
        assert status_bar_quota._remaining_style(10.0) == "class:status-bar-bad"
        assert status_bar_quota._remaining_style(0.0) == "class:status-bar-critical"


class TestPlanSegments:
    def test_a_hidden_field_produces_nothing(self, cached, monkeypatch):
        """A filtered-out field means the provider is never asked either."""
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=13.0)))
        asked: list[list[str]] = []
        monkeypatch.setattr("agent.account_usage_cache.has_account_usage", lambda slug: True)
        monkeypatch.setattr("agent.account_usage_cache.refresh_account_usage_async",
                            lambda providers: asked.append(list(providers)))
        monkeypatch.setattr(status_bar_quota, "_last_refresh", {})

        segments, has_clock = status_bar_quota.plan_segments("minimax", frozenset({"model"}))

        assert segments == []
        assert has_clock is False
        assert asked == []

    def test_a_visible_field_renders_and_refreshes(self, cached, monkeypatch):
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=13.0)))
        asked: list[list[str]] = []
        monkeypatch.setattr("agent.account_usage_cache.has_account_usage", lambda slug: True)
        monkeypatch.setattr("agent.account_usage_cache.refresh_account_usage_async",
                            lambda providers: asked.append(list(providers)))
        monkeypatch.setattr(status_bar_quota, "_last_refresh", {})

        segments, has_clock = status_bar_quota.plan_segments("minimax", frozenset({"quota"}))

        assert _texts(segments) == ["⊛ 5h 87%"]
        assert has_clock is False
        assert asked == [["minimax"]]

    def test_no_field_filter_shows_the_allowance(self, cached, monkeypatch):
        """An unfiltered bar (the default) shows the allowance."""
        cached("minimax", _snapshot(
            AccountUsageWindow(label="5h limit", short_label="5h", used_percent=13.0)))
        monkeypatch.setattr(status_bar_quota, "_last_refresh", {})

        segments, _has_clock = status_bar_quota.plan_segments("minimax", None)

        assert _texts(segments) == ["⊛ 5h 87%"]


class TestRefreshRemaining:
    def test_asks_for_a_background_refresh(self, monkeypatch):
        """The refresh runs off the repaint path: the bar only ever reads the cache."""
        asked: list[list[str]] = []
        monkeypatch.setattr("agent.account_usage_cache.has_account_usage", lambda slug: True)
        monkeypatch.setattr("agent.account_usage_cache.refresh_account_usage_async",
                            lambda providers: asked.append(list(providers)))
        monkeypatch.setattr(status_bar_quota, "_last_refresh", {})

        status_bar_quota.refresh_remaining("minimax")

        assert asked == [["minimax"]]

    def test_a_provider_without_an_allowance_api_is_never_asked(self, monkeypatch):
        asked: list[list[str]] = []
        monkeypatch.setattr("agent.account_usage_cache.has_account_usage", lambda slug: False)
        monkeypatch.setattr("agent.account_usage_cache.refresh_account_usage_async",
                            lambda providers: asked.append(list(providers)))
        monkeypatch.setattr(status_bar_quota, "_last_refresh", {})

        status_bar_quota.refresh_remaining("some-provider")

        assert asked == []

    def test_the_repaint_path_asks_at_most_once_per_interval(self, monkeypatch):
        """A status bar repaints constantly; the fetch must not follow it."""
        asked: list[list[str]] = []
        monkeypatch.setattr("agent.account_usage_cache.has_account_usage", lambda slug: True)
        monkeypatch.setattr("agent.account_usage_cache.refresh_account_usage_async",
                            lambda providers: asked.append(list(providers)))
        monkeypatch.setattr(status_bar_quota, "_last_refresh", {})

        for _ in range(10):
            status_bar_quota.refresh_remaining("minimax")

        assert len(asked) == 1
