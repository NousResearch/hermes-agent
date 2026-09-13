"""Status-bar ``limits`` segment: subscription rate limits for Codex and Claude Code."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import cli as cli_mod
from cli import HermesCLI
from hermes_cli.status_bar_limits import AccountLimitsPoller, bar_readings, format_limit


def _window(label, used):
    return SimpleNamespace(label=label, used_percent=used)


def _codex(session=None, weekly=None):
    return SimpleNamespace(windows=(_window("Session", session), _window("Weekly", weekly)))


def _claude(session=None, week=None, opus=None, fable=None):
    return SimpleNamespace(windows=(_window("Current session", session), _window("Current week", week),
                                    _window("Opus week", opus), _window("Fable week", fable)))


def _cli_with_poller(poller):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "openai-codex/gpt-5"
    cli_obj.session_start = datetime.now() - timedelta(minutes=3)
    cli_obj.conversation_history = []
    cli_obj.agent = None
    cli_obj._account_limits_poller = poller
    return cli_obj


def test_shortest_window_is_shown_and_account_week_is_not():
    assert bar_readings(_codex(session=22, weekly=100), "cx") == [("cx", 22.0)]
    assert bar_readings(_claude(session=27, week=90), "cc") == [("cc", 27.0)]
    assert bar_readings(_claude(), "cc") == []
    assert bar_readings(None, "cc") == []


def test_weekly_only_plan_falls_back_to_the_weekly_window():
    # Codex Pro reports no 5-hour window: the weekly cap is then the one that blocks you.
    assert bar_readings(_codex(session=None, weekly=27), "cx") == [("cx", 27.0)]


def test_per_model_weekly_caps_get_their_own_reading():
    assert bar_readings(_claude(session=36, week=30, fable=57), "cc") == [("cc", 36.0), ("fable", 57.0)]
    assert bar_readings(_claude(session=36, week=30, opus=12, fable=57), "cc") == [
        ("cc", 36.0), ("opus", 12.0), ("fable", 57.0)]


def test_format_is_fixed_width_across_magnitudes():
    widths = {len(format_limit("cx", v)) for v in (0, 7, 27, 100)}
    assert widths == {len("cx 100%")}


def test_poller_keeps_last_good_reading_when_a_fetch_fails():
    calls = {"n": 0}

    def fetch(provider):
        calls["n"] += 1
        if provider == "openai-codex":
            if calls["n"] > 2:
                raise RuntimeError("network")
            return _codex(session=22, weekly=100)
        return _claude(session=27, week=10)

    poller = AccountLimitsPoller(fetch=fetch)
    poller.refresh_once()
    assert poller.read() == [("cx", 22.0), ("cc", 27.0)]
    poller.refresh_once()  # codex fetch now raises; cx must survive
    assert poller.read() == [("cx", 22.0), ("cc", 27.0)]


def test_poller_drops_provider_that_reports_no_windows():
    poller = AccountLimitsPoller(fetch=lambda p: _claude(session=40) if p == "anthropic" else _codex())
    poller.refresh_once()
    assert poller.read() == [("cc", 40.0)]


def test_poller_requests_repaint_only_when_a_reading_changes():
    repaints = []
    poller = AccountLimitsPoller(fetch=lambda p: _codex(session=40), on_change=lambda: repaints.append(1))
    poller.refresh_once()
    poller.refresh_once()  # same values: no second repaint
    assert len(repaints) == 1


def test_limits_segment_renders_after_context_and_respects_field_gate():
    poller = AccountLimitsPoller(
        fetch=lambda p: _codex(session=100, weekly=22) if p == "openai-codex" else _claude(session=27, week=90, fable=57))
    poller.refresh_once()
    cli_obj = _cli_with_poller(poller)

    cli_obj._status_bar_field_set_cache = frozenset({"model", "context_pct", "limits"})
    text = cli_obj._build_status_bar_text(width=120)
    assert "cx 100%" in text and "cc  27%" in text and "fable  57%" in text
    assert text.index("--") < text.index("cx 100%") < text.index("cc  27%") < text.index("fable  57%")

    cli_obj._status_bar_field_set_cache = frozenset({"model", "context_pct"})
    text = cli_obj._build_status_bar_text(width=120)
    assert "cx" not in text and "cc" not in text


def test_limits_is_opt_in_and_never_polls_by_default():
    """The default field set (no ``fields`` list) hides the segment and must not start a network
    poller: users who never asked for it should not have Hermes calling provider APIs."""
    cli_obj = _cli_with_poller(None)
    with patch.object(cli_mod, "CLI_CONFIG", {"display": {"status_bar": {"fields": []}}}):
        text = cli_obj._build_status_bar_text(width=120)
    assert "cx" not in text and "cc" not in text
    assert cli_obj._account_limits_poller is None
