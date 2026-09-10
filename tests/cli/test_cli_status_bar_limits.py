"""Status-bar ``limits`` segment: subscription rate limits for Codex and Claude Code."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import cli as cli_mod
from cli import HermesCLI
from hermes_cli.status_bar_limits import AccountLimitsPoller, format_limit, session_window_percent


def _window(label, used):
    return SimpleNamespace(label=label, used_percent=used)


def _codex(session=None, weekly=None):
    return SimpleNamespace(windows=(_window("Session", session), _window("Weekly", weekly)))


def _claude(session=None, week=None, opus=None):
    return SimpleNamespace(windows=(_window("Current session", session), _window("Current week", week),
                                    _window("Opus week", opus)))


def _cli_with_poller(poller):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "openai-codex/gpt-5"
    cli_obj.session_start = datetime.now() - timedelta(minutes=3)
    cli_obj.conversation_history = []
    cli_obj.agent = None
    cli_obj._account_limits_poller = poller
    return cli_obj


def test_session_window_is_shown_not_the_weekly_one():
    assert session_window_percent(_codex(session=22, weekly=100)) == 22
    assert session_window_percent(_claude(session=27, week=90, opus=95)) == 27
    assert session_window_percent(_codex(session=None, weekly=27)) is None
    assert session_window_percent(_claude()) is None
    assert session_window_percent(None) is None


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


def test_poller_drops_provider_that_reports_no_session_window():
    poller = AccountLimitsPoller(fetch=lambda p: _claude(session=40) if p == "anthropic" else _codex(weekly=80))
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
        fetch=lambda p: _codex(session=100, weekly=22) if p == "openai-codex" else _claude(session=27, week=90))
    poller.refresh_once()
    cli_obj = _cli_with_poller(poller)

    cli_obj._status_bar_field_set_cache = frozenset({"model", "context_pct", "limits"})
    text = cli_obj._build_status_bar_text(width=120)
    assert "cx 100%" in text and "cc  27%" in text
    assert text.index("--") < text.index("cx 100%") < text.index("cc  27%")  # after the context read-out

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
