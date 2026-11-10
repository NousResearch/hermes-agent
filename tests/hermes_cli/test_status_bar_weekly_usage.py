"""The opt-in ``weekly_usage`` status-bar field: the share of the provider's weekly allowance spent.

The number comes from the provider's usage API, so the bar must read it from the shared cache and
refresh in the background: a slow usage endpoint can never be allowed to stall a repaint.
"""

import threading
from datetime import datetime, timedelta, UTC

from agent import account_usage
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.account_usage_cache import cached_account_usage, remember_account_usage
from cli import HermesCLI
from hermes_cli.status_bar_usage import WEEKLY_WINDOW_LABELS, weekly_usage_percent, weekly_window
from tests.waiting import wait_until

_OPTED_IN = frozenset({"model", "weekly_usage"})
_WIDE_COLUMNS = 120
_MID_COLUMNS = 60
_NARROW_COLUMNS = 40


class _Agent:
    provider = "anthropic"
    model = "claude-sonnet-4"


def _make_cli(field_set):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "claude-sonnet-4"
    cli_obj.session_start = datetime.now() - timedelta(minutes=3)
    cli_obj.conversation_history = []
    cli_obj.agent = _Agent()
    cli_obj._status_bar_field_set_cache = field_set
    return cli_obj


def _bar_text(cli_obj, snapshot, width=_WIDE_COLUMNS, field_set=_OPTED_IN):
    segments = cli_obj._status_bar_segments(snapshot, width, field_set, False, styled=False)
    return " | ".join("".join(text for _, text in segment) for segment in segments)


def _weekly_style(cli_obj, snapshot):
    segments = cli_obj._status_bar_segments(snapshot, _WIDE_COLUMNS, _OPTED_IN, False, styled=True)
    (segment,) = [segment for segment in segments if "".join(text for _, text in segment).startswith("wk ")]
    return segment[0][0]


def _snapshot(*windows, provider="anthropic", age=timedelta(0)):
    return AccountUsageSnapshot(provider=provider, source="test", fetched_at=datetime.now(UTC) - age,
                                windows=tuple(windows))


def _week(used, label="Current week", scope="account"):
    return AccountUsageWindow(label=label, used_percent=used, scope=scope,
                              reset_at=datetime.now(UTC) + timedelta(days=3))


def _no_refresh(monkeypatch):
    calls = []
    monkeypatch.setattr("agent.account_usage_cache.refresh_account_usage_async",
                        lambda providers: calls.append(list(providers)) or [])
    return calls


def test_segment_is_opt_in_and_the_default_bar_never_asks_the_usage_api(monkeypatch):
    calls = _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(49.0)))

    cli_obj = _make_cli(None)
    snapshot = cli_obj._get_status_bar_snapshot()

    assert snapshot["weekly_usage_percent"] is None
    assert "wk " not in _bar_text(cli_obj, snapshot, field_set=None)
    assert calls == [], "the default field set must not start a usage fetch"


def test_bar_reads_the_cache_and_refreshes_without_waiting(monkeypatch):
    release, called, answered = threading.Event(), threading.Event(), threading.Event()

    def slow_usage_api(_base_url, _api_key):
        called.set()
        release.wait(30)
        answered.set()
        return _snapshot(_week(48.2))

    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "anthropic", slow_usage_api)
    cli_obj = _make_cli(_OPTED_IN)

    first = cli_obj._get_status_bar_snapshot()
    assert not answered.is_set(), "the repaint waited on the usage API"
    assert first["weekly_usage_percent"] is None, "nothing cached yet, so nothing to show"
    assert "wk " not in _bar_text(cli_obj, first)
    assert called.wait(30), "an opted-in repaint must start a background usage refresh"

    release.set()
    wait_until(lambda: cached_account_usage("anthropic") is not None, timeout=30,
               what="the background refresh cached the usage snapshot")

    second = cli_obj._get_status_bar_snapshot()
    assert second["weekly_usage_percent"] == 49
    assert "wk 49%" in _bar_text(cli_obj, second)


def test_percent_rounds_up_and_only_the_account_weekly_window_counts(monkeypatch):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(
        _week(90.0, label="Current session"),
        _week(99.0, label="Opus week", scope="model"),
        _week(97.0, label="Weekly", scope="model"),
        _week(48.2),
    ))

    assert weekly_usage_percent("anthropic") == 49


def test_whole_percent_is_not_rounded_up(monkeypatch):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(21.0)))

    assert weekly_usage_percent("anthropic") == 21


def test_stale_snapshot_shows_nothing_rather_than_an_old_number(monkeypatch):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(49.0), age=timedelta(hours=1)))

    assert weekly_usage_percent("anthropic") is None


def test_snapshot_without_a_weekly_window_shows_nothing(monkeypatch):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(30.0, label="Current session")))

    assert weekly_usage_percent("anthropic") is None


def test_provider_without_a_usage_api_never_starts_a_fetch(monkeypatch):
    calls = _no_refresh(monkeypatch)

    assert weekly_usage_percent("custom") is None
    assert weekly_usage_percent(None) is None
    assert calls == []


def test_a_failing_usage_read_hides_the_segment_and_keeps_the_bar(monkeypatch):
    def broken(_provider):
        raise RuntimeError("usage cache unavailable")

    monkeypatch.setattr("agent.account_usage_cache.has_account_usage", broken)
    cli_obj = _make_cli(_OPTED_IN)

    assert weekly_usage_percent("anthropic") is None
    text = cli_obj._build_status_bar_text(width=_WIDE_COLUMNS)
    assert "claude-sonnet-4" in text and "wk " not in text


def test_colour_follows_how_much_of_the_week_is_spent(monkeypatch):
    _no_refresh(monkeypatch)
    cli_obj = _make_cli(_OPTED_IN)
    snapshot = cli_obj._get_status_bar_snapshot()

    for percent, style in ((10, "good"), (49, "good"), (50, "warn"), (79, "warn"), (80, "critical"), (100, "critical")):
        snapshot["weekly_usage_percent"] = percent
        assert _weekly_style(cli_obj, snapshot) == f"class:status-bar-{style}", percent


def test_narrow_bar_drops_the_segment(monkeypatch):
    _no_refresh(monkeypatch)
    cli_obj = _make_cli(_OPTED_IN)
    snapshot = cli_obj._get_status_bar_snapshot()
    snapshot["weekly_usage_percent"] = 49

    assert "wk 49%" in _bar_text(cli_obj, snapshot, width=_MID_COLUMNS)
    assert "wk 49%" not in _bar_text(cli_obj, snapshot, width=_NARROW_COLUMNS)


def test_plain_text_bar_shows_the_segment(monkeypatch):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(49.0)))
    cli_obj = _make_cli(_OPTED_IN)

    assert "wk 49%" in cli_obj._build_status_bar_text(width=_WIDE_COLUMNS)


def test_anthropic_fetcher_labels_its_weekly_window_as_the_bar_expects(monkeypatch):
    payload = {"five_hour": {"utilization": 12.0, "resets_at": None},
               "seven_day": {"utilization": 48.2, "resets_at": None},
               "seven_day_opus": {"utilization": 99.0, "resets_at": None}}
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "sk-ant-oat01-test")
    monkeypatch.setattr(account_usage, "_get_json", lambda *_args, **_kwargs: payload)

    window = weekly_window(account_usage._fetch_anthropic_account_usage())

    assert window is not None and window.used_percent == 48.2


def test_codex_weekly_window_label_is_one_the_bar_reads():
    labels = dict(account_usage._codex_window_labels({"primary_window": {"limit_window_seconds": 604800}}))

    assert labels["primary_window"] in WEEKLY_WINDOW_LABELS
