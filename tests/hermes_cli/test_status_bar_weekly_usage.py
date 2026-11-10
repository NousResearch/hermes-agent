"""The opt-in ``weekly_usage`` status-bar field: the share of the provider's weekly allowance spent.

The number comes from the provider's usage API, so the bar must read it from the shared cache and
refresh in the background: a slow usage endpoint can never be allowed to stall a repaint, and a
repaint every second must never become a request every second.
"""

import threading
from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock, patch

import pytest

import providers
from agent import account_usage
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.account_usage_cache import cached_account_usage, remember_account_usage
from cli import HermesCLI
from hermes_cli.status_bar_usage import WEEKLY_WINDOW_LABELS, weekly_usage_percent, weekly_window
from providers.base import ProviderProfile
from tests.waiting import wait_until

_OPTED_IN = frozenset({"model", "weekly_usage"})
_WIDE_COLUMNS = 120
_MID_COLUMNS = 60
_NARROW_COLUMNS = 40
# Width tiers of ``_status_bar_segments``: below 52 columns the bar keeps only model and duration.
_FIRST_MID_COLUMNS = 52
_LAST_NARROW_COLUMNS = 51
_FETCH_DEADLINE_SECONDS = 30
_OAUTH_SHAPED_TOKEN = "sk-ant-oat01-test"


class _Agent:
    def __init__(self, provider="anthropic"):
        self.provider = provider
        self.model = "claude-sonnet-4"


def _make_cli(field_set, provider: str | None = "anthropic"):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "claude-sonnet-4"
    cli_obj.session_start = datetime.now() - timedelta(minutes=3)
    cli_obj.conversation_history = []
    cli_obj.agent = _Agent(provider) if provider else None
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


class _CountingUsageApi:
    """A usage API stand-in that answers at once and counts every request."""

    def __init__(self, answer=None, error=None):
        self.answer, self.error = answer, error
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, _base_url, _api_key):
        with self.lock:
            self.calls += 1
        if self.error:
            raise self.error
        return self.answer


# ── Reading the cached number ──────────────────────────────────────────────────


def test_segment_is_opt_in_and_the_default_bar_never_asks_the_usage_api(monkeypatch):
    calls = _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(49.0)))

    cli_obj = _make_cli(None)
    snapshot = cli_obj._get_status_bar_snapshot()

    assert snapshot["weekly_usage_percent"] is None
    assert "wk " not in _bar_text(cli_obj, snapshot, field_set=None)
    assert calls == [], "the default field set must not start a usage fetch"


def test_a_fields_list_without_weekly_usage_never_asks_the_usage_api(monkeypatch):
    calls = _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(49.0)))
    fields = frozenset({"model", "context_pct", "git_branch"})

    cli_obj = _make_cli(fields)
    snapshot = cli_obj._get_status_bar_snapshot()

    assert snapshot["weekly_usage_percent"] is None
    assert "wk " not in _bar_text(cli_obj, snapshot, field_set=fields)
    assert calls == []


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


@pytest.mark.parametrize(("used", "shown"),
                         [(-1.0, 0), (0.0, 0), (0.01, 1), (99.01, 100), (100.0, 100), (104.0, 100)])
def test_reading_is_rounded_up_and_kept_within_0_to_100(monkeypatch, used, shown):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(used)))

    assert weekly_usage_percent("anthropic") == shown


def test_weekly_window_without_a_reading_is_skipped(monkeypatch):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(None), _week(48.2, label="Weekly")))

    assert weekly_usage_percent("anthropic") == 49


def test_stale_snapshot_shows_nothing_and_still_asks_for_a_fresh_one(monkeypatch):
    calls = _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(49.0), age=timedelta(hours=1)))

    assert weekly_usage_percent("anthropic") is None
    assert calls == [["anthropic"]]


def test_snapshot_without_a_weekly_window_shows_nothing(monkeypatch):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(30.0, label="Current session")))

    assert weekly_usage_percent("anthropic") is None


def test_provider_without_a_usage_api_never_starts_a_fetch(monkeypatch):
    calls = _no_refresh(monkeypatch)

    assert weekly_usage_percent("custom") is None
    assert weekly_usage_percent(None) is None
    assert weekly_usage_percent("   ") is None
    assert calls == []


def test_provider_name_is_normalised_before_the_cache_is_read(monkeypatch):
    calls = _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(49.0)))

    assert weekly_usage_percent("  Anthropic ") == 49
    assert calls == [["anthropic"]]


def test_a_failing_usage_read_hides_the_segment_and_keeps_the_bar(monkeypatch):
    def broken(_provider):
        raise RuntimeError("usage cache unavailable")

    monkeypatch.setattr("agent.account_usage_cache.has_account_usage", broken)
    cli_obj = _make_cli(_OPTED_IN)

    assert weekly_usage_percent("anthropic") is None
    text = cli_obj._build_status_bar_text(width=_WIDE_COLUMNS)
    assert "claude-sonnet-4" in text and "wk " not in text


# ── Background refresh ─────────────────────────────────────────────────────────


def test_bar_reads_the_cache_and_refreshes_without_waiting(monkeypatch):
    release, called, answered = threading.Event(), threading.Event(), threading.Event()

    def slow_usage_api(_base_url, _api_key):
        called.set()
        release.wait(_FETCH_DEADLINE_SECONDS)
        answered.set()
        return _snapshot(_week(48.2))

    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "anthropic", slow_usage_api)
    cli_obj = _make_cli(_OPTED_IN)

    first = cli_obj._get_status_bar_snapshot()
    assert not answered.is_set(), "the repaint waited on the usage API"
    assert first["weekly_usage_percent"] is None, "nothing cached yet, so nothing to show"
    assert "wk " not in _bar_text(cli_obj, first)
    assert called.wait(_FETCH_DEADLINE_SECONDS), "an opted-in repaint must start a background usage refresh"

    release.set()
    wait_until(lambda: cached_account_usage("anthropic") is not None, timeout=_FETCH_DEADLINE_SECONDS,
               what="the background refresh cached the usage snapshot")

    second = cli_obj._get_status_bar_snapshot()
    assert second["weekly_usage_percent"] == 49
    assert "wk 49%" in _bar_text(cli_obj, second)


def test_repaints_ask_the_usage_api_once_per_refresh_interval(monkeypatch):
    usage_api = _CountingUsageApi(answer=_snapshot(_week(49.0)))
    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "anthropic", usage_api)
    cli_obj = _make_cli(_OPTED_IN)

    for _ in range(50):  # fifty idle redraws: under a minute at the default one-second interval
        cli_obj._get_status_bar_snapshot()
    wait_until(lambda: cli_obj._get_status_bar_snapshot()["weekly_usage_percent"] == 49,
               timeout=_FETCH_DEADLINE_SECONDS, what="the reading reached the bar")
    assert usage_api.calls == 1, "repaints inside the refresh interval must share one request"

    monkeypatch.setattr("agent.account_usage_cache.REFRESH_AFTER_S", 0.0)  # the interval has passed
    cli_obj._get_status_bar_snapshot()
    wait_until(lambda: usage_api.calls == 2, timeout=_FETCH_DEADLINE_SECONDS,
               what="a repaint after the interval asked again")


def test_a_failed_fetch_is_not_retried_on_every_repaint(monkeypatch):
    usage_api = _CountingUsageApi(error=RuntimeError("usage API unreachable"))
    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "anthropic", usage_api)
    cli_obj = _make_cli(_OPTED_IN)

    cli_obj._get_status_bar_snapshot()
    wait_until(lambda: usage_api.calls == 1, timeout=_FETCH_DEADLINE_SECONDS, what="the first try")
    snapshots = [cli_obj._get_status_bar_snapshot() for _ in range(20)]

    assert usage_api.calls == 1
    assert all(snapshot["weekly_usage_percent"] is None for snapshot in snapshots)
    assert "claude-sonnet-4" in cli_obj._build_status_bar_text(width=_WIDE_COLUMNS)


def test_an_empty_answer_does_not_erase_the_last_reading(monkeypatch):
    remember_account_usage("anthropic", _snapshot(_week(49.0)))
    usage_api = _CountingUsageApi(answer=_snapshot())
    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "anthropic", usage_api)
    cli_obj = _make_cli(_OPTED_IN)

    cli_obj._get_status_bar_snapshot()
    wait_until(lambda: usage_api.calls == 1, timeout=_FETCH_DEADLINE_SECONDS, what="the refresh ran")

    assert cli_obj._get_status_bar_snapshot()["weekly_usage_percent"] == 49


def test_a_plugin_providers_weekly_window_reaches_the_bar(monkeypatch):
    class _PluginWithUsage(ProviderProfile):
        def fetch_account_usage(self, *, base_url=None, api_key=None):
            return _snapshot(_week(63.4, label="Weekly"), _week(10.0, label="Rolling window"),
                             provider="plugin-weekly")

    profile = _PluginWithUsage(name="plugin-weekly")
    real_lookup = providers.get_provider_profile
    monkeypatch.setattr(providers, "get_provider_profile",
                        lambda name: profile if name == "plugin-weekly" else real_lookup(name))
    cli_obj = _make_cli(_OPTED_IN, provider="plugin-weekly")

    wait_until(lambda: cli_obj._get_status_bar_snapshot()["weekly_usage_percent"] == 64,
               timeout=_FETCH_DEADLINE_SECONDS, what="the plugin's weekly reading reached the bar")


# ── Whose usage ────────────────────────────────────────────────────────────────


def test_the_cli_provider_stands_in_until_an_agent_exists(monkeypatch):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(49.0)))
    cli_obj = _make_cli(_OPTED_IN, provider=None)
    cli_obj.provider = "anthropic"

    assert cli_obj._get_status_bar_snapshot()["weekly_usage_percent"] == 49


def test_the_live_agents_provider_wins_over_the_cli_default(monkeypatch):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(49.0)))
    remember_account_usage("openai-codex", _snapshot(_week(70.0, label="Weekly"), provider="openai-codex"))
    cli_obj = _make_cli(_OPTED_IN, provider="openai-codex")
    cli_obj.provider = "anthropic"

    assert cli_obj._get_status_bar_snapshot()["weekly_usage_percent"] == 70


def test_switching_provider_mid_session_switches_the_reading(monkeypatch):
    calls = _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(49.0)))
    remember_account_usage("openai-codex", _snapshot(_week(70.0, label="Weekly"), provider="openai-codex"))
    cli_obj = _make_cli(_OPTED_IN)

    assert cli_obj._get_status_bar_snapshot()["weekly_usage_percent"] == 49
    cli_obj.agent.provider = "openai-codex"  # /model to a Codex model, or a fallback
    assert cli_obj._get_status_bar_snapshot()["weekly_usage_percent"] == 70
    cli_obj.agent.provider = "custom"
    assert cli_obj._get_status_bar_snapshot()["weekly_usage_percent"] is None
    assert calls == [["anthropic"], ["openai-codex"]]


# ── Fetcher labels the bar depends on ──────────────────────────────────────────


def test_anthropic_fetcher_labels_its_weekly_window_as_the_bar_expects(monkeypatch):
    payload = {"five_hour": {"utilization": 12.0, "resets_at": None},
               "seven_day": {"utilization": 48.2, "resets_at": None},
               "seven_day_opus": {"utilization": 99.0, "resets_at": None}}
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: _OAUTH_SHAPED_TOKEN)
    monkeypatch.setattr(account_usage, "_get_json", lambda *_args, **_kwargs: payload)

    window = weekly_window(account_usage._fetch_anthropic_account_usage())

    assert window is not None and window.used_percent == 48.2


def test_codex_weekly_window_label_is_one_the_bar_reads():
    labels = dict(account_usage._codex_window_labels({"primary_window": {"limit_window_seconds": 604800}}))

    assert labels["primary_window"] in WEEKLY_WINDOW_LABELS


# ── Rendering ──────────────────────────────────────────────────────────────────


def test_colour_follows_how_much_of_the_week_is_spent(monkeypatch):
    _no_refresh(monkeypatch)
    cli_obj = _make_cli(_OPTED_IN)
    snapshot = cli_obj._get_status_bar_snapshot()

    for percent, style in ((0, "good"), (49, "good"), (50, "warn"), (79, "warn"), (80, "critical"), (100, "critical")):
        snapshot["weekly_usage_percent"] = percent
        assert _weekly_style(cli_obj, snapshot) == f"class:status-bar-{style}", percent


def test_narrow_bar_drops_the_segment_at_the_same_width_as_cache_hit(monkeypatch):
    _no_refresh(monkeypatch)
    cli_obj = _make_cli(_OPTED_IN)
    snapshot = cli_obj._get_status_bar_snapshot()
    snapshot["weekly_usage_percent"] = 49

    for width in (_FIRST_MID_COLUMNS, _MID_COLUMNS, _WIDE_COLUMNS):
        assert "wk 49%" in _bar_text(cli_obj, snapshot, width=width), width
    for width in (_LAST_NARROW_COLUMNS, _NARROW_COLUMNS):
        assert "wk 49%" not in _bar_text(cli_obj, snapshot, width=width), width


def test_segment_sits_between_the_context_gauge_and_the_speed_readings(monkeypatch):
    _no_refresh(monkeypatch)
    fields = frozenset({"model", "context_pct", "cache_hit", "weekly_usage", "latency", "duration"})
    cli_obj = _make_cli(fields)
    snapshot = cli_obj._get_status_bar_snapshot()
    snapshot.update(weekly_usage_percent=49, cache_hit_pct=48.0, avg_latency_label="4.8s")

    text = _bar_text(cli_obj, snapshot, field_set=fields)

    assert text.index("◎ 48.0%") < text.index("wk 49%") < text.index("◷ 4.8s") < text.index("3m"), text


def test_plain_text_bar_shows_the_segment(monkeypatch):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(49.0)))
    cli_obj = _make_cli(_OPTED_IN)

    assert "wk 49%" in cli_obj._build_status_bar_text(width=_WIDE_COLUMNS)


def _prompt_toolkit_fragments(cli_obj, width=_WIDE_COLUMNS):
    cli_obj._status_bar_visible = True
    mock_app = MagicMock()
    mock_app.output.get_size.return_value = MagicMock(columns=width)
    with patch("prompt_toolkit.application.get_app", return_value=mock_app):
        return cli_obj._get_status_bar_fragments()


def test_prompt_toolkit_bar_paints_the_segment_in_its_threshold_colour(monkeypatch):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(83.0)))
    cli_obj = _make_cli(_OPTED_IN)

    assert ("class:status-bar-critical", "wk 83%") in _prompt_toolkit_fragments(cli_obj)


def test_modern_footer_shows_the_segment(monkeypatch):
    _no_refresh(monkeypatch)
    remember_account_usage("anthropic", _snapshot(_week(49.0)))
    cli_obj = _make_cli(_OPTED_IN)
    cli_obj._modern_layout = True

    fragments = _prompt_toolkit_fragments(cli_obj)

    weekly = [style for style, text in fragments if text == "wk 49%"]
    assert len(weekly) == 1 and "class:status-bar-good" in weekly[0].split(), fragments


@pytest.mark.parametrize(("language", "label"), [("en", "wk 49%"), ("de", "Wo. 49%"), ("ja", "週 49%")])
def test_label_follows_the_interface_language(monkeypatch, language, label):
    _no_refresh(monkeypatch)
    monkeypatch.setenv("HERMES_LANGUAGE", language)
    remember_account_usage("anthropic", _snapshot(_week(48.2)))
    cli_obj = _make_cli(_OPTED_IN)

    assert label in cli_obj._build_status_bar_text(width=_WIDE_COLUMNS)
