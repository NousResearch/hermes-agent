"""v5 desktop: hermes.desktop.{feature_use,friction,onboarding,dislike,mode_use,action_use}.

Every value the Desktop sends collapses onto a closed code-defined set, the backend latches again (durably,
in the profile's store, for feature use and the daily report) so a replay, a second window or a backend
restart cannot inflate a row, and a disabled profile writes nothing (not even a latch file)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_desktop as desktop
from hermes_cli.observability.shared_metrics import SharedMetricsStore
import tui_gateway.server as server

TODAY = "2026-09-27"


@pytest.fixture
def marks(tmp_path, monkeypatch):
    captured: list[tuple[str, dict]] = []
    policy = {"on": True}
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(relay_shared_metrics, "enabled", lambda: policy["on"])
    monkeypatch.setattr(relay_shared_metrics, "record_process_mark", lambda mark, data: captured.append((mark, data)))
    store = {"saves": True}

    def saved(rows):
        if not policy["on"]:
            return len(rows)
        if store["saves"]:
            captured.extend(rows)
        return len(rows) if store["saves"] else 0

    monkeypatch.setattr(relay_shared_metrics, "record_process_marks_saved", saved, raising=False)
    monkeypatch.setattr(desktop, "_daily", {})
    monkeypatch.setattr(desktop, "_utc_day", lambda: TODAY)
    yield SimpleNamespace(rows=captured, policy=policy, home=home, store=store)


def _stored(metric: str) -> list[tuple[str, dict, int]]:
    """``(period_start, dimensions, value)`` rows the profile's store holds for one metric."""
    return [(r["period_start"], r["dimensions"], r["value"])
            for r in SharedMetricsStore().counter_snapshot() if r["metric_name"] == metric]


def _rpc(method: str, **params) -> dict:
    return server.handle_request({"jsonrpc": "2.0", "id": "r1", "method": method, "params": params})


def _assert_valid(rows) -> None:
    for mark, data in rows:
        assert contract.counter_dimensions_are_valid(contract._DECISION_MARK_METRICS[mark], data), (mark, data)


def test_feature_use_counts_each_area_once_per_day_and_collapses_unknown_areas(marks):
    for area in ("terminal_pane", "terminal_pane", "settings_config_voice", "settings_brand_new_tab", "/Users/x"):
        assert _rpc("shared_metrics.desktop_feature_use", area=area)["result"] == {"ok": True}

    assert sorted((dims["area"], value) for _, dims, value in _stored(contract.DESKTOP_FEATURE_USE_METRIC)) == [
        ("other", 1), ("settings_config_voice", 1), ("settings_other", 1), ("terminal_pane", 1)]


def test_friction_detail_is_checked_against_its_own_kind_and_capped_per_day(marks):
    _rpc("shared_metrics.desktop_friction", kind="renderer_crash", detail="oom")
    _rpc("shared_metrics.desktop_friction", kind="renderer_crash", detail="network")  # a disconnect word
    _rpc("shared_metrics.desktop_friction", kind="error_toast", detail="ENOSPC: /home/alice/.hermes is full")
    _rpc("shared_metrics.desktop_friction", kind="made_up", detail="crash")
    for _ in range(desktop.FRICTION_DAILY_CAP + 5):
        _rpc("shared_metrics.desktop_friction", kind="slow_frame", detail="1s_to_5s")

    rows = [(d["kind"], d["detail"]) for _, d in marks.rows]
    assert rows[:3] == [("renderer_crash", "oom"), ("renderer_crash", "other"), ("error_toast", "other")]
    assert rows.count(("slow_frame", "1s_to_5s")) == desktop.FRICTION_DAILY_CAP
    _assert_valid(marks.rows)


def test_onboarding_is_once_per_step_event_across_processes(marks, monkeypatch):
    _rpc("shared_metrics.desktop_onboarding", step="provider_setup", event="reached")
    monkeypatch.setattr(desktop, "_daily", {})  # a fresh backend process: only the file latch remains
    _rpc("shared_metrics.desktop_onboarding", step="provider_setup", event="reached")
    _rpc("shared_metrics.desktop_onboarding", step="provider_setup", event="abandoned")
    _rpc("shared_metrics.desktop_onboarding", step="not_a_step", event="reached")

    assert [(d["step"], d["event"]) for _, d in marks.rows] == [
        ("provider_setup", "reached"), ("provider_setup", "abandoned")]
    _assert_valid(marks.rows)


def test_turning_collection_off_drops_the_onboarding_latches(marks):
    _rpc("shared_metrics.desktop_onboarding", step="guide", event="reached")
    latches = marks.home / "telemetry" / "shared_metrics" / desktop.ONBOARDING_LATCH_DIRNAME
    assert (latches / "guide.reached").exists()

    assert _rpc("shared_metrics.set", enabled=False)["result"]["enabled"] is False
    assert not latches.exists()


def test_disabled_profile_records_nothing_and_leaves_no_latch(marks):
    marks.policy["on"] = False
    _rpc("shared_metrics.desktop_feature_use", area="projects")
    _rpc("shared_metrics.desktop_friction", kind="renderer_crash", detail="crash")
    _rpc("shared_metrics.desktop_onboarding", step="consent", event="completed")
    _rpc("shared_metrics.desktop_dislike", signal="quick_close", target="terminal_pane")
    daily = _rpc("shared_metrics.desktop_daily", day="2026-09-26",
                 modes=[{"mode": "sessions", "active_ms": 60_000, "messages_sent": 3}])

    assert daily["result"] == {"recorded": False}  # never "settled" into another profile's off switch
    assert marks.rows == []
    assert desktop._daily == {}
    assert not (marks.home / "telemetry").exists()

    marks.policy["on"] = True  # the earlier calls claimed nothing, so opting in later still counts
    _rpc("shared_metrics.desktop_feature_use", area="projects")
    assert [dims["area"] for _, dims, _ in _stored(contract.DESKTOP_FEATURE_USE_METRIC)] == ["projects"]


def test_daily_report_records_used_modes_and_bucketed_actions_once_per_day(marks):
    params = {
        "day": "2026-09-26", "bot_count": 4,
        "modes": [
            {"mode": "sessions", "active_ms": 45 * 60_000, "messages_sent": 12},
            {"mode": "bots", "active_ms": 0, "messages_sent": 0},  # not used that day: no row
        ],
        "actions": [
            {"action": "view.toggleSidebar", "via": "shortcut", "count": 30},
            {"action": "view.toggleSidebar", "via": "click", "count": 2},
            {"action": "plugin.someContributedThing", "via": "palette", "count": 1},
        ],
    }
    assert _rpc("shared_metrics.desktop_daily", **params)["result"] == {"recorded": True}
    assert _rpc("shared_metrics.desktop_daily", **params)["result"] == {"recorded": True}  # replay: no rows

    assert _stored(contract.DESKTOP_MODE_USE_METRIC) == [
        ("2026-09-26", {"active_minutes_bucket": "30m_to_2h", "bot_count_bucket": "3_to_5",
                        "messages_sent_bucket": "11_to_25", "mode": "sessions"}, 1)]
    assert _stored(contract.DESKTOP_ACTION_USE_METRIC) == [
        ("2026-09-26", {"action": "other", "count_bucket": "1", "via": "palette"}, 1),
        ("2026-09-26", {"action": "view.toggleSidebar", "count_bucket": "2", "via": "click"}, 1),
        ("2026-09-26", {"action": "view.toggleSidebar", "count_bucket": "26_to_100", "via": "shortcut"}, 1),
    ]


def test_a_day_and_an_area_count_once_across_windows_and_backend_restarts_in_the_usage_days_period(marks, monkeypatch):
    day = {"day": "2026-09-20", "modes": [{"mode": "sessions", "active_ms": 60_000, "messages_sent": 3}],
           "actions": [{"action": "composer.send", "count": 4, "via": "shortcut"}]}
    for _ in range(2):  # two windows / a restarted backend: nothing of the first process survives
        monkeypatch.setattr(desktop, "_daily", {})
        monkeypatch.setattr(desktop, "_reported_days", set(), raising=False)
        assert _rpc("shared_metrics.desktop_daily", **day)["result"] == {"recorded": True}
        _rpc("shared_metrics.desktop_feature_use", area="terminal_pane")

    assert [(p, v) for p, _, v in _stored(contract.DESKTOP_ACTION_USE_METRIC)] == [("2026-09-20", 1)]
    assert [(p, v) for p, _, v in _stored(contract.DESKTOP_MODE_USE_METRIC)] == [("2026-09-20", 1)]
    assert [v for _, _, v in _stored(contract.DESKTOP_FEATURE_USE_METRIC)] == [1]


def test_daily_report_that_the_store_refuses_stays_with_the_client(marks, monkeypatch):
    real = SharedMetricsStore.update_rollup_state
    monkeypatch.setattr(SharedMetricsStore, "update_rollup_state",
                        lambda self, *a: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")))
    params = {"day": "2026-09-26", "modes": [{"mode": "bots", "active_ms": 1, "messages_sent": 0}]}
    assert _rpc("shared_metrics.desktop_daily", **params)["result"] == {"recorded": False}
    monkeypatch.setattr(SharedMetricsStore, "update_rollup_state", real)
    assert _rpc("shared_metrics.desktop_daily", **params)["result"] == {"recorded": True}
    assert [d["active_minutes_bucket"] for _, d, _ in _stored(contract.DESKTOP_MODE_USE_METRIC)] == ["lt_5m"]


def test_dislike_targets_are_closed_per_signal(marks):
    _rpc("shared_metrics.desktop_dislike", signal="quick_close", target="browser_pane")
    _rpc("shared_metrics.desktop_dislike", signal="cancelled", target="model_picker")
    _rpc("shared_metrics.desktop_dislike", signal="rage_click", target="composer.send")
    _rpc("shared_metrics.desktop_dislike", signal="undo", target="restored_draft")
    _rpc("shared_metrics.desktop_dislike", signal="feature_disabled", target="tips")
    _rpc("shared_metrics.desktop_dislike", signal="quick_close", target="keybind_capture")  # a flow, not an area
    _rpc("shared_metrics.desktop_dislike", signal="loathing", target="tips")

    assert [(d["signal"], d["target"]) for _, d in marks.rows] == [
        ("quick_close", "browser_pane"), ("cancelled", "model_picker"), ("rage_click", "composer.send"),
        ("undo", "restored_draft"), ("feature_disabled", "tips"), ("quick_close", "other")]
    assert {(d["setting"], d["direction"]) for _, d in marks.rows} == {("none", "none")}
    _assert_valid(marks.rows)


def test_setting_change_sends_only_the_schema_key_and_the_backend_decides_the_direction(marks):
    config = marks.home / "config.yaml"
    config.write_text("display:\n  compact: true\n  skin: default\n")
    _rpc("shared_metrics.desktop_dislike", signal="setting_off_default", setting="display.compact")
    _rpc("shared_metrics.desktop_dislike", signal="setting_off_default", setting="display.skin")
    _rpc("shared_metrics.desktop_dislike", signal="setting_off_default", setting="providers.my-secret-box.api_key")

    assert [(d["target"], d["setting"], d["direction"]) for _, d in marks.rows] == [
        ("setting", "display.compact", "away_from_default"), ("setting", "display.skin", "to_default"),
        ("setting", "other", "none")]
    _assert_valid(marks.rows)


def test_v3_schema_accepts_exactly_the_desktop_contract_values():
    import hermes_cli.observability as observability

    schema = json.loads((Path(observability.__file__).parent / "schemas/hermes.shared_metrics.v4.schema.json").read_text())
    by_name = {d["properties"]["name"]["const"]: d for d in schema["$defs"].values() if "properties" in d}
    for metric in (contract.DESKTOP_FEATURE_USE_METRIC, contract.DESKTOP_FRICTION_METRIC,
                   contract.DESKTOP_ONBOARDING_METRIC, contract.DESKTOP_MODE_USE_METRIC,
                   contract.DESKTOP_ACTION_USE_METRIC, contract.DESKTOP_DISLIKE_METRIC):
        dims = by_name[metric]["properties"]["dimensions"]["properties"]
        assert {f: set(spec["enum"]) for f, spec in dims.items() if "enum" in spec} == {
            f: set(values) for f, values in contract._COUNTER_DIMENSION_VALUES[metric].items()}
