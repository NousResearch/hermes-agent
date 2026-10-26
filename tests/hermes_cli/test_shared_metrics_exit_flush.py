"""Graceful ``os._exit`` paths run the shared-metrics exit flush their skipped atexit hook would have run.

The gateway (SIGTERM, update restart, idle exit all funnel through ``_exit_after_graceful_shutdown``),
the compute host, a SIGTERMed kanban worker and an orphaned slash worker leave via ``os._exit``, so a
session still open at the stop never wrote its summary rows to the opt-in local store. The flush is
bounded, creates nothing when collection is off, and runs every profile's runtime under that profile's
own consent (a multiplex gateway exits from the launch profile's scope).
"""

from __future__ import annotations

import functools
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from agent import relay_runtime
from hermes_cli import lifecycle, plugins
from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics_process as process_metrics
from hermes_cli.plugins import PluginManager
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.waiting import wait_until


class _Exited(Exception):
    pass


def _fake_exit(code):
    raise _Exited(code)


def _store(home: Path) -> Path:
    return home / "telemetry" / "shared_metrics" / "metrics.sqlite3"


def _metric_names(home: Path) -> set[str]:
    database = _store(home)
    if not database.exists():
        return set()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as conn:
        return {name for (name,) in conn.execute("SELECT metric_name FROM counter_aggregates")}


def _open_consent_windows(home: Path) -> list[tuple]:
    with sqlite3.connect(_store(home).as_uri() + "?mode=ro", uri=True) as conn:
        return conn.execute("SELECT opened_at FROM send_consent_windows WHERE closed_at IS NULL").fetchall()


def _require_native_relay() -> None:
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")


def _zero_plugin_runtimes(monkeypatch) -> None:
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()
    manager = PluginManager()
    manager._discovered = True  # a runtime with zero plugins (#64178)
    monkeypatch.setattr(plugins, "_plugin_manager", manager)


@pytest.fixture
def metrics_home(tmp_path, monkeypatch):
    """Real NeMo Relay runtime over a temp HERMES_HOME; ``policy['on']`` is the opt-in answer."""
    _require_native_relay()
    home = tmp_path / "hermes-home"
    policy = {"on": True}
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly",
        lambda: {"telemetry": {"shared_metrics": {"enabled": policy["on"], "send": False}}},
    )
    _zero_plugin_runtimes(monkeypatch)
    yield home, policy
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()


def _finished_task_in_open_session(session_id: str = "open-session") -> None:
    """One completed task in a session that stays open, as a messaging or Desktop chat does."""
    event = {"session_id": session_id, "task_id": "task-1", "platform": "cli"}
    lifecycle.invoke_hook("on_session_start", **event, turn_id="turn-1")
    relay_shared_metrics.start_task_run(**event)
    lifecycle.invoke_hook("pre_llm_call", **event, turn_id="turn-1", messages=["hello"])  # a user turn
    relay_shared_metrics.finish_task_run(**event, result={"completed": True, "final_response": "ok"})


def _wait_for(home: Path, metric: str) -> None:
    wait_until(lambda: metric in _metric_names(home), what=f"{metric} in the store")


def test_graceful_gateway_exit_writes_the_open_sessions_summary(metrics_home, monkeypatch):
    import gateway.run as gateway_run

    home, _ = metrics_home
    _finished_task_in_open_session()
    _wait_for(home, "hermes.task_run.finished")  # the task flush runs on its own worker
    assert "hermes.session.count" not in _metric_names(home)

    monkeypatch.setattr(gateway_run.os, "_exit", _fake_exit)
    with pytest.raises(_Exited):
        gateway_run._exit_after_graceful_shutdown(1)

    assert "hermes.session.count" in _metric_names(home)


def test_compute_host_shutdown_frame_writes_the_open_sessions_summary(metrics_home):
    import io

    from tui_gateway.compute_host import ComputeHost

    home, _ = metrics_home
    _finished_task_in_open_session()
    _wait_for(home, "hermes.task_run.finished")
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0)

    host.handle_frame({"type": "shutdown"})  # the reader thread os._exit(0)s right after this

    assert "hermes.session.count" in _metric_names(home)


def test_sigtermed_kanban_worker_writes_the_open_sessions_summary(metrics_home, monkeypatch):
    import signal

    import cli
    from hermes_cli import cli_single_query

    home, _ = metrics_home
    _finished_task_in_open_session()
    _wait_for(home, "hermes.task_run.finished")
    handlers = {}
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t-1")
    monkeypatch.setattr(signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler))
    monkeypatch.setattr(signal, "alarm", lambda seconds: 0)
    for name in ("_arm_exit_watchdog_on_shutdown_signal", "_flush_logging_and_stdio"):
        monkeypatch.setattr(cli, name, lambda: None)
    monkeypatch.setattr(cli, "_interrupt_agent_for_signal", lambda agent, signum: None)
    monkeypatch.setattr(cli, "_flush_one_shot_session_store", lambda c: None)
    monkeypatch.setattr(cli_single_query.os, "_exit", _fake_exit)
    cli_single_query._install_single_query_signal_handlers(object())

    with pytest.raises(_Exited):
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    assert "hermes.session.count" in _metric_names(home)


def test_orphaned_slash_worker_writes_the_open_sessions_summary(metrics_home, monkeypatch):
    from tui_gateway import slash_worker

    home, _ = metrics_home
    _finished_task_in_open_session()
    _wait_for(home, "hermes.task_run.finished")
    exited = threading.Event()
    monkeypatch.setattr(slash_worker, "_is_orphaned", lambda ppid: True)
    monkeypatch.setattr(slash_worker.os, "_exit", lambda code: exited.set())

    slash_worker._start_parent_death_watchdog(1)

    assert exited.wait(10)
    assert "hermes.session.count" in _metric_names(home)


def test_graceful_exit_with_collection_off_creates_no_store(metrics_home, monkeypatch):
    import gateway.run as gateway_run

    home, policy = metrics_home
    policy["on"] = False
    _finished_task_in_open_session()

    monkeypatch.setattr(gateway_run.os, "_exit", _fake_exit)
    with pytest.raises(_Exited):
        gateway_run._exit_after_graceful_shutdown(0)

    assert not (home / "telemetry").exists()


def _write_policy(home: Path, **policy) -> None:
    config = home / "config.yaml"
    config.write_text(json.dumps({"telemetry": {"shared_metrics": policy}}))
    later = time.time() + 5  # a rewrite inside one mtime tick must still read as a change
    os.utime(config, (later, later))


def test_exit_flush_applies_each_profiles_own_consent(tmp_path, monkeypatch):
    """A multiplex gateway exits from the launch profile's scope. A routed profile with sending off
    keeps it off (no consent window opened on its store), and one that turned collection off after
    its turn gets no exit rows; both rulings are that profile's own, not the launch profile's."""
    import gateway.run as gateway_run

    _require_native_relay()
    launch, send_off, revoked = (tmp_path / name for name in ("launch", "send-off", "revoked"))
    for home in (launch, send_off, revoked):
        home.mkdir()
    _write_policy(launch, enabled=True, send=True, endpoint="http://127.0.0.1:9/never")
    _write_policy(send_off, enabled=True, send=False)
    _write_policy(revoked, enabled=True, send=False)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    _zero_plugin_runtimes(monkeypatch)
    try:
        for home in (send_off, revoked):  # routed turns, each under its own profile scope
            token = set_hermes_home_override(str(home))
            try:
                _finished_task_in_open_session(f"open-{home.name}")
                _wait_for(home, "hermes.task_run.finished")
            finally:
                reset_hermes_home_override(token)
        _write_policy(revoked, enabled=False, send=False)
        rows_before_exit = _metric_names(revoked)

        monkeypatch.setattr(gateway_run.os, "_exit", _fake_exit)
        with pytest.raises(_Exited):
            gateway_run._exit_after_graceful_shutdown(0)

        assert "hermes.session.count" in _metric_names(send_off)  # its own consent still flushes it
        assert _open_consent_windows(send_off) == []
        assert _metric_names(revoked) == rows_before_exit
    finally:
        relay_shared_metrics._reset_for_tests()
        relay_runtime._reset_for_tests()


def test_a_wedged_exit_flush_is_bounded_and_runs_after_the_clean_exit_stamp(metrics_home, monkeypatch):
    """A managed tool call still in flight holds the Relay flush barrier (#53107): the exit still
    leaves within the flush bound, and the clean-exit sentinel is already written, so a supervisor
    killing the process during the flush never records a graceful stop as an unclean death."""
    import gateway.run as gateway_run
    from agent import relay_tools
    from gateway.lifecycle_ledger import get_lifecycle_sentinel_path

    home, _ = metrics_home
    _finished_task_in_open_session()
    _wait_for(home, "hermes.task_run.finished")
    entered, release = threading.Event(), threading.Event()

    def wedged_tool(_args):
        entered.set()
        release.wait(30)
        return "late"

    threading.Thread(
        target=relay_tools.execute, args=("terminal", {"command": "x"}, wedged_tool),
        kwargs={"session_id": "other-session"}, daemon=True,
    ).start()
    assert entered.wait(10)
    phase_at_flush, finished_in_time = [], []
    bounded_flush = functools.partial(process_metrics.flush_before_hard_exit, timeout_s=0.5)

    def observed_flush():
        sentinel = get_lifecycle_sentinel_path()
        phase_at_flush.append(json.loads(sentinel.read_text()).get("phase") if sentinel.exists() else None)
        finished_in_time.append(bounded_flush())

    monkeypatch.setattr(process_metrics, "flush_before_hard_exit", observed_flush)
    monkeypatch.setattr(gateway_run.os, "_exit", _fake_exit)
    started = time.monotonic()
    try:
        with pytest.raises(_Exited):
            gateway_run._exit_after_graceful_shutdown(0)
    finally:
        release.set()

    assert finished_in_time == [False]  # the barrier really was wedged
    assert phase_at_flush == ["exited"]
    assert time.monotonic() - started < 5


def _engagement_state(home: Path) -> dict | None:
    with sqlite3.connect(_store(home).as_uri() + "?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT value FROM telemetry_state WHERE key = 'engagement_day_state'").fetchone()
    return json.loads(row[0]) if row else None


def test_graceful_exit_drains_the_root_profiles_deferred_host_count(tmp_path, monkeypatch):
    """The host-level active-profiles accumulator lives in the ROOT profile's store, a store no
    per-profile runtime owns. When another process held it past the busy timeout, the profile's visit
    was deferred in memory; the graceful exit still writes it there (and only there) once the store
    frees, instead of losing it to ``os._exit``."""
    import gateway.run as gateway_run
    from hermes_cli.observability import shared_metrics_engagement as engagement
    from hermes_cli.observability.shared_metrics import SharedMetricsStore

    root = (tmp_path / "root").resolve()
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    _write_policy(root, enabled=True, send=False)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    resource = {"architecture": "x86_64", "hermes_version": "0.0.0", "install_method": "git", "os_family": "linux"}
    profile_store = SharedMetricsStore(_store(profile), _store(profile).parent / "outbox")
    engagement._root_store(root)  # the store exists; another process now holds its write lock
    holder = sqlite3.connect(_store(root), isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        engagement.record(profile_store, resource, surface="cli", route=None)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert _engagement_state(root) is None  # deferred, not written
    profile_state = _engagement_state(profile)

    monkeypatch.setattr(gateway_run.os, "_exit", _fake_exit)
    with pytest.raises(_Exited):
        gateway_run._exit_after_graceful_shutdown(0)

    root_state = wait_until(lambda: _engagement_state(root), what="the root's deferred host count")
    assert root_state["owner"] is True and len(root_state["profiles"]) == 1
    assert _engagement_state(profile) == profile_state  # the profile's own store never got the root's row
