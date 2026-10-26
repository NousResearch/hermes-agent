"""Graceful ``os._exit`` paths run the shared-metrics exit flush their skipped atexit hook would have run.

The gateway (SIGTERM, update restart, idle exit all funnel through ``_exit_after_graceful_shutdown``),
the compute host, a SIGTERMed kanban worker and an orphaned slash worker leave via ``os._exit``, so a
session still open at the stop never wrote its summary rows to the opt-in local store. The flush is
bounded and creates nothing when collection is off.
"""

from __future__ import annotations

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


@pytest.fixture
def metrics_home(tmp_path, monkeypatch):
    """Real NeMo Relay runtime over a temp HERMES_HOME; ``policy['on']`` is the opt-in answer."""
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")
    home = tmp_path / "hermes-home"
    policy = {"on": True}
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly",
        lambda: {"telemetry": {"shared_metrics": {"enabled": policy["on"], "send": False}}},
    )
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()
    manager = PluginManager()
    manager._discovered = True  # a runtime with zero plugins (#64178)
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    yield home, policy
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()


def _finished_task_in_open_session() -> None:
    """One completed task in a session that stays open, as a messaging or Desktop chat does."""
    event = {"session_id": "open-session", "task_id": "task-1", "platform": "cli"}
    lifecycle.invoke_hook("on_session_start", **event, turn_id="turn-1")
    relay_shared_metrics.start_task_run(**event)
    lifecycle.invoke_hook("pre_llm_call", **event, turn_id="turn-1", messages=["hello"])  # a user turn
    relay_shared_metrics.finish_task_run(**event, result={"completed": True, "final_response": "ok"})


def _wait_for(home: Path, metric: str, deadline_s: float = 10.0) -> None:
    deadline = time.monotonic() + deadline_s
    while metric not in _metric_names(home):
        assert time.monotonic() < deadline, f"{metric} never reached the store"
        time.sleep(0.05)


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


def test_a_wedged_exit_flush_never_holds_the_exit(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(relay_shared_metrics, "shutdown_runtimes", lambda: release.wait(30))
    started = time.monotonic()
    try:
        assert process_metrics.flush_before_hard_exit(timeout_s=0.2) is False
    finally:
        release.set()
    assert time.monotonic() - started < 5
