"""Tests for dead-PID kanban fleet failure alerts."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


@pytest.fixture
def captured_failure_alerts():
    mgr = get_plugin_manager()
    events: list[dict] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks["kanban_failure_alert"] = [lambda **kw: events.append(kw)]
    try:
        yield events
    finally:
        mgr._hooks = saved


def _crash_task(conn, *, consecutive_failures: int = 2, max_retries: int | None = None) -> str:
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="dead pid", assignee="worker", max_retries=max_retries)
    kb.claim_task(conn, tid, claimer=f"{host}:deadpid-test")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kb._set_worker_pid(conn, tid, dead.pid)
    conn.execute(
        "UPDATE tasks SET started_at = started_at - 9999, consecutive_failures = ? WHERE id = ?",
        (consecutive_failures, tid),
    )
    conn.execute(
        "UPDATE task_runs SET started_at = started_at - 9999 WHERE task_id = ?",
        (tid,),
    )
    conn.commit()
    kb._record_worker_exit(dead.pid, 1 << 8)  # nonzero exit -> "pid N exited with code 1"
    return tid


def test_kanban_failure_alert_is_valid_hook():
    assert "kanban_failure_alert" in VALID_HOOKS


def test_dead_pid_cf3_fires_exactly_one_failure_alert(conn, captured_failure_alerts):
    tid = _crash_task(conn, consecutive_failures=2, max_retries=10)

    crashed = kb.detect_crashed_workers(conn)

    assert tid in crashed
    assert len(captured_failure_alerts) == 1
    event = captured_failure_alerts[0]
    assert event["task_id"] == tid
    assert event["assignee"] == "worker"
    assert event["board"] == "default"
    assert event["consecutive_failures"] == 3
    assert event["fingerprint"] == "pid n exited with code 1"
    # Additive only: max_retries keeps the breaker from blocking at cf=3.
    row = conn.execute("SELECT status, consecutive_failures FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 3


def test_no_failure_alert_below_cf3(conn, captured_failure_alerts):
    tid = _crash_task(conn, consecutive_failures=1, max_retries=10)

    crashed = kb.detect_crashed_workers(conn)

    assert tid in crashed
    assert captured_failure_alerts == []


def test_failure_alert_plugin_error_is_not_swallowed_and_dead_letters(conn):
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def boom(**_kw):
        raise RuntimeError("relay down")

    mgr._hooks["kanban_failure_alert"] = [boom]
    try:
        tid = _crash_task(conn, consecutive_failures=2, max_retries=10)
        with pytest.raises(RuntimeError, match="relay down"):
            kb.detect_crashed_workers(conn)
    finally:
        mgr._hooks = saved

    event = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind = 'kanban_failure_alert_failed'",
        (tid,),
    ).fetchone()
    assert event is not None
    assert event["kind"] == "kanban_failure_alert_failed"
    assert "relay down" in event["payload"]


def _load_deadpid_plugin():
    plugin_path = (
        Path(__file__).resolve().parents[2]
        / "plugins"
        / "deadpid-fleet-alert"
        / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location("deadpid_fleet_alert_test", plugin_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_deadpid_plugin_dedups_by_fingerprint(monkeypatch):
    plugin = _load_deadpid_plugin()
    plugin._seen_by_fingerprint.clear()
    sent: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        sent.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(plugin.subprocess, "run", fake_run)
    monkeypatch.setattr(plugin, "_settings", lambda: ("discord:#fleet-reports", 30))
    now = {"t": 1000.0}
    monkeypatch.setattr(plugin.time, "time", lambda: now["t"])
    payload = {
        "task_id": "t_deadbeef",
        "board": "jarvis-os",
        "assignee": "devops",
        "run_id": 42,
        "consecutive_failures": 3,
        "fingerprint": "pid n not alive",
        "error": "pid 123 not alive",
    }

    first = plugin._on_kanban_failure_alert(**payload)
    second = plugin._on_kanban_failure_alert(**payload)
    now["t"] += 31
    third = plugin._on_kanban_failure_alert(**payload)

    assert first == {"sent": True, "target": "discord:#fleet-reports", "fingerprint": "pid n not alive"}
    assert second == {"deduped": True, "fingerprint": "pid n not alive"}
    assert third == {"sent": True, "target": "discord:#fleet-reports", "fingerprint": "pid n not alive"}
    assert len(sent) == 2
    assert sent[0][:5] == ["hermes", "send", "--to", "discord:#fleet-reports", "--quiet"]
