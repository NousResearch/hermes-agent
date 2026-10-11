"""Default-off coordinator contracts using real disposable board and processes."""
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_dispatch as dispatch
from hermes_cli import kanban_lane_coordinator as coordinator
from hermes_cli.kanban_provider_lanes import LaneLedger


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda name: True)
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "ok")
    monkeypatch.setattr(dispatch, "_system_memory_sample", lambda: {"mem_available_kib": 16 * 1024**2})
    kb.init_db()
    conn = kbc.connect()
    router = tmp_path / "models.toml"
    router.write_text('[stages.build]\ncandidates = [{provider = "openai", model = "fixture"}]\n')
    settings = {"enabled": True, "router_path": str(router)}
    yield conn, settings
    conn.close()


def task(conn, *, provider="openai-codex", status="ready"):
    tid = kb.create_task(conn, title="synthetic coordinator", assignee="builder",
                         model_override="fixture", provider_override=provider)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, tid))
    return tid


def tick(conn, settings, spawn):
    return dispatch.dispatch_once(conn, spawn_fn=spawn, provider_lanes=settings,
                                  dispatch_scheduling={}, reconcile_orphans=False, max_spawn=10)


def snapshot():
    ledger = LaneLedger(coordinator.ledger_path())
    try:
        return ledger.snapshot()
    finally:
        ledger.close()


def test_disabled_keeps_legacy_spawn_and_creates_no_ledger(board):
    conn, _ = board
    tid = task(conn, provider="anthropic")
    calls = []
    result = tick(conn, {}, lambda task, workspace: calls.append(task.id))
    assert calls == [tid]
    assert len(result.spawned) == 1
    assert not coordinator.ledger_path().exists()


@pytest.mark.platforms("linux", "macos")
def test_enabled_reserves_binds_and_caps_without_changing_pin(board):
    conn, settings = board
    tids = [task(conn) for _ in range(3)]
    calls = []
    def spawn(claimed, workspace, *, board=None):
        calls.append((claimed.provider_override, claimed.model_override))
        return os.getpid()
    result = tick(conn, settings, spawn)
    assert len(result.spawned) == 2
    assert calls == [("openai-codex", "fixture")] * 2
    rows = snapshot()
    assert len(rows) == 2
    assert all(row["worker"] and row["observed_provider"] is None for row in rows)
    waiting = [kb.get_task(conn, tid) for tid in tids if kb.get_task(conn, tid).status == "ready"]
    assert len(waiting) == 1 and waiting[0].consecutive_failures == 0


@pytest.mark.parametrize("status", ["ready", "review"])
def test_unknown_memory_defers_and_preserves_source_phase(board, monkeypatch, status):
    conn, settings = board
    tid = task(conn, status=status)
    monkeypatch.setattr(dispatch, "_system_memory_sample", lambda: {})
    calls = []
    assert not tick(conn, settings, lambda *args: calls.append(args)).spawned
    current = kb.get_task(conn, tid)
    assert current.status == status and current.claim_lock is None
    assert current.consecutive_failures == 0 and not calls and snapshot() == []


@pytest.mark.parametrize("provider", ["anthropic", "openai", "unknown"])
def test_incompatible_transport_never_substitutes_api_billing(board, provider):
    conn, settings = board
    tid = task(conn, provider=provider)
    calls = []
    tick(conn, settings, lambda *args: calls.append(args))
    assert not calls
    assert kb.get_task(conn, tid).consecutive_failures == 0
    assert not coordinator.ledger_path().exists()


def test_uncertain_spawn_does_not_retry_or_release_pending_capacity(board):
    conn, settings = board
    tid = task(conn)
    calls = []
    def spawn(*args):
        calls.append(args)
        raise TypeError("adapter may already have spawned")
    tick(conn, settings, spawn)
    assert len(calls) == 1
    assert kb.get_task(conn, tid).status == "running"
    rows = snapshot()
    assert len(rows) == 1 and rows[0]["worker"] is None
    assert rows[0]["observed_model"] is None
    assert conn.execute("SELECT COUNT(*) FROM task_events WHERE kind = 'lane_spawn_uncertain'").fetchone()[0] == 1


def test_existing_untracked_run_prevents_activation(board):
    conn, settings = board
    old = task(conn)
    kb.claim_task(conn, old)
    task(conn)
    calls = []
    tick(conn, settings, lambda *args: calls.append(args))
    assert not calls and snapshot() == []


def test_dry_run_does_not_create_host_ledger(board):
    conn, settings = board
    task(conn)
    dispatch.dispatch_once(conn, dry_run=True, provider_lanes=settings,
                           dispatch_scheduling={}, reconcile_orphans=False)
    assert not coordinator.ledger_path().exists()


def test_gateway_settings_are_explicit_and_do_not_leak_to_next_tick(board):
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings
    conn, settings = board
    resolved = _resolve_dispatcher_settings({"provider_lanes": settings}, kb)
    assert asdict(resolved)["provider_lanes"] == settings
    coordinator.with_settings(settings, lambda: None)
    assert coordinator.current_settings.get() is None


@pytest.mark.platforms("linux", "macos")
def test_terminal_card_retains_slot_until_real_child_exits(board, monkeypatch):
    # Simulate a delayed/failed terminal reaper: the ledger must independently
    # retain capacity until physical exit, not merely trust the ended card.
    monkeypatch.setattr(dispatch, "reap_terminal_workers", lambda conn: [])
    conn, settings = board
    children = []
    def spawn(*args):
        child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
        children.append(child)
        return child.pid
    try:
        tids = [task(conn) for _ in range(2)]
        assert len(tick(conn, settings, spawn).spawned) == 2
        completed = kb.get_task(conn, tids[0])
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?", (completed.id,))
            conn.execute("UPDATE task_runs SET ended_at = 1, outcome = 'done' WHERE id = ?", (completed.current_run_id,))
        waiting = task(conn)
        assert not tick(conn, settings, spawn).spawned
        assert len(children) == 2 and len(snapshot()) == 2
        children[0].terminate()
        children[0].wait(timeout=10)
        # Advance the disposable board's deferral timestamp beyond its normal
        # infrastructure cooldown; do not weaken the production retry guard.
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET ended_at = 1 WHERE task_id = ?", (waiting,))
        assert len(tick(conn, settings, spawn).spawned) == 1
        assert kb.get_task(conn, waiting).worker_pid == children[2].pid
        assert len(snapshot()) == 2
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)
            child.stdin.close()


@pytest.mark.parametrize("value", [None, "true", 1, []])
def test_invalid_enabled_setting_fails_closed(value):
    with pytest.raises(coordinator.LaneDeferred):
        coordinator.resolve_settings({"enabled": value})
