"""The dispatcher must claim ready work before awaited decomposition."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_ready_dispatch_precedes_auto_decompose(monkeypatch, tmp_path):
    from gateway.run import GatewayRunner
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_decompose as decomp
    import hermes_cli.config as config_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    calls = []

    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": True,
                "auto_decompose_per_tick": 1,
            }
        },
    )
    monkeypatch.setattr(kb, "list_boards", lambda include_archived=False: [{"slug": "default"}])
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kb, "dispatch_once", lambda *args, **kwargs: calls.append("dispatch"))
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(kb, "review_dispatch_enabled", lambda: False)
    monkeypatch.setattr(decomp, "list_triage_ids", lambda: ["t_atomic"])

    def _decompose(*args, **kwargs):
        calls.append("decompose")
        runner._running = False
        return decomp.DecomposeOutcome("t_atomic", False, "test stop")

    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    async def _sleep(_delay):
        return None

    monkeypatch.setattr(decomp, "decompose_task", _decompose)
    monkeypatch.setattr("gateway.run.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", _sleep)

    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=3.0))

    assert calls[:2] == ["dispatch", "decompose"]


def test_external_drain_stops_new_kanban_dispatch():
    """A Desktop/gateway drain must let workers finish without spawning more."""
    from gateway.kanban_watchers import _kanban_dispatch_allowed

    runner = SimpleNamespace(_draining=False, _external_drain_active=True)

    assert _kanban_dispatch_allowed(runner) is False

@pytest.fixture
def decomposition_tick(monkeypatch, tmp_path):
    """Real board and decomposer; only process dispatch and the LLM are fake."""
    from gateway.run import GatewayRunner
    from hermes_cli import kanban_db as kb
    import hermes_cli.config as config_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    runner = object.__new__(GatewayRunner)
    runner._running = True
    monkeypatch.setattr(config_mod, "load_config", lambda: {
        "kanban": {"dispatch_interval_seconds": 1,
                   "auto_decompose": True, "auto_decompose_per_tick": 1},
    })
    monkeypatch.setattr(kb, "list_boards", lambda **kw: [{"slug": "default"}])
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kb, "dispatch_once", lambda *a, **kw: None)
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(kb, "review_dispatch_enabled", lambda: False)

    requests = []

    def fake_llm(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"fanout": True, "tasks": [
                {"title": "bounded analysis", "body": "Inspect the supplied input."},
            ]}),
        ))])

    async def to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    async def stop_after_tick(delay):
        if delay != 5:  # The watcher has a separate startup grace period.
            runner._running = False

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_llm)
    monkeypatch.setattr("gateway.run.asyncio.to_thread", to_thread)
    monkeypatch.setattr("gateway.run.asyncio.sleep", stop_after_tick)

    def run_tick():
        asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=3))

    return run_tick, requests


def test_auto_decompose_allows_undecomposed_task_with_downstream_dependents(decomposition_tick):
    """Dependency children are consumers, not evidence of prior fan-out."""
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="structural extraction", triage=True)
        consumer = kb.create_task(conn, title="verify extraction", parents=[tid])
        assert kb.child_ids(conn, tid) == [consumer]
        assert not any(ev.kind == "decomposed" for ev in kb.list_events(conn, tid))

    run_tick, _ = decomposition_tick
    run_tick()

    with kb.connect() as conn:
        events = [ev for ev in kb.list_events(conn, tid) if ev.kind == "decomposed"]
        assert len(events) == 1
        assert kb.get_task(conn, tid).status == "todo"
        assert kb.get_task(conn, consumer).status == "todo"
        assert len(kb.parent_ids(conn, tid)) == 1
        assert kb.child_ids(conn, tid) == [consumer]


@pytest.mark.parametrize("has_downstream", [False, True])
def test_auto_decompose_skips_actual_decomposition_without_spending_budget(
    decomposition_tick, monkeypatch, has_downstream,
):
    """A real re-triaged root cannot duplicate its plan or starve fresh ideas."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_decompose as decomp

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="old root", triage=True)
        if has_downstream:
            kb.create_task(conn, title="downstream consumer", parents=[tid])
        plan = kb.decompose_triage_task(
            conn, tid, root_assignee="planner", children=[{"title": "existing step"}],
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
        fresh = kb.create_task(conn, title="new idea", triage=True)
        before_count = len(kb.list_tasks(conn))
        assert kb.parent_ids(conn, tid) == plan
        if not has_downstream:
            assert kb.child_ids(conn, tid) == []
    # Fix order to put the protected root ahead of fresh work at a budget of 1.
    monkeypatch.setattr(decomp, "list_triage_ids", lambda: [tid, fresh])
    run_tick, requests = decomposition_tick
    run_tick()

    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"
        assert kb.parent_ids(conn, tid) == plan
        assert len([ev for ev in kb.list_events(conn, tid) if ev.kind == "decomposed"]) == 1
        assert kb.get_task(conn, fresh).status == "todo"
        assert len(kb.list_tasks(conn)) == before_count + 1
    assert len(requests) == 1
    assert f"Task id: {fresh}" in requests[0]["messages"][1]["content"]


def test_auto_decompose_repromotes_existing_spec_without_auxiliary_model(
    decomposition_tick, monkeypatch,
):
    """A re-triaged concrete spec must not re-enter model-dependent triage."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_decompose as decomp

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="board-record receipt",
            body="Read the named Kanban records and emit a bounded no-op receipt.",
            triage=True,
        )
        assert kb.specify_triage_task(
            conn,
            tid,
            title="board-record receipt",
            body="Read the named Kanban records and emit a bounded no-op receipt.",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))

    monkeypatch.setattr(decomp, "list_triage_ids", lambda: [tid])
    run_tick, requests = decomposition_tick
    run_tick()

    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "ready"
        assert len([ev for ev in kb.list_events(conn, tid) if ev.kind == "specified"]) == 2
        assert not any(ev.kind == "decomposed" for ev in kb.list_events(conn, tid))
    assert requests == []


def test_auto_decompose_fails_closed_when_history_cannot_be_read(decomposition_tick, monkeypatch):
    """An unreadable guard must never authorize a new auxiliary request."""
    import sqlite3
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="history unavailable", triage=True)
    real_connect = kb.connect

    def connect_without_event_reads(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_authorizer(lambda action, table, *_: (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and table == "task_events"
            else sqlite3.SQLITE_OK
        ))
        return conn

    run_tick, requests = decomposition_tick
    with monkeypatch.context() as scoped:
        scoped.setattr(kb, "connect", connect_without_event_reads)
        run_tick()
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "triage"
        assert len(kb.list_tasks(conn)) == 1
    assert requests == []
