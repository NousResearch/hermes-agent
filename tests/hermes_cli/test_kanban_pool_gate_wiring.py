"""The pool gate must fire in the REAL dispatch path, not just as a helper.

A unit-green gate that no caller invokes is inert. These drive
``dispatch_once`` end to end and assert on spawn side effects.
"""
from __future__ import annotations

import pytest

import hermes_cli.kanban_db as kb
import hermes_cli.kanban_db_connect as kbc
import hermes_cli.kanban_db_dispatch as kbd


@pytest.fixture(autouse=True)
def _reset_gate_state():
    kbd._POOL_GATE_STATE.clear()
    yield
    kbd._POOL_GATE_STATE.clear()


def _spawn_recorder():
    spawned = []

    def spawn_fn(task, workspace):
        spawned.append(task.id)
        return 4242

    return spawn_fn, spawned


def _settings(monkeypatch, *, url="http://pool.invalid/health", min_eligible=1, trip=0):
    monkeypatch.setattr(kbd, "_kanban_pool_settings", lambda: (url, min_eligible, trip))


def test_pool_bound_task_is_not_spawned_when_pool_empty(monkeypatch):
    """daedalus-opus (claude-apr) must NOT spawn while the pool is exhausted."""
    _settings(monkeypatch)
    monkeypatch.setattr(kbd, "resolve_task_provider", lambda row, who: "claude-apr")
    monkeypatch.setattr(kbd, "pool_admits_spawn", lambda *a, **k: False)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))

    spawn_fn, spawned = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="pool-bound", assignee="daedalus-opus")
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn, max_spawn=5)

    assert spawned == [], "worker was spawned into an empty pool"
    assert [t for t, _who, _r in res.pool_gated] == [tid]
    assert res.pool_gated[0][2] == "pool_unavailable"
    assert [t for t, _w, _ws in res.spawned] == []


def test_gated_task_is_not_claimed(monkeypatch):
    """A held card keeps its ready status and gains no run row."""
    _settings(monkeypatch)
    monkeypatch.setattr(kbd, "resolve_task_provider", lambda row, who: "claude-apr")
    monkeypatch.setattr(kbd, "pool_admits_spawn", lambda *a, **k: False)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))

    spawn_fn, _ = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="untouched", assignee="daedalus-opus")
        kbd.dispatch_once(conn, spawn_fn=spawn_fn, max_spawn=5)

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.current_run_id is None
        runs = conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?", (tid,),
        ).fetchone()["n"]
        assert runs == 0, "gated card must not consume a run/retry"


def test_non_pool_task_still_spawns_while_pool_empty(monkeypatch):
    """openai-codex is unaffected by a dead claude pool."""
    _settings(monkeypatch)
    monkeypatch.setattr(kbd, "resolve_task_provider", lambda row, who: "openai-codex")

    def _boom(*a, **k):
        raise AssertionError("non-pool provider must never probe pool health")

    monkeypatch.setattr(kbd, "pool_admits_spawn", _boom)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))

    spawn_fn, spawned = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="codex", assignee="daedalus")
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn, max_spawn=5)

    assert spawned == [tid]
    assert res.pool_gated == []


def test_healthy_pool_admits_spawn(monkeypatch):
    _settings(monkeypatch)
    monkeypatch.setattr(kbd, "resolve_task_provider", lambda row, who: "claude-apr")
    monkeypatch.setattr(kbd, "pool_admits_spawn", lambda *a, **k: True)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))

    spawn_fn, spawned = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="healthy", assignee="daedalus-opus")
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn, max_spawn=5)

    assert spawned == [tid]
    assert res.pool_gated == []


def test_empty_health_url_disables_the_gate(monkeypatch):
    """Operator kill switch: kanban.pool_health_url="" admits everything."""
    _settings(monkeypatch, url="")
    monkeypatch.setattr(kbd, "resolve_task_provider", lambda row, who: "claude-apr")

    def _boom(*a, **k):
        raise AssertionError("gate disabled — must not probe")

    monkeypatch.setattr(kbd, "pool_admits_spawn", _boom)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))

    spawn_fn, spawned = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="disabled", assignee="daedalus-opus")
        kbd.dispatch_once(conn, spawn_fn=spawn_fn, max_spawn=5)

    assert spawned == [tid]


def test_circuit_holds_pool_spawns_board_wide(monkeypatch):
    """Gate 3: enough recent rate_limited closes hold pool spawns."""
    _settings(monkeypatch, trip=5)
    monkeypatch.setattr(kbd, "resolve_task_provider", lambda row, who: "claude-apr")
    monkeypatch.setattr(kbd, "pool_admits_spawn", lambda *a, **k: True)
    monkeypatch.setattr(kbd, "rate_limit_circuit_open", lambda *a, **k: True)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))

    spawn_fn, spawned = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="circuit", assignee="daedalus-opus")
        res = kbd.dispatch_once(conn, spawn_fn=spawn_fn, max_spawn=5)

    assert spawned == []
    assert res.pool_gated == [(tid, "daedalus-opus", "rate_limit_circuit")]


def test_gate_exception_fails_open(monkeypatch):
    """A broken gate must never stop the board."""
    _settings(monkeypatch)
    monkeypatch.setattr(kbd, "resolve_task_provider", lambda row, who: "claude-apr")

    def _explode(*a, **k):
        raise RuntimeError("health client blew up")

    monkeypatch.setattr(kbd, "pool_admits_spawn", _explode)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))

    spawn_fn, spawned = _spawn_recorder()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="explode", assignee="daedalus-opus")
        kbd.dispatch_once(conn, spawn_fn=spawn_fn, max_spawn=5)

    assert spawned == [tid], "gate error must fail OPEN, not strand the card"
