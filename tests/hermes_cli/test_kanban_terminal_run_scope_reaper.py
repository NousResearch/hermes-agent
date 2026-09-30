"""Invariant: a terminal run's WHOLE worker systemd scope is reaped by the
dispatcher itself, not just the worker's own pid (t_62f24f45, upstreaming the
``kanban-scope-reaper`` stopgap timer from t_0b8e9d3f).

A worker that starts a dev server / vite preview / browser daemon in the
background and loses track of it leaves that process -- and any double-forked
descendant reparented to init -- alive in the worker's transient systemd
scope forever. ``reap_terminal_workers`` only reaches the worker's own pid;
``reap_terminal_run_scopes`` stops the whole cgroup once the run has been
terminal long enough, unless something inside it is listening on a port
currently published via ``tailscale serve``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _closed_run(conn, *, ended_ago: int) -> tuple[str, int]:
    tid = kb.create_task(conn, title="finished", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?", (ended_ago, run_id))
    return tid, run_id


def _unit(tid: str, run_id: int) -> str:
    return f"hermes-worker-kanban-{tid}-run-{run_id}.scope"


def test_stopped_past_grace_with_no_listener(conn):
    tid, run_id = _closed_run(conn, ended_ago=kbd.TERMINAL_RUN_SCOPE_REAP_GRACE_SECONDS + 60)
    unit = _unit(tid, run_id)
    stopped = []

    result = kbd.reap_terminal_run_scopes(
        conn,
        list_units_fn=lambda: [unit],
        published_ports_fn=lambda: set(),
        scope_ports_fn=lambda u: set(),
        stop_fn=lambda u: (stopped.append(u) or True),
    )

    assert result == [tid]
    assert stopped == [unit]
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))]
    assert "terminal_scope_reaped" in kinds


def test_kept_within_grace_window(conn):
    tid, run_id = _closed_run(conn, ended_ago=60)
    unit = _unit(tid, run_id)
    stopped = []

    result = kbd.reap_terminal_run_scopes(
        conn,
        list_units_fn=lambda: [unit],
        published_ports_fn=lambda: set(),
        scope_ports_fn=lambda u: set(),
        stop_fn=lambda u: (stopped.append(u) or True),
    )

    assert result == []
    assert stopped == []


def test_kept_when_listening_port_is_published(conn):
    tid, run_id = _closed_run(conn, ended_ago=kbd.TERMINAL_RUN_SCOPE_REAP_GRACE_SECONDS + 60)
    unit = _unit(tid, run_id)
    stopped = []

    result = kbd.reap_terminal_run_scopes(
        conn,
        list_units_fn=lambda: [unit],
        published_ports_fn=lambda: {4184},
        scope_ports_fn=lambda u: {4184},
        stop_fn=lambda u: (stopped.append(u) or True),
    )

    assert result == []
    assert stopped == []


def test_reaped_when_listening_port_is_not_published(conn):
    """A stray listener that nobody published via tailscale is not a reason to keep it."""
    tid, run_id = _closed_run(conn, ended_ago=kbd.TERMINAL_RUN_SCOPE_REAP_GRACE_SECONDS + 60)
    unit = _unit(tid, run_id)
    stopped = []

    result = kbd.reap_terminal_run_scopes(
        conn,
        list_units_fn=lambda: [unit],
        published_ports_fn=lambda: {4184},
        scope_ports_fn=lambda u: {5173},  # some other, unpublished dev server
        stop_fn=lambda u: (stopped.append(u) or True),
    )

    assert result == [tid]
    assert stopped == [unit]


def test_kept_when_published_ports_unknown(conn):
    """tailscale status could not be read: fail closed, never guess "not published"."""
    tid, run_id = _closed_run(conn, ended_ago=kbd.TERMINAL_RUN_SCOPE_REAP_GRACE_SECONDS + 60)
    unit = _unit(tid, run_id)
    stopped = []

    result = kbd.reap_terminal_run_scopes(
        conn,
        list_units_fn=lambda: [unit],
        published_ports_fn=lambda: None,
        scope_ports_fn=lambda u: {5173},
        stop_fn=lambda u: (stopped.append(u) or True),
    )

    assert result == []
    assert stopped == []


def test_never_reaps_a_running_task(conn):
    tid = kb.create_task(conn, title="still going", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    unit = _unit(tid, run_id)
    stopped = []

    result = kbd.reap_terminal_run_scopes(
        conn,
        list_units_fn=lambda: [unit],
        published_ports_fn=lambda: set(),
        scope_ports_fn=lambda u: set(),
        stop_fn=lambda u: (stopped.append(u) or True),
    )

    assert result == []
    assert stopped == []


def test_never_reaps_an_unattributable_unit(conn):
    """A scope-shaped unit whose (task, run) this board's DB has no row for is left alone --
    it may belong to a sibling board's task, and must never be guessed at."""
    unit = "hermes-worker-kanban-t_deadbeef-run-999999.scope"
    stopped = []

    result = kbd.reap_terminal_run_scopes(
        conn,
        list_units_fn=lambda: [unit],
        published_ports_fn=lambda: set(),
        scope_ports_fn=lambda u: set(),
        stop_fn=lambda u: (stopped.append(u) or True),
    )

    assert result == []
    assert stopped == []


def test_stop_failure_is_not_counted_as_reaped(conn):
    tid, run_id = _closed_run(conn, ended_ago=kbd.TERMINAL_RUN_SCOPE_REAP_GRACE_SECONDS + 60)
    unit = _unit(tid, run_id)

    result = kbd.reap_terminal_run_scopes(
        conn,
        list_units_fn=lambda: [unit],
        published_ports_fn=lambda: set(),
        scope_ports_fn=lambda u: set(),
        stop_fn=lambda u: False,
    )

    assert result == []
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))]
    assert "terminal_scope_reaped" not in kinds


def test_unmatched_unit_names_are_ignored(conn):
    result = kbd.reap_terminal_run_scopes(
        conn,
        list_units_fn=lambda: ["hermes-worker-cron-job-42.scope", "not-a-scope-at-all"],
        published_ports_fn=lambda: set(),
        scope_ports_fn=lambda u: set(),
        stop_fn=lambda u: True,
    )
    assert result == []


def test_dispatch_once_reaps_terminal_scopes(conn, monkeypatch):
    """End-to-end: a normal dispatcher tick reaps a stale scope with no timer involved."""
    tid, run_id = _closed_run(conn, ended_ago=kbd.TERMINAL_RUN_SCOPE_REAP_GRACE_SECONDS + 60)
    unit = _unit(tid, run_id)
    stopped = []
    monkeypatch.setattr(kbd, "_loaded_kanban_worker_scopes", lambda: [unit])
    monkeypatch.setattr(kbd, "_tailscale_published_ports", lambda: set())
    monkeypatch.setattr(kbd, "_scope_listening_ports", lambda u: set())
    monkeypatch.setattr("tools.process_registry._stop_systemd_unit",
                         lambda u: (stopped.append(u) or True))

    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)

    assert result.reaped_terminal_scopes == [tid]
    assert stopped == [unit]
