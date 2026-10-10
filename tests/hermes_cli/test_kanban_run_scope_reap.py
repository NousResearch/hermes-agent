"""Invariant: a run that has ended leaves no loaded worker scope of its own.

Every worker runs in a transient ``hermes-worker-kanban-<task>-run-<n>.scope``
cgroup. ``--collect`` only removes that scope once its cgroup is empty, so a
worker that ends leaving a background child behind — a ``bun run dev``, a
``python3 -m http.server``, an ``Xvfb``, a ``while pgrep ...; do sleep 15; done``
loop — keeps the scope loaded, holds the child's port, and holds the child's
memory. Nine such scopes were stopped by hand on 2026-10-01 (ports 5177-5299,
one a four-day-old ``bun run dev`` from a card that finished on Sep 27, another a
reclaimed run's scope still holding 34 tasks and an orphaned ``sccache``).

``reap_terminal_run_scopes`` stops the scope of every run that has ended, and
never one whose run is still open, still inside the finalisation grace window, or
not this board's at all.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import shutil as _shutil

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


def _unit(task_id: str, run_id: int) -> str:
    return f"{kbd.WORKER_SCOPE_UNIT_PREFIX}{task_id}-run-{run_id}.scope"


def _claimed_run(conn) -> tuple[str, int]:
    tid = kb.create_task(conn, title="card", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    assert run_id is not None
    return tid, run_id


def _closed_run(conn, *, ended_ago: int = 600) -> tuple[str, int]:
    tid, run_id = _claimed_run(conn)
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    # Default: the run closed long enough ago that the grace window has passed.
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id = ?", (ended_ago, run_id))
    return tid, run_id


def _list_units(monkeypatch, units: list[str]) -> None:
    monkeypatch.setattr(kbd, "_loaded_worker_scope_units", lambda: list(units))


def _recorder(stops: list[str], *, fail_on: set[str] | None = None):
    def stop(unit: str) -> bool:
        stops.append(unit)
        if fail_on and unit in fail_on:
            raise RuntimeError("boom")
        return True

    return stop


def test_closed_run_scope_is_stopped_and_recorded(conn, monkeypatch):
    tid, run_id = _closed_run(conn)
    unit = _unit(tid, run_id)
    _list_units(monkeypatch, [unit])
    stops: list[str] = []

    assert kbd.reap_terminal_run_scopes(conn, stop_fn=_recorder(stops)) == [unit]

    assert stops == [unit]
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))]
    assert "terminal_run_scope_reaped" in kinds
    payload = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='terminal_run_scope_reaped'", (tid,)
    ).fetchone()["payload"]
    assert unit in payload and str(run_id) in payload


def test_open_run_scope_is_never_touched(conn, monkeypatch):
    """A live worker's scope holds the live worker: stopping it would kill the run."""
    tid, run_id = _claimed_run(conn)
    unit = _unit(tid, run_id)
    _list_units(monkeypatch, [unit])
    stops: list[str] = []

    assert kbd.reap_terminal_run_scopes(conn, stop_fn=_recorder(stops)) == []
    assert stops == []


def test_freshly_closed_scope_waits_for_the_grace_window(conn, monkeypatch):
    """A worker is still alive for a moment after its own kanban_complete returns
    (final turn, session persistence), so a just-closed run is left alone."""
    tid, run_id = _closed_run(conn, ended_ago=0)
    unit = _unit(tid, run_id)
    _list_units(monkeypatch, [unit])
    stops: list[str] = []

    assert kbd.reap_terminal_run_scopes(conn, stop_fn=_recorder(stops)) == []
    assert stops == []

    conn.execute(
        "UPDATE task_runs SET ended_at = ended_at - ? WHERE id = ?",
        (kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS + 1, run_id),
    )
    assert kbd.reap_terminal_run_scopes(conn, stop_fn=_recorder(stops)) == [unit]


def test_scope_of_another_board_is_left_alone(conn, monkeypatch):
    """Scopes carry the task id, not the board: a name no run row in this DB
    matches belongs to a board this dispatcher does not own."""
    _tid, run_id = _closed_run(conn)
    unit = _unit("t_11111111", run_id + 9000)
    _list_units(monkeypatch, [unit])
    stops: list[str] = []

    assert kbd.reap_terminal_run_scopes(conn, stop_fn=_recorder(stops)) == []
    assert stops == []


def test_one_failing_scope_does_not_abort_the_sweep(conn, monkeypatch):
    tid_a, run_a = _closed_run(conn)
    tid_b, run_b = _closed_run(conn)
    unit_a, unit_b = _unit(tid_a, run_a), _unit(tid_b, run_b)
    _list_units(monkeypatch, [unit_a, unit_b])
    stops: list[str] = []

    stopped = kbd.reap_terminal_run_scopes(conn, stop_fn=_recorder(stops, fail_on={unit_a}))

    assert stops == [unit_a, unit_b]
    assert stopped == [unit_b]


def test_a_scope_that_will_not_stop_is_not_reported(conn, monkeypatch):
    """``_stop_systemd_unit`` answers False when the stop failed outright; the
    sweep must not claim a reap it did not achieve."""
    tid, run_id = _closed_run(conn)
    unit = _unit(tid, run_id)
    _list_units(monkeypatch, [unit])

    assert kbd.reap_terminal_run_scopes(conn, stop_fn=lambda _u: False) == []


def test_dispatch_tick_reports_the_reaped_scopes(conn, monkeypatch):
    tid, run_id = _closed_run(conn)
    unit = _unit(tid, run_id)
    _list_units(monkeypatch, [unit])
    monkeypatch.setattr(kbd, "_stop_worker_scope_unit", lambda _u: True)

    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)

    assert result.reaped_run_scopes == [unit]


def test_lister_filters_to_launchable_worker_scopes(monkeypatch):
    """The pattern is the guard: a placeholder (``-run-missing``) scope is never a
    real run, and no other unit family is a worker scope."""
    lines = "\n".join([
        "hermes-worker-kanban-t_abc-run-12.scope loaded active running",
        "hermes-worker-kanban-t_abc-run-missing.scope loaded active running",
        "hermes-probe-scope-4242.scope loaded active running",
        "hermes-worker-cron-run-3.scope loaded active running",
        "",
    ])
    monkeypatch.setattr(
        _shutil, "which", lambda name: "/usr/bin/systemctl" if name == "systemctl" else None,
    )
    monkeypatch.setattr(
        kbd.subprocess, "run",
        lambda *a, **k: type("P", (), {"returncode": 0, "stdout": lines})(),
    )
    monkeypatch.setattr(
        "tools.process_registry.systemd_user_bus_env", lambda *a, **k: {},
    )

    assert kbd._loaded_worker_scope_units() == ["hermes-worker-kanban-t_abc-run-12.scope"]


def test_lister_is_a_noop_without_systemctl(monkeypatch):
    monkeypatch.setattr(_shutil, "which", lambda _name: None)

    assert kbd._loaded_worker_scope_units() == []


def test_real_unit_name_matches_the_spawn_suffix():
    """The sweep parses what ``_restart_safe_worker_argv`` mints
    (``hermes-worker-`` + ``kanban-<task>-run-<n>``): the two must not drift."""
    did = "t_9f36a783"
    run_id = 723
    unit = f"hermes-worker-kanban-{did}-run-{run_id}.scope"
    match = kbd.WORKER_SCOPE_UNIT_RE.match(unit)
    assert match is not None
    assert match.group("task_id") == did
    assert int(match.group("run_id")) == run_id
    assert int(time.time()) > 0  # the pattern is anchored to the real clock only for the grace cut
