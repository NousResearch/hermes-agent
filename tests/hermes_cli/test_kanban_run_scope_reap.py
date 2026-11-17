"""Invariant: a run that has ended leaves no loaded worker scope of its own.

Every worker of a systemd-hosted dispatcher runs in a transient
``hermes-worker-kanban-<task>-run-<n>.scope``. ``--collect`` removes that scope only once its
cgroup is empty, so a worker that ends leaving a background child behind (a dev server, a
``python3 -m http.server``, an ``Xvfb``, a ``while pgrep ...; do sleep 15; done`` loop) keeps the
scope loaded and holds the child's port and memory for as long as the child lives, while the
board shows a cleanly finished run.

``reap_terminal_run_scopes`` stops the scope of every run that has ended, and never one whose
run is still open, still inside the finalisation grace window, or not this board's at all.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_worker_scopes as scopes


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
    conn.commit()
    return tid, run_id


def _list_units(monkeypatch, units: list[str]) -> None:
    monkeypatch.setattr(scopes, "_loaded_worker_scope_units", lambda: list(units))


def _recorder(stops: list[str], *, fail_on: set[str] | None = None):
    def stop(unit: str) -> bool:
        stops.append(unit)
        if fail_on and unit in fail_on:
            raise RuntimeError("boom")
        return True

    return stop


def test_closed_run_scope_is_stopped_and_recorded(conn, monkeypatch):
    tid, run_id = _closed_run(conn)
    unit = scopes.worker_scope_unit(tid, run_id)
    _list_units(monkeypatch, [unit])
    stops: list[str] = []

    assert scopes.reap_terminal_run_scopes(conn, stop_fn=_recorder(stops)) == [unit]

    assert stops == [unit]
    payload = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='terminal_run_scope_reaped'", (tid,)
    ).fetchone()["payload"]
    assert unit in payload and str(run_id) in payload


def test_open_run_scope_is_never_touched(conn, monkeypatch):
    """A live worker's scope holds the live worker: stopping it would kill the run."""
    tid, run_id = _claimed_run(conn)
    _list_units(monkeypatch, [scopes.worker_scope_unit(tid, run_id)])
    stops: list[str] = []

    assert scopes.reap_terminal_run_scopes(conn, stop_fn=_recorder(stops)) == []
    assert stops == []


def test_freshly_closed_scope_waits_for_the_grace_window(conn, monkeypatch):
    """A worker is still alive for a moment after its own kanban_complete returns
    (final turn, session persistence), so a just-closed run is left alone."""
    tid, run_id = _closed_run(conn, ended_ago=0)
    unit = scopes.worker_scope_unit(tid, run_id)
    _list_units(monkeypatch, [unit])
    stops: list[str] = []

    assert scopes.reap_terminal_run_scopes(conn, stop_fn=_recorder(stops)) == []
    assert stops == []

    conn.execute(
        "UPDATE task_runs SET ended_at = ended_at - ? WHERE id = ?",
        (scopes.TERMINAL_WORKER_REAP_GRACE_SECONDS + 1, run_id),
    )
    conn.commit()
    assert scopes.reap_terminal_run_scopes(conn, stop_fn=_recorder(stops)) == [unit]


def test_scope_of_another_board_is_left_alone(conn, monkeypatch):
    """Scopes carry the task id, not the board: a name no run row in this DB
    matches belongs to a board this dispatcher does not own."""
    _tid, run_id = _closed_run(conn)
    _list_units(monkeypatch, [scopes.worker_scope_unit("t_00000001", run_id + 9000)])
    stops: list[str] = []

    assert scopes.reap_terminal_run_scopes(conn, stop_fn=_recorder(stops)) == []
    assert stops == []


def test_one_failing_scope_does_not_abort_the_sweep(conn, monkeypatch):
    tid_a, run_a = _closed_run(conn)
    tid_b, run_b = _closed_run(conn)
    unit_a, unit_b = scopes.worker_scope_unit(tid_a, run_a), scopes.worker_scope_unit(tid_b, run_b)
    _list_units(monkeypatch, [unit_a, unit_b])
    stops: list[str] = []

    stopped = scopes.reap_terminal_run_scopes(conn, stop_fn=_recorder(stops, fail_on={unit_a}))

    assert stops == [unit_a, unit_b]
    assert stopped == [unit_b]


def test_a_scope_that_will_not_stop_is_not_reported(conn, monkeypatch):
    """``_stop_systemd_unit`` answers False when the stop failed outright; the
    sweep must not claim a reap it did not achieve."""
    tid, run_id = _closed_run(conn)
    _list_units(monkeypatch, [scopes.worker_scope_unit(tid, run_id)])

    assert scopes.reap_terminal_run_scopes(conn, stop_fn=lambda _u: False) == []
    assert not conn.execute(
        "SELECT 1 FROM task_events WHERE task_id=? AND kind='terminal_run_scope_reaped'", (tid,)
    ).fetchone()


def test_dispatch_tick_reports_the_reaped_scopes(conn, monkeypatch):
    tid, run_id = _closed_run(conn)
    unit = scopes.worker_scope_unit(tid, run_id)
    _list_units(monkeypatch, [unit])
    monkeypatch.setattr(scopes, "_stop_worker_scope_unit", lambda _u: True)

    result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)

    assert result.reaped_run_scopes == [unit]


def test_lister_keeps_only_launchable_worker_scopes(monkeypatch):
    """The pattern is the guard: a placeholder (``-run-missing``) scope is never a
    real run, and no other unit family is a worker scope."""
    lines = "\n".join([
        "hermes-worker-kanban-t_abc12345-run-12.scope loaded active running",
        "hermes-worker-kanban-t_abc12345-run-missing.scope loaded active running",
        "hermes-probe-scope-4242.scope loaded active running",
        "hermes-worker-cron-run-3.scope loaded active running",
        "",
    ])
    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which",
        lambda name, *a, **k: "/usr/bin/systemctl" if name == "systemctl" else real_which(name, *a, **k),
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=lines),
    )
    monkeypatch.setattr("tools.process_registry.systemd_user_bus_env", lambda *a, **k: {})

    assert scopes._loaded_worker_scope_units() == ["hermes-worker-kanban-t_abc12345-run-12.scope"]


def test_lister_is_a_noop_without_systemctl(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)

    assert scopes._loaded_worker_scope_units() == []


def test_unit_name_matches_what_the_spawn_mints(monkeypatch):
    """The sweep and the OOM probe parse the unit ``_restart_safe_worker_argv`` asks for
    (``hermes-worker-`` + ``kanban-<task>-run-<n>``): the two must not drift."""
    seen: dict = {}

    def fake_child_argv(command, *, unit_suffix, **_kwargs):
        seen["suffix"] = unit_suffix
        return SimpleNamespace(mode="scoped", argv=command)

    monkeypatch.setattr("tools.process_registry.restart_safe_gateway_child_argv", fake_child_argv)
    task = SimpleNamespace(id="t_00000001", current_run_id=723)

    kbd._restart_safe_worker_argv(task, ["hermes"])

    unit = f"hermes-worker-{seen['suffix']}.scope"
    assert unit == scopes.worker_scope_unit("t_00000001", 723)
    match = scopes.WORKER_SCOPE_UNIT_RE.match(unit)
    assert match is not None
    assert (match.group("task_id"), int(match.group("run_id"))) == ("t_00000001", 723)
