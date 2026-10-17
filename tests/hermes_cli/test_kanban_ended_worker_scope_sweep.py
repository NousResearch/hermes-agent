"""Invariant: a Kanban worker's transient systemd scope does not outlive its run.

A worker that exits (or dies) while a descendant keeps running, e.g. a
browser-harness daemon reparented to the user manager, keeps its
``hermes-worker-kanban-<task>-run-<id>.scope`` alive with ``--collect`` never
firing. The dispatcher tick stops every such scope whose run ended more than
``TERMINAL_WORKER_REAP_GRACE_SECONDS`` ago, and nothing else.

``systemctl`` is replaced by a fake binary on PATH that lists the given units
and records every ``stop``.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_scopes as kbs


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


class FakeSystemctl:
    def __init__(self, tmp_path: Path, monkeypatch, *, fail_stop: bool = False, fail_list: bool = False):
        self.dir = tmp_path / "fakebin"
        self.dir.mkdir()
        self.units_file = tmp_path / "units.txt"
        self.units_file.write_text("")
        self.stops_file = tmp_path / "stops.txt"
        script = self.dir / "systemctl"
        script.write_text(
            "#!/bin/sh\n"
            'LAST=""; for x in "$@"; do LAST="$x"; done\n'
            'case " $* " in\n'
            f'  *" list-units "*) {"exit 1" if fail_list else f"cat {self.units_file}"} ;;\n'
            f'  *" stop "*) echo "$LAST" >> {self.stops_file}; {"exit 1" if fail_stop else "exit 0"} ;;\n'
            "esac\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{self.dir}{os.pathsep}{os.environ.get('PATH', '')}")

    def set_units(self, *names: str) -> None:
        self.units_file.write_text("".join(
            f"{n} loaded active running [systemd-run] python -m hermes_cli.main\n" for n in names))

    @property
    def stopped(self) -> list[str]:
        return self.stops_file.read_text().split() if self.stops_file.exists() else []


def _scope(task_id: str, run_id: int) -> str:
    return f"hermes-worker-{kbs.kanban_worker_unit_suffix(task_id, run_id)}.scope"


def _ended_run(conn, *, ended_ago: int = 600, claim_lock: str | None = None) -> tuple[str, int]:
    tid = kb.create_task(conn, title="finished", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?", (ended_ago, run_id))
    if claim_lock is not None:
        conn.execute("UPDATE task_runs SET claim_lock = ? WHERE id=?", (claim_lock, run_id))
    return tid, run_id


def _reclaimed_run(conn) -> tuple[str, int]:
    """A run closed by a reclaim path that clears its claim_lock (``_reclaim_dangling_run``)."""
    tid, run_id = _ended_run(conn)
    conn.execute("UPDATE task_runs SET claim_lock = NULL WHERE id=?", (run_id,))
    return tid, run_id


def _open_run(conn) -> tuple[str, int]:
    tid = kb.create_task(conn, title="working", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    return tid, kb._current_run_id(conn, tid)


def test_unit_suffix_is_the_spawn_format():
    assert kbs.kanban_worker_unit_suffix("t_abc", 42) == "kanban-t_abc-run-42"


def test_scope_of_ended_run_is_stopped_on_dispatch_tick(conn, tmp_path, monkeypatch):
    fake = FakeSystemctl(tmp_path, monkeypatch)
    tid, run_id = _ended_run(conn)
    fake.set_units(_scope(tid, run_id))

    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)

    assert fake.stopped == [_scope(tid, run_id)]


def test_open_run_fresh_run_foreign_host_and_strangers_are_untouched(conn, tmp_path, monkeypatch):
    fake = FakeSystemctl(tmp_path, monkeypatch)
    open_tid, open_run = _open_run(conn)
    fresh_tid, fresh_run = _ended_run(conn, ended_ago=0)
    foreign_tid, foreign_run = _ended_run(conn, claim_lock="some-other-host:1234")
    ended_tid, ended_run = _ended_run(conn)
    fake.set_units(
        _scope(open_tid, open_run),
        _scope(fresh_tid, fresh_run),
        _scope(foreign_tid, foreign_run),
        _scope(ended_tid, ended_run),
        f"hermes-worker-kanban-{ended_tid}-run-notanumber.scope",
        f"hermes-worker-kanban-{ended_tid}-run-{ended_run}-x.scope",
        f"hermes-worker-kanban-t_unknown-run-{ended_run}.scope",  # another board's card
        "hermes-worker-proc_abc123.scope",
        "session-4.scope",
    )

    assert kbs.stop_ended_worker_scopes(conn) == [_scope(ended_tid, ended_run)]

    assert fake.stopped == [_scope(ended_tid, ended_run)]


def test_scope_is_stopped_once_grace_has_passed(conn, tmp_path, monkeypatch):
    fake = FakeSystemctl(tmp_path, monkeypatch)
    tid, run_id = _ended_run(conn, ended_ago=0)
    fake.set_units(_scope(tid, run_id))
    assert kbs.stop_ended_worker_scopes(conn) == []

    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?",
                 (kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS, run_id))
    assert kbs.stop_ended_worker_scopes(conn) == [_scope(tid, run_id)]


def test_scope_of_reclaimed_run_is_stopped(conn, tmp_path, monkeypatch):
    """Reclaimed (hung/dead) workers are the leaking case; their run has no claim_lock left."""
    fake = FakeSystemctl(tmp_path, monkeypatch)
    tid, run_id = _reclaimed_run(conn)
    fake.set_units(_scope(tid, run_id))

    assert kbs.stop_ended_worker_scopes(conn) == [_scope(tid, run_id)]


def test_stops_per_tick_are_capped(conn, tmp_path, monkeypatch):
    """A backlog of leaked scopes is drained over several ticks, not inside one dispatch lock."""
    fake = FakeSystemctl(tmp_path, monkeypatch)
    cap = kbs.MAX_SCOPE_STOPS_PER_TICK
    fake.set_units(*(_scope(*_ended_run(conn)) for _ in range(cap + 2)))

    assert len(kbs.stop_ended_worker_scopes(conn)) == cap
    assert len(fake.stopped) == cap


@pytest.mark.parametrize("kwargs", [{"fail_stop": True}, {"fail_list": True}])
def test_failing_systemctl_never_breaks_the_tick(conn, tmp_path, monkeypatch, kwargs):
    FakeSystemctl(tmp_path, monkeypatch, **kwargs).set_units(_scope(*_ended_run(conn)))

    assert kbs.stop_ended_worker_scopes(conn) == []
    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)


def test_missing_systemctl_never_breaks_the_tick(conn, tmp_path, monkeypatch):
    _ended_run(conn)
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    assert kbs.stop_ended_worker_scopes(conn) == []
    kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, dry_run=True, max_spawn=0)
