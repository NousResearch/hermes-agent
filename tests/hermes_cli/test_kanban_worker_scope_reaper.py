"""Invariant: a finished run's worker scope is stopped, so the descendant processes
it leaked cannot outlive the run and hold a port forever (issue #70716).

A kanban worker runs in its own transient ``systemd-run --user --scope`` launched
``--collect``, which removes the scope only once *every* process in its cgroup has
exited.  ``reap_terminal_workers`` keys on the closed run row's retained pid — and
that pid is the dead *worker*, not the mock server / crawler it left behind — so a
leaked child keeps the scope ``active`` indefinitely and the next attempt at the
same card dies on ``EADDRINUSE`` looking like broken test infrastructure rather
than an orphan (``t_eca0aab6`` -> ``t_dcfc99ec``).

``reap_orphaned_worker_scopes`` therefore decides liveness from the live scope
list and the *open run rows*, never from PIDs.  These tests pin that filter: what
gets stopped, what is deliberately left alone, and that a real orphan in a real
transient scope is actually reaped.
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_reaping as kr
from tools import process_registry as pr
from tools.process_registry_scopes import WORKER_SCOPE_PREFIX, parse_kanban_worker_scope

# A unit that is not a kanban worker scope must never be a reap candidate.
_FOREIGN_UNITS = (
    "app-org.chromium.Chromium-716693.scope",
    "init.scope",
    "hermes-worker-other.scope",              # hermes prefix, but not a kanban scope
    "hermes-worker-kanban-t_deadbeef.scope",  # no -run-<n> suffix
    "hermes-worker-kanban-notatask-run-7.scope",
)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _claimed(conn, *, title="leaked") -> tuple[str, int]:
    """A task with an open run; returns ``(task_id, run_id)``."""
    tid = kb.create_task(conn, title=title, assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    assert run_id is not None
    return tid, int(run_id)


def _scope(tid: str, run_id: int) -> str:
    return f"hermes-worker-kanban-{tid}-run-{run_id}.scope"


def _close_run(conn, tid: str, run_id: int, *, ended_ago: int = 600) -> None:
    """Close the run and backdate ``ended_at`` past the reaper's grace window."""
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    conn.execute(
        "UPDATE task_runs SET ended_at = ? WHERE id = ?",
        (int(time.time()) - ended_ago, run_id),
    )


def _patch_units(monkeypatch, units):
    # ``_worker_scope_units`` is resolved in the reaping module's namespace, so patch
    # it there; the sweep itself is still called through the dispatch facade below,
    # which is what pins the re-export the reclaim phase depends on.
    monkeypatch.setattr(kr, "_worker_scope_units", lambda: set(units))


def _events(conn, tid):
    return [
        r["kind"]
        for r in conn.execute("SELECT kind FROM task_events WHERE task_id = ?", (tid,))
    ]


# The scope prefix is written in two places: minted by
# ``tools/process_registry._build_systemd_scope_argv`` (which sits at its FILE_LINES
# cap and therefore keeps the literal) and parsed by ``process_registry_scopes``.
# A rename on either side would leave the sweep running and matching nothing, so
# pin the two together rather than trusting the duplication.
def test_the_minted_scope_name_is_the_one_the_sweep_parses(monkeypatch):
    # ``_build_systemd_scope_argv`` imports ``shutil`` lazily, so pin ``which`` on the
    # module object rather than on a name the registry does not re-export.
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/systemd-run")
    argv = pr._build_systemd_scope_argv(["/bin/true"], unit_suffix="kanban-t_29b0d9cd-run-2424")
    unit = argv[argv.index("--unit") + 1]

    assert unit == f"{WORKER_SCOPE_PREFIX}kanban-t_29b0d9cd-run-2424"
    # ... and the sweep accepts exactly that name, task id and run id included.
    assert parse_kanban_worker_scope(f"{unit}.scope") == ("t_29b0d9cd", 2424)


# --------------------------------------------------------------------------- unit


def test_scope_of_a_finished_run_is_stopped(conn, monkeypatch):
    tid, run_id = _claimed(conn)
    _close_run(conn, tid, run_id)
    scope = _scope(tid, run_id)
    _patch_units(monkeypatch, {scope})
    stopped = []

    assert kbd.reap_orphaned_worker_scopes(conn, stop_fn=lambda u: stopped.append(u) or True) == [scope]
    assert stopped == [scope]
    assert "worker_scope_reaped" in _events(conn, tid)


def test_scope_of_a_live_run_is_never_stopped(conn, monkeypatch):
    """Mid-run, mid-``kanban_complete``, or between claim and ``worker_registered``
    all have an open row holding the scope."""
    tid, run_id = _claimed(conn)
    scope = _scope(tid, run_id)
    _patch_units(monkeypatch, {scope})
    stopped = []

    assert kbd.reap_orphaned_worker_scopes(conn, stop_fn=lambda u: stopped.append(u) or True) == []
    assert stopped == []
    assert "worker_scope_reaped" not in _events(conn, tid)


def test_superseded_run_scope_is_left_alone_while_a_newer_run_owns_the_card(conn, monkeypatch):
    """Run 1797 was replaced by 1800 while 1797's scope was still occupied.

    ``t_eca0aab6``: run 1797 timed out at 01:56:19 and run 1800 claimed the card at
    01:56:51 with 1797's scope still holding its orphan. A *differing* current run
    id means the worker was replaced, not that it is dead — the sweep must not act
    on a scope the card no longer points at. Once every run of the card has closed,
    ``_current_run_id`` is None and the leaked scopes become reapable (the next
    tick collects them), so this guard delays a reap, it never strands one.
    """
    tid, first = _claimed(conn)
    _close_run(conn, tid, first)
    # Replace it: a fresh open run becomes the card's current one.
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at) VALUES (?, 'coder', 'running', ?)",
        (tid, int(time.time())),
    )
    second = conn.execute(
        "SELECT MAX(id) AS id FROM task_runs WHERE task_id = ?", (tid,)
    ).fetchone()["id"]
    conn.execute(
        "UPDATE tasks SET status = 'running', current_run_id = ?, claim_lock = 'live', claim_expires = ? WHERE id = ?",
        (second, int(time.time()) + 3600, tid),
    )
    assert second != first

    scope = _scope(tid, first)
    _patch_units(monkeypatch, {scope})
    stopped = []

    assert kbd.reap_orphaned_worker_scopes(conn, stop_fn=lambda u: stopped.append(u) or True) == []
    assert stopped == []

    # And once that newer run closes too, the same scope is reaped — not stranded.
    conn.execute("UPDATE task_runs SET status = 'done', ended_at = ? WHERE id = ?", (int(time.time()) - 600, second))
    conn.execute("UPDATE tasks SET status = 'done', current_run_id = NULL WHERE id = ?", (tid,))
    assert kbd.reap_orphaned_worker_scopes(conn, stop_fn=lambda u: stopped.append(u) or True) == [scope]
    assert stopped == [scope]


def test_scope_inside_the_grace_window_is_left_alone(conn, monkeypatch):
    """A run that closed moments ago may still be doing its final turn: same grace
    as ``reap_terminal_workers``, same reason."""
    tid, run_id = _claimed(conn)
    _close_run(conn, tid, run_id, ended_ago=0)
    scope = _scope(tid, run_id)
    _patch_units(monkeypatch, {scope})
    stopped = []

    assert kbd.reap_orphaned_worker_scopes(conn, stop_fn=lambda u: stopped.append(u) or True) == []
    assert stopped == []


def test_non_worker_units_are_never_candidates(conn, monkeypatch):
    _patch_units(monkeypatch, set(_FOREIGN_UNITS))
    stopped = []

    assert kbd.reap_orphaned_worker_scopes(conn, stop_fn=lambda u: stopped.append(u) or True) == []
    assert stopped == []


def test_a_failed_stop_is_not_reported_as_reaped(conn, monkeypatch):
    tid, run_id = _claimed(conn)
    _close_run(conn, tid, run_id)
    scope = _scope(tid, run_id)
    _patch_units(monkeypatch, {scope})

    assert kbd.reap_orphaned_worker_scopes(conn, stop_fn=lambda u: False) == []
    assert "worker_scope_reaped" not in _events(conn, tid)


def test_one_dead_scope_does_not_block_another(conn, monkeypatch):
    tids = []
    scopes = []
    for i in range(2):
        tid, run_id = _claimed(conn, title=f"leaked{i}")
        _close_run(conn, tid, run_id)
        tids.append(tid)
        scopes.append(_scope(tid, run_id))
    _patch_units(monkeypatch, set(scopes))
    # First scope's stop raises: the sweep must skip only that one.
    stop_fn = lambda u: (_ for _ in ()).throw(RuntimeError("boom")) if u == scopes[0] else True

    assert kbd.reap_orphaned_worker_scopes(conn, stop_fn=stop_fn) == [scopes[1]]


def test_empty_enumeration_is_a_noop(conn, monkeypatch):
    """``systemctl`` unavailable / bus unreachable reads as "nothing to touch"."""
    tid, run_id = _claimed(conn)
    _close_run(conn, tid, run_id)
    _patch_units(monkeypatch, set())

    assert kbd.reap_orphaned_worker_scopes(conn, stop_fn=lambda u: True) == []


# -------------------------------------------------------------------- integration


@pytest.mark.integration
def test_real_transient_scope_with_orphaned_child_is_reaped(conn, monkeypatch):
    """End-to-end on the real user bus: start a transient scope mirroring the
    worker->mock shape, kill the leader (the dead worker), and prove the sweep
    reaps the *orphaned child* by stopping the scope's whole cgroup."""
    if not (subprocess.run(["bash", "-lc", "command -v systemctl"], capture_output=True, check=False).returncode == 0):
        pytest.skip("no systemctl")
    tid, run_id = _claimed(conn)
    _close_run(conn, tid, run_id)
    scope = _scope(tid, run_id)
    unit = scope[:-len(".scope")]

    # ``sleep 300 &`` then exit: the leader *exits* while the backgrounded child
    # stays in the scope — exactly the worker->mock shape (the worker ends, its
    # server keeps serving).  ``--collect`` keeps the scope alive for that child.
    leader = subprocess.Popen(
        ["systemd-run", "--user", "--scope", "--quiet", "--unit", unit, "--collect",
         "--", "/bin/sh", "-c", "sleep 300 >/dev/null 2>&1 & exit 0"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    leader.wait(timeout=15)
    try:
        pids = []
        for _ in range(40):
            probe = subprocess.run(
                ["systemctl", "--user", "show", scope, "-p", "PIDs", "--value"],
                capture_output=True, text=True, check=False,
            )
            pids = probe.stdout.split()
            if pids:
                break
            time.sleep(0.25)
        if not pids:
            pytest.skip("could not keep an orphan alive in a transient user scope")
        assert pids, "scope should hold the orphaned child after the leader exited"

        # Real enumeration + real _stop_systemd_unit (no monkeypatching here).
        assert kbd.reap_orphaned_worker_scopes(conn) == [scope]

        for _ in range(40):
            probe = subprocess.run(
                ["systemctl", "--user", "show", scope, "-p", "ActiveState", "--value"],
                capture_output=True, text=True, check=False,
            )
            if probe.stdout.strip() in ("", "inactive", "failed"):
                break
            time.sleep(0.25)
        for pid in pids:
            assert not os.path.exists(f"/proc/{pid}"), f"orphan {pid} survived the sweep"
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait()
        subprocess.run(
            ["systemctl", "--user", "stop", scope],
            capture_output=True, stdin=subprocess.DEVNULL, check=False,
        )
