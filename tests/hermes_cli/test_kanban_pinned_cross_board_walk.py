"""Cross-board load walks must survive the worker's ``HERMES_KANBAN_DB`` pin.

The dispatcher spawns every worker with ``HERMES_KANBAN_DB`` pinned to *that
worker's* board file (``kanban_db_dispatch._default_spawn``), and the pin
outranks the ``board=`` slug inside ``kanban_db_path``. So a cross-board walk
that resolves each slug through that resolver maps every board onto the worker's
own file, dedups them all away, and reports ``0`` / ``{}`` — from inside exactly
the process the host-level (``max_in_progress``) and per-profile
(``max_in_progress_per_profile``) caps exist to bound.

The pin does not mean "all boards share one file". It means "this process
belongs to ONE board", so enumerating the others must not go through it.

Reachability of the pinned path, with evidence: ``hermes kanban dispatch`` run
in a worker's terminal inherits the pin — the worker's shell has
``HERMES_KANBAN_DB=<its board>/kanban.db`` (verified on a live worker) and a
child process does not scrub it (``agent/secret_scope.py`` lists
``HERMES_KANBAN_*`` as genuinely-global, and the terminal tool passes it
through). ``kanban_ops._cmd_dispatch`` then connects with ``connect_closing()``
— the pinned file — and calls ``dispatch_once(conn, ...)`` with no ``board``.

These tests mirror the unpinned pair in ``test_kanban_host_cap.py``
(``test_max_in_progress_counts_other_boards``) with the pin set, so the two
suites read as one before/after pair.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB and NO board pin."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.init_db()
    return home


def _pin_to(monkeypatch, board=None):
    """Reproduce the dispatcher's worker env: pin THIS process to one board."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path(board=board)))


def _running_on_other_board(slug: str, *, assignee: str = "alice", count: int = 1):
    """``count`` claimed (status='running') tasks on the additional board ``slug``."""
    kb.create_board(slug)
    with kbc.connect(board=slug) as conn:
        for i in range(count):
            tid = kb.create_task(conn, title=f"busy-{i}", assignee=assignee)
            assert kb.claim_task(conn, tid) is not None


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


# ---------------------------------------------------------------------------
# 1. The counts themselves (unit level)
# ---------------------------------------------------------------------------


def test_pinned_env_still_counts_running_tasks_on_other_boards(
    kanban_home, monkeypatch,
):
    """A worker on board A must count board B's running workers.

    Under the pin every slug resolves to board A's file, so the path-equality
    dedup skips every board and this reads 0 — the host cap silently
    under-counts from inside the only processes that dispatch from a pin.
    """
    _running_on_other_board("second", count=2)
    _pin_to(monkeypatch)

    assert kbd.count_running_tasks_other_boards() == 2


def test_pinned_env_still_counts_other_boards_by_assignee(
    kanban_home, monkeypatch,
):
    """Same walk, per-assignee: a profile's fan-out is not board-local."""
    _running_on_other_board("second", assignee="alice", count=2)
    _running_on_other_board("third", assignee="bob", count=1)
    _pin_to(monkeypatch)

    assert kbd.count_running_tasks_other_boards_by_assignee() == {"alice": 2, "bob": 1}


def test_pinned_env_excludes_only_the_processs_own_board(
    kanban_home, monkeypatch,
):
    """The excluded board is the one the process is on, not "all of them".

    Board A holds a running worker of its own: it is counted board-locally by
    the tick (``count_running_tasks(conn)``), so the other-board walk must skip
    A and only A.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="busy-here", assignee="alice")
        assert kb.claim_task(conn, tid) is not None
    _running_on_other_board("second", count=1)
    _pin_to(monkeypatch)

    assert kbd.count_running_tasks_other_boards() == 1


# ---------------------------------------------------------------------------
# 2. The caps the walk exists to enforce (the defect's actual bite)
# ---------------------------------------------------------------------------


def test_pinned_env_host_cap_counts_other_boards(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Pinned dispatch: another board's workers still consume the host budget.

    Unpinned counterpart: ``test_max_in_progress_counts_other_boards``.
    """
    _running_on_other_board("second", count=2)
    _pin_to(monkeypatch)

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="wants-to-run", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # Host budget (2) is already consumed by the second board → nothing spawns.
    assert not spawns
    assert not res.spawned


def test_pinned_env_per_profile_cap_counts_other_boards(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Pinned dispatch: the per-profile cap spans boards too."""
    _running_on_other_board("second", assignee="alice", count=1)
    _pin_to(monkeypatch)

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="wants-to-run", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns),
            max_in_progress_per_profile=1,
        )

    assert not spawns
    assert [c[1] for c in res.skipped_per_profile_capped] == ["alice"]
