"""Per-profile concurrency cap counts running workers across boards (#135515).

``kanban.max_in_progress_per_profile`` is documented as a machine-level cap
(one profile's model/API quota/browser pool lives on the host, not inside a
board), but the dispatcher counted each assignee's ``running`` tasks with a
SELECT on the CURRENT board's connection only — so N boards multiplied the
per-profile ceiling by N, exactly the fan-out the cap exists to prevent.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    from pathlib import Path
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_per_profile_cap_counts_other_boards(
    kanban_home, all_assignees_spawnable,
):
    """A running worker for assignee X on board B counts against X's cap when
    board A's dispatcher ticks — the cap bounds the profile (host), not each
    board's local view of it."""
    kb.create_board("second")

    # Board B: X already has one running worker (cap=1 → X is at cap host-wide).
    with kbc.connect(board="second") as conn:
        tid = kb.create_task(conn, title="busy-on-second", assignee="worker-x")
        assert kb.claim_task(conn, tid) is not None

    # Board A (default): a ready task for the SAME assignee.
    spawns: list = []
    with kbc.connect() as conn:
        ready_id = kb.create_task(conn, title="wants-to-run", assignee="worker-x")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns),
            max_in_progress_per_profile=1,
        )

    # X is at the cap via board B → deferred, not spawned.
    assert not spawns
    assert not res.spawned
    assert res.skipped_per_profile_capped == [(ready_id, "worker-x", 1)]


def test_per_profile_cap_same_board_behaviour_unchanged(
    kanban_home, all_assignees_spawnable,
):
    """Same-board counting keeps its historical shape: the busy assignee is
    deferred to skipped_per_profile_capped while a different assignee spawns."""
    spawns: list = []
    with kbc.connect() as conn:
        busy_id = kb.create_task(conn, title="busy-here", assignee="worker-x")
        assert kb.claim_task(conn, busy_id) is not None
        ready_id = kb.create_task(conn, title="wants-to-run", assignee="worker-x")
        other_id = kb.create_task(conn, title="other-profile", assignee="worker-y")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns),
            max_in_progress_per_profile=1,
        )

    # X capped at 1 by its own board's running task; Y unaffected.
    assert res.skipped_per_profile_capped == [(ready_id, "worker-x", 1)]
    assert [task_id for task_id, *_ in res.spawned] == [other_id]
