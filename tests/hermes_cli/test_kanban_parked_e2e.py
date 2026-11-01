"""End-to-end: the parked column against a real board DB.

Reproduces the ticket's measured shape (20 unassigned cards in ``ready``,
7 higher-priority assigned cards in front of them) and verifies the board
reports them as parked rather than as runnable ready work.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def isolated_kanban_home(tmp_path, monkeypatch):
    """Fresh HERMES_HOME with a clean kanban DB."""
    test_home = tmp_path / ".hermes"
    test_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(test_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db
    yield kanban_db, test_home


def _fake_spawn(*args, **kwargs):
    return 4242


def test_parked_column_end_to_end(isolated_kanban_home, capsys):
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        parked_ids = []
        for i in range(20):
            tid = kb.create_task(conn, title=f"guarden ticket {i}")
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
            parked_ids.append(tid)
        # 7 higher-priority assigned cards sit IN FRONT of the unassigned
        # ones in the lane order, so a budget break never reached them.
        for i in range(7):
            kb.create_task(conn, title=f"assigned {i}", assignee="default", priority=1000)

    with kbc.connect_closing() as conn:
        before = kb.board_stats(conn)
        assert before["by_status"]["ready"] == 27, before["by_status"]
        assert before["ready_runnable"] == 7, before["ready_runnable"]

    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
        after = kb.board_stats(conn)
        parked_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE status = 'parked'"
        ).fetchone()["n"]

    # The 20 unassigned cards left the ready lane even though 7 assigned
    # cards outranked them and the tick had no room to spawn.
    assert parked_rows == 20, parked_rows
    # The 7 assigned cards spawned, so ``ready`` is empty: every card that
    # remains in the ready lane is one the dispatcher can actually run.
    assert after["by_status"].get("ready", 0) == 0, after["by_status"]
    assert after["by_status"].get("running") == 7, after["by_status"]
    assert after["ready_runnable"] == 0, after["ready_runnable"]
    assert len(res.parked_unassigned) == 20, len(res.parked_unassigned)
    assert not res.skipped_unassigned

    # A parked card names no one.
    with kbc.connect_closing() as conn:
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (parked_ids[0],),
        ).fetchone()
        assert not row["assignee"]

    # Routing one parked card puts it back in the runnable lane.
    with kbc.connect_closing() as conn:
        kb.assign_task(conn, parked_ids[0], "default")
        row = kb.get_task(conn, parked_ids[0])
        assert row.status == "ready", row.status
        assert row.assignee == "default"
        kinds = [e.kind for e in kb.list_events(conn, parked_ids[0])]
        assert "unparked" in kinds, kinds

    with kbc.connect_closing() as conn:
        res2 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
        assert any(s[0] == parked_ids[0] for s in res2.spawned)
