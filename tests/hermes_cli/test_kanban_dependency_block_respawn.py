"""Regression tests for #107784 — dependency-kind block must not re-dispatch.

``kanban_block(kind="dependency")`` parks the card in ``todo`` (not
``blocked``) and closes the run with ``outcome='blocked'``. ``#28712``
sticky-block only looks at status=``blocked``, so ``recompute_ready``
promotes a parentless todo back to ready and ``check_respawn_guard``
(pre-fix) has no blocked-outcome branch — ``dispatch_once`` then
re-spawns the worker with no new input.

These tests pin:

* Parentless dependency-block + bare ``promoted`` from ``recompute_ready``
  is held (no spawn).
* A strictly-later comment / ``unblocked`` / ``promoted_manual`` is a
  release and may spawn.
* A child with a linked parent is fail-open (parent completion is the
  new input).
* The review lane is never held by this branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _age_ended_at(conn, task_id: str, seconds: int = 5) -> None:
    """Backdate ``task_runs.ended_at`` so later comments/events are strictly after."""
    conn.execute(
        "UPDATE task_runs SET ended_at = ended_at - ? WHERE task_id=?",
        (seconds, task_id),
    )
    conn.commit()


def _spawn_tracker():
    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    return spawned, spawn


def _dependency_block_parentless(conn, *, title: str = "waiting on an external reply"):
    tid = kb.create_task(conn, title=title, assignee="marketing")
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    assert kb.block_task(
        conn, tid, reason="third party has not replied yet",
        kind="dependency", expected_run_id=claimed.current_run_id,
    )
    assert kb.get_task(conn, tid).status == "todo"
    assert kb.latest_run(conn, tid).outcome == "blocked"
    _age_ended_at(conn, tid)
    return tid


def test_parentless_dependency_block_is_not_respawned(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#107784: parentless dependency-block + recompute_ready ``promoted``
    must not re-dispatch. This is the RED on pre-fix main."""
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="waiting on an external reply", assignee="marketing")
        claimed = kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid, reason="third party has not replied yet",
            kind="dependency", expected_run_id=claimed.current_run_id,
        )
        assert kb.get_task(conn, tid).status == "todo"
        assert kb.latest_run(conn, tid).outcome == "blocked"
        # age ended_at by a few seconds
        conn.execute("UPDATE task_runs SET ended_at = ended_at - 5 WHERE task_id=?", (tid,))
        conn.commit()
        spawned = []

        def spawn(task, workspace, board=None):
            spawned.append(task.id)
            return 4242

        for _ in range(3):
            kbd.dispatch_once(conn, spawn_fn=spawn)
        assert tid not in spawned


def test_bare_promoted_from_recompute_ready_does_not_release(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ``promoted`` event written by ``recompute_ready`` is not new input."""
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

    with kbc.connect() as conn:
        tid = _dependency_block_parentless(conn)
        assert kb.recompute_ready(conn) == 1
        kinds = [
            r["kind"] for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (tid,),
            ).fetchall()
        ]
        assert "promoted" in kinds
        assert "promoted_manual" not in kinds
        spawned, spawn = _spawn_tracker()
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert tid not in spawned
        assert kbd.check_respawn_guard(conn, tid) == "blocked_outcome"


def test_comment_after_dependency_block_releases(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A comment strictly after the blocked run is new input — may spawn."""
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

    with kbc.connect() as conn:
        tid = _dependency_block_parentless(conn)
        kb.add_comment(conn, tid, author="op", body="third party replied, retry")
        spawned, spawn = _spawn_tracker()
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert tid in spawned


def test_unblocked_event_after_dependency_block_releases(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``unblocked`` event strictly after the blocked run is a release."""
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

    with kbc.connect() as conn:
        tid = _dependency_block_parentless(conn)
        # dependency-kind parks in todo, so unblock_task (blocked/scheduled
        # only) is a no-op; write the equivalent release event the guard keys on.
        ended_at = kb.latest_run(conn, tid).ended_at
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "unblocked")
            conn.execute(
                "UPDATE task_events SET created_at = ? "
                "WHERE task_id=? AND kind='unblocked'",
                (int(ended_at) + 1, tid),
            )
        spawned, spawn = _spawn_tracker()
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert tid in spawned


def test_promote_task_after_dependency_block_releases(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator ``promote_task`` writes ``promoted_manual`` — may spawn."""
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

    with kbc.connect() as conn:
        tid = _dependency_block_parentless(conn)
        ok, reason = kb.promote_task(conn, tid, actor="op")
        assert ok, reason
        spawned, spawn = _spawn_tracker()
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert tid in spawned


def test_linked_child_spawns_after_parent_completes(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linked-parent fail-open: parent completion is the new input."""
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="marketing")
        child = kb.create_task(conn, title="child", assignee="marketing")
        claimed = kb.claim_task(conn, child)
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        assert kb.block_task(
            conn, child, reason="wait", kind="dependency",
            expected_run_id=claimed.current_run_id,
        )
        assert kb.get_task(conn, child).status == "todo"
        _age_ended_at(conn, child)

        spawned, spawn = _spawn_tracker()
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert child not in spawned, "parent still open — must not spawn"

        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, result="done")
        spawned.clear()
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert child in spawned
        assert kb.get_task(conn, child).status == "running"


def test_review_lane_not_held_by_blocked_outcome(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review lane must never apply the blocked-outcome hold."""
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)

    with kbc.connect() as conn:
        tid = _dependency_block_parentless(conn)
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None
