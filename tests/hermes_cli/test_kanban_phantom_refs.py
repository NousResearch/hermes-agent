"""Phantom-vs-cross-board prose reference detection (``_flag_phantom_prose_refs``).

Invariants:

1. A ``t_<hex>`` id cited in a completion summary that exists on ANOTHER
   board is a legitimate cross-board citation — it must emit
   ``cross_board_references`` and must NEVER emit
   ``suspected_hallucinated_references`` nor spawn a ``verify:`` child.
2. An id that resolves on NO board is a phantom — it emits
   ``suspected_hallucinated_references`` and spawns exactly one ``verify:``
   child assigned to the completing task's profile, parented on the task.
3. The two kinds are structurally distinct: the cross-board payload carries
   ``refs`` (id -> board slug) and never ``phantom_refs``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.init_db()
    return home


def _kinds(conn, task_id):
    return [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
    ).fetchall()]


def _payload(conn, task_id, kind):
    for r in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id", (task_id, kind)
    ).fetchall():
        return json.loads(r["payload"])
    raise AssertionError(f"no {kind!r} event on {task_id}")


def _verify_children(conn):
    return conn.execute(
        "SELECT id, title, assignee, status FROM tasks WHERE title LIKE 'verify:%'"
    ).fetchall()


def test_cross_board_reference_is_not_a_phantom(kanban_home):
    """A citation to a task on another board is legitimate, not hallucinated."""
    kb.create_board("other-board")
    other = connect(board="other-board")
    remote_id = kb.create_task(other, title="lives elsewhere", assignee="sysadmin")

    conn = connect(board="default")
    tid = kb.create_task(conn, title="cites another board", assignee="sysadmin")
    assert kb.complete_task(conn, tid, summary=f"Probe {remote_id} ran on the shared board.") is True

    kinds = _kinds(conn, tid)
    assert "cross_board_references" in kinds
    assert "suspected_hallucinated_references" not in kinds
    assert _verify_children(conn) == []

    payload = _payload(conn, tid, "cross_board_references")
    assert payload["refs"] == {remote_id: "other-board"}
    assert "phantom_refs" not in payload


def test_unresolvable_reference_is_phantom_and_spawns_verify_child(kanban_home):
    """An id on no board at all is a phantom and gets re-checked."""
    conn = connect(board="default")
    tid = kb.create_task(conn, title="invents a task", assignee="coder")
    fake = "t_deadbeef99"
    assert kb.complete_task(conn, tid, summary=f"Verified via {fake} which passed.") is True

    kinds = _kinds(conn, tid)
    assert "suspected_hallucinated_references" in kinds
    assert _payload(conn, tid, "suspected_hallucinated_references")["phantom_refs"] == [fake]

    children = _verify_children(conn)
    assert len(children) == 1
    child = children[0]
    assert child["assignee"] == "coder"
    assert child["title"] == "verify: invents a task"
    # Default initial_status normalizes from the (done) parent -> ready.
    assert child["status"] == "ready"
    parents = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ?", (child["id"],)
    ).fetchall()
    assert [p["parent_id"] for p in parents] == [tid]


def test_mixed_refs_report_both_kinds_separately(kanban_home):
    """One summary citing a real remote id AND a fake id yields both kinds."""
    kb.create_board("other-board")
    other = connect(board="other-board")
    remote_id = kb.create_task(other, title="real remote task", assignee="sysadmin")

    conn = connect(board="default")
    tid = kb.create_task(conn, title="mixed citations", assignee="sysadmin")
    fake = "t_feedface01"
    assert kb.complete_task(
        conn, tid, summary=f"Saw {remote_id} on the shared board and invented {fake}."
    ) is True

    kinds = _kinds(conn, tid)
    assert "cross_board_references" in kinds
    assert "suspected_hallucinated_references" in kinds
    assert _payload(conn, tid, "cross_board_references")["refs"] == {remote_id: "other-board"}
    assert _payload(conn, tid, "suspected_hallucinated_references")["phantom_refs"] == [fake]
    assert len(_verify_children(conn)) == 1


def test_local_reference_is_silent(kanban_home):
    """An id that exists on THIS board is neither phantom nor cross-board."""
    conn = connect(board="default")
    other_local = kb.create_task(conn, title="local sibling", assignee="sysadmin")
    tid = kb.create_task(conn, title="cites local", assignee="sysadmin")
    assert kb.complete_task(conn, tid, summary=f"See {other_local} for the handoff.") is True

    kinds = _kinds(conn, tid)
    assert "cross_board_references" not in kinds
    assert "suspected_hallucinated_references" not in kinds
    assert _verify_children(conn) == []
