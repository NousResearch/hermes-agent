"""Regression tests for t_a3bebeea — non-blocking 'tracking' link kind.

The wedge: a *tracking epic* (organizational parent, e.g. the Loopy AI epic
t_6875775d) linked over its work items made the kernel's ``parents_not_done``
invariant gate every child until the epic itself was done — but the epic
can't be done until its children are: a structural standoff that previously
forced the operator into the unlink + force-promote dance
(``reference_kanban_force_promoted_child_wedge``).

The fix: ``task_links.link_kind`` ('blocking' default | 'tracking').
Only 'blocking' edges gate a child, at all five enforcement sites
(claim_task, recompute_ready, promote_task, unblock_task, link_tasks's
ready->todo demotion). 'tracking' edges are membership only.

These tests pin down:

* A child whose only parent edge is 'tracking' claims + completes while the
  parent is open — the wedge scenario, eliminated.
* The default 'blocking' edge still gates exactly as before (no behavior
  change for existing graphs; the migration backfills legacy rows as
  'blocking').
* ``recompute_ready`` promotes a todo child whose only parent is tracking.
* ``set_link_kind`` blocking->tracking is the operator affordance: the child
  un-gates immediately via ``recompute_ready``.
* The additive migration is idempotent on a legacy DB lacking ``link_kind``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# (a) The wedge scenario: tracking parent never gates the child
# ---------------------------------------------------------------------------


def test_tracking_parent_does_not_block_child_claim(kanban_home: Path) -> None:
    """A child tracking-linked under an OPEN epic must claim and complete
    while the epic is still open — previously claim_task demoted it to todo
    with ``claim_rejected (parents_not_done)`` every attempt."""
    with kb.connect() as conn:
        epic = kb.create_task(conn, title="tracking epic (stays open)")
        child = kb.create_task(conn, title="work item")
        kb.link_tasks(conn, epic, child, kind="tracking")

        # Tracking link must not demote the ready child.
        assert kb.get_task(conn, child).status == "ready"

        claimed = kb.claim_task(conn, child)
        assert claimed is not None, "tracking parent must not gate the claim"
        kb.complete_task(conn, child, result="done while epic open")

        assert kb.get_task(conn, child).status == "done"
        assert kb.get_task(conn, epic).status != "done", "epic untouched"


# ---------------------------------------------------------------------------
# (b) Default 'blocking' semantics are unchanged
# ---------------------------------------------------------------------------


def test_blocking_parent_still_gates_child(kanban_home: Path) -> None:
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="blocking parent")
        child = kb.create_task(conn, title="dependent child")
        kb.link_tasks(conn, parent, child)  # default kind='blocking'

        # link_tasks demotes a ready child under an undone blocking parent.
        assert kb.get_task(conn, child).status == "todo"
        assert kb.claim_task(conn, child) is None

        # Parent completion releases the child (recompute_ready runs inside).
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, result="ok")
        assert kb.get_task(conn, child).status == "ready"
        assert kb.claim_task(conn, child) is not None


# ---------------------------------------------------------------------------
# (c) recompute_ready promotes a todo child whose only parent is tracking
# ---------------------------------------------------------------------------


def test_recompute_ready_promotes_child_with_only_tracking_parent(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        epic = kb.create_task(conn, title="tracking epic")
        child = kb.create_task(conn, title="work item")
        kb.link_tasks(conn, epic, child, kind="tracking")

        # Force the child into todo (simulating any racy writer / legacy
        # state); the dispatcher's recompute must promote it back.
        conn.execute(
            "UPDATE tasks SET status = 'todo' WHERE id = ?", (child,)
        )
        conn.commit()

        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# (d) Operator affordance: convert a wedged blocking edge in place
# ---------------------------------------------------------------------------


def test_set_link_kind_unwedges_blocked_child(kanban_home: Path) -> None:
    with kb.connect() as conn:
        epic = kb.create_task(conn, title="accidental blocking epic")
        child = kb.create_task(conn, title="wedged work item")
        kb.link_tasks(conn, epic, child)  # blocking -> child demoted to todo
        assert kb.get_task(conn, child).status == "todo"

        assert kb.set_link_kind(conn, epic, child, "tracking") is True
        # set_link_kind runs recompute_ready: the child un-gates immediately.
        assert kb.get_task(conn, child).status == "ready"
        assert kb.claim_task(conn, child) is not None

        # Converting a non-existent edge reports failure, mutates nothing.
        ghost = kb.create_task(conn, title="unlinked")
        assert kb.set_link_kind(conn, epic, ghost, "tracking") is False


def test_link_kind_rejects_unknown_kind(kanban_home: Path) -> None:
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        with pytest.raises(ValueError):
            kb.link_tasks(conn, a, b, kind="organizational")
        with pytest.raises(ValueError):
            kb.set_link_kind(conn, a, b, "soft")


# ---------------------------------------------------------------------------
# (e) Additive migration: legacy DB without link_kind
# ---------------------------------------------------------------------------


def test_migration_backfills_legacy_links_as_blocking(kanban_home: Path) -> None:
    """A pre-migration task_links row (no link_kind column) must come out of
    the migration as 'blocking' — zero behavior change for existing graphs —
    and re-running the migration must be a no-op."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="legacy parent")
        child = kb.create_task(conn, title="legacy child", parents=[parent])

        # Rewind the schema to its legacy shape (SQLite >= 3.35 DROP COLUMN);
        # the existing edge survives, now without a kind.
        conn.execute("ALTER TABLE task_links DROP COLUMN link_kind")
        conn.commit()
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(task_links)")}
        assert "link_kind" not in cols

        # Migration is additive + idempotent.
        kb._migrate_add_optional_columns(conn)
        kb._migrate_add_optional_columns(conn)
        conn.commit()

        row = conn.execute(
            "SELECT link_kind FROM task_links "
            "WHERE parent_id = ? AND child_id = ?",
            (parent, child),
        ).fetchone()
        assert row["link_kind"] == "blocking"

        # And the backfilled edge still gates: the child stays gated until
        # the parent completes.
        assert kb.claim_task(conn, child) is None
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, result="ok")
        assert kb.get_task(conn, child).status == "ready"
