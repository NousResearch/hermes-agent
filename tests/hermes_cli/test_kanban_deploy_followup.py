"""Deploy-follow-up auto-creation on approval of a platform (hermes-agent) card.

A review approval (``metadata["review_outcome"] == "approved"``) of a task whose
workspace is a git worktree under this hermes-agent repo must auto-create a
single, idempotency-keyed follow-up card that merges the branch and restarts the
gateway. Non-platform approvals and non-approval completions create nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


# The real repo worktrees root — the production branch path the code detects.
_WT = (Path(kb.__file__).resolve().parents[1] / ".worktrees").resolve()


def _platform_review_completion(conn, ttl_seconds: int | None = None):
    """Drive a card configured as a hermes-agent worktree through review, then
    approve it. Returns (task_id, review_run)."""
    task_id = kb.create_task(
        conn,
        title="Fix duplicate-send on gateway",
        assignee="builder",
        workspace_kind="worktree",
        workspace_path=str(_WT / "t_deploytest"),
        branch_name="wt/t_deploytest",
    )
    implementation = kb.claim_task(conn, task_id, claimer="builder:1")
    assert implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        summary="ready for independent review",
        reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, task_id, ttl_seconds=ttl_seconds)
    assert review is not None
    return task_id, review


def _children_of(conn, parent_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT child_id FROM task_links WHERE parent_id = ?", (parent_id,)
    ).fetchall()
    return [row["child_id"] for row in rows]


def test_approved_platform_card_creates_one_deploy_followup(conn):
    task_id, review = _platform_review_completion(conn)
    assert kb.complete_task(
        conn,
        task_id,
        summary="Reviewed and approved. Checks passed.",
        metadata={"review_outcome": "approved", "reviewer_checks": ["gate_passed"]},
        expected_run_id=review.current_run_id,
    )

    children = _children_of(conn, task_id)
    assert len(children) == 1
    child_id = children[0]
    child = kb.get_task(conn, child_id)
    assert child is not None
    # Charter §8 (2026-09-03): approval precedes deploy. The card is minted HELD
    # and only a human `kanban unblock` releases it — never dispatch-ready.
    assert child.status == "blocked"
    assert child.block_kind == "operator_hold"
    assert child.assignee == "default"
    assert child.title == "Deploy: merge wt/t_deploytest + restart gateway"
    assert "operator_hold" in (child.body or "")
    assert "--no-ff" in (child.body or "") and "deploy/" in (child.body or "")
    kinds = [r[0] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (child_id,)).fetchall()]
    assert "blocked" in kinds  # dates the hold for escalation-watch / pre-flight
    assert "restart-root-gateway.sh" in (child.body or "")
    assert "git merge" in (child.body or "")
    assert task_id in (child.body or "")
    assert child.idempotency_key == f"deploy-followup:{task_id}"
    # Watchdog install step is wired into the deploy path.
    assert "install_fleet_watchdogs.py" in (child.body or "")
    assert "--dest ~/.hermes/scripts" in (child.body or "")
    assert "scripts/fleet-watchdogs/" in (child.body or "")


def test_repeat_approval_cannot_create_second_deploy_followup(conn):
    """Idempotency: the follow-up key is stable, so a re-completion (or a
    re-fired approval attempt with the same key) returns the SAME card."""
    task_id, review = _platform_review_completion(conn)
    assert kb.complete_task(
        conn,
        task_id,
        metadata={"review_outcome": "approved"},
        expected_run_id=review.current_run_id,
    )
    children = _children_of(conn, task_id)
    assert len(children) == 1

    # Second approval attempt with the same idempotency key -> same card.
    again = kb.create_task(
        conn,
        title="Deploy: merge wt/t_deploytest + restart gateway",
        assignee="default",
        body="deploy",
        parents=[task_id],
        idempotency_key=f"deploy-followup:{task_id}",
    )
    assert again == children[0]
    assert _children_of(conn, task_id) == children


def test_non_platform_approval_creates_nothing(conn):
    """A worktree card NOT under the hermes-agent repo is not a platform card."""
    task_id = kb.create_task(
        conn,
        title="Fix billing API",
        assignee="builder",
        workspace_kind="worktree",
        workspace_path=str(Path("/tmp/other-repo/.worktrees/t_billing")),
        branch_name="wt/t_billing",
    )
    implementation = kb.claim_task(conn, task_id, claimer="builder:1")
    assert implementation is not None
    assert kb.request_review(
        conn, task_id, summary="ready", reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, task_id)
    assert review is not None
    assert kb.complete_task(
        conn, task_id, metadata={"review_outcome": "approved"},
        expected_run_id=review.current_run_id,
    )
    assert _children_of(conn, task_id) == []


def test_scratch_worktree_approval_creates_nothing(conn):
    """A scratch-workspace approval is not a platform worktree card."""
    task_id = kb.create_task(
        conn, title="Docs update", assignee="builder", workspace_kind="scratch",
    )
    implementation = kb.claim_task(conn, task_id, claimer="builder:1")
    assert implementation is not None
    assert kb.request_review(
        conn, task_id, summary="ready", reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, task_id)
    assert review is not None
    assert kb.complete_task(
        conn, task_id, metadata={"review_outcome": "approved"},
        expected_run_id=review.current_run_id,
    )
    assert _children_of(conn, task_id) == []


def test_non_approval_completion_creates_nothing(conn):
    """A review that ends without review_outcome==approved creates no follow-up."""
    task_id, review = _platform_review_completion(conn)
    assert kb.complete_task(
        conn,
        task_id,
        metadata={"review_outcome": "escalated", "note": "needs human"},
        expected_run_id=review.current_run_id,
    )
    assert _children_of(conn, task_id) == []

def test_create_task_hold_sets_block_kind_and_validates(conn):
    """`block_kind` at creation (the --hold / hold=true path)."""
    held = kb.create_task(conn, title="Job parent", assignee="jobsy",
                          initial_status="blocked", block_kind="operator_hold")
    t = kb.get_task(conn, held)
    assert t.status == "blocked" and t.block_kind == "operator_hold"
    with pytest.raises(ValueError):
        kb.create_task(conn, title="bad", block_kind="operator_hold")  # needs initial_status=blocked
    with pytest.raises(ValueError):
        kb.create_task(conn, title="bad", initial_status="blocked", block_kind="not_a_kind")
    # a held card is not promoted by parent-readiness recomputation — even one
    # parked by hand in SQL with no `blocked` event row (the 09-02 park scripts)
    kb.recompute_ready(conn)
    assert kb.get_task(conn, held).status == "blocked"
    by_hand = kb.create_task(conn, title="parked by SQL", assignee="jobsy")
    conn.execute("UPDATE tasks SET status='blocked', block_kind='operator_hold' WHERE id=?", (by_hand,))
    conn.commit()
    kb.recompute_ready(conn)
    assert kb.get_task(conn, by_hand).status == "blocked"
    # and a human unblock releases it to the work pool
    assert kb.unblock_task(conn, held)
    assert kb.get_task(conn, held).status in ("ready", "todo")


def test_worktree_toolchain_provisioning_symlinks_only_what_exists(tmp_path: Path):
    repo = tmp_path / "repo"; (repo / "venv" / "bin").mkdir(parents=True)
    (repo / "node_modules").mkdir()
    wt = tmp_path / "repo" / ".worktrees" / "t_x"; wt.mkdir(parents=True)
    linked = kb._provision_worktree_toolchain(repo, wt)
    assert sorted(linked) == ["node_modules", "venv"]
    assert (wt / "venv").is_symlink() and (wt / "venv" / "bin").is_dir()
    assert not (wt / ".venv").exists()
    # idempotent, and never clobbers an existing dir
    assert kb._provision_worktree_toolchain(repo, wt) == []
