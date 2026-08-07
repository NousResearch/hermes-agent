
import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb
from hermes_cli.kanban_swarm import (
    SwarmWorkerSpec,
    create_swarm,
    latest_blackboard,
    post_blackboard_update,
)


def test_create_swarm_builds_parallel_workers_verifier_and_synthesizer(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Map the target market and produce a decision memo.",
            workers=[
                SwarmWorkerSpec(profile="researcher-a", title="Market scan", body="Find competitors"),
                SwarmWorkerSpec(profile="researcher-b", title="Customer scan", body="Find customer pains"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            tenant="intel",
            created_by="orchestrator",
        )

        root = kb.get_task(conn, created.root_id)
        workers = [kb.get_task(conn, tid) for tid in created.worker_ids]
        verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)

        assert root.status == "done"
        assert root.assignee == "orchestrator"
        assert [task.status for task in workers] == ["ready", "ready"]
        assert [task.assignee for task in workers] == ["researcher-a", "researcher-b"]
        assert verifier.status == "todo"
        assert synthesizer.status == "todo"
        assert set(kb.parent_ids(conn, created.verifier_id)) == set(created.worker_ids)
        assert kb.parent_ids(conn, created.synthesizer_id) == [created.verifier_id]
        assert all(created.root_id in (task.body or "") for task in workers)
    finally:
        conn.close()


def test_swarm_blackboard_merges_structured_updates(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Collect evidence.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Evidence", body="Find proof")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        post_blackboard_update(
            conn,
            created.root_id,
            author="researcher",
            key="sources",
            value=["https://example.com/a"],
        )
        post_blackboard_update(
            conn,
            created.root_id,
            author="reviewer",
            key="risks",
            value={"missing_primary_source": True},
        )

        board = latest_blackboard(conn, created.root_id)
        assert board["sources"] == ["https://example.com/a"]
        assert board["risks"] == {"missing_primary_source": True}
        assert board["_authors"]["sources"] == "researcher"
    finally:
        conn.close()


def test_swarm_verifier_and_synthesis_are_dependency_gated(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Research two branches then verify and synthesize.",
            workers=[
                SwarmWorkerSpec(profile="a", title="Branch A", body="A"),
                SwarmWorkerSpec(profile="b", title="Branch B", body="B"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        kb.complete_task(
            conn,
            created.worker_ids[0],
            summary="A done",
            metadata={"confidence": 0.8},
        )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.verifier_id).status == "todo"
        assert kb.get_task(conn, created.synthesizer_id).status == "todo"

        kb.complete_task(conn, created.worker_ids[1], summary="B done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.verifier_id).status == "ready"
        assert kb.get_task(conn, created.synthesizer_id).status == "todo"

        kb.complete_task(
            conn,
            created.verifier_id,
            summary="Verified both branches",
            metadata={"gate": "pass"},
        )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.synthesizer_id).status == "ready"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Worktree swarms: every card shares one checkout
# ---------------------------------------------------------------------------

def _make_project(name="Web App", repo="/tmp/webapp"):
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name=name, folders=[repo])
        return pdb.get_project(pc, pid)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def test_worktree_swarm_shares_explicit_checkout_across_all_cards(tmp_path):
    """An explicit worktree path/branch reaches the verifier and synthesizer.

    Without this the verifier opens its own checkout and reviews an empty
    tree instead of what the worker committed.
    """
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Wire the login endpoint and review it.",
            workers=[SwarmWorkerSpec(profile="programmer", title="Wire login", body="Implement")],
            verifier_assignee="code-reviewer",
            synthesizer_assignee="writer",
            workspace_kind="worktree",
            workspace_path="/repo/checkout",
            branch_name="wt/login",
        )

        cards = [
            kb.get_task(conn, created.worker_ids[0]),
            kb.get_task(conn, created.verifier_id),
            kb.get_task(conn, created.synthesizer_id),
        ]
        assert {c.workspace_kind for c in cards} == {"worktree"}
        assert {c.workspace_path for c in cards} == {"/repo/checkout"}
        # The branch matters as much as the path: resolve_workspace only reuses
        # an existing checkout when the branch matches, and otherwise forks a
        # fresh worktree — which would defeat the shared path.
        assert {c.branch_name for c in cards} == {"wt/login"}
    finally:
        conn.close()


def test_worktree_swarm_pins_project_derived_checkout_for_downstream_cards(tmp_path):
    """With only --project, the first worker's *resolved* worktree is reused.

    ``create_task`` hands each project-linked task its own fresh
    ``<repo>/.worktrees/<task-id>``, so without pinning every card in the swarm
    would land somewhere different.
    """
    proj = _make_project()
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Add login and review it.",
            workers=[SwarmWorkerSpec(profile="programmer", title="Add login", body="Implement")],
            verifier_assignee="code-reviewer",
            synthesizer_assignee="writer",
            workspace_kind="worktree",
            project_id=proj.slug,
        )

        worker = kb.get_task(conn, created.worker_ids[0])
        verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)

        # The worker got the deterministic project worktree...
        assert worker.workspace_path == os.path.join(
            proj.primary_path, ".worktrees", worker.id
        )
        # ...and the downstream cards were pinned to that exact path + branch,
        # not to fresh dirs keyed on their own task ids.
        assert verifier.workspace_path == worker.workspace_path
        assert synthesizer.workspace_path == worker.workspace_path
        assert verifier.branch_name == worker.branch_name
        assert synthesizer.branch_name == worker.branch_name
    finally:
        conn.close()


def test_worktree_swarm_pins_a_branch_when_none_was_given(tmp_path):
    """``--workspace worktree:<path>`` with no ``--branch`` still shares a branch.

    The path alone is not enough: ``_resolve_worktree_workspace`` falls back to
    a per-card ``wt/<task-id>``, finds the worker's checkout on a different
    branch, and forks the verifier into a fresh worktree of its own.
    """
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Wire the login endpoint and review it.",
            workers=[SwarmWorkerSpec(profile="programmer", title="Wire login", body="Implement")],
            verifier_assignee="code-reviewer",
            synthesizer_assignee="writer",
            workspace_kind="worktree",
            workspace_path="/repo/checkout",  # inside a repo, no --branch
        )

        cards = [
            kb.get_task(conn, created.worker_ids[0]),
            kb.get_task(conn, created.verifier_id),
            kb.get_task(conn, created.synthesizer_id),
        ]
        # A path inside a repo is already shared by every card, so it is kept
        # verbatim — only the missing branch had to be settled.
        assert {c.workspace_path for c in cards} == {"/repo/checkout"}
        assert {c.branch_name for c in cards} == {f"wt/{created.root_id}"}
    finally:
        conn.close()


def test_worktree_swarm_worker_and_verifier_resolve_to_one_checkout(tmp_path):
    """End-to-end at the resolver: the verifier opens the worker's checkout.

    This is the bug in one assertion — dispatch, not card creation, is where
    the fork used to happen, so the cards are run through
    ``_resolve_worktree_workspace`` against a real repo.
    """
    repo = _make_repo(tmp_path)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Add login and review it.",
            workers=[SwarmWorkerSpec(profile="programmer", title="Add login", body="Implement")],
            verifier_assignee="code-reviewer",
            synthesizer_assignee="writer",
            workspace_kind="worktree",
            workspace_path=str(repo),  # a repo root, no --branch
        )
        worker = kb.get_task(conn, created.worker_ids[0])
        verifier = kb.get_task(conn, created.verifier_id)
    finally:
        conn.close()

    worker_ws, worker_branch = kb._resolve_worktree_workspace(worker)
    (worker_ws / "login.py").write_text("def login(): ...\n", encoding="utf-8")
    _git(worker_ws, "add", "login.py")
    _git(worker_ws, "commit", "-m", "wire login")

    verifier_ws, verifier_branch = kb._resolve_worktree_workspace(verifier)

    assert verifier_ws.resolve() == worker_ws.resolve()
    assert verifier_branch == worker_branch == f"wt/{created.root_id}"
    # The whole point: there is something to review.
    assert (verifier_ws / "login.py").exists()
    # A repo root is not itself a worktree, so the swarm was anchored on one
    # dir under it instead of letting each card claim `.worktrees/<its own id>`.
    assert worker_ws.resolve() == (repo / ".worktrees" / created.root_id).resolve()


def test_bare_worktree_swarm_shares_the_board_default_checkout(tmp_path):
    """``--workspace worktree`` with no path anchors on the board default once.

    ``create_task`` copies the board's ``default_workdir`` onto each card, and
    that is a repo root, so dispatch would still hand every card its own
    ``.worktrees/<task-id>``.
    """
    repo = _make_repo(tmp_path)
    kb.write_board_metadata(None, default_workdir=str(repo))
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Patch the parser and review it.",
            workers=[SwarmWorkerSpec(profile="programmer", title="Patch parser", body="Implement")],
            verifier_assignee="code-reviewer",
            synthesizer_assignee="writer",
            workspace_kind="worktree",
        )
        cards = [
            kb.get_task(conn, created.worker_ids[0]),
            kb.get_task(conn, created.verifier_id),
            kb.get_task(conn, created.synthesizer_id),
        ]
    finally:
        conn.close()

    expected = str(repo / ".worktrees" / created.root_id)
    assert {c.workspace_path for c in cards} == {expected}
    assert {c.branch_name for c in cards} == {f"wt/{created.root_id}"}


def test_project_swarm_on_default_scratch_workspace_shares_one_checkout(tmp_path):
    """``--project`` without ``--workspace`` is a worktree swarm too.

    ``create_task`` promotes a resolved project link from ``scratch`` to
    ``worktree`` on its own, which used to happen *after* the swarm had already
    decided not to pin anything.
    """
    proj = _make_project()
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Add login and review it.",
            workers=[SwarmWorkerSpec(profile="programmer", title="Add login", body="Implement")],
            verifier_assignee="code-reviewer",
            synthesizer_assignee="writer",
            project_id=proj.slug,  # workspace_kind left at its "scratch" default
        )
        worker = kb.get_task(conn, created.worker_ids[0])
        verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)

        assert worker.workspace_kind == "worktree"
        assert worker.workspace_path == os.path.join(
            proj.primary_path, ".worktrees", worker.id
        )
        for card in (verifier, synthesizer):
            assert card.workspace_kind == "worktree"
            assert card.workspace_path == worker.workspace_path
            assert card.branch_name == worker.branch_name
    finally:
        conn.close()


def test_project_swarm_on_default_scratch_workspace_rejects_multiple_workers(tmp_path):
    """The one-worker guard must see through the project promotion as well."""
    proj = _make_project()
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        with pytest.raises(ValueError, match="exactly one"):
            create_swarm(
                conn,
                goal="Two people editing one branch.",
                workers=[
                    SwarmWorkerSpec(profile="programmer", title="A", body="a"),
                    SwarmWorkerSpec(profile="programmer", title="B", body="b"),
                ],
                verifier_assignee="code-reviewer",
                synthesizer_assignee="writer",
                project_id=proj.slug,
            )
    finally:
        conn.close()


def test_unresolvable_project_swarm_stays_on_scratch(tmp_path):
    """A dangling project link is dropped by ``create_task`` — don't promote it.

    Otherwise parallel scratch workers would start being rejected because of a
    project that does not exist.
    """
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Two independent research tracks.",
            workers=[
                SwarmWorkerSpec(profile="researcher-a", title="A", body="a"),
                SwarmWorkerSpec(profile="researcher-b", title="B", body="b"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            project_id="no-such-project",
        )
        cards = [kb.get_task(conn, tid) for tid in created.worker_ids]
        cards.append(kb.get_task(conn, created.verifier_id))
        assert {c.workspace_kind for c in cards} == {"scratch"}
        assert {c.branch_name for c in cards} == {None}
    finally:
        conn.close()


def test_worktree_swarm_rejects_multiple_workers(tmp_path):
    """Parallel workers on one checkout would clobber each other — fail loudly."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        with pytest.raises(ValueError, match="exactly one"):
            create_swarm(
                conn,
                goal="Two people editing one branch.",
                workers=[
                    SwarmWorkerSpec(profile="programmer", title="A", body="a"),
                    SwarmWorkerSpec(profile="programmer", title="B", body="b"),
                ],
                verifier_assignee="code-reviewer",
                synthesizer_assignee="writer",
                workspace_kind="worktree",
                workspace_path="/repo/checkout",
            )
    finally:
        conn.close()


def test_scratch_swarm_keeps_independent_workspaces(tmp_path):
    """Regression: the default swarm is unchanged — no branch, no sharing."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Two independent research tracks.",
            workers=[
                SwarmWorkerSpec(profile="researcher-a", title="A", body="a"),
                SwarmWorkerSpec(profile="researcher-b", title="B", body="b"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        cards = [kb.get_task(conn, tid) for tid in created.worker_ids]
        cards += [kb.get_task(conn, created.verifier_id),
                  kb.get_task(conn, created.synthesizer_id)]
        assert {c.workspace_kind for c in cards} == {"scratch"}
        assert {c.branch_name for c in cards} == {None}
    finally:
        conn.close()
