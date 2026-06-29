"""Guard: the Kanban dispatcher must refuse to spawn a *mutating* coder or
reviewer worker whose resolved workspace is a protected
``crypto-bot-platform-service`` checkout.

A mutating agent in the live working tree can clobber uncommitted state, fight
a human's branch, or corrupt the shared checkout. Coder/reviewer work must run
in an isolated git worktree (``.../.worktrees/<task>``). Inspection-only roles
(architect / research) may still read these checkouts, so the guard is
role-scoped.

Protected:
  * exactly ``/Users/rwest/work/crypto-bot-platform-service`` (the root path
    itself — but NOT its descendants, so the sanctioned
    ``<root>/.worktrees/<task>`` worktree stays allowed), and
  * anything at or under ``/Users/rwest/service-checkouts/crypto-bot-platform-service``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# kanban_home fixture (mirrors tests/hermes_cli/test_kanban_db.py)
# ---------------------------------------------------------------------------
@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "kanban@example.com"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Kanban Test"], check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Pure guard: workspace_is_protected (uses the REAL protected constants; pure
# string comparison, no filesystem mutation).
# ---------------------------------------------------------------------------
def test_exact_work_root_is_protected():
    assert kb.workspace_is_protected(
        "/Users/rwest/work/crypto-bot-platform-service"
    )


def test_work_root_isolated_worktree_is_allowed():
    # Only the work-root path itself is protected, NOT its descendants — so the
    # sanctioned per-task worktree under it stays allowed.
    assert not kb.workspace_is_protected(
        "/Users/rwest/work/crypto-bot-platform-service/.worktrees/task-123"
    )


def test_service_checkouts_root_is_protected():
    assert kb.workspace_is_protected(
        "/Users/rwest/service-checkouts/crypto-bot-platform-service"
    )


def test_service_checkouts_descendant_is_protected():
    assert kb.workspace_is_protected(
        "/Users/rwest/service-checkouts/crypto-bot-platform-service/src/app"
    )


def test_service_checkouts_worktree_descendant_is_protected():
    # The whole service-checkouts subtree is protected — even a .worktrees dir
    # under it. The sanctioned escape hatch is the work-root worktree.
    assert kb.workspace_is_protected(
        "/Users/rwest/service-checkouts/crypto-bot-platform-service/.worktrees/t1"
    )


def test_unrelated_path_is_not_protected():
    assert not kb.workspace_is_protected("/Users/rwest/work/some-other-service")
    assert not kb.workspace_is_protected("/tmp/whatever")


def test_sibling_path_prefix_is_not_protected():
    # Guard must not treat a sibling that merely shares the string prefix as
    # being "under" the protected tree.
    assert not kb.workspace_is_protected(
        "/Users/rwest/service-checkouts/crypto-bot-platform-service-staging"
    )


def test_symlink_alias_to_protected_tree_is_protected(tmp_path, monkeypatch):
    protected_tree = tmp_path / "service-checkouts" / "crypto-bot-platform-service"
    protected_tree.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(protected_tree, target_is_directory=True)
    _patch_protected_root(monkeypatch, trees=[protected_tree])
    assert kb.workspace_is_protected(alias)


@pytest.mark.skipif(__import__("sys").platform != "darwin", reason="macOS case-insensitive path hardening")
def test_macos_case_variant_of_protected_path_is_protected(monkeypatch):
    monkeypatch.setattr(kb, "PROTECTED_WORKSPACE_EXACT", ("/Users/rwest/work/crypto-bot-platform-service",))
    assert kb.workspace_is_protected("/Users/rwest/WORK/crypto-bot-platform-service")


# ---------------------------------------------------------------------------
# Pure guard: check_protected_workspace (role-scoped refusal message).
# ---------------------------------------------------------------------------
def test_coder_in_protected_dir_is_refused_with_clear_message():
    msg = kb.check_protected_workspace(
        task_id="t1",
        assignee="coder",
        workspace="/Users/rwest/work/crypto-bot-platform-service",
        lane="ready",
    )
    assert msg is not None
    # The message must instruct the operator to use an isolated worktree.
    assert ".worktrees/t1" in msg
    assert "isolated" in msg.lower()
    assert "worktree" in msg.lower()


def test_reviewer_review_lane_in_protected_subtree_is_refused():
    msg = kb.check_protected_workspace(
        task_id="t2",
        assignee="anyone",  # review lane is always a mutating reviewer role
        workspace="/Users/rwest/service-checkouts/crypto-bot-platform-service/svc",
        lane="review",
    )
    assert msg is not None
    assert "worktree" in msg.lower()


def test_coder_in_isolated_worktree_is_allowed():
    assert kb.check_protected_workspace(
        task_id="t3",
        assignee="coder",
        workspace="/Users/rwest/work/crypto-bot-platform-service/.worktrees/t3",
        lane="ready",
    ) is None


def test_architect_may_inspect_protected_checkout():
    assert kb.check_protected_workspace(
        task_id="t4",
        assignee="architect",
        workspace="/Users/rwest/work/crypto-bot-platform-service",
        lane="ready",
    ) is None


def test_research_may_inspect_protected_checkout():
    assert kb.check_protected_workspace(
        task_id="t5",
        assignee="crypto-research",  # token-based: "research" present
        workspace="/Users/rwest/service-checkouts/crypto-bot-platform-service",
        lane="ready",
    ) is None


def test_hybrid_research_coder_is_guarded_as_mutating():
    assert kb.check_protected_workspace(
        task_id="t5a",
        assignee="research-coder",
        workspace="/Users/rwest/work/crypto-bot-platform-service",
        lane="ready",
    ) is not None


def test_unrelated_ready_lane_assignee_is_not_guarded_yet():
    # The first implementation is intentionally scoped to coder/reviewer roles.
    assert kb.check_protected_workspace(
        task_id="t5b",
        assignee="designer",
        workspace="/Users/rwest/work/crypto-bot-platform-service",
        lane="ready",
    ) is None


def test_coder_outside_protected_is_allowed():
    assert kb.check_protected_workspace(
        task_id="t6",
        assignee="coder",
        workspace="/Users/rwest/work/some-other-service",
        lane="ready",
    ) is None


# ---------------------------------------------------------------------------
# DispatchResult schema invariant.
# ---------------------------------------------------------------------------
def test_dispatch_result_has_skipped_protected_workspace_field():
    r = kb.DispatchResult()
    assert hasattr(r, "skipped_protected_workspace")
    assert r.skipped_protected_workspace == []


# ---------------------------------------------------------------------------
# Dispatch integration. To avoid ever touching the real protected checkouts,
# the protected roots are monkeypatched onto fake tmp paths.
# ---------------------------------------------------------------------------
def _patch_protected_root(monkeypatch, exact=(), trees=()):
    monkeypatch.setattr(kb, "PROTECTED_WORKSPACE_EXACT", tuple(str(p) for p in exact))
    monkeypatch.setattr(kb, "PROTECTED_WORKSPACE_TREES", tuple(str(p) for p in trees))


def test_dispatch_blocks_coder_dir_workspace_at_protected_root(kanban_home, tmp_path, monkeypatch):
    """A coder ``dir`` task resolving to the protected root is refused: not
    spawned, blocked, and the protected dir is never materialized."""
    import hermes_cli.profiles as profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)

    protected_root = tmp_path / "fake-work" / "crypto-bot-platform-service"
    _patch_protected_root(monkeypatch, exact=[protected_root])

    spawns: list[tuple[str, str]] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append((task.id, workspace))
        return 4242

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="mutate the live checkout",
            assignee="coder",
            workspace_kind="dir",
            workspace_path=str(protected_root),
        )
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        task = kb.get_task(conn, tid)

    assert spawns == [], "guarded coder must NOT spawn into the protected checkout"
    assert not any(s[0] == tid for s in result.spawned)
    assert any(t[0] == tid for t in result.skipped_protected_workspace)
    assert task is not None and task.status == "blocked"
    # The guard previews without materializing — the protected dir stays absent.
    assert not protected_root.exists()


def test_dispatch_allows_coder_isolated_worktree_under_protected_root(kanban_home, tmp_path, monkeypatch):
    """Even when the repo root is protected, a ``worktree`` task anchored on it
    resolves to ``<root>/.worktrees/<task>`` — an isolated worktree — and is
    allowed to spawn."""
    import hermes_cli.profiles as profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)

    repo = tmp_path / "crypto-bot-platform-service"
    _init_git_repo(repo)
    # The repo root itself is protected; the per-task worktree under it is not.
    _patch_protected_root(monkeypatch, exact=[repo])
    kb.create_board("wt-allowed", default_workdir=str(repo))

    spawns: list[tuple[str, str]] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append((task.id, workspace))
        return 99

    with kb.connect(board="wt-allowed") as conn:
        tid = kb.create_task(
            conn,
            title="ship in an isolated worktree",
            assignee="coder",
            workspace_kind="worktree",
            board="wt-allowed",
        )
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn, board="wt-allowed")
        task = kb.get_task(conn, tid)

    expected = repo / ".worktrees" / tid
    assert spawns == [(tid, str(expected))]
    assert result.spawned == [(tid, "coder", str(expected))]
    assert not any(t[0] == tid for t in result.skipped_protected_workspace)
    assert task is not None and task.status == "running"


def test_dispatch_blocks_reviewer_in_review_lane_at_protected_subtree(kanban_home, tmp_path, monkeypatch):
    """A review-lane task whose workspace is under the protected subtree is
    refused before the review agent spawns."""
    import hermes_cli.profiles as profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)

    protected_tree = tmp_path / "service-checkouts" / "crypto-bot-platform-service"
    _patch_protected_root(monkeypatch, trees=[protected_tree])

    spawns: list[tuple[str, str]] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append((task.id, workspace))
        return 7

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="review the PR",
            assignee="reviewer",
            workspace_kind="dir",
            workspace_path=str(protected_tree / "checkout"),
        )
        # Move the task into the review lane (PR opened), claim_lock cleared.
        conn.execute(
            "UPDATE tasks SET status='review', claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        task = kb.get_task(conn, tid)

    assert spawns == [], "guarded reviewer must NOT spawn into the protected subtree"
    assert any(t[0] == tid for t in result.skipped_protected_workspace)
    assert task is not None and task.status == "blocked"
    assert not (protected_tree / "checkout").exists()


def test_dispatch_allows_architect_inspecting_protected_dir(kanban_home, tmp_path, monkeypatch):
    """An architect (inspection-only) task may run against the protected
    checkout — the guard does not block read-only roles."""
    import hermes_cli.profiles as profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)

    protected_root = tmp_path / "service-checkouts" / "crypto-bot-platform-service"
    _patch_protected_root(monkeypatch, trees=[protected_root])

    spawns: list[tuple[str, str]] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append((task.id, workspace))
        return 11

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="inspect architecture",
            assignee="architect",
            workspace_kind="dir",
            workspace_path=str(protected_root),
        )
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        task = kb.get_task(conn, tid)

    assert spawns == [(tid, str(protected_root))]
    assert result.spawned == [(tid, "architect", str(protected_root))]
    assert not any(t[0] == tid for t in result.skipped_protected_workspace)
    assert task is not None and task.status == "running"


def test_preview_resolution_error_falls_back_to_normal_spawn_failure(
    kanban_home, tmp_path, monkeypatch
):
    """Malformed workspaces must not abort the dispatcher tick before the
    existing workspace-error path can record a failure and continue."""
    import hermes_cli.profiles as profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)

    spawns: list[tuple[str, str]] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append((task.id, workspace))
        return 12

    with kb.connect() as conn:
        bad_id = kb.create_task(
            conn,
            title="bad workspace",
            assignee="coder",
            workspace_kind="dir",
            workspace_path="relative-path-is-invalid",
            priority=10,
        )
        good_id = kb.create_task(
            conn,
            title="good workspace still runs",
            assignee="coder",
            workspace_kind="dir",
            workspace_path=str(tmp_path / "good"),
            priority=1,
        )
        result = kb.dispatch_once(conn, spawn_fn=fake_spawn)
        bad_task = kb.get_task(conn, bad_id)

    assert any(task_id == good_id for task_id, _workspace in spawns)
    assert not any(task_id == bad_id for task_id, _workspace in spawns)
    assert bad_task is not None and bad_task.status == "ready"
    assert "non-absolute workspace_path" in (bad_task.last_failure_error or "")
    assert result.spawned == [(good_id, "coder", str(tmp_path / "good"))]
