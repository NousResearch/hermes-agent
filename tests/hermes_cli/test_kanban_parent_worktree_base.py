"""Regression coverage for parent-derived Kanban worktree bases."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "kanban@example.com")
    _git(repo, "config", "user.name", "Kanban Test")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_single_parent_worktree_starts_at_completed_parent_commit(tmp_path: Path) -> None:
    repo, stale_head = _repo(tmp_path)
    db = kb.connect(tmp_path / "kanban.db")
    try:
        parent_id = kb.create_task(db, title="parent", assignee="builder")
        parent = kb.claim_task(db, parent_id)
        assert parent is not None

        (repo / "parent.txt").write_text("reviewed parent work\n", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "parent work")
        parent_sha = _git(repo, "rev-parse", "HEAD")
        assert kb.complete_task(
            db,
            parent_id,
            summary="reviewed",
            metadata={"commit": parent_sha},
            expected_run_id=parent.current_run_id,
        )

        _git(repo, "reset", "--hard", stale_head)
        child_id = kb.create_task(
            db,
            title="child",
            assignee="builder",
            parents=[parent_id],
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="fix/child",
        )
        child = kb.claim_task(db, child_id)
        assert child is not None

        workspace, branch = kb._resolve_worktree_workspace(child, conn=db)

        assert branch == "fix/child"
        assert _git(workspace, "rev-parse", "HEAD") == parent_sha
        run = kb.get_run(db, child.current_run_id)
        assert run is not None
        assert run.resolved_base_sha == parent_sha
        claimed = [event for event in kb.list_events(db, child_id) if event.kind == "claimed"][-1]
        assert claimed.payload["resolved_base_sha"] == parent_sha
    finally:
        db.close()


@pytest.mark.parametrize("commit", [None, "abc123", "f" * 40])
def test_single_parent_worktree_rejects_unusable_parent_commit(
    tmp_path: Path, commit: str | None
) -> None:
    repo, _ = _repo(tmp_path)
    db = kb.connect(tmp_path / "kanban.db")
    try:
        parent_id = kb.create_task(db, title="parent", assignee="builder")
        parent = kb.claim_task(db, parent_id)
        assert parent is not None
        metadata = {"commit": commit} if commit is not None else {}
        assert kb.complete_task(
            db,
            parent_id,
            metadata=metadata,
            expected_run_id=parent.current_run_id,
        )
        child_id = kb.create_task(
            db,
            title="child",
            parents=[parent_id],
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        child = kb.claim_task(db, child_id)
        assert child is not None

        with pytest.raises(ValueError, match="metadata.commit|unavailable"):
            kb._resolve_worktree_workspace(child, conn=db)
    finally:
        db.close()


def test_dispatch_records_parent_base_diagnostic_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = _repo(tmp_path)
    db = kb.connect(tmp_path / "kanban.db")
    spawned = False

    def spawn(*_args, **_kwargs):
        nonlocal spawned
        spawned = True
        return 1

    try:
        parent_id = kb.create_task(db, title="parent", assignee="dev")
        assert kb.complete_task(db, parent_id, metadata={"commit": "abc123"})
        child_id = kb.create_task(
            db,
            title="child",
            assignee="dev",
            parents=[parent_id],
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)

        result = kb._dispatch_once_locked(db, spawn_fn=spawn, max_spawn=1)

        assert spawned is False
        assert not result.spawned
        failure = [
            event
            for event in kb.list_events(db, child_id)
            if event.kind == "spawn_failed"
        ][-1]
        assert "metadata.commit" in failure.payload["error"]
    finally:
        db.close()


def test_multi_parent_worktree_rejects_implicit_base(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    db = kb.connect(tmp_path / "kanban.db")
    try:
        parents = []
        for title in ("left", "right"):
            parent_id = kb.create_task(db, title=title, assignee="builder")
            assert kb.complete_task(db, parent_id, metadata={"commit": _git(repo, "rev-parse", "HEAD")})
            parents.append(parent_id)
        child_id = kb.create_task(
            db,
            title="integration",
            parents=parents,
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        child = kb.claim_task(db, child_id)
        assert child is not None

        with pytest.raises(ValueError, match="2 parents.*ambiguous"):
            kb._resolve_worktree_workspace(child, conn=db)
    finally:
        db.close()


@pytest.mark.parametrize("dirty", [False, True])
def test_existing_worktree_not_descending_from_parent_is_preserved(
    tmp_path: Path, dirty: bool
) -> None:
    repo, stale_head = _repo(tmp_path)
    existing = repo / ".worktrees" / "child"
    _git(repo, "worktree", "add", "-b", "fix/child", str(existing), stale_head)

    (repo / "parent.txt").write_text("parent\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "parent")
    parent_sha = _git(repo, "rev-parse", "HEAD")
    if dirty:
        (existing / "wip.txt").write_text("keep me\n", encoding="utf-8")

    db = kb.connect(tmp_path / "kanban.db")
    try:
        parent_id = kb.create_task(db, title="parent", assignee="builder")
        assert kb.complete_task(db, parent_id, metadata={"commit": parent_sha})
        child_id = kb.create_task(
            db,
            title="child",
            parents=[parent_id],
            workspace_kind="worktree",
            workspace_path=str(existing),
            branch_name="fix/child",
        )
        child = kb.claim_task(db, child_id)
        assert child is not None

        with pytest.raises(RuntimeError, match=f"existing {'dirty' if dirty else 'clean'} worktree"):
            kb._resolve_worktree_workspace(child, conn=db)

        assert _git(existing, "rev-parse", "HEAD") == stale_head
        if dirty:
            assert (existing / "wip.txt").read_text(encoding="utf-8") == "keep me\n"
    finally:
        db.close()


def test_canonical_child_worktree_on_wrong_branch_still_checks_parent_base(
    tmp_path: Path,
) -> None:
    repo, stale_head = _repo(tmp_path)
    (repo / "parent.txt").write_text("parent\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "parent")
    parent_sha = _git(repo, "rev-parse", "HEAD")
    db = kb.connect(tmp_path / "kanban.db")
    try:
        parent_id = kb.create_task(db, title="parent", assignee="builder")
        assert kb.complete_task(db, parent_id, metadata={"commit": parent_sha})
        child_id = kb.create_task(
            db,
            title="child",
            parents=[parent_id],
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="fix/child",
        )
        canonical = repo / ".worktrees" / child_id
        _git(repo, "worktree", "add", "-b", "wrong/branch", str(canonical), stale_head)
        kb.set_workspace_path(db, child_id, canonical)
        child = kb.claim_task(db, child_id)
        assert child is not None

        with pytest.raises(RuntimeError, match="does not descend"):
            kb._resolve_worktree_workspace(child, conn=db)
        assert _git(canonical, "rev-parse", "HEAD") == stale_head
    finally:
        db.close()
