from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import kanban_db as kb


def git(cwd: Path, *args: str) -> str:
    cp = subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)
    return cp.stdout.strip()


def make_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"; repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "guard@example.invalid")
    git(repo, "config", "user.name", "Workspace Guard Test")
    (repo / "README.md").write_text("base\n")
    git(repo, "add", "README.md"); git(repo, "commit", "-m", "base")
    worktree = tmp_path / "wt"
    git(repo, "worktree", "add", "-b", "card/t_12345678/change", str(worktree), "HEAD")
    return repo, worktree


def task(kind: str, path: Path | None, branch: str | None = None):
    if kind == "worktree" and branch is None:
        branch = "card/t_12345678/change"
    return SimpleNamespace(id="t_12345678", workspace_kind=kind,
                           workspace_path=str(path) if path is not None else None,
                           branch_name=branch)


def config(repo: Path) -> dict:
    return {"canonical_roots": [str(repo)],
            "required_worktree_branch_prefix": "card/{task_id}/"}


def test_workspace_guard_absent_config_preserves_legacy_behavior():
    assert kb._prepare_workspace_guard(None) is None
    assert kb._prepare_workspace_guard({}) is None


def test_workspace_guard_falsey_non_mapping_config_fails_closed():
    for malformed in ([], "", 0):
        state = kb._prepare_workspace_guard(malformed)
        reason = kb._workspace_guard_reason(task("scratch", None), state)
        assert reason and "kanban.dispatcher must be a mapping" in reason


def test_workspace_guard_canonical_checkout_blocks_even_when_clean(tmp_path: Path):
    repo, _ = make_repo(tmp_path)
    reason = kb._workspace_guard_reason(task("dir", repo), kb._prepare_workspace_guard(config(repo)))
    assert reason and "configured canonical root" in reason


def test_workspace_guard_dirty_canonical_checkout_blocks(tmp_path: Path):
    repo, _ = make_repo(tmp_path); (repo / "README.md").write_text("dirty\n")
    reason = kb._workspace_guard_reason(task("dir", repo), kb._prepare_workspace_guard(config(repo)))
    assert reason and "BLOCKED" in reason


def test_workspace_guard_registered_card_worktree_is_allowed(tmp_path: Path):
    repo, worktree = make_repo(tmp_path)
    state = kb._prepare_workspace_guard(config(repo))
    assert kb._workspace_guard_reason(task("worktree", worktree), state) is None


def test_workspace_guard_wrong_branch_blocks(tmp_path: Path):
    repo, worktree = make_repo(tmp_path); git(worktree, "switch", "-c", "wrong/branch")
    reason = kb._workspace_guard_reason(task("worktree", worktree), kb._prepare_workspace_guard(config(repo)))
    assert reason and "must start with card/t_12345678/" in reason


def test_workspace_guard_missing_root_fails_closed(tmp_path: Path):
    state = kb._prepare_workspace_guard(config(tmp_path / "missing"))
    reason = kb._workspace_guard_reason(task("scratch", None), state)
    assert reason and "missing canonical root" in reason


def test_workspace_guard_non_git_root_fails_closed(tmp_path: Path):
    root = tmp_path / "plain"; root.mkdir()
    state = kb._prepare_workspace_guard(config(root))
    reason = kb._workspace_guard_reason(task("scratch", None), state)
    assert reason and "non-git canonical root" in reason


def test_workspace_guard_canonical_root_symlink_loop_fails_closed(tmp_path: Path):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    state = kb._prepare_workspace_guard(config(loop))
    reason = kb._workspace_guard_reason(task("scratch", None), state)
    assert reason and "cannot resolve canonical root" in reason


def test_workspace_guard_explicit_workspace_symlink_loop_fails_closed(tmp_path: Path):
    repo, _ = make_repo(tmp_path)
    loop = tmp_path / "workspace-loop"
    loop.symlink_to(loop)
    reason = kb._workspace_guard_reason(
        task("dir", loop), kb._prepare_workspace_guard(config(repo))
    )
    assert reason and "cannot resolve workspace path" in reason


def test_workspace_guard_unmaterialized_worktree_blocks_real_start(tmp_path: Path):
    repo, _ = make_repo(tmp_path)
    reason = kb._workspace_guard_reason(task("worktree", None), kb._prepare_workspace_guard(config(repo)))
    assert reason and "no existing registered workspace_path" in reason


def test_workspace_guard_preflight_blocks_missing_or_wrong_declared_branch(tmp_path: Path):
    repo, _ = make_repo(tmp_path)
    state = kb._prepare_workspace_guard(config(repo))
    missing = task("worktree", None, branch="")
    wrong = task("worktree", None, branch="wt/t_12345678")
    assert "must declare branch_name" in (kb._workspace_guard_preflight_reason(missing, state) or "")
    assert "does not start with" in (kb._workspace_guard_preflight_reason(wrong, state) or "")


def test_workspace_guard_malformed_branch_prefix_fails_closed(tmp_path: Path):
    repo, _ = make_repo(tmp_path)
    for malformed_prefix in ("{bogus}/", "{}", "{task_id.foo}"):
        malformed = config(repo)
        malformed["required_worktree_branch_prefix"] = malformed_prefix
        state = kb._prepare_workspace_guard(malformed)
        reason = kb._workspace_guard_preflight_reason(task("worktree", None), state)
        assert reason and "invalid required_worktree_branch_prefix template" in reason


def test_workspace_guard_resolves_default_scratch_path_before_root_check(
    tmp_path: Path, monkeypatch,
):
    repo, _ = make_repo(tmp_path)
    scratch_link = tmp_path / "scratch-link"
    scratch_link.symlink_to(repo, target_is_directory=True)
    monkeypatch.setattr(kb, "workspaces_root", lambda: scratch_link)
    reason = kb._workspace_guard_reason(
        task("scratch", None), kb._prepare_workspace_guard(config(repo))
    )
    assert reason and "configured canonical root" in reason


def test_workspace_guard_refresh_observes_newly_created_worktree(tmp_path: Path):
    repo = tmp_path / "repo"; repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "guard@example.invalid")
    git(repo, "config", "user.name", "Workspace Guard Test")
    (repo / "README.md").write_text("base\n")
    git(repo, "add", "README.md"); git(repo, "commit", "-m", "base")
    stale = kb._prepare_workspace_guard(config(repo))
    worktree = tmp_path / "new-wt"
    git(repo, "worktree", "add", "-b", "card/t_12345678/change", str(worktree), "HEAD")
    assert "not a registered linked Git worktree" in (
        kb._workspace_guard_reason(task("worktree", worktree), stale) or ""
    )
    refreshed = kb._prepare_workspace_guard(config(repo))
    assert kb._workspace_guard_reason(task("worktree", worktree), refreshed) is None
