"""Behavior contracts for hermes_cli.worktree_gc (attended reclaim).

Each guard gets its own contract against a REAL git repo fixture (no mocks —
the entire value of these tests is exercising actual git verdicts):

- clean + fully merged tree        → reap
- untracked-only dirt              → reap-archive (files archived, then removed)
- tracked modifications            → keep, any age
- unique unpushed commits          → keep
- patch-equivalent commits (rebase/squash-merge leak) → reap
- live-locked tree                 → keep
- kanban t_<hex> tree              → keep (owned by kanban gc)
- branch GC: merged branch deleted, unique-commit branch kept,
  checked-out branch kept, protected names kept
- reclaim operates ONLY on the frozen audit list (concurrent-session trap)
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import worktree_gc


def _git(args, cwd, env=None):
    e = dict(os.environ)
    e.update({
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    })
    if env:
        e.update(env)
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=str(cwd), env=e,
    )
    assert result.returncode == 0, f"git {args} failed: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """origin (bare) + clone with .worktrees/, HOME redirected for archives."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()

    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(["init", "--bare", "-b", "main"], origin)

    clone = tmp_path / "repo"
    _git(["clone", str(origin), str(clone)], tmp_path)
    (clone / "README.md").write_text("hello\n")
    _git(["add", "."], clone)
    _git(["commit", "-m", "init"], clone)
    _git(["push", "origin", "main"], clone)
    # origin/HEAD so upstream resolution works like a real clone.
    _git(["remote", "set-head", "origin", "main"], clone)
    (clone / ".worktrees").mkdir()
    return clone


def _add_worktree(repo_path, name, branch=None):
    tree = repo_path / ".worktrees" / name
    branch = branch or f"hermes/{name}"
    _git(["worktree", "add", str(tree), "-b", branch], repo_path)
    return tree, branch


def _verdict(records, name):
    match = [record for record in records if record.name == name]
    assert match, f"no record for {name}"
    return match[0]


def _add_managed_conversation_worktree(repo_path, tmp_path, root_id="aged-root"):
    from agent.conversation_worktree import ConversationWorktreeManager
    from agent.conversation_worktree_policy import ConversationWorktreePolicy
    from hermes_state import SessionDB

    stable_source = tmp_path / f"{root_id}-stable-source"
    _git(
        ["worktree", "add", str(stable_source), "-b", f"stable/{root_id}", "HEAD"],
        repo_path,
    )
    db = SessionDB(tmp_path / f"{root_id}-state.db")
    manager = ConversationWorktreeManager(
        ConversationWorktreePolicy(
            enabled=True,
            source_worktree=stable_source,
            worktree_root=repo_path / ".worktrees",
            branch_prefix="hermes/session",
            bootstrap=False,
            bootstrap_command=(),
            bootstrap_timeout=1.0,
            create_timeout=3.0,
            retain_until_explicit_cleanup=True,
        ),
        db,
    )
    binding = manager.bind_new_root_session(root_id, conversation_kind="interactive")
    assert binding is not None
    old = __import__("time").time() - (365 * 86400)
    os.utime(binding.path, (old, old))
    return db, binding


def test_marker_write_failure_after_worktree_add_keeps_durable_manager_claim(
    repo, tmp_path, monkeypatch
):
    """A retained failed creation must never become generic-GC fodder.

    Removing the durable common-repository claim would make this fail: the
    worktree has no per-worktree owner marker, is clean and merged, and the
    generic reclaimer would otherwise force-remove it.
    """
    from agent.conversation_worktree import (
        ConversationWorktreeError,
        ConversationWorktreeManager,
    )
    from agent.conversation_worktree_policy import ConversationWorktreePolicy
    from hermes_state import SessionDB

    stable_source = tmp_path / "failed-marker-stable-source"
    _git(
        ["worktree", "add", str(stable_source), "-b", "stable/failed-marker", "HEAD"],
        repo,
    )
    db = SessionDB(tmp_path / "failed-marker-state.db")
    worktree_manager = ConversationWorktreeManager(
        ConversationWorktreePolicy(
            enabled=True,
            source_worktree=stable_source,
            worktree_root=repo / ".worktrees",
            branch_prefix="hermes/session",
            bootstrap=False,
            bootstrap_command=(),
            bootstrap_timeout=1.0,
            create_timeout=3.0,
            retain_until_explicit_cleanup=True,
        ),
        db,
    )

    def fail_marker(_record):
        raise ConversationWorktreeError("marker unavailable", phase="create")

    original_common_claim = worktree_manager._ensure_common_owner_claim

    def claim_before_worktree_creation(record):
        assert not Path(record.worktree_path).exists()
        original_common_claim(record)

    monkeypatch.setattr(
        worktree_manager, "_ensure_common_owner_claim", claim_before_worktree_creation
    )
    monkeypatch.setattr(worktree_manager, "_ensure_owner_marker", fail_marker)
    try:
        with pytest.raises(ConversationWorktreeError, match="marker unavailable"):
            worktree_manager.bind_new_root_session(
                "failed-marker", conversation_kind="interactive"
            )

        failed = db.get_conversation_worktree("failed-marker")
        assert failed is not None
        assert failed.state == "creation_failed"
        failed_path = Path(failed.worktree_path)
        assert failed_path.is_dir()

        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        record = _verdict(records, failed_path.name)
        assert record.verdict == "keep"
        assert record.reason == "manager-owned conversation worktree"

        actions = worktree_gc.reclaim_worktrees(str(repo), records=records)
        assert failed_path.is_dir()
        assert actions == []
    finally:
        db.close()


class TestAuditVerdicts:
    def test_clean_merged_tree_reaps(self, repo):
        _add_worktree(repo, "hermes-clean")
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        assert _verdict(records, "hermes-clean").verdict == "reap"

    def test_tracked_modifications_keep(self, repo):
        tree, _ = _add_worktree(repo, "hermes-dirty")
        (tree / "README.md").write_text("edited\n")
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        record = _verdict(records, "hermes-dirty")
        assert record.verdict == "keep"
        assert "tracked" in record.reason

    def test_untracked_only_is_reap_archive(self, repo):
        tree, _ = _add_worktree(repo, "hermes-scratch")
        (tree / "PR_BODY_DRAFT.md").write_text("draft\n")
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        record = _verdict(records, "hermes-scratch")
        assert record.verdict == "reap-archive"
        assert record.untracked == ["PR_BODY_DRAFT.md"]

    def test_unique_unpushed_commits_keep(self, repo):
        tree, _ = _add_worktree(repo, "hermes-work")
        (tree / "new.py").write_text("x = 1\n")
        _git(["add", "."], tree)
        _git(["commit", "-m", "unique work"], tree)
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        record = _verdict(records, "hermes-work")
        assert record.verdict == "keep"
        assert "unpushed" in record.reason

    def test_patch_equivalent_commits_reap(self, repo):
        """The squash/rebase-merge leak: local commit unreachable from any
        remote ref but patch-equivalent to an upstream commit → merged work."""
        tree, _ = _add_worktree(repo, "hermes-merged")
        (tree / "feat.py").write_text("y = 2\n")
        _git(["add", "."], tree)
        _git(["commit", "-m", "feat"], tree)
        sha = _git(["rev-parse", "HEAD"], tree)
        # "Merge" it to main with a DIFFERENT committer so the cherry-pick
        # produces a distinct sha (same-second identical-committer cherry
        # picks can produce the identical sha — pitfall from the skill).
        _git(["cherry-pick", sha], repo,
             env={"GIT_COMMITTER_NAME": "other", "GIT_COMMITTER_EMAIL": "o@o"})
        _git(["push", "origin", "main"], repo)
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        assert _verdict(records, "hermes-merged").verdict == "reap"

    def test_live_locked_tree_keeps(self, repo):
        tree, _ = _add_worktree(repo, "hermes-live")
        _git(["worktree", "lock", str(tree),
              "--reason", f"hermes pid={os.getpid()}"], repo)
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        record = _verdict(records, "hermes-live")
        assert record.verdict == "keep"
        assert "in use" in record.reason

    def test_kanban_tree_untouched(self, repo):
        _add_worktree(repo, "t_deadbeef", branch="kanban/t_deadbeef")
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        record = _verdict(records, "t_deadbeef")
        assert record.verdict == "keep"
        assert "kanban" in record.reason

    def test_aged_manager_owned_conversation_tree_is_always_kept(self, repo, tmp_path):
        db, binding = _add_managed_conversation_worktree(repo, tmp_path)
        try:
            records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
            record = _verdict(records, binding.path.name)
            assert record.verdict == "keep"
            assert record.reason == "manager-owned conversation worktree"
            assert binding.path.exists()
        finally:
            db.close()


class TestReclaim:
    def test_reap_removes_tree_and_branch(self, repo):
        tree, branch = _add_worktree(repo, "hermes-clean")
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        actions = worktree_gc.reclaim_worktrees(str(repo), records=records)
        assert any("removed hermes-clean" in a for a in actions)
        assert not tree.exists()
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", branch],
            capture_output=True, text=True, cwd=str(repo),
        )
        assert probe.returncode != 0, "branch should be gone with its tree"

    def test_untracked_files_archived_before_removal(self, repo):
        tree, _ = _add_worktree(repo, "hermes-scratch")
        (tree / "NOTES.md").write_text("important scribbles\n")
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        worktree_gc.reclaim_worktrees(str(repo), records=records)
        assert not tree.exists()
        archive_root = Path.home() / ".hermes" / "archive" / "worktree-prune"
        archived = list(archive_root.rglob("NOTES.md"))
        assert archived, "untracked file must be archived, not destroyed"
        assert archived[0].read_text() == "important scribbles\n"

    def test_dry_run_changes_nothing(self, repo):
        tree, _ = _add_worktree(repo, "hermes-clean")
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        actions = worktree_gc.reclaim_worktrees(
            str(repo), dry_run=True, records=records
        )
        assert any("would remove" in a for a in actions)
        assert tree.exists()

    def test_frozen_list_ignores_trees_created_after_audit(self, repo):
        """Concurrent-session trap: a tree created between audit and reclaim
        must be out of scope by construction."""
        _add_worktree(repo, "hermes-old")
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        late_tree, _ = _add_worktree(repo, "hermes-late")
        worktree_gc.reclaim_worktrees(str(repo), records=records)
        assert late_tree.exists(), "tree created after the audit must survive"

    def test_frozen_reap_record_rechecks_conversation_ownership_before_remove(
        self, repo
    ):
        tree, _ = _add_worktree(repo, "hermes-raced-owner")
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        assert _verdict(records, "hermes-raced-owner").verdict == "reap"
        marker = Path(
            _git(
                ["-C", str(tree), "rev-parse", "--git-path", "hermes-conversation-owner-v1"],
                repo,
            )
        )
        marker.write_text(
            json.dumps(
                {
                    "owner": "conversation-worktree-manager",
                    "root_session_id": "raced-owner",
                    "worktree_path": str(tree.resolve()),
                    "repo_common_dir": _git(
                        [
                            "-C",
                            str(tree),
                            "rev-parse",
                            "--path-format=absolute",
                            "--git-common-dir",
                        ],
                        repo,
                    ),
                }
            ),
            encoding="utf-8",
        )

        actions = worktree_gc.reclaim_worktrees(str(repo), records=records)

        assert tree.exists()
        assert actions == ["kept hermes-raced-owner (manager-owned conversation worktree)"]

    def test_dead_locked_tree_is_unlocked_and_reaped(self, repo):
        tree, _ = _add_worktree(repo, "hermes-zombie")
        _git(["worktree", "lock", str(tree),
              "--reason", "hermes pid=999999999"], repo)
        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        assert _verdict(records, "hermes-zombie").verdict == "reap"
        worktree_gc.reclaim_worktrees(str(repo), records=records)
        assert not tree.exists()


class TestBranchGC:
    def test_merged_branch_deleted_any_name(self, repo):
        """Branch GC is content-gated, not name-gated: any fully-merged local
        branch is safe to delete regardless of prefix."""
        _git(["branch", "salv-12345", "main"], repo)
        _git(["branch", "feat/some-old-thing", "main"], repo)
        records = worktree_gc.audit_branches(str(repo))
        by_name = {record.name: record for record in records}
        assert by_name["salv-12345"].verdict == "delete"
        assert by_name["feat/some-old-thing"].verdict == "delete"
        worktree_gc.reclaim_branches(str(repo), records=records)
        out = _git(["branch", "--format=%(refname:short)"], repo)
        assert "salv-12345" not in out
        assert "feat/some-old-thing" not in out

    def test_unique_commit_branch_kept(self, repo):
        _git(["checkout", "-b", "feat/real-work"], repo)
        (repo / "wip.py").write_text("z = 3\n")
        _git(["add", "."], repo)
        _git(["commit", "-m", "wip"], repo)
        _git(["checkout", "main"], repo)
        records = worktree_gc.audit_branches(str(repo))
        by_name = {record.name: record for record in records}
        assert by_name["feat/real-work"].verdict == "keep"
        assert "unique" in by_name["feat/real-work"].reason

    def test_patch_equivalent_branch_deleted(self, repo):
        """Rebase-merged PR branch: SHAs differ from main but every commit is
        patch-equivalent — the dominant branch leak."""
        _git(["checkout", "-b", "fix/landed"], repo)
        (repo / "fix.py").write_text("a = 4\n")
        _git(["add", "."], repo)
        _git(["commit", "-m", "fix"], repo)
        sha = _git(["rev-parse", "HEAD"], repo)
        _git(["checkout", "main"], repo)
        _git(["cherry-pick", sha], repo,
             env={"GIT_COMMITTER_NAME": "other", "GIT_COMMITTER_EMAIL": "o@o"})
        _git(["push", "origin", "main"], repo)
        records = worktree_gc.audit_branches(str(repo))
        by_name = {record.name: record for record in records}
        assert by_name["fix/landed"].verdict == "delete"
        assert "patch-equivalent" in by_name["fix/landed"].reason

    def test_checked_out_and_protected_kept(self, repo):
        _tree, branch = _add_worktree(repo, "hermes-active")
        records = worktree_gc.audit_branches(str(repo))
        by_name = {record.name: record for record in records}
        assert by_name["main"].verdict == "keep"
        assert by_name[branch].verdict == "keep"
