"""Fail-closed explicit cleanup for durable conversation worktrees."""

from __future__ import annotations

from pathlib import Path
import multiprocessing
import subprocess

import pytest

from agent.conversation_worktree import (
    ConversationWorktreeError,
    ConversationWorktreeManager,
    acquire_conversation_root_lease,
)
from agent.conversation_worktree_policy import ConversationWorktreePolicy
from hermes_state import SessionDB


def _hold_root_lease(binding_data: dict[str, str], ready, release) -> None:
    from agent import conversation_worktree as worktrees

    lease = worktrees.acquire_conversation_root_lease(
        root_session_id=binding_data["root_session_id"],
        worktree_path=Path(binding_data["worktree_path"]),
        repo_common_dir=Path(binding_data["repo_common_dir"]),
        surface="cleanup-test-child",
    )
    ready.set()
    release.wait(timeout=15)
    lease.release()


def git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_worktree_is_locked(repo: Path, worktree: Path) -> bool:
    current: Path | None = None
    for line in git(repo, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            current = Path(line.removeprefix("worktree ")).resolve()
        elif current == worktree.resolve() and line.startswith("locked"):
            return True
    return False


@pytest.fixture
def prepared_binding(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)

    source = tmp_path / "stable"
    source.mkdir()
    git(source, "init")
    git(source, "config", "user.email", "test@example.invalid")
    git(source, "config", "user.name", "Hermes Test")
    git(source, "checkout", "-b", "stable")
    source.joinpath("tracked.txt").write_text("base\n", encoding="utf-8")
    git(source, "add", "tracked.txt")
    git(source, "commit", "-m", "base")
    git(source, "remote", "add", "origin", str(remote))
    git(source, "push", "-u", "origin", "stable")

    database = SessionDB(tmp_path / "state.db")
    policy = ConversationWorktreePolicy(
        enabled=True,
        source_worktree=source,
        worktree_root=tmp_path / "conversation-worktrees",
        branch_prefix="hermes/session",
        bootstrap=False,
        bootstrap_command=(),
        bootstrap_timeout=1.0,
        create_timeout=3.0,
        retain_until_explicit_cleanup=True,
    )
    manager = ConversationWorktreeManager(policy, database)
    binding = manager.bind_new_root_session("root", conversation_kind="interactive")
    assert binding is not None
    git(binding.path, "config", "user.email", "test@example.invalid")
    git(binding.path, "config", "user.name", "Hermes Test")

    sibling = tmp_path / "sibling-worktree"
    git(source, "worktree", "add", "-b", "sibling", str(sibling), binding.base_commit)

    yield manager, database, source, remote, binding, sibling

    database.close()


def commit_binding(binding, message: str = "conversation change") -> str:
    binding.path.joinpath("tracked.txt").write_text(f"{message}\n", encoding="utf-8")
    git(binding.path, "add", "tracked.txt")
    git(binding.path, "commit", "-m", message)
    return git(binding.path, "rev-parse", "HEAD")


def push_binding(binding) -> None:
    git(binding.path, "push", "origin", f"HEAD:refs/heads/{binding.branch}")


def integrate_binding(source: Path, binding) -> None:
    git(source, "merge", "--ff-only", binding.branch)


def certify_safe(source: Path, binding) -> None:
    commit_binding(binding)
    push_binding(binding)
    integrate_binding(source, binding)
    git(source, "push", "origin", "stable")


def test_cleanup_reports_dirty_unintegrated_and_unpushed_together(prepared_binding):
    manager, _db, _source, _remote, binding, _sibling = prepared_binding
    commit_binding(binding)
    binding.path.joinpath("untracked.txt").write_text("keep\n", encoding="utf-8")

    verdict = manager.inspect_cleanup(binding.root_session_id)

    assert verdict.allowed is False
    assert verdict.reasons == ("dirty", "unintegrated", "unpushed")
    assert binding.path.exists()


def test_cleanup_refuses_active_session_binding(prepared_binding):
    manager, _db, source, _remote, binding, _sibling = prepared_binding
    certify_safe(source, binding)

    verdict = manager.inspect_cleanup(
        binding.root_session_id, active_session_bound=True
    )

    assert verdict.allowed is False
    assert verdict.reasons == ("active",)


def test_cleanup_refuses_in_progress_git_state(prepared_binding):
    manager, _db, source, _remote, binding, _sibling = prepared_binding
    certify_safe(source, binding)
    merge_head = Path(git(binding.path, "rev-parse", "--git-path", "MERGE_HEAD"))
    merge_head.write_text(binding.base_commit + "\n", encoding="utf-8")

    verdict = manager.inspect_cleanup(binding.root_session_id)

    assert "in-progress" in verdict.reasons
    assert binding.path.exists()


@pytest.mark.parametrize(
    "marker",
    ("index.lock", "HEAD.lock", "packed-refs.lock", "branch-ref.lock"),
)
def test_cleanup_refuses_real_git_lock_files(prepared_binding, marker):
    manager, _db, source, _remote, binding, _sibling = prepared_binding
    certify_safe(source, binding)
    if marker == "branch-ref.lock":
        lock_path = Path(
            git(
                binding.path,
                "rev-parse",
                "--git-path",
                f"refs/heads/{binding.branch}.lock",
            )
        )
    else:
        lock_path = Path(git(binding.path, "rev-parse", "--git-path", marker))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("held\n", encoding="utf-8")

    verdict = manager.inspect_cleanup(binding.root_session_id)

    assert "in-progress" in verdict.reasons
    assert binding.path.exists()


def test_cleanup_refuses_root_lease_held_by_another_process(prepared_binding):
    manager, _db, source, _remote, binding, _sibling = prepared_binding
    certify_safe(source, binding)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_root_lease,
        args=(
            {
                "root_session_id": binding.root_session_id,
                "worktree_path": str(binding.path),
                "repo_common_dir": str(binding.repo_common_dir),
            },
            ready,
            release,
        ),
    )
    process.start()
    try:
        assert ready.wait(timeout=10), "child did not acquire root lease"

        verdict = manager.inspect_cleanup(binding.root_session_id)

        assert verdict.allowed is False
        assert verdict.reasons == ("active",)
        assert binding.path.exists()
    finally:
        release.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_cleanup_refuses_corrupt_root_lease_registry_as_unknown(prepared_binding):
    manager, _db, source, _remote, binding, _sibling = prepared_binding
    certify_safe(source, binding)
    from agent.conversation_worktree import _root_lease_paths

    state_path, _lock_path = _root_lease_paths(
        binding.repo_common_dir, binding.root_session_id
    )
    state_path.write_text("not-json", encoding="utf-8")

    verdict = manager.inspect_cleanup(binding.root_session_id)

    assert verdict.allowed is False
    assert verdict.reasons == ("unknown",)


def test_root_lease_lock_failure_is_controlled(monkeypatch, prepared_binding):
    _manager, _db, _source, _remote, binding, _sibling = prepared_binding
    from agent import conversation_worktree as worktrees

    @worktrees.contextmanager
    def unavailable(_path, *, timeout):
        raise OSError("denied")
        yield

    monkeypatch.setattr(worktrees, "_lease_file_lock", unavailable)

    with pytest.raises(ConversationWorktreeError) as exc_info:
        acquire_conversation_root_lease(
            root_session_id=binding.root_session_id,
            worktree_path=binding.path,
            repo_common_dir=binding.repo_common_dir,
            surface="test",
        )

    assert exc_info.value.phase == "lease"
    assert str(exc_info.value) == "conversation root lease registry is unavailable"


def test_root_lease_release_fails_closed_when_registry_is_uncertain(prepared_binding):
    _manager, _db, _source, _remote, binding, _sibling = prepared_binding
    lease = acquire_conversation_root_lease(
        root_session_id=binding.root_session_id,
        worktree_path=binding.path,
        repo_common_dir=binding.repo_common_dir,
        surface="test",
    )
    lease.state_path.write_text("{corrupt", encoding="utf-8")

    with pytest.raises(ConversationWorktreeError) as exc_info:
        lease.release()

    assert exc_info.value.phase == "lease"
    assert lease.released is False


def test_cleanup_distinguishes_unintegrated_from_unpushed(prepared_binding):
    manager, _db, source, _remote, binding, _sibling = prepared_binding
    commit_binding(binding)
    push_binding(binding)

    verdict = manager.inspect_cleanup(binding.root_session_id)

    assert "unintegrated" in verdict.reasons
    assert "unpushed" not in verdict.reasons
    assert source.exists()


def test_cleanup_refuses_integrated_but_unpushed_head(prepared_binding):
    manager, _db, source, _remote, binding, _sibling = prepared_binding
    commit_binding(binding)
    integrate_binding(source, binding)

    verdict = manager.inspect_cleanup(binding.root_session_id)

    assert "unintegrated" not in verdict.reasons
    assert "unpushed" in verdict.reasons


def test_cleanup_refuses_mismatched_worktree_identity(prepared_binding):
    manager, _db, _source, _remote, binding, _sibling = prepared_binding
    git(binding.path, "checkout", "--detach")

    verdict = manager.inspect_cleanup(binding.root_session_id)

    assert verdict.allowed is False
    assert verdict.reasons == ("mismatched identity",)
    assert binding.path.exists()


def test_cleanup_refuses_missing_remote_evidence(prepared_binding):
    manager, _db, _source, _remote, binding, _sibling = prepared_binding
    git(binding.path, "remote", "remove", "origin")

    verdict = manager.inspect_cleanup(binding.root_session_id)

    assert "missing remote evidence" in verdict.reasons
    assert binding.path.exists()


def test_cleanup_maps_git_inspection_failure_to_unknown(prepared_binding, monkeypatch):
    manager, _db, _source, _remote, binding, _sibling = prepared_binding
    original = manager._run_git

    def fail_status(cwd, args, timeout, phase):
        if args[:1] == ["status"]:
            raise ConversationWorktreeError("git inspection failed", phase=phase)
        return original(cwd, args, timeout, phase)

    monkeypatch.setattr(manager, "_run_git", fail_status)

    verdict = manager.inspect_cleanup(binding.root_session_id)

    assert verdict.allowed is False
    assert "unknown" in verdict.reasons
    assert binding.path.exists()


def test_explicit_cleanup_removes_only_exact_safe_worktree(prepared_binding):
    manager, db, source, _remote, binding, sibling = prepared_binding
    certify_safe(source, binding)
    from agent.conversation_worktree import _common_owner_claim_path

    common_claim = _common_owner_claim_path(binding.repo_common_dir, binding.path)
    assert common_claim.exists()
    claimed = db.get_conversation_worktree(binding.root_session_id)
    assert claimed is not None

    result = manager.remove_after_explicit_request(binding.root_session_id)

    assert result.removed is True
    assert result.verdict.allowed is True
    assert not binding.path.exists()
    assert sibling.exists()
    assert source.exists()
    assert git(source, "show-ref", "--verify", f"refs/heads/{binding.branch}")
    removed = db.get_conversation_worktree(binding.root_session_id)
    assert removed is not None
    assert removed.state == "removed"
    assert removed.worktree_path == claimed.worktree_path
    assert removed.branch == claimed.branch
    worktrees = git(source, "worktree", "list", "--porcelain")
    assert str(binding.path) not in worktrees
    assert str(sibling) in worktrees
    assert not common_claim.exists()


def test_blocked_cleanup_never_removes_or_marks_binding(prepared_binding):
    manager, db, _source, _remote, binding, sibling = prepared_binding
    binding.path.joinpath("untracked.txt").write_text("keep\n", encoding="utf-8")

    result = manager.remove_after_explicit_request(binding.root_session_id)

    assert result.removed is False
    assert "dirty" in result.verdict.reasons
    assert binding.path.exists()
    assert sibling.exists()
    record = db.get_conversation_worktree(binding.root_session_id)
    assert record is not None
    assert record.state == "ready"


def test_remove_failure_returns_bounded_sanitized_evidence_and_keeps_ledger(
    prepared_binding, monkeypatch
):
    manager, db, source, _remote, binding, sibling = prepared_binding
    certify_safe(source, binding)
    original = manager._run_git

    def fail_remove(cwd, args, timeout, phase):
        if args[:2] == ["worktree", "remove"]:
            return subprocess.CompletedProcess(
                ["git", *args],
                1,
                stdout="",
                stderr="token=super-secret\x00\n" + ("remove denied " * 100),
            )
        return original(cwd, args, timeout, phase)

    monkeypatch.setattr(manager, "_run_git", fail_remove)

    assert git_worktree_is_locked(source, binding.path)

    result = manager.remove_after_explicit_request(binding.root_session_id)

    assert result.removed is False
    assert result.verdict.reasons == ("remove_failed",)
    assert result.failure_phase == "remove"
    assert result.failure_message is not None
    assert "super-secret" not in result.failure_message
    assert "\x00" not in result.failure_message
    assert len(result.failure_message) <= 300
    assert binding.path.exists()
    assert git_worktree_is_locked(source, binding.path)
    assert sibling.exists()
    record = db.get_conversation_worktree(binding.root_session_id)
    assert record is not None
    assert record.state == "ready"
