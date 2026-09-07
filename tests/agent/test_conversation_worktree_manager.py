"""Real-Git coverage for conversation-root worktree lifecycle ownership."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from multiprocessing import get_context

import pytest

from agent.conversation_worktree import (
    ConversationWorktreeError,
    ConversationWorktreeManager,
)
from agent.conversation_worktree_policy import ConversationWorktreePolicy
from hermes_state import SessionDB


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
    current_path = None
    for line in git(repo, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            current_path = Path(line.removeprefix("worktree ")).resolve()
        elif current_path == worktree.resolve() and line.startswith("locked"):
            return True
    return False


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "stable"
    path.mkdir()
    git(path, "init")
    git(path, "config", "user.email", "test@example.invalid")
    git(path, "config", "user.name", "Hermes Test")
    (path / "tracked.txt").write_text("committed\n", encoding="utf-8")
    git(path, "add", "tracked.txt")
    git(path, "commit", "-m", "initial")
    return path


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def policy(repo: Path, tmp_path: Path, **overrides) -> ConversationWorktreePolicy:
    values = {
        "enabled": True,
        "source_worktree": repo,
        "worktree_root": tmp_path / "conversation-worktrees",
        "branch_prefix": "hermes/session",
        "bootstrap": False,
        "bootstrap_command": (),
        "bootstrap_timeout": 1.0,
        "create_timeout": 2.0,
        "retain_until_explicit_cleanup": True,
    }
    values.update(overrides)
    return ConversationWorktreePolicy(**values)


def manager(repo: Path, db: SessionDB, tmp_path: Path, **overrides):
    return ConversationWorktreeManager(policy(repo, tmp_path, **overrides), db)


def _process_bind(
    repo_path: str,
    db_path: str,
    worktree_root: str,
    root_session_id: str,
    start,
    results,
):
    """Spawn-safe child used to exercise real process-level repository locks."""
    start.wait(10)
    session_db = SessionDB(Path(db_path))
    try:
        worktree_manager = ConversationWorktreeManager(
            ConversationWorktreePolicy(
                enabled=True,
                source_worktree=Path(repo_path),
                worktree_root=Path(worktree_root),
                branch_prefix="hermes/session",
                bootstrap=False,
                bootstrap_command=(),
                bootstrap_timeout=1.0,
                create_timeout=5.0,
                retain_until_explicit_cleanup=True,
            ),
            session_db,
        )
        binding = worktree_manager.bind_new_root_session(
            root_session_id, conversation_kind="interactive"
        )
        assert binding is not None
        results.put(("ok", str(binding.path), binding.branch))
    except BaseException as exc:
        results.put(("error", type(exc).__name__, str(exc)))
    finally:
        session_db.close()


def _process_bind_with_head_pause(
    repo_path: str,
    db_path: str,
    worktree_root: str,
    root_session_id: str,
    pause_after_head: bool,
    head_observed,
    release_head,
    done,
    results,
):
    """Expose first-claim HEAD selection without replacing real Git or SQLite."""

    class HeadPausingManager(ConversationWorktreeManager):
        def _git_stdout(self, cwd, args, phase):
            value = super()._git_stdout(cwd, args, phase)
            if (
                args == ["rev-parse", "HEAD"]
                and Path(cwd).resolve() == Path(repo_path).resolve()
            ):
                head_observed.set()
                if pause_after_head and not release_head.wait(10):
                    raise RuntimeError(
                        "timed out waiting to release first HEAD selection"
                    )
            return value

    session_db = SessionDB(Path(db_path))
    try:
        worktree_manager = HeadPausingManager(
            ConversationWorktreePolicy(
                enabled=True,
                source_worktree=Path(repo_path),
                worktree_root=Path(worktree_root),
                branch_prefix="hermes/session",
                bootstrap=False,
                bootstrap_command=(),
                bootstrap_timeout=1.0,
                create_timeout=5.0,
                retain_until_explicit_cleanup=True,
            ),
            session_db,
        )
        binding = worktree_manager.bind_new_root_session(
            root_session_id, conversation_kind="interactive"
        )
        assert binding is not None
        results.put(("ok", binding.base_commit, str(binding.path), binding.branch))
    except BaseException as exc:
        results.put(("error", type(exc).__name__, str(exc)))
    finally:
        done.set()
        session_db.close()


def test_binding_pins_committed_head_without_copying_dirty_stable_files(repo, db, tmp_path):
    base = git(repo, "rev-parse", "HEAD")
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("do not copy\n", encoding="utf-8")

    binding = manager(repo, db, tmp_path).bind_new_root_session(
        "root-1", conversation_kind="interactive"
    )

    assert binding is not None
    assert binding.base_commit == base
    assert git(binding.path, "rev-parse", "HEAD") == base
    assert (binding.path / "tracked.txt").read_text(encoding="utf-8") == "committed\n"
    assert not (binding.path / "untracked.txt").exists()
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "dirty\n"
    assert (repo / "untracked.txt").read_text(encoding="utf-8") == "do not copy\n"


def test_task_kind_bypasses_conversation_isolation(repo, db, tmp_path):
    assert manager(repo, db, tmp_path).bind_new_root_session(
        "task", conversation_kind="task"
    ) is None
    assert db.get_conversation_worktree("task") is None


def test_bootstrap_failure_is_retained_and_never_ready(repo, db, tmp_path):
    worktree_manager = manager(
        repo,
        db,
        tmp_path,
        bootstrap=True,
        bootstrap_command=(sys.executable, "-c", "raise SystemExit(7)"),
    )

    with pytest.raises(ConversationWorktreeError, match="bootstrap"):
        worktree_manager.bind_new_root_session("root", conversation_kind="interactive")

    record = db.get_conversation_worktree("root")
    assert record is not None
    assert record.state == "creation_failed"
    assert record.failure_phase == "bootstrap"
    assert Path(record.worktree_path).exists()


def test_repeated_root_reuses_same_validated_ready_binding(repo, db, tmp_path):
    worktree_manager = manager(repo, db, tmp_path)
    first = worktree_manager.bind_new_root_session("root", conversation_kind="interactive")
    second = worktree_manager.bind_new_root_session("root", conversation_kind="interactive")

    assert first == second
    assert worktree_manager.resolve_existing_session("root") == first


def test_resolve_accepts_exact_owned_ready_worktree_after_branch_rename(
    repo, db, tmp_path
):
    worktree_manager = manager(repo, db, tmp_path)
    first = worktree_manager.bind_new_root_session(
        "root", conversation_kind="interactive"
    )
    assert first is not None

    git(first.path, "branch", "-m", "hermes/resumed-conversation-work")

    resolved = worktree_manager.resolve_existing_session("root")

    assert resolved is not None
    assert resolved.path == first.path
    assert resolved.branch == "hermes/resumed-conversation-work"
    assert resolved.base_commit == first.base_commit


def test_resolve_rejects_branch_rename_when_exact_owner_marker_is_missing(
    repo, db, tmp_path
):
    worktree_manager = manager(repo, db, tmp_path)
    first = worktree_manager.bind_new_root_session(
        "root", conversation_kind="interactive"
    )
    assert first is not None
    marker = Path(
        git(first.path, "rev-parse", "--git-path", "hermes-conversation-owner-v1")
    )
    marker.unlink()
    git(first.path, "branch", "-m", "hermes/unowned-rename")

    with pytest.raises(ConversationWorktreeError, match="identity validation"):
        worktree_manager.resolve_existing_session("root")


def test_resolve_accepts_exact_owned_rewrite_when_base_is_in_worktree_reflog(
    repo, db, tmp_path
):
    worktree_manager = manager(repo, db, tmp_path)
    first = worktree_manager.bind_new_root_session(
        "root", conversation_kind="interactive"
    )
    assert first is not None
    unrelated = git(first.path, "commit-tree", "HEAD^{tree}", "-m", "rewritten root")
    git(first.path, "reset", "--hard", unrelated)

    resolved = worktree_manager.resolve_existing_session("root")

    assert resolved is not None
    assert resolved.path == first.path
    assert resolved.base_commit == first.base_commit


def test_resolve_rejects_rewrite_when_base_is_absent_from_worktree_reflog(
    repo, db, tmp_path
):
    worktree_manager = manager(repo, db, tmp_path)
    first = worktree_manager.bind_new_root_session(
        "root", conversation_kind="interactive"
    )
    assert first is not None
    unrelated = git(first.path, "commit-tree", "HEAD^{tree}", "-m", "unrelated root")
    git(first.path, "reset", "--hard", unrelated)
    git_dir = Path(git(first.path, "rev-parse", "--path-format=absolute", "--git-dir"))
    (git_dir / "logs" / "HEAD").unlink()

    with pytest.raises(ConversationWorktreeError, match="no longer descends"):
        worktree_manager.resolve_existing_session("root")


def test_active_managed_worktree_is_git_locked_against_external_removal(
    repo, db, tmp_path
):
    binding = manager(repo, db, tmp_path).bind_new_root_session(
        "root", conversation_kind="interactive"
    )

    assert binding is not None
    removal = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "worktree",
            "remove",
            "--force",
            str(binding.path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert removal.returncode != 0
    assert "locked" in removal.stderr.lower()
    assert binding.path.exists()


def test_bind_reuse_relocks_externally_unlocked_ready_worktree(repo, db, tmp_path):
    worktree_manager = manager(repo, db, tmp_path)
    first = worktree_manager.bind_new_root_session("root", conversation_kind="interactive")
    assert first is not None
    git(repo, "worktree", "unlock", str(first.path))
    assert not git_worktree_is_locked(repo, first.path)

    second = worktree_manager.bind_new_root_session("root", conversation_kind="interactive")

    assert second == first
    assert git_worktree_is_locked(repo, first.path)


def test_resolve_relocks_externally_unlocked_ready_worktree(repo, db, tmp_path):
    worktree_manager = manager(repo, db, tmp_path)
    first = worktree_manager.bind_new_root_session("root", conversation_kind="interactive")
    assert first is not None
    git(repo, "worktree", "unlock", str(first.path))
    assert not git_worktree_is_locked(repo, first.path)

    resolved = worktree_manager.resolve_existing_session("root")

    assert resolved == first
    assert git_worktree_is_locked(repo, first.path)


def test_concurrent_same_root_claim_creates_one_binding(repo, tmp_path):
    path = tmp_path / "state.db"
    first_db = SessionDB(path)
    second_db = SessionDB(path)
    first_manager = manager(repo, first_db, tmp_path)
    second_manager = manager(repo, second_db, tmp_path)
    results = []
    errors = []

    def bind(worktree_manager):
        try:
            results.append(
                worktree_manager.bind_new_root_session(
                    "root", conversation_kind="interactive"
                )
            )
        except BaseException as exc:  # surfaced below with exact failure
            errors.append(exc)

    left = threading.Thread(target=bind, args=(first_manager,))
    right = threading.Thread(target=bind, args=(second_manager,))
    left.start()
    right.start()
    left.join(timeout=10)
    right.join(timeout=10)
    try:
        assert not left.is_alive()
        assert not right.is_alive()
        assert errors == []
        assert len(results) == 2
        assert results[0] == results[1]
        assert results[0] is not None
        assert results[0].path.exists()
    finally:
        first_db.close()
        second_db.close()


def test_common_dir_mismatch_is_rejected_before_reuse(repo, db, tmp_path):
    worktree_manager = manager(repo, db, tmp_path)
    base = git(repo, "rev-parse", "HEAD")
    expected_path = tmp_path / "conversation-worktrees" / "conversation-root"
    db.claim_conversation_worktree(
        root_session_id="root",
        worktree_path=str(expected_path),
        branch="hermes/session/not-the-manager-branch",
        base_commit=base,
        repo_common_dir="/not/the/source/common-dir",
    )
    db.mark_conversation_worktree_ready("root")

    with pytest.raises(ConversationWorktreeError, match="identity"):
        worktree_manager.resolve_existing_session("root")


def test_existing_partial_path_is_retained_and_marked_failed(repo, db, tmp_path):
    worktree_manager = manager(repo, db, tmp_path)
    root_id = "root"
    expected_path, _ = worktree_manager._expected_identity(root_id)
    expected_path.mkdir(parents=True)
    expected_path.joinpath("partial.txt").write_text("keep me\n", encoding="utf-8")

    with pytest.raises(ConversationWorktreeError, match="already exists"):
        worktree_manager.bind_new_root_session(root_id, conversation_kind="interactive")

    record = db.get_conversation_worktree(root_id)
    assert record is not None
    assert record.state == "creation_failed"
    assert expected_path.joinpath("partial.txt").exists()


@pytest.mark.live_system_guard_bypass
def test_bootstrap_timeout_is_retained(repo, db, tmp_path):
    worktree_manager = manager(
        repo,
        db,
        tmp_path,
        bootstrap=True,
        bootstrap_command=(sys.executable, "-c", "import time; time.sleep(60)"),
        bootstrap_timeout=0.05,
    )

    with pytest.raises(ConversationWorktreeError, match="bootstrap"):
        worktree_manager.bind_new_root_session("root", conversation_kind="interactive")

    record = db.get_conversation_worktree("root")
    assert record is not None
    assert record.state == "creation_failed"
    assert record.failure_phase == "bootstrap"
    assert "timed out" in (record.failure_message or "")


def test_root_id_is_deterministically_sanitized_for_path_and_branch(repo, db, tmp_path):
    worktree_manager = manager(repo, db, tmp_path)
    binding = worktree_manager.bind_new_root_session(
        "root/../with spaces", conversation_kind="interactive"
    )

    assert binding is not None
    assert binding == worktree_manager.resolve_existing_session("root/../with spaces")
    assert binding.path.parent == tmp_path / "conversation-worktrees"
    assert git(repo, "check-ref-format", "--branch", binding.branch) == binding.branch


def test_bootstrap_runs_inside_worktree(repo, db, tmp_path):
    marker = tmp_path / "bootstrap-cwd.txt"
    script = (
        "from pathlib import Path; import os; "
        f"Path({str(marker)!r}).write_text(os.getcwd(), encoding='utf-8')"
    )
    binding = manager(
        repo,
        db,
        tmp_path,
        bootstrap=True,
        bootstrap_command=(sys.executable, "-c", script),
    ).bind_new_root_session("root", conversation_kind="interactive")

    assert binding is not None
    assert marker.read_text(encoding="utf-8") == str(binding.path)


def test_worktree_root_in_unrelated_repository_is_rejected_before_claim(repo, db, tmp_path):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    git(unrelated, "init")
    git(unrelated, "config", "user.email", "test@example.invalid")
    git(unrelated, "config", "user.name", "Hermes Test")
    (unrelated / "tracked.txt").write_text("other\n", encoding="utf-8")
    git(unrelated, "add", "tracked.txt")
    git(unrelated, "commit", "-m", "initial")

    with pytest.raises(ConversationWorktreeError, match="unrelated repository"):
        manager(
            repo, db, tmp_path, worktree_root=unrelated / "conversation-worktrees"
        ).bind_new_root_session("root", conversation_kind="interactive")

    assert db.get_conversation_worktree("root") is None
    assert git(unrelated, "status", "--porcelain") == ""


def test_worktree_root_inside_sibling_registered_worktree_is_rejected(
    repo, db, tmp_path
):
    sibling = tmp_path / "sibling-worktree"
    git(repo, "worktree", "add", "-b", "sibling", str(sibling), "HEAD")
    worktree_manager = manager(
        repo,
        db,
        tmp_path,
        worktree_root=sibling / "conversation-worktrees",
    )
    expected_path, _ = worktree_manager._expected_identity("root")

    try:
        with pytest.raises(ConversationWorktreeError, match="registered worktree"):
            worktree_manager.bind_new_root_session(
                "root", conversation_kind="interactive"
            )
    finally:
        if expected_path.exists():
            git(repo, "worktree", "remove", "--force", str(expected_path))

    assert db.get_conversation_worktree("root") is None
    assert git(sibling, "status", "--porcelain") == ""


def test_unbound_target_cannot_reuse_registered_worktree_under_external_root(
    repo, db, tmp_path
):
    worktree_manager = manager(repo, db, tmp_path)
    expected_path, _ = worktree_manager._expected_identity("root")
    git(repo, "worktree", "add", "-b", "occupied", str(expected_path), "HEAD")

    with pytest.raises(ConversationWorktreeError, match="registered worktree"):
        worktree_manager.bind_new_root_session("root", conversation_kind="interactive")

    assert db.get_conversation_worktree("root") is None


def test_ready_binding_is_rejected_when_configured_root_is_inside_sibling(
    repo, db, tmp_path
):
    sibling = tmp_path / "sibling-ready"
    git(repo, "worktree", "add", "-b", "sibling-ready", str(sibling), "HEAD")
    worktree_manager = manager(
        repo,
        db,
        tmp_path,
        worktree_root=sibling / "conversation-worktrees",
    )
    expected_path, expected_branch = worktree_manager._expected_identity("root")
    base = git(repo, "rev-parse", "HEAD")
    common_dir = git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    git(repo, "worktree", "add", "-b", expected_branch, str(expected_path), base)
    db.claim_conversation_worktree(
        root_session_id="root",
        worktree_path=str(expected_path),
        branch=expected_branch,
        base_commit=base,
        repo_common_dir=common_dir,
    )
    db.mark_conversation_worktree_ready("root")

    with pytest.raises(ConversationWorktreeError, match="registered worktree"):
        worktree_manager.bind_new_root_session("root", conversation_kind="interactive")


def test_resolve_rejects_ready_binding_when_configured_root_is_inside_sibling(
    repo, db, tmp_path
):
    sibling = tmp_path / "sibling-resolve"
    git(repo, "worktree", "add", "-b", "sibling-resolve", str(sibling), "HEAD")
    worktree_manager = manager(
        repo,
        db,
        tmp_path,
        worktree_root=sibling / "conversation-worktrees",
    )
    expected_path, expected_branch = worktree_manager._expected_identity("root")
    base = git(repo, "rev-parse", "HEAD")
    common_dir = git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    git(repo, "worktree", "add", "-b", expected_branch, str(expected_path), base)
    db.claim_conversation_worktree(
        root_session_id="root",
        worktree_path=str(expected_path),
        branch=expected_branch,
        base_commit=base,
        repo_common_dir=common_dir,
    )
    db.mark_conversation_worktree_ready("root")

    with pytest.raises(ConversationWorktreeError, match="registered worktree"):
        worktree_manager.resolve_existing_session("root")


@pytest.mark.live_system_guard_bypass
def test_bootstrap_timeout_kills_its_process_tree(repo, db, tmp_path):
    leaked = tmp_path / "leaked-grandchild.txt"
    child = (
        "from pathlib import Path; import time; "
        "time.sleep(0.4); "
        f"Path({str(leaked)!r}).write_text('leaked', encoding='utf-8')"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "time.sleep(60)"
    )
    worktree_manager = manager(
        repo,
        db,
        tmp_path,
        bootstrap=True,
        bootstrap_command=(sys.executable, "-c", parent),
        bootstrap_timeout=0.05,
    )

    started = time.monotonic()
    with pytest.raises(ConversationWorktreeError, match="bootstrap timed out"):
        worktree_manager.bind_new_root_session("root", conversation_kind="interactive")
    assert time.monotonic() - started < 2.0
    record = db.get_conversation_worktree("root")
    assert record is not None
    assert record.state == "creation_failed"
    assert record.failure_phase == "bootstrap"
    assert "timed out" in (record.failure_message or "")
    time.sleep(0.7)
    assert not leaked.exists()


def test_blocked_bootstrap_does_not_hold_repository_create_lock(repo, db, tmp_path):
    started = tmp_path / "blocked-bootstrap-started"
    blocker_name, _ = manager(repo, db, tmp_path)._expected_identity("blocked")
    script = (
        "from pathlib import Path; import os, time; "
        f"blocked = {blocker_name.name!r}; "
        f"marker = Path({str(started)!r}); "
        "marker.write_text('started', encoding='utf-8') if os.path.basename(os.getcwd()) == blocked else None; "
        "time.sleep(1.0) if os.path.basename(os.getcwd()) == blocked else None"
    )
    worktree_manager = manager(
        repo,
        db,
        tmp_path,
        bootstrap=True,
        bootstrap_command=(sys.executable, "-c", script),
        bootstrap_timeout=3.0,
    )
    failures = []

    def bind_blocked():
        try:
            worktree_manager.bind_new_root_session("blocked", conversation_kind="interactive")
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=bind_blocked)
    thread.start()
    assert _wait_for_path(started)
    try:
        began = time.monotonic()
        fast = worktree_manager.bind_new_root_session("fast", conversation_kind="interactive")
        assert fast is not None
        assert time.monotonic() - began < 0.5
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []


def test_lock_setup_failure_is_controlled_before_first_claim(
    repo, db, tmp_path, monkeypatch
):
    worktree_manager = manager(repo, db, tmp_path)

    def fail_open(_path):
        raise OSError("fd exhausted")

    monkeypatch.setattr(worktree_manager, "_open_lock_file", fail_open)
    with pytest.raises(ConversationWorktreeError, match="lock.*unavailable"):
        worktree_manager.bind_new_root_session("root", conversation_kind="interactive")

    assert db.get_conversation_worktree("root") is None


def test_same_root_is_idempotent_across_real_processes(repo, tmp_path):
    context = get_context("spawn")
    start = context.Event()
    results = context.Queue()
    args = (str(repo), str(tmp_path / "state.db"), str(tmp_path / "worktrees"), "root", start, results)
    first = context.Process(target=_process_bind, args=args)
    second = context.Process(target=_process_bind, args=args)
    first.start()
    second.start()
    start.set()
    first.join(timeout=20)
    second.join(timeout=20)

    assert first.exitcode == 0
    assert second.exitcode == 0
    observed = [results.get(timeout=3), results.get(timeout=3)]
    assert all(result[0] == "ok" for result in observed)
    assert observed[0] == observed[1]


def test_first_binding_identity_is_selected_under_repository_lock(repo, tmp_path):
    context = get_context("spawn")
    old_head_observed = context.Event()
    new_head_observed = context.Event()
    release_old_head = context.Event()
    unused_release = context.Event()
    first_done = context.Event()
    second_done = context.Event()
    results = context.Queue()
    common = (
        str(repo),
        str(tmp_path / "state.db"),
        str(tmp_path / "worktrees"),
        "root",
    )
    old_base = git(repo, "rev-parse", "HEAD")
    first = context.Process(
        target=_process_bind_with_head_pause,
        args=(
            *common,
            True,
            old_head_observed,
            release_old_head,
            first_done,
            results,
        ),
    )
    first.start()
    assert old_head_observed.wait(10)

    (repo / "advanced.txt").write_text("advanced\n", encoding="utf-8")
    git(repo, "add", "advanced.txt")
    git(repo, "commit", "-m", "advance source while first contender waits")
    assert git(repo, "rev-parse", "HEAD") != old_base

    second = context.Process(
        target=_process_bind_with_head_pause,
        args=(
            *common,
            False,
            new_head_observed,
            unused_release,
            second_done,
            results,
        ),
    )
    second.start()
    if new_head_observed.wait(1.0):
        assert second_done.wait(10)
    release_old_head.set()
    first.join(timeout=20)
    second.join(timeout=20)

    assert first.exitcode == 0
    assert second.exitcode == 0
    observed = [results.get(timeout=3), results.get(timeout=3)]
    assert all(result[0] == "ok" for result in observed)
    assert observed[0] == observed[1]
    assert observed[0][1] == old_base


def test_different_roots_create_unique_worktrees_across_real_processes(repo, tmp_path):
    context = get_context("spawn")
    start = context.Event()
    results = context.Queue()
    common = (str(repo), str(tmp_path / "state.db"), str(tmp_path / "worktrees"))
    first = context.Process(target=_process_bind, args=(*common, "root-a", start, results))
    second = context.Process(target=_process_bind, args=(*common, "root-b", start, results))
    first.start()
    second.start()
    start.set()
    first.join(timeout=20)
    second.join(timeout=20)

    assert first.exitcode == 0
    assert second.exitcode == 0
    observed = [results.get(timeout=3), results.get(timeout=3)]
    assert all(result[0] == "ok" for result in observed)
    assert observed[0][1] != observed[1][1]
    assert observed[0][2] != observed[1][2]


def _wait_for_path(path: Path, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return path.exists()
