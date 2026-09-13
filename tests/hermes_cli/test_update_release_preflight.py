"""Release-apply Git-state preflight: ``hermes update --version`` fails closed, first.

Contract under test:
- the APPLY path refuses (exit 1, static guidance naming ``git status`` and ``git stash
  list``) when the checkout carries uncommitted tracked or untracked changes, an in-progress
  merge/rebase/cherry-pick/revert/bisect, an unmerged index, or parked Hermes update
  autostash entries — BEFORE the receipt, the pre-update backup, the Windows gateway pause,
  the lockfile/EOL churn cleanup, any network fetch, the autostash, or the checkout
- ordinary user stashes and a clean tree pass the preflight; branch mode keeps its existing
  autostash behavior and is never gated
- ``--check`` and ``--plan`` stay read-only and keep answering on a dirty tree
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import update_cmd as hermes_update_cmd


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, capture_output=True, text=True,
        encoding="utf-8", errors="replace")


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "hermes@example.invalid")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "base")
    return repo


def _guard_release_side_effects(monkeypatch, repo: Path) -> list[str]:
    """Point the impl at *repo* and turn every downstream side effect into a recorded call."""
    effects: list[str] = []

    def record(name):
        def _record(*_a, **_k):
            effects.append(name)
        return _record

    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", record("backup"))
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", record("gateway-pause"))
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", record("autostash"))
    monkeypatch.setattr(hermes_update_cmd, "_begin_update_receipt_and_plan", record("receipt"))
    monkeypatch.setattr(hermes_update_cmd, "_fetch_official_release_tag", record("network-fetch"))
    monkeypatch.setattr(hermes_update_cmd, "_discard_lockfile_churn", record("lockfile-churn"))
    monkeypatch.setattr(hermes_update_cmd, "_normalize_managed_eol", record("eol-normalize"))
    return effects


def _release_args() -> SimpleNamespace:
    return SimpleNamespace(version="v2026.7.30", branch=None)


def _conflicting_branches(repo: Path) -> None:
    """``feature`` and ``main`` both rewrite tracked.txt from the same base commit."""
    _git(repo, "checkout", "-b", "feature")
    (repo / "tracked.txt").write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature edit")
    _git(repo, "checkout", "main")
    (repo / "tracked.txt").write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main edit")


def _prepare_dirty_tracked(repo: Path) -> None:
    (repo / "tracked.txt").write_text("edited\n", encoding="utf-8")


def _prepare_untracked_only(repo: Path) -> None:
    (repo / "new-work.txt").write_text("untracked\n", encoding="utf-8")


def _prepare_staged_only(repo: Path) -> None:
    (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "staged.txt")


def _prepare_merge_in_progress(repo: Path) -> None:
    _conflicting_branches(repo)
    _git(repo, "merge", "feature", check=False)


def _prepare_unmerged_index_without_marker(repo: Path) -> None:
    _prepare_merge_in_progress(repo)
    # Drop the in-progress marker: the unmerged index entries alone must still refuse.
    (repo / ".git" / "MERGE_HEAD").unlink()


def _prepare_rebase_in_progress(repo: Path) -> None:
    _conflicting_branches(repo)
    _git(repo, "checkout", "feature")
    _git(repo, "rebase", "main", check=False)


def _prepare_cherry_pick_in_progress(repo: Path) -> None:
    _conflicting_branches(repo)
    _git(repo, "cherry-pick", "feature", check=False)


def _prepare_revert_in_progress(repo: Path) -> None:
    (repo / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "commit", "-am", "one")
    (repo / "tracked.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "commit", "-am", "two")
    _git(repo, "revert", "--no-edit", "HEAD~1", check=False)


def _prepare_bisect_in_progress(repo: Path) -> None:
    _git(repo, "bisect", "start")


def _prepare_sequencer_pending(repo: Path) -> None:
    """A multi-commit cherry-pick whose conflicted pick was committed manually: the
    CHERRY_PICK_HEAD marker is consumed and ``git status`` is clean, but ``.git/sequencer``
    still holds the remaining pick — ``--continue``/``--abort`` still apply."""
    _git(repo, "checkout", "-b", "side")
    (repo / "tracked.txt").write_text("side one\n", encoding="utf-8")
    _git(repo, "commit", "-am", "side one")
    (repo / "second.txt").write_text("side two\n", encoding="utf-8")
    _git(repo, "add", "second.txt")
    _git(repo, "commit", "-m", "side two")
    _git(repo, "checkout", "main")
    (repo / "tracked.txt").write_text("main one\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main one")
    _git(repo, "cherry-pick", "side~1", "side", check=False)
    (repo / "tracked.txt").write_text("resolved\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "-c", "core.editor=true", "commit", "--no-edit")
    # Preconditions for the gap under test: nothing left behind but the sequencer.
    assert not (repo / ".git" / "CHERRY_PICK_HEAD").exists()
    assert (repo / ".git" / "sequencer").exists()
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""


def _prepare_parked_hermes_autostash(repo: Path) -> None:
    (repo / "tracked.txt").write_text("parked update work\n", encoding="utf-8")
    stash_ref = hermes_update_cmd._stash_local_changes_if_needed(["git"], repo)
    assert stash_ref is not None
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""


@pytest.mark.parametrize(
    "prepare",
    [
        _prepare_dirty_tracked,
        _prepare_untracked_only,
        _prepare_staged_only,
        _prepare_merge_in_progress,
        _prepare_unmerged_index_without_marker,
        _prepare_rebase_in_progress,
        _prepare_cherry_pick_in_progress,
        _prepare_revert_in_progress,
        _prepare_bisect_in_progress,
        _prepare_sequencer_pending,
        _prepare_parked_hermes_autostash,
    ],
    ids=[
        "dirty-tracked", "untracked-only", "staged-only", "merge-in-progress",
        "unmerged-index", "rebase-in-progress", "cherry-pick-in-progress",
        "revert-in-progress", "bisect-in-progress", "sequencer-pending",
        "parked-hermes-autostash",
    ],
)
def test_release_apply_refuses_ambiguous_git_state_before_any_side_effect(
        monkeypatch, tmp_path, capsys, prepare):
    repo = _repo(tmp_path)
    prepare(repo)
    effects = _guard_release_side_effects(monkeypatch, repo)
    stashes_before = _git(repo, "stash", "list").stdout
    status_before = _git(repo, "status", "--porcelain").stdout

    with pytest.raises(SystemExit, match="1"):
        hermes_update_cmd._cmd_update_impl(_release_args(), gateway_mode=False)

    assert effects == []
    out = capsys.readouterr().out
    assert "✗" in out
    assert "git status" in out
    assert "git stash list" in out
    # Read-only refusal: the checkout is byte-for-byte where the user left it.
    assert _git(repo, "stash", "list").stdout == stashes_before
    assert _git(repo, "status", "--porcelain").stdout == status_before


@pytest.mark.parametrize(
    "stash_message",
    [
        None,
        "my own parked work",
        # Substring is not identity: an ordinary stash merely CONTAINING the generated
        # name (or the words) must never block a release update.
        "notes about hermes-update-autostash-20260913-101010 backup",
        "cleanup notes for hermes-update autostash entries",
    ],
    ids=["clean-tree", "ordinary-user-stash", "substring-generated-name", "substring-words"],
)
def test_release_apply_preflight_passes_clean_and_ordinary_stash(
        monkeypatch, tmp_path, stash_message):
    repo = _repo(tmp_path)
    if stash_message is not None:
        (repo / "tracked.txt").write_text("parked by hand\n", encoding="utf-8")
        _git(repo, "stash", "push", "-m", stash_message)
    _guard_release_side_effects(monkeypatch, repo)

    def sentinel(*_a, **_k):
        raise RuntimeError("passed-preflight")

    monkeypatch.setattr(hermes_update_cmd, "_resolve_update_options", sentinel)

    with pytest.raises(RuntimeError, match="passed-preflight"):
        hermes_update_cmd._cmd_update_impl(_release_args(), gateway_mode=False)


def test_preflight_stash_classification_is_exact_not_substring(tmp_path):
    """The blocker matches the complete generated subject grammar (prefix + UTC timestamp),
    never a substring — and a manually parked entry carrying the exact generated name blocks
    just like one the updater created."""
    repo = _repo(tmp_path)

    (repo / "tracked.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "stash", "push", "-m", "notes about hermes-update-autostash-20260913-101010 backup")
    (repo / "tracked.txt").write_text("b\n", encoding="utf-8")
    _git(repo, "stash", "push", "-m", "hermes-update-autostash-20260913-101010-mine")
    assert hermes_update_cmd._release_apply_git_state_block_reason(["git"], repo) is None

    (repo / "tracked.txt").write_text("c\n", encoding="utf-8")
    _git(repo, "stash", "push", "-m", "hermes-update-autostash-20250101-101010")
    reason = hermes_update_cmd._release_apply_git_state_block_reason(["git"], repo)
    assert reason is not None
    assert "autostash" in reason


def test_preflight_detects_exact_autostash_parked_from_detached_head(tmp_path):
    """The updater stashes from detached checkouts too — subject ``On (no branch): <name>``
    must classify as a Hermes updater autostash."""
    repo = _repo(tmp_path)
    _git(repo, "checkout", "--detach", "HEAD")
    (repo / "tracked.txt").write_text("detached work\n", encoding="utf-8")
    stash_ref = hermes_update_cmd._stash_local_changes_if_needed(["git"], repo)
    assert stash_ref is not None
    _git(repo, "checkout", "main")

    reason = hermes_update_cmd._release_apply_git_state_block_reason(["git"], repo)
    assert reason is not None
    assert "autostash" in reason


def test_preflight_resolves_git_paths_in_linked_worktree(tmp_path):
    """Operation markers live under ``.git/worktrees/<name>/`` in a linked worktree; the
    probes must resolve them through Git, per checkout — the primary stays unblocked while
    the worktree mid-cherry-pick refuses."""
    repo = _repo(tmp_path)
    _conflicting_branches(repo)
    worktree = tmp_path / "linked-worktree"
    _git(repo, "worktree", "add", "--detach", str(worktree), "main")
    _git(worktree, "cherry-pick", "feature", check=False)
    assert not (worktree / ".git").is_dir()  # layout under test: .git is a gitfile pointer

    assert hermes_update_cmd._release_apply_git_state_block_reason(["git"], repo) is None
    reason = hermes_update_cmd._release_apply_git_state_block_reason(["git"], worktree)
    assert reason is not None
    assert "cherry-pick" in reason


def test_branch_apply_keeps_autostash_behavior_on_dirty_tree(monkeypatch, tmp_path):
    """Ordinary branch updates are not gated: a dirty tree must sail past where the release
    preflight would refuse (the existing autostash owns it downstream)."""
    repo = _repo(tmp_path)
    _prepare_dirty_tracked(repo)
    _guard_release_side_effects(monkeypatch, repo)

    def sentinel(*_a, **_k):
        raise RuntimeError("passed-preflight")

    monkeypatch.setattr(hermes_update_cmd, "_resolve_update_options", sentinel)

    with pytest.raises(RuntimeError, match="passed-preflight"):
        hermes_update_cmd._cmd_update_impl(
            SimpleNamespace(version=None, branch=None), gateway_mode=False)


def test_update_check_version_still_answers_on_dirty_tree(monkeypatch, tmp_path, capsys):
    """``--check`` reports without requiring a clean worktree and mutates nothing."""
    repo = _repo(tmp_path)
    _git(repo, "tag", "v2026.7.30")
    (repo / "tracked.txt").write_text("dirty while checking\n", encoding="utf-8")
    (repo / "scratch.txt").write_text("untracked while checking\n", encoding="utf-8")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    monkeypatch.setattr(
        "hermes_cli.update_contract.evaluate_update_admission", lambda _root: None)
    monkeypatch.setattr(
        hermes_update_cmd, "_fetch_official_release_tag",
        lambda *_a, **_k: subprocess.CompletedProcess(["git"], 0, stdout="", stderr=""))

    hermes_update_cmd._cmd_update_check(version="v2026.7.30")

    out = capsys.readouterr().out
    assert "Already at version v2026.7.30" in out or "Update available" in out
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "dirty while checking\n"
    assert (repo / "scratch.txt").read_text(encoding="utf-8") == "untracked while checking\n"
    assert _git(repo, "stash", "list").stdout.strip() == ""


def test_update_plan_version_stays_read_only_on_dirty_tree(monkeypatch, tmp_path):
    """``--plan`` inventories without touching git even when the tree is dirty."""
    repo = _repo(tmp_path)
    _prepare_dirty_tracked(repo)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.update_inventory.collect_runtime_inventory", lambda: "inventory")
    printed = []
    monkeypatch.setattr(
        "hermes_cli.update_inventory.print_update_plan", lambda plan: printed.append(plan))

    def fail_run(cmd, **_kwargs):  # pragma: no cover - plan mode is read-only/no-network
        raise AssertionError(f"plan mode may not spawn a subprocess: {cmd}")

    monkeypatch.setattr(subprocess, "run", fail_run)

    handled = hermes_main._update_preflight_handled(
        SimpleNamespace(plan=True, update_version="v2026.7.30", branch=None))

    assert handled is True
    assert printed == ["inventory"]
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "edited\n"
