"""Real Git proof for folded ancestry and fork updater recovery boundaries."""
import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_history as history


def git(repo, *args, input=None):
    return subprocess.check_output(
        ["git", "-c", "core.hooksPath=NUL", "-C", str(repo), *args],
        input=input, text=True, encoding="utf-8", stderr=subprocess.PIPE,
    ).strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "Contributor")
    git(tmp_path, "config", "user.email", "contributor@example.invalid")
    (tmp_path / "behavior.txt").write_text("preserve me\n", encoding="utf-8")
    git(tmp_path, "add", "behavior.txt")
    git(tmp_path, "commit", "-m", "original behavior")
    git(tmp_path, "remote", "add", "origin", "https://example.invalid/owner/hermes-agent")
    # This throwaway repo stands in for a fork checkout: its first commit is the fork root.
    monkeypatch.setattr(history, "FORK_ROOT_COMMIT", git(tmp_path, "rev-parse", "HEAD"))
    return tmp_path


def folded_tip(repo, *, related):
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    parents = ["-p", git(repo, "rev-parse", "HEAD")] if related else []
    tip = git(repo, "commit-tree", tree, *parents, input="folded review\n")
    git(repo, "update-ref", "refs/remotes/origin/main", tip)
    return tip


def test_plan_is_read_only_and_distinguishes_tree_from_ancestry(repo):
    old = git(repo, "rev-parse", "HEAD")
    tip = folded_tip(repo, related=False)
    refs = git(repo, "show-ref")
    result = history.assess_history(["git"], repo, "origin/main")
    assert result.relationship == "unrelated"
    assert result.same_tree and result.head == old and result.target_head == tip
    assert git(repo, "show-ref") == refs


def test_fold_refusal_preserves_dirty_files_index_and_both_histories(repo, capsys):
    old = git(repo, "rev-parse", "HEAD")
    tip = folded_tip(repo, related=False)
    (repo / "behavior.txt").write_text("uncommitted intent\n", encoding="utf-8")
    git(repo, "add", "behavior.txt")
    (repo / "untracked.txt").write_text("untracked intent", encoding="utf-8")
    staged = git(repo, "diff", "--cached")
    status = git(repo, "status", "--porcelain")
    with pytest.raises(SystemExit) as refused:
        history.guard_fork_history(["git"], repo, "origin/main")
    # Exit 2: upstream's desktop updaters treat it as a non-retryable refusal.
    assert refused.value.code == 2
    assert git(repo, "rev-parse", "HEAD") == old
    assert git(repo, "rev-parse", "origin/main") == tip
    assert git(repo, "diff", "--cached") == staged
    assert git(repo, "status", "--porcelain") == status
    rescued = git(repo, "for-each-ref", "--format=%(objectname)", "refs/hermes-update-backups/").splitlines()
    assert sorted(rescued) == sorted([old, tip])
    out = capsys.readouterr().out
    assert "tracked trees match" in out
    # scripts/desktop-update/{posix.sh,windows.ps1} key their refusal text on this literal.
    assert "HERMES_UPDATE_HISTORY_REVIEW_REQUIRED" in out.splitlines()


def test_genuine_divergence_is_not_assumed_to_be_a_fold(repo):
    base = git(repo, "rev-parse", "HEAD")
    tip = folded_tip(repo, related=True)
    (repo / "behavior.txt").write_text("local change", encoding="utf-8")
    git(repo, "add", "behavior.txt")
    git(repo, "commit", "-m", "local contributor")
    result = history.assess_history(["git"], repo, "origin/main")
    assert result.relationship == "diverged" and not result.same_tree
    assert result.local_only == 1 and result.remote_only == 1
    assert git(repo, "merge-base", "HEAD", tip) == base
    with pytest.raises(SystemExit):
        history.guard_fork_history(["git"], repo, "origin/main")


@pytest.mark.parametrize("relationship", ["equal", "fast_forward", "local_ahead"])
def test_normal_ancestry_remains_available_without_recovery_writes(repo, relationship):
    old = git(repo, "rev-parse", "HEAD")
    tip = folded_tip(repo, related=True)
    if relationship == "equal":
        git(repo, "update-ref", "refs/remotes/origin/main", old)
    elif relationship == "local_ahead":
        git(repo, "merge", "--ff-only", tip)
        git(repo, "update-ref", "refs/remotes/origin/main", old)
    assert history.guard_fork_history(["git"], repo, "origin/main").relationship == relationship
    assert not git(repo, "for-each-ref", "refs/hermes-update-backups/")


def test_backup_failure_never_allows_reset(repo, monkeypatch, capsys):
    folded_tip(repo, related=False)
    old = git(repo, "rev-parse", "HEAD")
    real = history._git

    def fail_backup(cmd, cwd, *args, **kwargs):
        if args[0] == "update-ref":
            return subprocess.CompletedProcess(args, 128, "", "write refused")
        return real(cmd, cwd, *args, **kwargs)

    monkeypatch.setattr(history, "_git", fail_backup)
    with pytest.raises(SystemExit):
        history.guard_fork_history(["git"], repo, "origin/main")
    assert git(repo, "rev-parse", "HEAD") == old
    assert "could not be written" in capsys.readouterr().out


def test_prepare_refuses_before_stash_or_branch_switch(repo, monkeypatch):
    from hermes_cli import main, update_cmd
    folded_tip(repo, related=False)
    monkeypatch.setattr(main, "PROJECT_ROOT", repo)
    monkeypatch.setattr(update_cmd, "_apply_parked_branch_guard", lambda *a, **k: pytest.fail("switched before review"))
    monkeypatch.setattr(main, "_stash_local_changes_if_needed", lambda *a: pytest.fail("stashed before review"))
    with pytest.raises(SystemExit):
        update_cmd._prepare_checkout_for_update(
            ["git"], "main", "main", is_fork=True, assume_yes=True,
            gateway_mode=False, gw_input_fn=None, switch_branch=False, _windows_gateway_resume=None,
        )


def test_late_fork_reconcile_cannot_reset_equal_tree_fold(repo, monkeypatch):
    from hermes_cli import main, update_cmd
    folded_tip(repo, related=False)
    old = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(main, "PROJECT_ROOT", repo)
    with pytest.raises(SystemExit):
        update_cmd._reconcile_diverged_checkout(["git"], "main", old)
    assert git(repo, "rev-parse", "HEAD") == old


def test_a_checkout_without_fork_history_is_not_guarded(repo, monkeypatch):
    """Upstream's throwaway repos run upstream's own rescue path: no refusal, no refs."""
    folded_tip(repo, related=False)
    monkeypatch.setattr(history, "FORK_ROOT_COMMIT", "0" * 40)  # unknown object: not fork history
    assert history.guard_fork_history(["git"], repo, "origin/main").relationship == "unrelated"
    assert git(repo, "for-each-ref", "refs/hermes-update-backups") == ""


@pytest.mark.parametrize("guarded", [True, False])
def test_plan_describes_the_path_the_guard_will_take(repo, monkeypatch, capsys, guarded):
    """--plan must not promise a stop the guard will not make (review on #125265)."""
    from hermes_cli.update_inventory import UpdatePlan, print_update_plan
    folded_tip(repo, related=False)
    if not guarded:
        monkeypatch.setattr(history, "FORK_ROOT_COMMIT", "0" * 40)
    plan = UpdatePlan(history={
        **history.assess_history(["git"], repo, "origin/main").to_dict(),
        "guarded": history.has_fork_ancestry(["git"], repo),
    })
    print_update_plan(plan)
    out = capsys.readouterr().out
    assert ("stop before stashing" in out) is guarded
    assert ("standard rescue-ref" in out) is not guarded
