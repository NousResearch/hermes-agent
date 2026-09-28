"""Real Git regression coverage for updates that land via an existing branch."""
import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd
from tests.hermes_cli.test_update_target_identity import git, update_tree  # noqa: F401


def init_repo(root, monkeypatch):
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.com")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    init_repo(root, monkeypatch)
    (root / "cli.py").write_text("value = 1\n", encoding="utf8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "old")
    old = git(root, "rev-parse", "HEAD")
    (root / "cli.py").write_text("value = 2\n", encoding="utf8")
    git(root, "commit", "-qam", "new")
    tip = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", "refs/remotes/origin/main", tip)
    git(root, "checkout", "-q", "--detach", old)
    return root, old, tip


def prepare(root, *, is_fork=False):
    return update_cmd._prepare_checkout_for_update(
        ["git"], "main", update_cmd._current_branch_name(["git"], check=True),
        is_fork=is_fork, assume_yes=True, gateway_mode=False, gw_input_fn=None,
        switch_branch=False, _windows_gateway_resume=None,
    )


def pull(plan, **kwargs):
    return update_cmd._pull_updates(
        ["git"], "main", plan.auto_stash_ref, prompt_for_restore=False,
        gw_input_fn=None, discard_local_changes=False, keep_stash=False,
        pre_sync_sha=plan.pre_sync_sha, rollback_branch=plan.rollback_branch, **kwargs,
    )


def test_existing_main_counts_from_running_detached_code(checkout):
    root, old, tip = checkout
    plan = prepare(root)
    assert git(root, "rev-parse", "HEAD") == tip
    assert plan.commit_count == 1
    assert plan.pre_sync_sha == old
    pull(plan)
    assert git(root, "rev-parse", "HEAD") == tip


def test_same_commit_switch_is_still_a_noop(checkout):
    root, old, tip = checkout
    git(root, "checkout", "-q", "--detach", tip)
    assert prepare(root).commit_count == 0


@pytest.mark.parametrize("parked", [False, True])
def test_syntax_failure_returns_to_original_checkout_without_rewriting_main(checkout, parked):
    root, old, tip = checkout
    if parked:
        git(root, "checkout", "-qb", "feature")
        (root / "local.txt").write_text("local work\n", encoding="utf8")
        git(root, "add", ".")
        git(root, "commit", "-qm", "local")
        old = git(root, "rev-parse", "HEAD")
    # Commit an actually invalid critical file on main; no mocked syntax verdict.
    git(root, "checkout", "-q", "main")
    (root / "cli.py").write_text("def broken(\n", encoding="utf8")
    git(root, "commit", "-qam", "bad upstream")
    bad = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", "refs/remotes/origin/main", bad)
    git(root, "checkout", "-q", "feature" if parked else old)
    plan = prepare(root)
    with pytest.raises(SystemExit) as exc:
        pull(plan)
    assert exc.value.code == 1
    assert git(root, "rev-parse", "HEAD") == old
    assert git(root, "rev-parse", "--abbrev-ref", "HEAD") == ("feature" if parked else "HEAD")
    assert git(root, "rev-parse", "main") == bad
    assert (root / "cli.py").read_text(encoding="utf8") == "value = 1\n"



def test_rollback_restores_the_commit_when_the_parked_branch_is_taken(checkout, tmp_path):
    """Another worktree holding the parked branch must not leave the install on broken code."""
    root, old, tip = checkout
    git(root, "checkout", "-qb", "feature")
    git(root, "commit", "--allow-empty", "-qm", "local")
    feature_tip = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "-q", "main")
    (root / "cli.py").write_text("def broken(\n", encoding="utf8")
    git(root, "commit", "-qam", "bad upstream")
    git(root, "update-ref", "refs/remotes/origin/main", git(root, "rev-parse", "HEAD"))
    git(root, "checkout", "-q", "feature")
    plan = prepare(root)
    git(root, "worktree", "add", "-q", str(tmp_path / "other"), "feature")
    with pytest.raises(SystemExit):
        pull(plan)
    assert git(root, "rev-parse", "HEAD") == feature_tip
    assert git(root, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    assert (root / "cli.py").read_text(encoding="utf8") == "value = 1\n"

def test_locally_ahead_switch_still_needs_completion(checkout):
    root, old, tip = checkout
    git(root, "checkout", "-q", "--detach", tip)
    git(root, "commit", "--allow-empty", "-qm", "local detached commit")
    before = git(root, "rev-parse", "HEAD")
    plan = prepare(root)
    assert plan.commit_count != 0
    assert plan.switched_without_new_commits
    assert plan.pre_sync_sha == before
    assert git(root, "rev-parse", "HEAD") == tip


def test_merge_that_reports_success_without_moving_stale_main_is_refused(checkout, monkeypatch):
    """Detached on a side commit with a stale local main: the switch moves HEAD, so a baseline
    taken from the running code (or a bypass for "the switch already moved") would accept a
    merge that returned 0 but left HEAD on the stale main."""
    root, old, tip = checkout
    git(root, "checkout", "-q", "--detach", old)
    git(root, "commit", "--allow-empty", "-qm", "side")
    git(root, "branch", "-f", "main", old)
    plan = prepare(root)
    assert git(root, "rev-parse", "HEAD") == old
    real_git_run = update_cmd._git_run

    def merge_is_a_silent_noop(git_cmd, args, *a, **k):
        if args[:2] == ["merge", "--ff-only"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        return real_git_run(git_cmd, args, *a, **k)

    monkeypatch.setattr(update_cmd, "_git_run", merge_is_a_silent_noop)
    with pytest.raises(SystemExit):
        pull(plan)
    assert git(root, "rev-parse", "HEAD") == old


def test_stale_local_main_can_catch_up_to_original_detached_tip(checkout):
    root, old, tip = checkout
    git(root, "branch", "-f", "main", old)
    git(root, "checkout", "-q", "--detach", tip)
    plan = prepare(root)
    assert plan.commit_count != 0
    pull(plan)
    assert git(root, "rev-parse", "main") == tip


@pytest.mark.parametrize("upstream_result", ["original", "unchanged", "wrong-branch", "reverted"])
def test_fork_sync_after_stale_branch_repair(checkout, monkeypatch, upstream_result):
    root, old, tip = checkout
    git(root, "checkout", "-q", "main")
    git(root, "commit", "--allow-empty", "-qm", "upstream")
    upstream_tip = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "-q", "--detach", upstream_tip)
    git(root, "branch", "-f", "main", old)
    plan = prepare(root)

    def sync(*args, **kwargs):
        assert git(root, "rev-parse", "HEAD") == tip
        if upstream_result == "original":
            git(root, "merge", "--ff-only", upstream_tip)
        elif upstream_result == "wrong-branch":
            git(root, "checkout", "-qb", "wrong")
        elif upstream_result == "reverted":
            git(root, "reset", "--hard", old)
        return True

    monkeypatch.setattr(update_cmd._m(), "_sync_with_upstream_if_needed", sync)
    kwargs = dict(prompt_for_restore=False, gw_input_fn=None,
                  discard_local_changes=False, keep_stash=False,
                  pre_sync_sha=plan.pre_sync_sha, rollback_branch=plan.rollback_branch,
                  sync_upstream=True)
    if upstream_result in {"wrong-branch", "reverted"}:
        with pytest.raises(SystemExit) as exc:
            update_cmd._pull_updates(["git"], "main", plan.auto_stash_ref, **kwargs)
        assert exc.value.code == 1
    else:
        before_pull = update_cmd._pull_updates(["git"], "main", plan.auto_stash_ref, **kwargs)
        completed = []
        monkeypatch.setattr(update_cmd, "_complete_source_update", completed.append)
        request = {}
        update_cmd._apply_pulled_update(
            ["git"], "main", before_pull, plan,
            _windows_gateway_resume=None, completion_request=request,
        )
        assert completed == [request]
        assert request["expected_sha"] == (upstream_tip if upstream_result == "original" else tip)
        assert git(root, "rev-parse", "main") == (upstream_tip if upstream_result == "original" else tip)
        assert git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_early_fork_sync_without_push_still_completes(checkout, monkeypatch):
    root, old, tip = checkout
    git(root, "checkout", "-q", "main")
    git(root, "commit", "--allow-empty", "-qm", "upstream")
    upstream_tip = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "-q", "--detach", tip)
    git(root, "branch", "-f", "main", tip)

    def sync(*args, **kwargs):
        git(root, "merge", "--ff-only", upstream_tip)
        # Successful local sync need not push the new SHA to origin.
        return True

    monkeypatch.setattr(update_cmd._m(), "_sync_with_upstream_if_needed", sync)
    plan = prepare(root, is_fork=True)
    assert plan.upstream_checked
    assert git(root, "rev-parse", "HEAD") == upstream_tip
    assert git(root, "rev-parse", "origin/main") == tip
    before_pull = pull(plan, sync_upstream=True)
    completed = []
    monkeypatch.setattr(update_cmd, "_complete_source_update", completed.append)
    request = {}
    update_cmd._apply_pulled_update(
        ["git"], "main", before_pull, plan,
        _windows_gateway_resume=None, completion_request=request,
    )
    assert completed == [request]
    assert request["expected_sha"] == upstream_tip


def test_command_hands_off_after_existing_main_switch(update_tree, monkeypatch, capsys):
    t = update_tree
    monkeypatch.setattr("hermes_cli.update_owning_install.retarget_to_owning_install", lambda *_: None)
    run = subprocess.run

    def local_git_only(command, *args, **kwargs):
        assert Path(command[0]).name.lower() in {"git", "git.exe"}, command
        assert Path(kwargs["cwd"]).resolve() in {t.clone, t.origin}, command
        return run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", local_git_only)
    git(t.clone, "fetch", "-q", "origin", "main")
    git(t.clone, "branch", "-f", "main", "origin/main")
    git(t.clone, "checkout", "-q", "--detach", t.base)
    t.args.channel = "main"
    monkeypatch.setattr(update_cmd._m(), "_sync_with_upstream_if_needed", lambda *a, **k: True)
    update_cmd._m().cmd_update(t.args)
    assert git(t.clone, "rev-parse", "HEAD") == t.newer
    assert len(t.requests) == 1
    assert "Already up to date" not in capsys.readouterr().out


def test_fork_sync_round_trip_is_not_misclassified_as_noop(tmp_path, monkeypatch):
    """origin/main moves first, then the upstream sync returns HEAD to the SHA that was
    running before the update: still a successful branch repair, not a no-op."""
    root = tmp_path / "fork"
    init_repo(root, monkeypatch)

    def commit(value):
        (root / "state.txt").write_text(value + "\n", encoding="utf-8")
        git(root, "add", "state.txt")
        git(root, "commit", "-qm", value)
        return git(root, "rev-parse", "HEAD")

    old, fork_tip, upstream_tip = commit("old"), commit("fork"), commit("upstream")
    git(root, "checkout", "-q", "--detach", upstream_tip)
    git(root, "branch", "-f", "main", old)
    git(root, "update-ref", "refs/remotes/origin/main", fork_tip)
    git(root, "update-ref", "refs/remotes/upstream/main", upstream_tip)

    plan = prepare(root, is_fork=True)
    assert plan.commit_count != 0
    assert plan.pre_sync_sha == upstream_tip
    assert git(root, "rev-parse", "HEAD") == old

    def sync_upstream(*_args, **_kwargs):
        git(root, "merge", "--ff-only", "refs/remotes/upstream/main")
        return True

    monkeypatch.setattr(update_cmd._m(), "_sync_with_upstream_if_needed", sync_upstream)
    pull(plan, sync_upstream=True)

    assert git(root, "rev-parse", "HEAD") == upstream_tip
    assert git(root, "rev-parse", "main") == upstream_tip
    assert git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"
