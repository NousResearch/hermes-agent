"""Real Git regression coverage for updates that land via an existing branch."""
import subprocess

import pytest

from hermes_cli import update_cmd
from tests.hermes_cli.test_update_target_identity import update_tree  # noqa: F401


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.com")
    (root / "cli.py").write_text("value = 1\n", encoding="utf8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "old")
    old = git(root, "rev-parse", "HEAD")
    (root / "cli.py").write_text("value = 2\n", encoding="utf8")
    git(root, "commit", "-qam", "new")
    tip = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", "refs/remotes/origin/main", tip)
    git(root, "checkout", "-q", "--detach", old)
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
    return root, old, tip


def prepare(root):
    return update_cmd._prepare_checkout_for_update(
        ["git"], "main", update_cmd._current_branch_name(["git"], check=True),
        is_fork=False, assume_yes=True, gateway_mode=False, gw_input_fn=None,
        switch_branch=False, _windows_gateway_resume=None,
    )


def test_existing_main_counts_from_running_detached_code(checkout):
    root, old, tip = checkout
    plan = prepare(root)
    assert git(root, "rev-parse", "HEAD") == tip
    assert plan.commit_count == 1
    assert plan.pre_sync_sha == old
    update_cmd._pull_updates(
        ["git"], "main", plan.auto_stash_ref, prompt_for_restore=False,
        gw_input_fn=None, discard_local_changes=False, keep_stash=False,
        pre_sync_sha=plan.pre_sync_sha,
        rollback_branch=plan.rollback_branch,
    )
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
        update_cmd._pull_updates(
            ["git"], "main", plan.auto_stash_ref, prompt_for_restore=False,
            gw_input_fn=None, discard_local_changes=False, keep_stash=False,
            pre_sync_sha=plan.pre_sync_sha, rollback_branch=plan.rollback_branch,
        )
    assert exc.value.code == 1
    assert git(root, "rev-parse", "HEAD") == old
    assert git(root, "rev-parse", "--abbrev-ref", "HEAD") == ("feature" if parked else "HEAD")
    assert git(root, "rev-parse", "main") == bad
    assert (root / "cli.py").read_text(encoding="utf8") == "value = 1\n"


def test_locally_ahead_switch_still_needs_completion(checkout):
    root, old, tip = checkout
    git(root, "checkout", "-q", "--detach", tip)
    git(root, "commit", "--allow-empty", "-qm", "local detached commit")
    before = git(root, "rev-parse", "HEAD")
    plan = prepare(root)
    assert plan.commit_count != 0
    assert plan.pre_sync_sha == before
    assert git(root, "rev-parse", "HEAD") == tip


def test_stale_local_main_can_catch_up_to_original_detached_tip(checkout):
    root, old, tip = checkout
    git(root, "branch", "-f", "main", old)
    git(root, "checkout", "-q", "--detach", tip)
    plan = prepare(root)
    assert plan.commit_count != 0
    update_cmd._pull_updates(
        ["git"], "main", plan.auto_stash_ref, prompt_for_restore=False,
        gw_input_fn=None, discard_local_changes=False, keep_stash=False,
        pre_sync_sha=plan.pre_sync_sha, rollback_branch=plan.rollback_branch,
    )
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


def test_command_hands_off_after_existing_main_switch(update_tree, monkeypatch, capsys):
    t = update_tree
    monkeypatch.setattr("hermes_cli.update_owning_install.retarget_to_owning_install", lambda *_: None)
    run = subprocess.run

    def local_git_only(command, *args, **kwargs):
        from pathlib import Path
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
