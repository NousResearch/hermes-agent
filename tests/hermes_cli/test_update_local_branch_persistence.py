"""Committed operator fixes survive the supported in-place branch update policy."""
from pathlib import Path
import subprocess

import pytest

# Import before the per-test I/O guard: bootstrap is interpreter initialization.
from hermes_cli import main, update_cmd


BRANCH = "local/macos-launchdaemon-update"


def git(root, *args, check=True):
    return subprocess.run(["git", *args], cwd=root, check=check,
                          capture_output=True, text=True)


def commit(root, name, content):
    (root / name).write_text(content, encoding="utf-8")
    git(root, "add", name)
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", name)
    return git(root, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def maintained_checkout(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-qb", "main")
    commit(upstream, "gateway-fix.txt", "base\n")
    root = tmp_path / "checkout"
    git(tmp_path, "clone", "-q", str(upstream), str(root))
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "checkout", "-qb", BRANCH)
    fix = commit(root, "gateway-fix.txt", "operator fix\n")
    monkeypatch.setattr(main, "PROJECT_ROOT", root)
    # Exercise the real configuration reader, not a stub of the branch guard.
    home = tmp_path / "operator-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "updates:\n  parked_branch_strategy: update_in_place\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return upstream, root, fix


def prepare(root):
    return update_cmd._prepare_checkout_for_update(
        ["git"], "main", BRANCH, is_fork=False, assume_yes=True,
        gateway_mode=False, gw_input_fn=None, switch_branch=False,
        _windows_gateway_resume=None)


def apply(plan):
    return update_cmd._pull_updates(
        ["git"], "main", plan.auto_stash_ref, prompt_for_restore=False,
        gw_input_fn=None, discard_local_changes=False, keep_stash=True,
        in_place_update=plan.in_place_update)


def test_keep_stash_updates_merge_committed_fix_without_parking_it(maintained_checkout):
    upstream, root, fix = maintained_checkout
    for n in range(2):
        target = commit(upstream, f"upstream-{n}.txt", "new upstream work\n")
        git(root, "fetch", "-q", "origin", "main")
        plan = prepare(root)
        assert plan.in_place_update and not plan.parked_branch_switched
        assert plan.auto_stash_ref is None and plan.commit_count > 0
        apply(plan)
        assert git(root, "branch", "--show-current").stdout.strip() == BRANCH
        assert git(root, "merge-base", "--is-ancestor", fix, "HEAD", check=False).returncode == 0
        assert git(root, "merge-base", "--is-ancestor", target, "HEAD", check=False).returncode == 0
        assert (root / "gateway-fix.txt").read_text() == "operator fix\n"
        assert (root / f"upstream-{n}.txt").exists()
        assert git(root, "status", "--porcelain").stdout == ""
        assert git(root, "stash", "list").stdout == ""


def test_conflicting_upstream_update_keeps_committed_fix_active(maintained_checkout):
    upstream, root, fix = maintained_checkout
    commit(upstream, "gateway-fix.txt", "conflicting upstream change\n")
    git(root, "fetch", "-q", "origin", "main")
    plan = prepare(root)
    assert plan.in_place_update and plan.auto_stash_ref is None
    with pytest.raises(SystemExit) as error:
        apply(plan)
    assert error.value.code == 1
    assert git(root, "rev-parse", "HEAD").stdout.strip() == fix
    assert git(root, "branch", "--show-current").stdout.strip() == BRANCH
    assert git(root, "status", "--porcelain").stdout == ""
    assert (root / "gateway-fix.txt").read_text() == "operator fix\n"
