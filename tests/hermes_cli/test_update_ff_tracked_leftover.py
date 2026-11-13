"""A tracked edit left on disk blocks ``merge --ff-only`` even though HEAD is an ancestor.

``hermes update`` stashes first. ``git stash push`` can still exit 0 and leave
that path modified (``Saved working directory and index state``). The
fast-forward then refuses to reset, because history divergence was not proven,
and the pull stops. The leftover has to be parked and the fast-forward retried
once. A reset that did not first stash those bytes would delete them.
"""

import subprocess
from pathlib import Path

import pytest

import hermes_cli.main as cli_main
from hermes_cli import update_cmd, update_cmd_git, update_cmd_stash

_TRACKED = "plugin-catalog/board-insights.yaml"


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def _ancestor_checkout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    tracked = repo / _TRACKED
    tracked.parent.mkdir()
    tracked.write_text("upstream-old\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "old")
    old = _git(repo, "rev-parse", "HEAD").strip()
    tracked.write_text("upstream-new\n")
    _git(repo, "commit", "-qam", "new")
    new = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "update-ref", "refs/remotes/origin/main", new)
    _git(repo, "reset", "-q", "--hard", old)
    return repo, old, new


def test_fast_forward_parks_a_tracked_leftover_and_retries(tmp_path, monkeypatch):
    """HEAD is an ancestor, one tracked file is still dirty, and the pull moves."""
    repo, old, new = _ancestor_checkout(tmp_path)
    (repo / _TRACKED).write_text("mine\n")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", repo)

    update_cmd._move_checkout_to(["git"], "main", "origin/main", new, old)

    assert _git(repo, "rev-parse", "HEAD").strip() == new
    assert (repo / _TRACKED).read_text() == "upstream-new\n"
    assert "hermes-update-autostash-" in _git(repo, "stash", "list")
    assert "mine" in _git(repo, "stash", "show", "-p")


def test_fast_forward_refusal_does_not_stash_a_clean_tree(tmp_path, monkeypatch):
    """A failed fast-forward with a clean tree still exits and creates no stash."""
    repo, old, _new = _ancestor_checkout(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", repo)

    with pytest.raises(SystemExit):
        update_cmd._move_checkout_to(
            ["git"], "main", "origin/main", "deadbeef" * 5, old
        )

    assert _git(repo, "stash", "list") == ""
    assert _git(repo, "rev-parse", "HEAD").strip() == old
    assert (repo / _TRACKED).read_text() == "upstream-old\n"


def _stash_push_leaves_the_file(monkeypatch):
    """``stash push`` exits 0 and the tracked path is modified again afterwards."""
    real = update_cmd_git._git_run

    def leave_dirty(git_cmd, args, cwd=None, **kwargs):
        result = real(git_cmd, args, cwd=cwd, **kwargs)
        if list(args)[:2] == ["stash", "push"] and result.returncode == 0 and cwd is not None:
            (Path(cwd) / _TRACKED).write_text("mine\n")
        return result

    monkeypatch.setattr(update_cmd_git, "_git_run", leave_dirty)


def test_fast_forward_retries_when_stash_exits_zero_and_leaves_the_file(
        tmp_path, monkeypatch):
    """The bytes survive in the autostash, then the fast-forward runs."""
    repo, old, new = _ancestor_checkout(tmp_path)
    (repo / _TRACKED).write_text("mine\n")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", repo)
    _stash_push_leaves_the_file(monkeypatch)

    update_cmd._move_checkout_to(["git"], "main", "origin/main", new, old)

    assert _git(repo, "rev-parse", "HEAD").strip() == new
    assert (repo / _TRACKED).read_text() == "upstream-new\n"
    assert "mine" in _git(repo, "stash", "show", "-p")


def test_fast_forward_keeps_bytes_the_new_stash_does_not_list(tmp_path, monkeypatch):
    """A dirty path missing from the stash is not reset, and the pull stops."""
    repo, old, new = _ancestor_checkout(tmp_path)
    (repo / _TRACKED).write_text("mine\n")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", repo)
    _stash_push_leaves_the_file(monkeypatch)
    monkeypatch.setattr(update_cmd_stash, "_stash_tracked_names", lambda *_a, **_k: set())

    with pytest.raises(SystemExit):
        update_cmd._move_checkout_to(["git"], "main", "origin/main", new, old)

    assert (repo / _TRACKED).read_text() == "mine\n"
    assert _git(repo, "rev-parse", "HEAD").strip() == old
