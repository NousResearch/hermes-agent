"""Orphaned update-autostash surfacing (#63717 problem 6).

``hermes update`` can legitimately leave an autostash behind (--keep-stash
parks it; a conflicted restore preserves it), but nothing ever mentioned those
entries again — they persisted invisibly for weeks. ``hermes update`` now
warns about ``hermes-update-autostash-*`` entries older than the threshold.
Behavioral tests use real git repos; no production mocking of the code under
test.
"""

import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli import update_cmd


def _git(cwd, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check
    )


def _make_repo_with_autostash(tmp_path, age_days: float):
    """Real repo with one hermes-update-autostash entry aged ``age_days``."""
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git not available")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tracked.txt").write_text("v1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")

    (tmp_path / "tracked.txt").write_text("local change\n")
    stamp = (
        datetime.now(timezone.utc) - timedelta(days=age_days)
    ).strftime("%Y%m%d-%H%M%S")
    name = f"hermes-update-autostash-{stamp}"
    _git(tmp_path, "stash", "push", "--include-untracked", "-m", name)
    return name


def test_old_autostash_is_surfaced(tmp_path, capsys):
    name = _make_repo_with_autostash(tmp_path, age_days=9)
    count = update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path)
    out = capsys.readouterr().out
    assert count == 1
    assert "leftover update autostash" in out
    assert name in out
    assert "git stash apply" in out
    # Never a GC: the entry must still exist.
    listed = _git(tmp_path, "stash", "list").stdout
    assert name in listed


def test_fresh_autostash_is_not_flagged(tmp_path, capsys):
    _make_repo_with_autostash(tmp_path, age_days=1)
    count = update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path)
    assert count == 0
    assert "leftover update autostash" not in capsys.readouterr().out


def test_non_hermes_stash_is_ignored(tmp_path, capsys):
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git not available")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tracked.txt").write_text("v1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    (tmp_path / "tracked.txt").write_text("user's own WIP\n")
    _git(tmp_path, "stash", "push", "-m", "my own stash from 20200101-000000")
    count = update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path)
    assert count == 0
    assert "leftover update autostash" not in capsys.readouterr().out


def test_unparseable_autostash_timestamp_left_alone(tmp_path, capsys):
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git not available")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tracked.txt").write_text("v1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    (tmp_path / "tracked.txt").write_text("change\n")
    _git(tmp_path, "stash", "push", "-m", "hermes-update-autostash-notadate")
    count = update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path)
    assert count == 0


def test_git_failure_is_nonfatal(tmp_path):
    # Not a git repo at all — must return 0, not raise.
    assert update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path) == 0
