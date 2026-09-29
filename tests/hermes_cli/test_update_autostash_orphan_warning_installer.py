"""The installer parks a dirty tree under its own stash name, and that park
has to age out the same way an update's does.

``hermes update`` names its autostash ``hermes-update-autostash-<stamp>``, and
``_warn_orphaned_update_autostashes`` surfaces entries older than the threshold
(#63717 problem 6). A re-run of the installer over an existing checkout
(``scripts/install.sh`` / ``scripts/install.ps1``) writes
``hermes-install-autostash-<stamp>`` with the same stamp before it switches or
resets branches — a user's local work parked by a path the update-side check
never looked at, so it aged out silently, exactly like the entries the check
was written to surface.
"""

import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli import update_cmd


def _git(cwd, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check
    )


def _make_repo_with_stash(tmp_path, name, age_days: float):
    """Real repo with one stash entry named ``name`` aged ``age_days``.

    Safe to call more than once on the same ``tmp_path``: a second entry is
    stashed on top of the first, so one repo can hold several.
    """
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git not available")
    if not (tmp_path / "tracked.txt").exists():
        _git(tmp_path, "init", "-q", "-b", "main")
        _git(tmp_path, "config", "user.email", "t@example.com")
        _git(tmp_path, "config", "user.name", "t")
        (tmp_path / "tracked.txt").write_text("v1\n")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-qm", "init")

    (tmp_path / "tracked.txt").write_text(f"local change {name}\n")
    _git(tmp_path, "stash", "push", "--include-untracked", "-m", name)
    return name


def _stamped(prefix: str, age_days: float) -> str:
    stamp = (datetime.now(timezone.utc) - timedelta(days=age_days)).strftime(
        "%Y%m%d-%H%M%S"
    )
    return f"{prefix}{stamp}"


def test_old_installer_autostash_is_surfaced(tmp_path, capsys):
    name = _make_repo_with_stash(
        tmp_path, _stamped("hermes-install-autostash-", 9), age_days=9
    )
    count = update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path)
    out = capsys.readouterr().out
    assert count == 1
    assert "leftover update autostash" in out
    # The notice must print the entry under its real name, not the update prefix.
    assert name in out
    assert "git stash apply" in out
    # Never a GC: the entry must still exist.
    assert name in _git(tmp_path, "stash", "list").stdout


def test_installer_and_update_autostashes_are_counted_together(tmp_path, capsys):
    _make_repo_with_stash(tmp_path, _stamped("hermes-install-autostash-", 9), 9)
    _make_repo_with_stash(tmp_path, _stamped("hermes-update-autostash-", 12), 12)
    count = update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path)
    out = capsys.readouterr().out
    assert count == 2
    assert "hermes-install-autostash-" in out
    assert "hermes-update-autostash-" in out


def test_fresh_installer_autostash_is_not_flagged(tmp_path, capsys):
    _make_repo_with_stash(tmp_path, _stamped("hermes-install-autostash-", 1), 1)
    count = update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path)
    assert count == 0
    assert "leftover update autostash" not in capsys.readouterr().out


def test_unstamped_plugin_autostash_is_left_alone(tmp_path, capsys):
    # plugins_cmd_git.py writes this name with no stamp; an entry whose age
    # cannot be read is left alone rather than guessed at.
    _make_repo_with_stash(tmp_path, "hermes-plugin-update-autostash", 0)
    count = update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path)
    assert count == 0
    assert "leftover update autostash" not in capsys.readouterr().out


def test_installer_name_without_a_stamp_is_left_alone(tmp_path, capsys):
    _make_repo_with_stash(tmp_path, "hermes-install-autostash-notadate", 0)
    count = update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path)
    assert count == 0
