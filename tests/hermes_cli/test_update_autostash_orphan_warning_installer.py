"""An installer's parked stash has to age out the way the updater's does.

``hermes update`` names its autostash ``hermes-update-autostash-<stamp>`` and
surfaces entries older than the threshold (#63717 problem 6). A re-run of the
installer over an existing checkout (``scripts/install.sh``,
``scripts/install.ps1``) writes ``hermes-install-autostash-<stamp>`` with the
same stamp before it switches or resets branches — the user's own unmerged
patches, parked just as permanently and invisible to the check that exists to
surface exactly this.
"""

import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli import update_cmd


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


def _stamped(prefix: str, age_days: int) -> str:
    stamp = (datetime.now(timezone.utc) - timedelta(days=age_days)).strftime(
        "%Y%m%d-%H%M%S"
    )
    return f"{prefix}{stamp}"


def _repo_with_stashes(tmp_path, *names: str) -> None:
    """Real repo holding one stash entry per name. Repeatable on one path."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    if not (tmp_path / "tracked.txt").exists():
        _git(tmp_path, "init", "-q", "-b", "main")
        _git(tmp_path, "config", "user.email", "t@example.com")
        _git(tmp_path, "config", "user.name", "t")
        (tmp_path / "tracked.txt").write_text("v1\n")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-qm", "init")
    for i, name in enumerate(names):
        (tmp_path / "tracked.txt").write_text(f"local change {i}\n")
        _git(tmp_path, "stash", "push", "--include-untracked", "-m", name)


def test_installer_and_update_autostashes_are_surfaced_together(tmp_path, capsys):
    installed = _stamped("hermes-install-autostash-", 9)
    updated = _stamped("hermes-update-autostash-", 12)
    _repo_with_stashes(tmp_path, installed, updated)

    assert update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path) == 2

    out = capsys.readouterr().out
    assert "leftover update autostash" in out
    # Each entry is named by the prefix it was actually created with.
    assert installed in out
    assert updated in out
    assert "git stash apply" in out
    # Never a GC: the entries must still exist.
    assert installed in _git(tmp_path, "stash", "list").stdout


def test_fresh_installer_autostash_is_not_flagged(tmp_path, capsys):
    _repo_with_stashes(tmp_path, _stamped("hermes-install-autostash-", 1))

    assert update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path) == 0
    assert "leftover update autostash" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "name",
    [
        "hermes-plugin-update-autostash",  # plugins_cmd_git.py: no stamp
        "hermes-install-autostash-notadate",  # installer prefix, unreadable age
        "my own stash from 20200101-000000",  # not a Hermes name at all
    ],
)
def test_entries_without_a_readable_age_are_left_alone(tmp_path, capsys, name):
    _repo_with_stashes(tmp_path, name)

    assert update_cmd._warn_orphaned_update_autostashes(["git"], tmp_path) == 0
    assert "leftover update autostash" not in capsys.readouterr().out
