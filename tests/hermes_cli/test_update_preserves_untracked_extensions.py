"""Regression for #120179: update autostash must not park untracked extensions."""

import subprocess
from pathlib import Path

from hermes_cli.update_cmd_stash import _stash_local_changes_if_needed


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def test_autostash_leaves_untracked_extension_during_tracked_restore(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    source = tmp_path / "tracked.py"
    source.write_text("original\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-qm", "baseline")

    extension = tmp_path / "gateway" / "local_extension" / "server.py"
    extension.parent.mkdir(parents=True)
    extension.write_text("installed\n")
    source.write_text("customized\n")

    stash_ref = _stash_local_changes_if_needed(["git"], tmp_path)
    assert stash_ref
    assert source.read_text() == "original\n"
    assert extension.read_text() == "installed\n"
    # No untracked parent is created; even a parked or discarded stash cannot remove the extension.
    parent = subprocess.run(["git", "rev-parse", "--verify", "-q", f"{stash_ref}^3"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert parent.returncode != 0
    _git(tmp_path, "stash", "apply", stash_ref)
    assert source.read_text() == "customized\n"
    assert extension.read_text() == "installed\n"


def test_untracked_only_does_not_create_or_reuse_autostash(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "tracked.py").write_text("original\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-qm", "baseline")

    extension = tmp_path / "gateway" / "local_extension" / "server.py"
    extension.parent.mkdir(parents=True)
    extension.write_text("installed\n")
    assert _stash_local_changes_if_needed(["git"], tmp_path) is None
    assert _git(tmp_path, "stash", "list").stdout == ""
    assert extension.read_text() == "installed\n"
