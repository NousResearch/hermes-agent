"""#120179: `hermes update` autostash must not sweep untracked files/dirs.

Real-git regression test: a checkout with a tracked modification PLUS an
untracked in-tree package (third-party adapter) must keep the untracked
files in the working tree across stash (+-restore); the autostash must
carry tracked changes only (no ``^3`` untracked parent).
"""
import shutil
import subprocess

import pytest

from hermes_cli import main as hermes_main


def _git(cwd, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=check
    )


@pytest.fixture()
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tracked.txt").write_text("v1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_autostash_leaves_untracked_package_in_tree(repo):
    """Tracked edit + untracked package: stash keeps untracked files in place."""
    (repo / "tracked.txt").write_text("v2 local change\n")
    pkg = repo / "gateway" / "butler_bridge"
    pkg.mkdir(parents=True)
    (pkg / "server.py").write_text("print('adapter')\n")
    (pkg / "__init__.py").write_text("")

    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], repo)
    assert stash_ref, "tracked modification must still be stashed"

    # THE BUG (#120179): untracked files were swept into the stash and the
    # tree was left with an empty directory shell.
    assert (pkg / "server.py").read_text() == "print('adapter')\n"
    assert (pkg / "__init__.py").exists()

    # The stash must be tracked-only: no ^3 untracked parent.
    has_u3 = _git(repo, "rev-parse", "--verify", "-q", f"{stash_ref}^3", check=False)
    assert has_u3.returncode != 0, "autostash must not contain untracked files"

    # Tracked change really was stashed (update window sees a clean tree).
    assert (repo / "tracked.txt").read_text() == "v1\n"

    # Restore round-trips the tracked edit without touching the package.
    _git(repo, "stash", "apply", stash_ref)
    assert (repo / "tracked.txt").read_text() == "v2 local change\n"
    assert (pkg / "server.py").read_text() == "print('adapter')\n"


def test_untracked_only_tree_needs_no_stash(repo):
    """Untracked-only dirt must not trigger a stash at all."""
    (repo / "new_adapter.py").write_text("print('new')\n")

    assert hermes_main._stash_local_changes_if_needed(["git"], repo) is None
    assert (repo / "new_adapter.py").read_text() == "print('new')\n"
    assert _git(repo, "stash", "list", check=False).stdout.strip() == ""
