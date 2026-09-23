"""A `hermes update` killed while git writes the new tree must leave a recoverable install.

Git rewrites the checkout file by file and moves HEAD last, so a kill in between leaves HEAD on the
old commit with some files already new — a mix that fails at import in every entry point. The
updater brackets the move with a marker; the next launch (``_early_recovery``, before any other
checkout import) puts the old tree back so ``hermes update`` can simply run again.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import _early_recovery as er
from hermes_cli import update_cmd


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """An install at commit A whose fetched ``origin/main`` is B (modifies, deletes, adds a package)."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@example.invalid")
    _git(origin, "config", "user.name", "t")
    for name, body in {"utils.py": "OLD = 1\n", "gone.py": "x = 1\n", "notes.md": "user file\n"}.items():
        (origin / name).write_text(body)
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "A")
    (origin / "utils.py").write_text("NEW = 1\n")
    (origin / "gone.py").unlink()
    (origin / "newpkg").mkdir()
    (origin / "newpkg" / "__init__.py").write_text("from utils import NEW\n")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "B")
    root = tmp_path / "install"
    _git(tmp_path, "clone", "-q", str(origin), str(root))
    _git(root, "reset", "-q", "--hard", "HEAD~1")
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", root)
    return root, _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "origin/main")


def _pull(root: Path) -> None:
    update_cmd._pull_updates(["git"], "main", None, prompt_for_restore=False, gw_input_fn=None,
                             discard_local_changes=False, keep_stash=False)


def _kill_mid_pull(root: Path, monkeypatch) -> None:
    """The fast-forward writes one new file, holds index.lock, and the process dies."""
    real = update_cmd._git_run

    def dying_git_run(git_cmd, args, *a, **kw):
        if args[:1] == ["merge"]:
            (root / "newpkg").mkdir()
            (root / "newpkg" / "__init__.py").write_text("from utils import NEW\n")
            (root / ".git" / "index.lock").touch()
            raise KeyboardInterrupt  # SIGKILL: nothing after this line of the updater runs
        return real(git_cmd, args, *a, **kw)

    monkeypatch.setattr(update_cmd, "_git_run", dying_git_run)
    with pytest.raises(KeyboardInterrupt):
        _pull(root)
    monkeypatch.setattr(update_cmd, "_git_run", real)


def test_killed_pull_is_restored_on_next_launch_and_update_reruns(checkout, monkeypatch):
    root, a, b = checkout
    (root / "notes.md").write_text("edited while bricked\n")  # a path the update never touches
    _kill_mid_pull(root, monkeypatch)
    assert _git(root, "rev-parse", "HEAD") == a  # the torn state: HEAD old, newpkg already new
    marker = er.interrupted_pull_marker(root)
    # A retry in a container gets the killed updater's pid: our own pid is never a live owner.
    assert f"pid={os.getpid()}" in marker.read_text()

    assert er.restore_interrupted_pull(root) is True, "restored files mean the caller must relaunch"

    assert _git(root, "rev-parse", "HEAD") == a
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == "M notes.md"
    assert not (root / "newpkg").exists() and not (root / ".git" / "index.lock").exists()
    assert not marker.exists()
    (root / "notes.md").write_text("user file\n")
    _pull(root)  # `hermes update` again: a normal fast-forward
    assert _git(root, "rev-parse", "HEAD") == b and not marker.exists()


def test_pull_marker_of_a_live_updater_is_left_alone(checkout, monkeypatch):
    root, a, _b = checkout
    _kill_mid_pull(root, monkeypatch)
    marker = er.interrupted_pull_marker(root)
    # Another `hermes` launched while an update is mid-pull must not race its git.
    marker.write_text(marker.read_text().replace(f"pid={os.getpid()}", f"pid={os.getppid()}"))

    assert er.restore_interrupted_pull(root) is False

    assert marker.exists() and (root / "newpkg" / "__init__.py").exists()
    assert (root / ".git" / "index.lock").exists()
