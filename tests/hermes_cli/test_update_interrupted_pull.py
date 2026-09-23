"""A `hermes update` killed while git writes the new tree must leave a recoverable install.

Git rewrites the checkout file by file and moves HEAD last, so a kill in between leaves HEAD on the
old commit with some files already new — a mix that fails at import in every entry point. The
updater brackets the move with a marker; the next launch (``_early_recovery``, before any other
checkout import) puts the old tree back so ``hermes update`` can simply run again.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import _early_recovery as er
from hermes_cli import update_cmd


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """An install at commit A whose fetched ``origin/main`` is B (modifies, deletes, adds, flips a mode)."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@example.invalid")
    _git(origin, "config", "user.name", "t")
    files = {"utils.py": "OLD = 1\n", "other.py": "a = 1\n", "gone.py": "x = 1\n", "cut.py": "c = 1\n",
             "blank.py": "b = 1\n", "half.py": "h = 1\n", "tool.sh": "echo\n"}
    for name, body in files.items():
        (origin / name).write_text(body, encoding="utf-8", newline="")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "A")
    for name, body in {"utils.py": "NEW = 1\n", "other.py": "a = 2\n", "cut.py": "c = 2\n",
                       "blank.py": "b = 2\n", "half.py": "h = 2  # long enough to span pages\n"}.items():
        (origin / name).write_text(body, encoding="utf-8", newline="")
    (origin / "gone.py").unlink()
    (origin / "newpkg").mkdir()
    (origin / "newpkg" / "__init__.py").write_text("from utils import NEW\n", encoding="utf-8", newline="")
    _git(origin, "add", "-A")
    _git(origin, "update-index", "--chmod=+x", "tool.sh")
    _git(origin, "commit", "-qm", "B")
    root = tmp_path / "install"
    _git(tmp_path, "clone", "-q", str(origin), str(root))
    _git(root, "reset", "-q", "--hard", "HEAD~1")
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", root)
    return root, _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "origin/main")


def _pull(root: Path) -> None:
    update_cmd._pull_updates(["git"], "main", None, prompt_for_restore=False, gw_input_fn=None,
                             discard_local_changes=False, keep_stash=False)


def test_killed_pull_is_restored_on_next_launch_and_update_reruns(checkout, monkeypatch):
    root, a, b = checkout
    real = update_cmd._git_run

    def dying_git_run(git_cmd, args, *rest, **kw):
        if args[:1] == ["merge"]:
            # Git rewrites a file as unlink, create, write: the kill lands inside one of those.
            (root / "utils.py").write_text("NEW = 1\n", encoding="utf-8", newline="")
            (root / "cut.py").unlink()
            (root / "blank.py").write_bytes(b"")
            (root / "half.py").write_bytes(b"h = 2  # long")  # a multi-page write cut short
            (root / "newpkg").mkdir()
            (root / "newpkg" / "__init__.py").write_text("from utils import NEW\n", encoding="utf-8", newline="")
            (root / ".git" / "index.lock").touch()
            raise KeyboardInterrupt  # SIGKILL: nothing after this line of the updater runs
        return real(git_cmd, args, *rest, **kw)

    monkeypatch.setattr(update_cmd, "_git_run", dying_git_run)
    with pytest.raises(KeyboardInterrupt):
        _pull(root)
    monkeypatch.setattr(update_cmd, "_git_run", real)
    assert _git(root, "rev-parse", "HEAD") == a  # the torn state: HEAD old, some files already new
    marker = er.interrupted_pull_marker(root)
    recorded = marker.read_text(encoding="utf-8")
    assert f"pid={os.getpid()}" in recorded and f"target={b}" in recorded  # the commit, not the ref name
    # The user re-applies their stash to a file the update also changes (git had not written it yet).
    (root / "other.py").write_text("a = 1  # my edit\n", encoding="utf-8", newline="")

    # Another `hermes` launched while an update is mid-pull must not race its git.
    updater = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        marker.write_text(recorded.replace(f"pid={os.getpid()}", f"pid={updater.pid}"), encoding="utf-8",
                          newline="")
        assert er.restore_interrupted_pull(root) is False
        assert marker.exists() and (root / ".git" / "index.lock").exists()
    finally:
        updater.kill()
        updater.wait()

    # A retry in a container gets the killed updater's pid: our own pid is never a live owner.
    marker.write_text(recorded, encoding="utf-8", newline="")
    assert er.restore_interrupted_pull(root) is True, "restored files mean the caller must relaunch"

    assert _git(root, "rev-parse", "HEAD") == a
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == "M other.py"
    assert (root / "other.py").read_text(encoding="utf-8") == "a = 1  # my edit\n", "the user's edit survives"
    assert not (root / "newpkg").exists() and not (root / ".git" / "index.lock").exists()
    assert not marker.exists()
    (root / "other.py").write_text("a = 1\n", encoding="utf-8", newline="")
    _pull(root)  # `hermes update` again: a normal fast-forward
    assert _git(root, "rev-parse", "HEAD") == b and not marker.exists()


def test_restore_never_touches_user_work_when_git_wrote_nothing(checkout):
    """sys.exit on a merge conflict is not a kill, and a marker git never acted on restores nothing."""
    root, a, b = checkout
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    _git(root, "checkout", "-q", "-b", "mywork")
    (root / "other.py").write_text("a = 'mine'\n", encoding="utf-8", newline="")
    _git(root, "commit", "-qam", "local work that conflicts upstream")
    with pytest.raises(SystemExit):
        _pull(root)
    marker = er.interrupted_pull_marker(root)
    assert not marker.exists()

    # Even a leftover marker (an older updater, or a kill mid-reconcile) stays out of the user's way:
    # following the printed advice leaves a merge in progress, and edits git never wrote are theirs.
    stale = f"pid=0\npre={_git(root, 'rev-parse', 'HEAD')}\ntarget={b}\nstash=\n"
    marker.write_text(stale, encoding="utf-8", newline="")
    merge = subprocess.run(["git", "-C", str(root), "merge", "origin/main"],
                           capture_output=True, text=True, encoding="utf-8")
    assert (root / ".git" / "MERGE_HEAD").exists(), merge.stdout + merge.stderr
    (root / "utils.py").write_text("OLD = 1  # resolved by hand\n", encoding="utf-8", newline="")
    before = _git(root, "status", "--porcelain", "--untracked-files=all")
    assert er.restore_interrupted_pull(root) is False
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == before
    _git(root, "reset", "-q", "--hard")  # the user gives up on the merge
    (root / "utils.py").write_text("OLD = 1  # my stash, re-applied\n", encoding="utf-8", newline="")
    # tool.sh only changes mode upstream and git never reached it: nothing to restore, no relaunch.
    assert er.restore_interrupted_pull(root) is False
    assert (root / "utils.py").read_text(encoding="utf-8") == "OLD = 1  # my stash, re-applied\n"
    assert not marker.exists(), "git wrote nothing: the marker is spent"
    # A target git no longer knows (gc, re-clone) can never be compared against: drop the marker.
    marker.write_text(stale.replace(b, "0" * 40), encoding="utf-8", newline="")
    assert er.restore_interrupted_pull(root) is False and not marker.exists()
