"""The Desktop hand-off window names the git step of `hermes update` while it runs.

posix.sh publishes "Updating code and dependencies" before it starts `hermes update`,
and nothing on the Python side published again until the dependency sync. On a
treeless (``--filter=tree:0``) install the fast-forward in between is where the new
tree and the changed files are lazy-fetched from origin, one round trip each; a real
Desktop update spent 2m20s there under the unchanged shim stage and was taken for hung.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd, update_stage


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """An install at commit A whose fetched ``origin/main`` is B."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@example.invalid")
    _git(origin, "config", "user.name", "t")
    (origin / "utils.py").write_text("OLD = 1\n", encoding="utf-8")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "A")
    (origin / "utils.py").write_text("NEW = 1\n", encoding="utf-8")
    _git(origin, "commit", "-qam", "B")
    root = tmp_path / "install"
    _git(tmp_path, "clone", "-q", str(origin), str(root))
    _git(root, "reset", "-q", "--hard", "HEAD~1")
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", root)
    return root, _git(root, "rev-parse", "origin/main")


def test_the_window_names_the_git_step_while_git_moves_the_tree(checkout, tmp_path, monkeypatch):
    root, target = checkout
    status = tmp_path / "hermes-update-status.4242"
    status.write_text(json.dumps({"status": "running", "message": "Updating code and dependencies"}))
    monkeypatch.setenv(update_stage.STATUS_FILE_ENV, str(status))
    on_screen = []
    real = update_cmd._git_run

    def watching_git_run(git_cmd, args, *rest, **kw):
        if args[:1] == ["merge"]:
            on_screen.append(json.loads(status.read_text(encoding="utf-8")))
        return real(git_cmd, args, *rest, **kw)

    monkeypatch.setattr(update_cmd, "_git_run", watching_git_run)
    update_cmd._pull_updates(["git"], "main", None, prompt_for_restore=False, gw_input_fn=None,
                             discard_local_changes=False, keep_stash=False)

    assert _git(root, "rev-parse", "HEAD") == target
    assert on_screen == [{"status": "running", "message": "Applying code changes"}]

