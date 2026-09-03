"""Tests for the CLI ``/worktree`` command handler.

``/worktree`` (Copilot CLI-inspired) inspects or creates isolated git
worktrees mid-session. These drive the mixin handler against real git repos:
status/list output, ``new`` creation with cwd + TERMINAL_CWD retargeting,
named-tree collision refusal, and graceful non-repo degradation.
"""

import contextlib
import io
import os
import shutil
import subprocess

import pytest

import cli as cli_mod
from hermes_cli.cli_commands_mixin import CLICommandsMixin

requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git required"
)


class _Stub(CLICommandsMixin):
    def __init__(self):
        self.agent = None


def _run(stub, command):
    """Capture handler output.

    Worktree messages route through ``cli._cprint`` (the ANSI-aware
    prompt_toolkit renderer) since the garbled-escape fix; its output binds
    to the prompt_toolkit session output object, not ``sys.stdout``, so
    ``redirect_stdout`` alone cannot capture it. Mirror ``_cprint`` into the
    same buffer to keep asserting the user-visible text.
    """
    import re
    from unittest.mock import patch as _patch

    buf = io.StringIO()
    ansi = re.compile(r"\x1b\[[0-9;]*m")

    def _fake_cprint(text):
        buf.write(ansi.sub("", str(text)) + "\n")

    with contextlib.redirect_stdout(buf), _patch.object(cli_mod, "_cprint", _fake_cprint):
        stub._handle_worktree_command(command)
    return buf.getvalue()


def _git(repo, *args):
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
             "HOME": str(repo), "PATH": os.environ["PATH"]},
    )


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "a.txt").write_text("hello\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(cli_mod, "_active_worktree", None)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    yield repo
    # Leave tmp dir before pytest tears it down (worktree cwd may be inside).
    os.chdir(tmp_path)


@requires_git
def test_status_no_active_worktree(repo):
    out = _run(_Stub(), "/worktree")
    assert "No active worktree" in out
    assert "/worktree new" in out


def test_status_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "_active_worktree", None)
    out = _run(_Stub(), "/worktree")
    assert "not inside a git repository" in out


def test_new_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = _run(_Stub(), "/worktree new")
    assert "requires being inside a git repository" in out


@requires_git
def test_list_shows_worktrees(repo):
    out = _run(_Stub(), "/worktree list")
    assert str(repo) in out


@requires_git
def test_new_named_creates_and_retargets(repo):
    out = _run(_Stub(), "/worktree new fix-login")
    assert "Worktree ready" in out
    wt = repo / ".worktrees" / "fix-login"
    assert wt.is_dir()
    assert os.environ["TERMINAL_CWD"] == str(wt)
    assert os.path.realpath(os.getcwd()) == os.path.realpath(str(wt))
    assert cli_mod._active_worktree is not None
    assert cli_mod._active_worktree["branch"] == "hermes/fix-login"
    # Same file contents as the base commit.
    assert (wt / "a.txt").read_text() == "hello\n"


@requires_git
def test_new_named_collision_refused(repo):
    _run(_Stub(), "/worktree new dup")
    first = cli_mod._active_worktree
    os.chdir(repo)
    out = _run(_Stub(), "/worktree new dup")
    assert "already exists" in out
    # Active worktree not clobbered by the failed attempt.
    assert cli_mod._active_worktree == first


@requires_git
def test_new_unnamed_uses_random_hermes_prefix(repo):
    out = _run(_Stub(), "/worktree new")
    assert "Worktree ready" in out
    name = os.path.basename(cli_mod._active_worktree["path"])
    assert name.startswith("hermes-")


@requires_git
def test_new_sanitizes_name(repo):
    _run(_Stub(), "/worktree new " + "weird name!@#")
    name = os.path.basename(cli_mod._active_worktree["path"])
    assert name == "weird-name"


@requires_git
def test_unknown_subcommand(repo):
    out = _run(_Stub(), "/worktree frobnicate")
    assert "Unknown /worktree subcommand" in out
