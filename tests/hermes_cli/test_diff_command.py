"""Tests for the CLI ``/diff`` command handler.

``/diff`` shows git changes in the working directory (unstaged + untracked by
default; ``staged``/``all`` modes) and ``/diff session`` shows the cumulative
checkpoint-baseline diff of everything Hermes changed. These drive the mixin
handler against real git repos (default modes) and a stubbed checkpoint
manager (session mode), asserting rendering, ``--stat``, and graceful
degradation.
"""

import contextlib
import io
import shutil
import subprocess

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin

requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git required"
)


class _Console:
    def __init__(self, sink):
        self._sink = sink

    def print(self, obj, **kwargs):
        self._sink.write(getattr(obj, "plain", str(obj)) + "\n")


class _Mgr:
    def __init__(self, result, enabled=True):
        self.enabled = enabled
        self._result = result
        self.calls = []

    def session_diff(self, cwd):
        self.calls.append(cwd)
        return self._result


class _Agent:
    def __init__(self, mgr):
        self._checkpoint_mgr = mgr


class _Stub(CLICommandsMixin):
    def __init__(self, agent=None):
        self.agent = agent


def _run(stub, command):
    buf = io.StringIO()
    stub.console = _Console(buf)
    with contextlib.redirect_stdout(buf):
        stub._handle_diff_command(command)
    return buf.getvalue()


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "HOME": str(repo),
                        "PATH": __import__("os").environ["PATH"]})


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    d = tmp_path / "repo"
    d.mkdir()
    _git(d, "init", "-q")
    (d / "main.py").write_text("print('hello')\n")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "init")
    monkeypatch.setenv("TERMINAL_CWD", str(d))
    return d


# ---------------------------------------------------------------------------
# Default (working-tree) mode — real git
# ---------------------------------------------------------------------------

@requires_git
def test_diff_clean_repo_reports_no_changes(repo):
    out = _run(_Stub(), "/diff")
    assert "No changes" in out


@requires_git
def test_diff_shows_unstaged_changes(repo):
    (repo / "main.py").write_text("print('changed')\n")
    out = _run(_Stub(), "/diff")
    assert "Unstaged" in out
    assert "-print('hello')" in out
    assert "+print('changed')" in out


@requires_git
def test_diff_lists_untracked_files(repo):
    (repo / "newfile.py").write_text("n = 1\n")
    out = _run(_Stub(), "/diff")
    assert "Untracked" in out
    assert "newfile.py" in out
    assert "+n = 1" in out


@requires_git
def test_diff_stat_suppresses_body(repo):
    (repo / "main.py").write_text("print('changed')\n")
    out = _run(_Stub(), "/diff --stat")
    assert "main.py" in out
    assert "+print('changed')" not in out


@requires_git
def test_diff_staged_mode(repo):
    (repo / "main.py").write_text("print('staged')\n")
    _git(repo, "add", "main.py")
    out = _run(_Stub(), "/diff staged")
    assert "Staged" in out
    assert "+print('staged')" in out


@requires_git
def test_diff_non_git_directory_is_graceful(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(plain))
    out = _run(_Stub(), "/diff")
    assert "not a git repository" in out.lower()


# ---------------------------------------------------------------------------
# Session mode — stubbed checkpoint manager
# ---------------------------------------------------------------------------

def test_diff_session_prints_stat_and_diff(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    mgr = _Mgr({
        "success": True,
        "stat": " main.py | 2 +-",
        "diff": "--- a/main.py\n+++ b/main.py\n-print('hello')\n+print('v3')\n",
    })
    out = _run(_Stub(_Agent(mgr)), "/diff session")
    assert " main.py | 2 +-" in out
    assert "+print('v3')" in out
    assert mgr.calls  # session_diff was consulted


def test_diff_session_stat_only_suppresses_body(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    mgr = _Mgr({
        "success": True,
        "stat": " main.py | 2 +-",
        "diff": "+print('v3')\n",
    })
    out = _run(_Stub(_Agent(mgr)), "/diff session --stat")
    assert " main.py | 2 +-" in out
    assert "+print('v3')" not in out


def test_diff_session_empty_reports_no_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    mgr = _Mgr({"success": True, "stat": "", "diff": "", "empty": True})
    out = _run(_Stub(_Agent(mgr)), "/diff session")
    assert "No changes" in out


def test_diff_session_disabled_explains_how_to_enable(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    mgr = _Mgr({"success": True, "stat": "", "diff": ""}, enabled=False)
    out = _run(_Stub(_Agent(mgr)), "/diff session")
    assert "not enabled" in out.lower()
    assert not mgr.calls  # short-circuits before touching the store


def test_diff_session_without_agent_is_graceful(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    out = _run(_Stub(agent=None), "/diff session")
    assert "No active agent session" in out


def test_diff_session_failure_surfaces_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    mgr = _Mgr({"success": False, "error": "boom"})
    out = _run(_Stub(_Agent(mgr)), "/diff session")
    assert "boom" in out
