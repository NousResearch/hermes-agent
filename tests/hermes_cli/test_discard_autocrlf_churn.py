"""Tests for _discard_autocrlf_churn — the CRLF line-ending auto-heal in _cmd_update_impl.

The function gates on four conditions:
  1. Windows-only (no-op on other platforms)
  2. core.autocrlf=true
  3. Working tree is dirty (git status --porcelain)
  4. ≥95% of dirty files are pure CRLF noise (git diff --ignore-cr-at-eol ratio)

These tests use monkeypatch/mock to simulate git output without touching the
real filesystem or git config.  No live git subprocesses — everything is a
``subprocess.run`` mock.
"""

import platform
import subprocess

import pytest

from hermes_cli.main import _discard_autocrlf_churn


# ── helpers ────────────────────────────────────────────────────────────────


def _run_output(stdout="", returncode=0):
    """Return a subprocess.CompletedProcess with the given stdout text."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def _make_git_cmd():
    """Replicate the git_cmd built in _cmd_update_impl on Windows."""
    if platform.system() == "Windows":
        return ["git", "-c", "windows.appendAtomically=false"]
    return ["git"]


class _SequentialRun:
    """Callable that returns pre-built CompletedProcess values in order.

    Each call consumes one result from the list.  Useful when the function
    under test makes multiple subprocess.run() calls in sequence.
    """

    def __init__(self, *results):
        self._results = list(results)
        self._calls = []

    def __call__(self, args, **kwargs):
        self._calls.append(args)
        if not self._results:
            return _run_output("", returncode=0)
        return self._results.pop(0)

    @property
    def calls(self):
        return self._calls


# ── tests ──────────────────────────────────────────────────────────────────


def test_noop_on_non_windows(monkeypatch):
    """Should return immediately when sys.platform is not win32."""
    monkeypatch.setattr("hermes_cli.main.sys.platform", "darwin")
    run_mock = _SequentialRun()
    monkeypatch.setattr("hermes_cli.main.subprocess.run", run_mock)

    git_cmd = _make_git_cmd()
    _discard_autocrlf_churn(git_cmd, repo_root=None)

    # No subprocess calls should have been made.
    assert len(run_mock.calls) == 0


def test_noop_when_autocrlf_not_true(monkeypatch):
    """Should return when core.autocrlf is not the string 'true'."""
    monkeypatch.setattr("hermes_cli.main.sys.platform", "win32")

    results = [
        _run_output("input\n"),   # git config core.autocrlf → "input"
    ]
    run_mock = _SequentialRun(*results)
    monkeypatch.setattr("hermes_cli.main.subprocess.run", run_mock)

    git_cmd = _make_git_cmd()
    _discard_autocrlf_churn(git_cmd, repo_root=None)

    # Only the autocrlf check should have been called.
    assert len(run_mock.calls) == 1
    assert "config" in run_mock.calls[0]


def test_noop_when_tree_clean(monkeypatch):
    """Should return when git status --porcelain produces no output."""
    monkeypatch.setattr("hermes_cli.main.sys.platform", "win32")

    results = [
        _run_output("true\n"),   # core.autocrlf = true
        _run_output(""),          # status --porcelain → empty → clean tree
    ]
    run_mock = _SequentialRun(*results)
    monkeypatch.setattr("hermes_cli.main.subprocess.run", run_mock)

    git_cmd = _make_git_cmd()
    _discard_autocrlf_churn(git_cmd, repo_root=None)

    # autocrlf + status calls, then early return (no diff calls).
    assert len(run_mock.calls) == 2


def test_noop_when_below_threshold(monkeypatch):
    """Should return when <95% of dirty files are CRLF-only."""
    monkeypatch.setattr("hermes_cli.main.sys.platform", "win32")

    # 100 dirty files total, 80 are CRLF-only → 80% ratio → below 95% threshold
    raw_names = "\n".join([f"file_{i}.py" for i in range(100)])
    ignored_names = "\n".join([f"file_{i}.py" for i in range(20)])  # only 20 survive

    results = [
        _run_output("true\n"),          # core.autocrlf = true
        _run_output(" M file_0.py\n"),  # status --porcelain → dirty
        _run_output(raw_names),          # git diff --name-only HEAD
        _run_output(ignored_names),      # git diff --ignore-cr-at-eol --name-only HEAD
    ]
    run_mock = _SequentialRun(*results)
    monkeypatch.setattr("hermes_cli.main.subprocess.run", run_mock)

    git_cmd = _make_git_cmd()
    _discard_autocrlf_churn(git_cmd, repo_root=None)

    # All four gates checked, but no config set or reset called.
    assert len(run_mock.calls) == 4


def test_heals_when_above_threshold(monkeypatch):
    """Should set core.autocrlf=input and reset --hard when ≥95% CRLF noise."""
    monkeypatch.setattr("hermes_cli.main.sys.platform", "win32")

    # 4,500 dirty files, 4,400 are CRLF-only → 97.8% → fires
    raw_names = "\n".join([f"file_{i}.py" for i in range(4500)])
    ignored_names = "\n".join([f"file_{i}.py" for i in range(100)])

    results = [
        _run_output("true\n"),          # core.autocrlf = true
        _run_output(" M file_0.py\n"),  # status --porcelain → dirty
        _run_output(raw_names),          # git diff --name-only HEAD
        _run_output(ignored_names),      # git diff --ignore-cr-at-eol --name-only HEAD
        _run_output(""),                 # git config core.autocrlf input
        _run_output("HEAD is now at ..."),  # git reset --hard HEAD
    ]
    run_mock = _SequentialRun(*results)
    monkeypatch.setattr("hermes_cli.main.subprocess.run", run_mock)

    git_cmd = _make_git_cmd()
    _discard_autocrlf_churn(git_cmd, repo_root=None)

    # All six gates + heal calls.
    assert len(run_mock.calls) == 6

    # Verify the heal commands.
    assert run_mock.calls[4] == git_cmd + ["config", "core.autocrlf", "input"]
    assert run_mock.calls[5] == git_cmd + ["reset", "--hard", "HEAD"]


def test_exactly_at_threshold_heals(monkeypatch):
    """95% exactly should trigger healing (≥, not >)."""
    monkeypatch.setattr("hermes_cli.main.sys.platform", "win32")

    raw_names = "\n".join([f"file_{i}.py" for i in range(100)])
    ignored_names = "\n".join([f"file_{i}.py" for i in range(5)])  # 95% exactly

    results = [
        _run_output("true\n"),
        _run_output(" M file_0.py\n"),
        _run_output(raw_names),
        _run_output(ignored_names),
        _run_output(""),
        _run_output("HEAD is now at ..."),
    ]
    run_mock = _SequentialRun(*results)
    monkeypatch.setattr("hermes_cli.main.subprocess.run", run_mock)

    git_cmd = _make_git_cmd()
    _discard_autocrlf_churn(git_cmd, repo_root=None)

    assert len(run_mock.calls) == 6  # Healed


def test_autocrlf_check_exception_is_silent(monkeypatch):
    """If git config core.autocrlf raises, the function should return silently."""
    monkeypatch.setattr("hermes_cli.main.sys.platform", "win32")

    def raise_os_error(*args, **kwargs):
        raise OSError("git not found")

    run_mock = _SequentialRun()
    run_mock.__call__ = raise_os_error
    monkeypatch.setattr("hermes_cli.main.subprocess.run", run_mock)

    git_cmd = _make_git_cmd()
    # Should not raise — returns silently.
    _discard_autocrlf_churn(git_cmd, repo_root=None)
