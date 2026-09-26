"""A docker CLI session records its launch cwd; a resume must find the same workspace.

The bug: ``_launch_cwd_for_session`` returned None for EVERY non-local backend, so
``sessions.cwd`` stayed NULL for docker CLI sessions (41/56 on a real profile) and
``hermes --resume`` had nothing to restore. The resume path
(``_restore_session_cwd`` / ``main.py``) reads that column and does two things with
it: ``os.chdir()`` AND re-exports ``TERMINAL_CWD`` — the latter is what makes the
docker ``/workspace`` bind fall on the same directory it did at launch. With the
column empty the session silently re-attaches to whatever the backend process cwd
happens to be: on a desktop backend that was the whole home directory, mounted rw
into the sandbox.

The fix distinguishes backends that CONSUME a host cwd (local + container backends,
whose sanitizers already discard it where it cannot apply) from backends whose
runtime is a different machine (ssh, managed_modal) and must keep the row empty.
"""

import os

import pytest

import run_agent
from run_agent import _launch_cwd_for_session, _launch_cwd_is_a_host_signal

# The classification the container-cwd sanitizers use, so the assertion below cannot
# drift from them: recording the launch dir is safe exactly when the sanitizers are
# the ones that police it.
from tools.terminal_tool_config import _CONTAINER_BACKENDS


@pytest.fixture(autouse=True)
def _host_cwd(monkeypatch, tmp_path):
    """A real, absolute host directory standing in for the user's worktree."""
    worktree = tmp_path / "bioextr-llm-situation"
    worktree.mkdir()
    monkeypatch.chdir(worktree)
    return worktree


@pytest.mark.parametrize("backend", ["", "local", *sorted(_CONTAINER_BACKENDS)])
def test_backend_that_consumes_a_host_cwd_records_the_launch_dir(monkeypatch, backend, _host_cwd):
    """Docker included: the launch dir IS the /workspace bind source, so it must be recorded."""
    monkeypatch.setenv("TERMINAL_ENV", backend)
    assert _launch_cwd_is_a_host_signal() is True
    assert _launch_cwd_for_session("cli") == str(_host_cwd)


@pytest.mark.parametrize("backend", ["ssh", "managed_modal"])
def test_backend_on_another_machine_keeps_the_row_empty(monkeypatch, backend, _host_cwd):
    """A remote sandbox has no host cwd to restore — re-recording one is the exit-126 class."""
    monkeypatch.setenv("TERMINAL_ENV", backend)
    assert _launch_cwd_is_a_host_signal() is False
    assert _launch_cwd_for_session("cli") is None


def test_container_check_failure_stays_conservative(monkeypatch, _host_cwd):
    """An import failure must not start stamping host paths onto remote sandbox rows."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(run_agent, "_launch_cwd_is_a_host_signal", run_agent._launch_cwd_is_a_host_signal)
    import builtins

    real_import = builtins.__import__

    def _fail(name, *args, **kwargs):
        if name == "tools.terminal_tool_config":
            raise ImportError("partial install")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail)
    assert _launch_cwd_is_a_host_signal() is False


@pytest.mark.parametrize("source", ["tui", "desktop", "gateway", "cron", "kanban"])
def test_non_cli_sources_records_nothing(monkeypatch, source, _host_cwd):
    """Only a human CLI run's launch dir is a workspace; other sources must not claim one."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    assert _launch_cwd_for_session(source) is None


@pytest.mark.parametrize("source", ["cli", "oneshot"])
def test_cli_family_sources_are_covered(monkeypatch, source, _host_cwd):
    """``hermes chat -q`` is a CLI run too — its resume reads the same column (main.py)."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    assert _launch_cwd_for_session(source) == str(_host_cwd)


def test_deleted_cwd_returns_none(monkeypatch, _host_cwd):
    """A cwd removed out from under the process must not be stamped (and must not raise)."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    gone = os.path.join(str(_host_cwd), "gone")
    os.mkdir(gone)
    os.chdir(gone)
    os.rmdir(gone)
    assert _launch_cwd_for_session("cli") is None
