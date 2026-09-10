import os

import pytest

from gateway.session_context import _UNSET, _VAR_MAP, clear_session_vars, set_session_vars
from run_agent import _launch_cwd_for_session, _session_source_for_agent


@pytest.fixture(autouse=True)
def _reset_contextvars():
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


def test_session_source_context_overrides_platform(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    tokens = set_session_vars(source="tool")
    try:
        assert _session_source_for_agent("tui") == "tool"
    finally:
        clear_session_vars(tokens)


def test_session_source_falls_back_to_platform(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    assert _session_source_for_agent("tui") == "tui"


def test_launch_cwd_records_local_cli_session(monkeypatch, tmp_path):
    """A local CLI session's shell cwd is the session's working directory."""
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.chdir(tmp_path)

    assert _launch_cwd_for_session("cli") == os.getcwd()


def test_launch_cwd_records_local_kanban_worker(monkeypatch, tmp_path):
    """The dispatcher spawns kanban workers with ``cwd=<card workspace>``, so the process
    cwd is the workspace the card's work happens in — cwd-based attribution needs it."""
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.chdir(tmp_path)

    assert _launch_cwd_for_session("kanban") == os.getcwd()


@pytest.mark.parametrize("source", ["cron", "gateway", "telegram", "webhook", "tui", "desktop"])
def test_launch_cwd_none_for_sources_without_a_host_cwd(monkeypatch, tmp_path, source):
    """Sources whose process cwd is not a stable host directory for the agent's tools stay NULL."""
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.chdir(tmp_path)

    assert _launch_cwd_for_session(source) is None


@pytest.mark.parametrize("source", ["cli", "kanban"])
def test_launch_cwd_none_on_non_local_terminal_backend(monkeypatch, tmp_path, source):
    """A remote backend's host cwd says nothing about where the agent's tools run."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.chdir(tmp_path)

    assert _launch_cwd_for_session(source) is None


@pytest.mark.parametrize("terminal_env", ["", "local", "LOCAL", " local "])
def test_launch_cwd_treats_blank_and_explicit_local_as_local(monkeypatch, tmp_path, terminal_env):
    monkeypatch.setenv("TERMINAL_ENV", terminal_env)
    monkeypatch.chdir(tmp_path)

    assert _launch_cwd_for_session("kanban") == os.getcwd()


