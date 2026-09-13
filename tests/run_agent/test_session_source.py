import os

import pytest

from gateway.session_context import _UNSET, _VAR_MAP, clear_session_vars, set_session_vars
from run_agent import _launch_cwd_for_session, _session_source_for_agent


@pytest.fixture(autouse=True)
def _reset_session_vars():
    tokens = set_session_vars()
    yield
    clear_session_vars(tokens)


@pytest.fixture(autouse=True)
def _local_terminal_backend(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)


def test_session_source_falls_back_to_platform(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    assert _session_source_for_agent("tui") == "tui"


def test_launch_cwd_records_local_cli_session(monkeypatch, tmp_path):
    """A local CLI session's shell cwd is the session's working directory."""
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)

    assert _launch_cwd_for_session("cli") == os.getcwd()


def test_launch_cwd_records_kanban_workspace_not_process_cwd(monkeypatch, tmp_path):
    """A kanban worker stamps its recorded workspace, not the inherited process cwd.

    The dispatcher spawns workers with ``cwd=workspace if os.path.isdir(workspace) else None``;
    when that falls back to None the worker inherits the dispatcher's cwd, which is NOT the
    card's workspace. HERMES_KANBAN_WORKSPACE is the authoritative value.
    """
    workspace = tmp_path / "card-workspace"
    workspace.mkdir()
    inherited = tmp_path / "gateway-cwd"
    inherited.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.chdir(inherited)

    assert _launch_cwd_for_session("kanban") == str(workspace)


def test_launch_cwd_none_for_kanban_worker_without_workspace_env(monkeypatch, tmp_path):
    """No workspace env -> the spawn path is unknown; do not guess from the process cwd.

    A missing HERMES_KANBAN_WORKSPACE means the worker was not spawned by today's dispatcher
    (or a legacy/foreign spawn path); the inherited cwd could be anything.
    """
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)

    assert _launch_cwd_for_session("kanban") is None


def test_launch_cwd_none_when_kanban_workspace_dir_missing(monkeypatch, tmp_path):
    """A workspace dir that does not exist at spawn is exactly the dispatcher's cwd=None
    path (kanban_db_dispatch.py spawns with ``cwd=None``); there is no directory to
    attribute the session to, so record nothing rather than the inherited cwd."""
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path / "gone"))
    monkeypatch.chdir(tmp_path)

    assert _launch_cwd_for_session("kanban") is None


@pytest.mark.parametrize("source", ["cron", "gateway", "telegram", "webhook", "tui", "desktop"])
def test_launch_cwd_none_for_sources_without_a_host_cwd(monkeypatch, tmp_path, source):
    """Sources whose process cwd is not a stable host directory for the agent's tools stay NULL."""
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
    workspace = tmp_path / "card-workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("TERMINAL_ENV", terminal_env)
    monkeypatch.chdir(tmp_path)

    assert _launch_cwd_for_session("kanban") == str(workspace)


def test_create_session_wiring_kanban_env_stamps_workspace(tmp_path, monkeypatch):
    """End-to-end wiring: a kanban worker (HERMES_SESSION_SOURCE=kanban + HERMES_KANBAN_WORKSPACE)
    creates a session row whose cwd is the workspace, through the real SessionDB."""
    from hermes_state import SessionDB

    workspace = tmp_path / "card-workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.chdir(tmp_path)  # inherited cwd differs from the workspace on purpose

    db = SessionDB(db_path=tmp_path / "state.db")
    # The autouse _reset_session_vars fixture stamps empty ContextVars over os.environ (empty
    # string wins in get_session_env), so bind the source the way a real kanban worker is bound:
    # through the session context itself.
    tokens = set_session_vars(source="kanban")
    try:
        source = _session_source_for_agent(None)
        db.create_session(session_id="sess-kanban-1", source=source, cwd=_launch_cwd_for_session(source))
    finally:
        clear_session_vars(tokens)

    row = db.get_session("sess-kanban-1")
    assert row is not None
    assert row["source"] == "kanban"
    assert row["cwd"] == str(workspace)


def test_create_session_wiring_cli_stamps_process_cwd(tmp_path, monkeypatch):
    """End-to-end wiring: a plain local CLI session stamps the shell cwd."""
    from hermes_state import SessionDB

    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)

    db = SessionDB(db_path=tmp_path / "state.db")
    source = _session_source_for_agent(None)
    db.create_session(session_id="sess-cli-1", source=source, cwd=_launch_cwd_for_session(source))

    row = db.get_session("sess-cli-1")
    assert row is not None
    assert row["cwd"] == os.getcwd()
