"""A routed execution context lands session state on the side it routes to.

When ``command_prefix`` sends commands into another filesystem (a VM guest, a
remote machine), every host path the terminal tool would otherwise pick names a
directory the routed shell cannot see. The failures are silent — the snapshot
bootstrap's ``mktemp`` failure only warns, and env vars quietly stop persisting
between commands — so these pin the routing rather than the symptom.
"""
import importlib
from pathlib import Path

import pytest


def _routing_context(api, **kwargs):
    return api.SessionExecutionContext(
        command_prefix=("/usr/bin/env", "--"), **kwargs)


@pytest.mark.linux_only
def test_backend_paths_require_a_routing_prefix_and_absolute_values():
    """Declaring a far-side path without routing is a contradiction, not a default."""
    api = importlib.import_module("hermes_cli.session_execution")
    for field in ("backend_temp_dir", "backend_cwd"):
        with pytest.raises(ValueError, match="routing command_prefix"):
            api.SessionExecutionContext(**{field: "/tmp"})
        with pytest.raises(ValueError, match="absolute"):
            _routing_context(api, **{field: "relative/path"})
    assert _routing_context(api, backend_temp_dir="/tmp").backend_temp_dir == "/tmp"
    assert _routing_context(api, backend_cwd="/home/guest").backend_cwd == "/home/guest"


@pytest.mark.linux_only
def test_local_backend_keeps_session_state_where_commands_actually_run(tmp_path, monkeypatch):
    """The snapshot must live on the routed filesystem, not the host's temp dir."""
    api = importlib.import_module("hermes_cli.session_execution")
    local = importlib.import_module("tools.environments.local")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "host-temp"))

    plain = local.LocalEnvironment.__new__(local.LocalEnvironment)
    plain.env = {}
    plain.execution_context = None
    host_temp = plain.get_temp_dir()

    routed = local.LocalEnvironment.__new__(local.LocalEnvironment)
    routed.env = {}
    routed.execution_context = api.SessionExecutionLease(
        home=str(tmp_path), session_id="routed",
        context=_routing_context(api, backend_temp_dir="/guest-tmp"))

    assert routed.get_temp_dir() == "/guest-tmp"
    assert host_temp != "/guest-tmp"


@pytest.mark.linux_only
def test_routed_session_starts_in_the_far_sides_directory(tmp_path, monkeypatch):
    """A host cwd fallback would make the wrapper's ``cd`` fail on the far side.

    Only the fallback is replaced: an explicit per-command workdir is the
    caller's own choice, and a recorded session cwd was read from the routed
    filesystem's own ``pwd``, so both must still win.
    """
    api = importlib.import_module("hermes_cli.session_execution")
    terminal_tool = importlib.import_module("tools.terminal_tool")

    host_cwd = str(tmp_path / "host-project")
    Path(host_cwd).mkdir()
    monkeypatch.setenv("TERMINAL_CWD", host_cwd)
    monkeypatch.setattr(terminal_tool, "get_session_cwd", lambda *a, **k: None)

    context = _routing_context(api, backend_cwd="/home/guest")
    api.register_session_execution_context("routed-session", context)
    try:
        plan = terminal_tool._plan_execution(
            "pwd", task_id="routed-session", session_id="routed-session",
            timeout=None, background=False, _host_local=False)
        assert plan.cwd == "/home/guest"

        # An explicit per-task cwd override is the caller's own choice and wins.
        terminal_tool.register_task_env_overrides("routed-session", {"cwd": "/etc"})
        try:
            explicit = terminal_tool._plan_execution(
                "pwd", task_id="routed-session", session_id="routed-session",
                timeout=None, background=False, _host_local=False)
            assert explicit.cwd == "/etc"
        finally:
            terminal_tool.register_task_env_overrides("routed-session", {})
    finally:
        api.remove_session_execution_context("routed-session")


@pytest.mark.linux_only
def test_unrouted_sessions_keep_the_configured_host_directory(tmp_path, monkeypatch):
    """The far-side fallback must not reach sessions that route nowhere."""
    terminal_tool = importlib.import_module("tools.terminal_tool")
    host_cwd = str(tmp_path / "host-project")
    Path(host_cwd).mkdir()
    monkeypatch.setenv("TERMINAL_CWD", host_cwd)
    monkeypatch.setattr(terminal_tool, "get_session_cwd", lambda *a, **k: None)

    plan = terminal_tool._plan_execution(
        "pwd", task_id="plain-session", session_id="plain-session",
        timeout=None, background=False, _host_local=False)
    assert plan.cwd == host_cwd
