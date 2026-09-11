"""Logical working directories attached to Hermes-owned Relay scopes."""

from pathlib import Path

import pytest

from agent import relay_cwd, runtime_cwd
from agent.relay_runtime import ConversationLease
from tools.terminal_tool import clear_session_cwd, record_session_cwd


def test_conversation_lease_keeps_released_in_its_original_position():
    lease = ConversationLease(
        "profile", "session", "cli", object(), None, "parent", True
    )

    assert lease.released is True
    assert lease.turn_cwd == ""


def test_session_context_and_task_record_remain_distinct(monkeypatch):
    token = runtime_cwd.set_session_cwd("/workspace/session")
    record_session_cwd("task-1", "/workspace/task")
    monkeypatch.setattr(
        "gateway.session_context.get_session_env", lambda _name, _default: ""
    )
    try:
        assert relay_cwd.resolve_relay_scope_cwds(
            object(), "task-1", "session-1", "cli"
        ) == ("/workspace/session", "/workspace/task")
        assert relay_cwd.resolve_relay_scope_cwds(
            object(), "task-1", "session-1", "subagent"
        ) == ("/workspace/task", "/workspace/task")
    finally:
        clear_session_cwd("task-1")
        runtime_cwd._SESSION_CWD.reset(token)


@pytest.mark.parametrize(
    "logical_cwd", ["~/remote-project", r"C:\work\project", r"\\server\share\project"]
)
def test_remote_scoped_cwd_is_preserved_without_host_resolution(
    monkeypatch, logical_cwd
):
    token = runtime_cwd.set_session_cwd(logical_cwd)
    monkeypatch.setattr("tools.terminal_tool.get_session_cwd", lambda _key: None)
    monkeypatch.setattr(
        runtime_cwd,
        "resolve_agent_cwd",
        lambda: (_ for _ in ()).throw(AssertionError("must not resolve on this host")),
    )
    try:
        assert relay_cwd.resolve_relay_scope_cwds(
            object(), "task-1", "session-1", "gateway"
        ) == (logical_cwd, logical_cwd)
    finally:
        runtime_cwd._SESSION_CWD.reset(token)


def test_gateway_session_key_uses_its_recorded_cwd(monkeypatch):
    token = runtime_cwd.set_session_cwd(None)
    record_session_cwd("gateway-key", "/remote/gateway-workspace")
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda name, default: (
            "gateway-key" if name == "HERMES_SESSION_KEY" else default
        ),
    )
    monkeypatch.setattr(
        runtime_cwd,
        "resolve_agent_cwd",
        lambda: (_ for _ in ()).throw(AssertionError("must not resolve on this host")),
    )
    try:
        assert relay_cwd.resolve_relay_scope_cwds(
            object(), "task-1", "session-1", "gateway"
        ) == ("/remote/gateway-workspace", "/remote/gateway-workspace")
    finally:
        clear_session_cwd("gateway-key")
        runtime_cwd._SESSION_CWD.reset(token)


@pytest.mark.parametrize(
    ("platform", "backend", "resolved", "expected", "uses_host"),
    [
        ("cli", "local", Path("/workspace"), "/workspace", True),
        ("", "local", Path("."), str(Path.cwd().resolve()), True),
        ("cli", "ssh", Path("/host"), "", False),
        ("gateway", "local", Path("/host"), "", False),
    ],
)
def test_only_local_cli_uses_host_cwd(
    monkeypatch, platform, backend, resolved, expected, uses_host
):
    token = runtime_cwd.set_session_cwd(None)
    host_lookups = []

    def resolve_host_cwd():
        host_lookups.append(True)
        return resolved

    monkeypatch.setattr("tools.terminal_tool.get_session_cwd", lambda _key: None)
    monkeypatch.setattr(
        "gateway.session_context.get_session_env", lambda _name, default: default
    )
    monkeypatch.setattr(
        "tools.terminal_scope.terminal_env",
        lambda name, default: backend if name == "TERMINAL_ENV" else default,
    )
    monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", resolve_host_cwd)
    try:
        assert relay_cwd.resolve_relay_scope_cwds(
            object(), "task-1", "session-1", platform
        ) == (expected, expected)
        assert bool(host_lookups) is uses_host
    finally:
        runtime_cwd._SESSION_CWD.reset(token)


def test_cwd_lookup_failures_do_not_break_the_turn(monkeypatch):
    def unavailable(*_args):
        raise RuntimeError("unavailable")

    monkeypatch.setattr("tools.terminal_tool.get_session_cwd", unavailable)
    monkeypatch.setattr(runtime_cwd, "scoped_session_cwd", unavailable)
    monkeypatch.setattr("gateway.session_context.get_session_env", unavailable)
    monkeypatch.setattr("tools.terminal_scope.terminal_env", unavailable)

    assert relay_cwd.resolve_relay_scope_cwds(
        object(), "task-1", "session-1", "gateway"
    ) == ("", "")
