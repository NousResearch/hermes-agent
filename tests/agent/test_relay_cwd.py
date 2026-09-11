"""Logical working directories attached to Hermes-owned Relay scopes."""

from agent import relay_cwd, runtime_cwd
from tools.terminal_tool import clear_session_cwd, record_session_cwd


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


def test_remote_scoped_cwd_is_preserved_without_host_resolution(monkeypatch):
    token = runtime_cwd.set_session_cwd("~/remote-project")
    monkeypatch.setattr("tools.terminal_tool.get_session_cwd", lambda _key: None)
    monkeypatch.setattr(
        runtime_cwd,
        "resolve_agent_cwd",
        lambda: (_ for _ in ()).throw(AssertionError("must not resolve on this host")),
    )
    try:
        assert relay_cwd.resolve_relay_scope_cwds(
            object(), "task-1", "session-1", "gateway"
        ) == ("~/remote-project", "~/remote-project")
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
