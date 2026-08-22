"""Regression test for cron-session approval isolation.

A cron job must use ``approvals.cron_mode`` for its own ``execute_code`` call,
without leaving process-global state that changes a later interactive gateway
turn handled by the same Python process.
"""

from __future__ import annotations

import os

import pytest

import cron.scheduler as cron_scheduler
from gateway.session_context import (
    clear_session_vars,
    get_session_env,
    reset_session_vars,
    set_session_vars,
)
from tools import approval as approval_module


class _DummySessionDB:
    def set_session_title(self, *args, **kwargs):
        pass

    def end_session(self, *args, **kwargs):
        pass

    def close(self):
        pass


class _FakeCronAgent:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def run_conversation(self, prompt):
        result = approval_module.check_execute_code_guard(
            "import os; print(1)", "local"
        )
        assert result["approved"] is False
        assert result["outcome"] == "blocked"
        assert get_session_env("HERMES_CRON_SESSION") == "1"
        return {
            "completed": True,
            "failed": False,
            "final_response": "cron execute_code blocked",
            "turn_exit_reason": "",
        }

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _clear_approval_state(monkeypatch):
    reset_session_vars()
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("cron-isolation-session")
    yield
    approval_module._permanent_approved.clear()
    approval_module.clear_session("default")
    approval_module.clear_session("cron-isolation-session")
    reset_session_vars()


def test_run_job_does_not_inherit_delegated_child_context(monkeypatch, tmp_path):
    """Cron is a root session even when its caller is a delegated child."""
    from agent.delegation_context import (
        delegated_child_context,
        is_delegated_child_context,
    )

    class _DelegationProbeAgent:
        def __init__(self, *args, **kwargs):
            assert is_delegated_child_context() is False
            assert get_session_env("HERMES_CRON_SESSION") == "1"

        def run_conversation(self, prompt):
            assert is_delegated_child_context() is False
            assert get_session_env("HERMES_CRON_SESSION") == "1"
            return {
                "completed": True,
                "failed": False,
                "final_response": "cron root context",
                "turn_exit_reason": "",
            }

        def close(self):
            pass

    monkeypatch.setenv("HERMES_MODEL", "test-model")
    monkeypatch.setattr("hermes_state.SessionDB", _DummySessionDB)
    monkeypatch.setattr("run_agent.AIAgent", _DelegationProbeAgent)
    monkeypatch.setattr(
        "hermes_constants.resolve_reasoning_config", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "test-key", "base_url": None, "provider": "test-provider",
            "api_mode": None, "command": None, "args": None,
        },
    )
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: [])
    monkeypatch.setattr(cron_scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cron_scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(cron_scheduler, "_guard_job_credential_exfil", lambda _job: None)

    with delegated_child_context():
        success, _output, final_response, error = cron_scheduler.run_job({
            "id": "delegated-cron", "name": "Delegated Cron", "prompt": "Run safely",
            "schedule_display": "manual",
        })
        assert is_delegated_child_context() is True

    assert success is True
    assert error is None
    assert final_response == "cron root context"


def test_run_job_restores_delegated_context_when_cron_reset_raises(monkeypatch, tmp_path):
    """A failing cron reset must not leak root identity back to the caller."""
    from agent.delegation_context import (
        delegated_child_context,
        is_delegated_child_context,
    )

    class _RaisingResetVar:
        def set(self, _value):
            return object()

        def reset(self, _token):
            raise RuntimeError("cron reset failed")

    monkeypatch.setenv("HERMES_MODEL", "test-model")
    monkeypatch.setattr("hermes_state.SessionDB", _DummySessionDB)
    monkeypatch.setattr("run_agent.AIAgent", _FakeCronAgent)
    monkeypatch.setattr(
        "hermes_constants.resolve_reasoning_config", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "***", "base_url": None, "provider": "test-provider",
            "api_mode": None, "command": None, "args": None,
        },
    )
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: [])
    monkeypatch.setattr(cron_scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cron_scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(cron_scheduler, "_guard_job_credential_exfil", lambda _job: None)
    from gateway.session_context import _VAR_MAP

    monkeypatch.setitem(_VAR_MAP, "HERMES_CRON_SESSION", _RaisingResetVar())

    with delegated_child_context():
        with pytest.raises(RuntimeError, match="cron reset failed"):
            cron_scheduler.run_job(
                {
                    "id": "delegated-cron-reset-failure",
                    "name": "Delegated Cron Reset Failure",
                    "prompt": "Run safely",
                    "schedule_display": "manual",
                }
            )
        assert is_delegated_child_context() is True


def _register_gateway_auto_approve(session_key: str) -> None:
    def _notify(_approval_data):
        with approval_module._lock:
            entries = approval_module._gateway_queues.get(session_key, [])
            if entries:
                entry = entries[-1]
                entry.result = "once"
                entry.event.set()

    with approval_module._lock:
        approval_module._gateway_notify_cbs[session_key] = _notify


def test_run_job_cron_execute_code_deny_does_not_pollute_later_gateway_execute_code(
    monkeypatch, tmp_path
):
    """Cron deny stays scoped; a later gateway approval still reaches its user."""
    monkeypatch.setenv("HERMES_MODEL", "test-model")
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_module, "_get_cron_approval_mode", lambda: "deny")
    monkeypatch.setattr("hermes_state.SessionDB", _DummySessionDB)
    monkeypatch.setattr("run_agent.AIAgent", _FakeCronAgent)
    monkeypatch.setattr(
        "hermes_constants.resolve_reasoning_config", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "test-key",
            "base_url": None,
            "provider": "test-provider",
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: [])
    monkeypatch.setattr(cron_scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(cron_scheduler, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr(
        cron_scheduler, "_guard_job_credential_exfil", lambda _job: None
    )

    success, _output, final_response, error = cron_scheduler.run_job(
        {
            "id": "ctx-isolation",
            "name": "Context Isolation",
            "prompt": "Run safely",
            "schedule_display": "manual",
        }
    )

    assert success is True
    assert error is None
    assert final_response == "cron execute_code blocked"
    assert os.environ.get("HERMES_CRON_SESSION") is None
    assert get_session_env("HERMES_CRON_SESSION") == ""

    # A completed in-process job must restore the truly-unset ContextVar state,
    # not leave an explicit empty value that shadows the standalone cron env
    # fallback in this reused context.
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    assert get_session_env("HERMES_CRON_SESSION") == "1"
    monkeypatch.delenv("HERMES_CRON_SESSION")

    session_key = "cron-isolation-session"
    key_token = approval_module.set_current_session_key(session_key)
    session_tokens = set_session_vars(
        platform="discord",
        chat_id="123",
        session_key=session_key,
        cron_session="",
    )
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    try:
        _register_gateway_auto_approve(session_key)
        result = approval_module.check_execute_code_guard(
            "import os; print(2)", "local"
        )
        assert result["approved"] is True
        assert result.get("user_approved") is True
    finally:
        clear_session_vars(session_tokens)
        approval_module.reset_current_session_key(key_token)
        with approval_module._lock:
            approval_module._gateway_queues.pop(session_key, None)
            approval_module._gateway_notify_cbs.pop(session_key, None)
