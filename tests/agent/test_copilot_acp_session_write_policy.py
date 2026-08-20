from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent.copilot_acp_client import CopilotACPClient
from agent.session_write_policy import (
    CallerType,
    PolicyDenied,
    SessionWritePolicy,
    SessionWritePolicyDecision,
    SessionWritePolicyDecisionResult,
    SessionWritePolicyMode,
    session_write_policy_scope,
)


class _FakeReadable:
    def __init__(self, lines: list[str] | None = None) -> None:
        self._lines = list(lines or [])

    def __iter__(self):
        return iter(self._lines)


class _FakePopenProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.stdout = _FakeReadable(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n",
                json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "fake-session"}}) + "\n",
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"text": "fake-acp-response"},
                            }
                        },
                    }
                )
                + "\n",
                json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}}) + "\n",
            ]
        )
        self.stderr = _FakeReadable([])
        self.killed = False
        self.terminated = False

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


@pytest.fixture
def client(tmp_path):
    return CopilotACPClient(
        acp_command="fake-acp",
        acp_args=["--acp", "--stdio", "--flag=value"],
        acp_cwd=str(tmp_path),
    )


def _allow_decision() -> SessionWritePolicyDecision:
    return SessionWritePolicyDecision(
        result=SessionWritePolicyDecisionResult.ALLOW,
        reason="policy_allow",
        operation_kind="terminal_exec",
        origin="DELEGATION",
    )


def _deny_decision(reason: str = "terminal_exec_denied_protected_mode") -> SessionWritePolicyDecision:
    return SessionWritePolicyDecision(
        result=SessionWritePolicyDecisionResult.DENY,
        reason=reason,
        operation_kind="terminal_exec",
        origin="DELEGATION",
    )


def _run_with_fakes(monkeypatch, client, events: list[str], *, decision: Any = None):
    policy_calls: list[dict[str, Any]] = []
    popen_calls: list[dict[str, Any]] = []
    env = {"SAFE_ENV": "1"}

    def fake_pre_spawn_consult(**kwargs):
        events.append("policy_consult")
        policy_calls.append(kwargs)
        if isinstance(decision, BaseException):
            raise decision
        return decision if decision is not None else _allow_decision()

    def fake_build_env():
        events.append("env_build")
        return env

    def fake_popen(argv, **kwargs):
        events.append("popen")
        popen_calls.append({"argv": argv, "kwargs": kwargs})
        return _FakePopenProcess()

    monkeypatch.setattr("agent.session_write_policy.pre_spawn_consult", fake_pre_spawn_consult)
    monkeypatch.setattr("agent.copilot_acp_client._build_subprocess_env", fake_build_env)
    monkeypatch.setattr("agent.copilot_acp_client._acp_supported", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("agent.copilot_acp_client.subprocess.Popen", fake_popen)

    result = client._run_prompt("hello", timeout_seconds=1)
    return result, policy_calls, popen_calls, env


def test_consult_arguments_are_exact(monkeypatch, client):
    events: list[str] = []
    result, policy_calls, _popen_calls, _env = _run_with_fakes(monkeypatch, client, events)

    assert result == ("fake-acp-response", "")
    assert len(policy_calls) == 1
    call = policy_calls[0]
    assert call["caller_type"] is CallerType.DELEGATION
    assert call["operation_kind"] == "terminal_exec"
    assert call["argv"] == [client._acp_command, *client._acp_args]
    assert call["raw_command"] is None
    assert call["cwd"] == client._acp_cwd
    assert call["env_subset"] is None
    assert call["target_path"] is None


def test_allow_builds_env_after_policy_and_reaches_popen(monkeypatch, client):
    events: list[str] = []
    result, policy_calls, popen_calls, env = _run_with_fakes(monkeypatch, client, events)

    assert result == ("fake-acp-response", "")
    assert events == ["policy_consult", "env_build", "popen"]
    assert len(policy_calls) == 1
    assert len(popen_calls) == 1
    popen_call = popen_calls[0]
    assert popen_call["argv"] == [client._acp_command, *client._acp_args]
    assert popen_call["kwargs"]["cwd"] == client._acp_cwd
    assert popen_call["kwargs"]["env"] is env
    assert popen_call["kwargs"]["stdin"] is not None
    assert popen_call["kwargs"]["stdout"] is not None
    assert popen_call["kwargs"]["stderr"] is not None


def test_returned_deny_blocks_before_env_and_popen(monkeypatch, client):
    events: list[str] = []

    with pytest.raises(RuntimeError) as excinfo:
        _run_with_fakes(monkeypatch, client, events, decision=_deny_decision("deny-reason"))

    message = str(excinfo.value)
    assert events == ["policy_consult"]
    assert "acp_subprocess_blocked_by_session_write_policy" in message
    assert "disposition=DENY_POLICY" in message
    assert "reason=deny-reason" in message


def test_policy_denied_blocks_before_env_and_popen(monkeypatch, client):
    events: list[str] = []
    denied = PolicyDenied(
        disposition=PolicyDenied.DISPOSITION_POLICY_DENY,
        caller_type=CallerType.DELEGATION,
        operation_kind="terminal_exec",
        reason="policy-denied-reason",
        detail={"unsafe": "not-rendered"},
    )

    with pytest.raises(RuntimeError) as excinfo:
        _run_with_fakes(monkeypatch, client, events, decision=denied)

    message = str(excinfo.value)
    assert events == ["policy_consult"]
    assert "acp_subprocess_blocked_by_session_write_policy" in message
    assert "disposition=DENY_POLICY" in message
    assert "reason=policy-denied-reason" in message
    assert "not-rendered" not in message


def test_unexpected_policy_error_fails_closed_without_sensitive_text(monkeypatch, client):
    events: list[str] = []

    with pytest.raises(RuntimeError) as excinfo:
        _run_with_fakes(
            monkeypatch,
            client,
            events,
            decision=RuntimeError("SECRET_TOKEN=do-not-render"),
        )

    message = str(excinfo.value)
    assert events == ["policy_consult"]
    assert "acp_subprocess_blocked_by_session_write_policy" in message
    assert "acp_pre_spawn_policy_evaluation_failed:RuntimeError" in message
    assert "SECRET_TOKEN" not in message
    assert "do-not-render" not in message


def _run_real_policy_context(monkeypatch, tmp_path, policy: SessionWritePolicy) -> tuple[tuple[str, str] | None, list[str]]:
    events: list[str] = []
    client = CopilotACPClient(acp_command="python", acp_args=["--version"], acp_cwd=str(tmp_path))

    def fake_build_env():
        events.append("env_build")
        return {"SAFE_ENV": "1"}

    def fake_popen(argv, **kwargs):
        events.append("popen")
        return _FakePopenProcess()

    monkeypatch.setattr("agent.copilot_acp_client._build_subprocess_env", fake_build_env)
    monkeypatch.setattr("agent.copilot_acp_client._acp_supported", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("agent.copilot_acp_client.subprocess.Popen", fake_popen)

    with session_write_policy_scope(policy):
        try:
            return client._run_prompt("hello", timeout_seconds=1), events
        except RuntimeError:
            return None, events


def test_real_normal_context_allows_simple_non_git_acp(monkeypatch, tmp_path):
    result, events = _run_real_policy_context(
        monkeypatch,
        tmp_path,
        SessionWritePolicy.normal("normal-session"),
    )

    assert result == ("fake-acp-response", "")
    assert events == ["env_build", "popen"]


def test_real_deny_all_context_blocks_before_popen(monkeypatch, tmp_path):
    result, events = _run_real_policy_context(
        monkeypatch,
        tmp_path,
        SessionWritePolicy.deny_all("deny-session"),
    )

    assert result is None
    assert events == []


def test_real_allowlist_context_denies_terminal_exec(monkeypatch, tmp_path):
    result, events = _run_real_policy_context(
        monkeypatch,
        tmp_path,
        SessionWritePolicy(
            session_id="allowlist-session",
            mode=SessionWritePolicyMode.ALLOWLIST,
            allowed_roots=(str(tmp_path),),
            protected=True,
        ),
    )

    assert result is None
    assert events == []


def test_no_delegate_or_terminal_tool_used_and_no_duplicate_env_or_popen(monkeypatch, client):
    events: list[str] = []

    def fail_terminal_tool(*args, **kwargs):
        raise AssertionError("terminal_tool must not be called")

    def fail_delegate_tool(*args, **kwargs):
        raise AssertionError("delegate_tool must not be called")

    monkeypatch.setattr("tools.terminal_tool.terminal", fail_terminal_tool, raising=False)
    monkeypatch.setattr("tools.delegate_tool.delegate_task", fail_delegate_tool)

    result, policy_calls, popen_calls, _env = _run_with_fakes(monkeypatch, client, events)

    assert result == ("fake-acp-response", "")
    assert events == ["policy_consult", "env_build", "popen"]
    assert len(policy_calls) == 1
    assert len(popen_calls) == 1
    assert len([event for event in events if event == "env_build"]) == 1
    assert len([event for event in events if event == "popen"]) == 1
