"""Behavioral seam tests for successful conversation-call cleanup."""

import builtins
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import conversation_loop
from agent.conversation_loop_success_cleanup import complete_successful_call
from run_agent import AIAgent


class _AgentStub:
    provider = "nous"

    def __init__(self):
        self.activity = []

    def _touch_activity(self, message):
        self.activity.append(message)


def test_success_cleanup_resets_retry_and_completes_logical_call(monkeypatch):
    agent = _AgentStub()
    retry = SimpleNamespace(has_retried_429=True)
    retry_before = retry
    cleared = []
    completed = []

    monkeypatch.setattr(
        "agent.nous_rate_guard.clear_nous_rate_limit",
        lambda: cleared.append(True),
    )
    monkeypatch.setattr(
        "agent.relay_llm.complete_logical_call",
        lambda request_id, *, outcome: completed.append((request_id, outcome)),
    )

    result = complete_successful_call(agent, retry, "request-7", 7)

    assert result is None
    assert retry is retry_before
    assert retry.has_retried_429 is False
    assert cleared == [True]
    assert completed == [("request-7", "success")]
    assert agent.activity == ["API call #7 completed"]


def test_success_cleanup_keeps_clear_failure_local(monkeypatch):
    agent = _AgentStub()
    retry = SimpleNamespace(has_retried_429=True)
    completed = []

    def fail_clear():
        raise RuntimeError("rate guard unavailable")

    monkeypatch.setattr("agent.nous_rate_guard.clear_nous_rate_limit", fail_clear)
    monkeypatch.setattr(
        "agent.relay_llm.complete_logical_call",
        lambda request_id, *, outcome: completed.append((request_id, outcome)),
    )

    complete_successful_call(agent, retry, "request-8", 8)

    assert completed == [("request-8", "success")]
    assert agent.activity == ["API call #8 completed"]


def test_success_cleanup_non_nous_does_not_clear_rate_limit(monkeypatch):
    agent = _AgentStub()
    agent.provider = "openai"
    retry = SimpleNamespace(has_retried_429=True)
    clear_nous_rate_limit = Mock()
    monkeypatch.setattr(
        "agent.nous_rate_guard.clear_nous_rate_limit", clear_nous_rate_limit
    )
    monkeypatch.setattr("agent.relay_llm.complete_logical_call", Mock())

    complete_successful_call(agent, retry, "request-non-nous", 10)

    clear_nous_rate_limit.assert_not_called()


def test_success_cleanup_lazily_resolves_nous_guard_once(monkeypatch):
    agent = _AgentStub()
    retry = SimpleNamespace(has_retried_429=True)
    resolved = []
    clear_nous_rate_limit = Mock(side_effect=RuntimeError("guard failed"))
    monkeypatch.setattr(
        "agent.nous_rate_guard.clear_nous_rate_limit", clear_nous_rate_limit
    )
    monkeypatch.setattr("agent.relay_llm.complete_logical_call", Mock())

    real_import = builtins.__import__

    def recording_import(name, *args, **kwargs):
        if name == "agent.nous_rate_guard":
            resolved.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", recording_import)
    complete_successful_call(agent, retry, "request-lazy", 11)

    assert resolved == ["agent.nous_rate_guard"]
    clear_nous_rate_limit.assert_called_once_with()


def test_success_cleanup_rate_guard_failure_still_completes_and_touches(monkeypatch):
    agent = _AgentStub()
    retry = SimpleNamespace(has_retried_429=True)
    completed = []

    def fail_clear():
        raise RuntimeError("rate guard unavailable")

    monkeypatch.setattr("agent.nous_rate_guard.clear_nous_rate_limit", fail_clear)
    monkeypatch.setattr(
        "agent.relay_llm.complete_logical_call",
        lambda request_id, *, outcome: completed.append((request_id, outcome)),
    )

    complete_successful_call(agent, retry, "exact-request", 42)

    assert retry.has_retried_429 is False
    assert completed == [("exact-request", "success")]
    assert agent.activity == ["API call #42 completed"]


def test_success_cleanup_does_not_swallow_relay_exception(monkeypatch):
    agent = _AgentStub()
    retry = SimpleNamespace(has_retried_429=True)
    monkeypatch.setattr("agent.nous_rate_guard.clear_nous_rate_limit", Mock())

    relay = Mock(side_effect=RuntimeError("relay failed"))
    monkeypatch.setattr("agent.relay_llm.complete_logical_call", relay)

    with pytest.raises(RuntimeError, match="relay failed"):
        complete_successful_call(agent, retry, "request-relay-error", 13)

    relay.assert_called_once_with("request-relay-error", outcome="success")
    assert agent.activity == []


def test_production_caller_breaks_once_after_helper_success(monkeypatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="test/model",
    )
    with (
        monkeypatch.context() as constructor_patches,
    ):
        constructor_patches.setattr(
            "run_agent.get_tool_definitions", lambda **_kwargs: []
        )
        constructor_patches.setattr(
            "run_agent.check_toolset_requirements", lambda **_kwargs: {}
        )
        constructor_patches.setattr("run_agent.OpenAI", Mock())
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=1,
            enabled_toolsets=[],
        )
    agent.client = Mock()
    agent._api_max_retries = 1
    relay_execute = Mock(return_value=response)
    relay_complete = Mock()
    monkeypatch.setattr("agent.relay_llm.execute", relay_execute)
    monkeypatch.setattr("agent.relay_llm.complete_logical_call", relay_complete)
    monkeypatch.setattr(agent, "_persist_session", Mock())
    monkeypatch.setattr(agent, "_save_trajectory", Mock())
    monkeypatch.setattr(agent, "_cleanup_task_resources", Mock())
    helper = Mock(wraps=complete_successful_call)
    monkeypatch.setattr(conversation_loop, "complete_successful_call", helper)

    result = agent.run_conversation("hello", conversation_history=[])

    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert helper.call_count == 1
    assert relay_execute.call_count == 1
    relay_complete.assert_called_once_with(helper.call_args.args[2], outcome="success")


def test_original_conversation_loop_patch_target_remains_patchable(monkeypatch):
    replacement = object()
    monkeypatch.setattr(conversation_loop, "run_conversation", replacement)
    assert conversation_loop.run_conversation is replacement


def test_importing_cleanup_module_has_no_runtime_side_effects(monkeypatch):
    module_name = "agent.conversation_loop_success_cleanup"
    original = sys.modules.pop(module_name)
    imported = []
    real_import = builtins.__import__

    def recording_import(name, *args, **kwargs):
        if name in {"agent.relay_llm", "agent.nous_rate_guard", "run_agent"}:
            imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", recording_import)
    try:
        reloaded = importlib.import_module(module_name)
        assert reloaded.complete_successful_call is not None
        assert imported == []
    finally:
        sys.modules.pop(module_name, None)
        sys.modules[module_name] = original


def test_existing_relay_and_rate_guard_monkeypatches_remain_visible(monkeypatch):
    agent = _AgentStub()
    retry = SimpleNamespace(has_retried_429=True)
    clear = Mock()
    relay = Mock()
    monkeypatch.setattr("agent.nous_rate_guard.clear_nous_rate_limit", clear)
    monkeypatch.setattr("agent.relay_llm.complete_logical_call", relay)

    complete_successful_call(agent, retry, "patched-request", 14)

    clear.assert_called_once_with()
    relay.assert_called_once_with("patched-request", outcome="success")
