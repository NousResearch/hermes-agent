import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.conversation_loop import _emit_turn_dead_marker
from hermes_cli.plugins import VALID_HOOKS
from run_agent import AIAgent


def _agent(printed):
    return SimpleNamespace(
        session_id="session-123",
        log_prefix="[agent] ",
        _vprint=lambda message, force=False: printed.append((message, force)),
    )


def _runtime_agent():
    tool_defs = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "search",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{"provider": "openrouter", "model": "fallback-model"}],
        )
    agent.client = MagicMock()
    agent._api_max_retries = 1
    return agent


def test_turn_dead_marker_prints_stable_json_and_fires_hook(monkeypatch):
    printed = []
    hook_calls = []

    def _fake_hook(name, **kwargs):
        hook_calls.append((name, kwargs))

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    payload = _emit_turn_dead_marker(
        _agent(printed),
        failure_reason="rate_limit",
        error_summary="HTTP 429: temporarily overloaded",
        provider="zai",
        model="glm-5.2",
        http_status="429",
        max_retries=8,
        api_call_count=1,
    )

    assert len(printed) == 1
    marker, force = printed[0]
    assert force is True
    assert "HERMES-TURN-DEAD: " in marker
    marker_payload = json.loads(marker.split("HERMES-TURN-DEAD: ", 1)[1])
    assert marker_payload == payload
    assert marker_payload["failure_reason"] == "rate_limit"
    assert marker_payload["http_status"] == 429
    assert marker_payload["provider"] == "zai"
    assert marker_payload["model"] == "glm-5.2"
    assert marker_payload["max_retries"] == 8
    assert hook_calls == [("on_turn_failed", payload)]


def test_turn_dead_marker_survives_hook_failure(monkeypatch, caplog):
    printed = []

    def _boom(name, **kwargs):
        raise RuntimeError("hook broke")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _boom)

    with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
        payload = _emit_turn_dead_marker(
            _agent(printed),
            failure_reason="timeout",
            error_summary="connection reset",
            provider="openrouter",
            model="reasoning-model",
            http_status=None,
            max_retries=3,
            api_call_count=2,
        )

    assert len(printed) == 1
    marker_payload = json.loads(printed[0][0].split("HERMES-TURN-DEAD: ", 1)[1])
    assert marker_payload == payload
    assert "http_status" not in marker_payload
    assert "on_turn_failed hook failed" in caplog.text


def test_on_turn_failed_is_a_registered_hook():
    assert "on_turn_failed" in VALID_HOOKS


def test_retry_and_fallback_exhaustion_returns_and_emits_turn_dead(monkeypatch):
    class RateLimitError(Exception):
        status_code = 429

        def __init__(self):
            super().__init__("HTTP 429 rate limit exceeded")
            self.response = SimpleNamespace(headers={})
            self.body = {"error": {"message": "rate limit exceeded"}}

    agent = _runtime_agent()
    printed = []
    hooks = []
    monkeypatch.setattr(agent, "_vprint", lambda message, force=False: printed.append(message))

    def _capture_hook(name, **payload):
        hooks.append((name, payload))
        return []

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        _capture_hook,
    )

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=RateLimitError()),
        patch.object(agent, "_try_activate_fallback", return_value=False) as fallback,
        patch.object(agent, "_try_recover_primary_transport", return_value=False),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_dump_api_request_debug"),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_dead"]["failure_reason"] == "rate_limit"
    assert result["turn_dead"]["http_status"] == 429
    assert result["turn_dead"]["provider"] == "openrouter"
    assert fallback.call_count >= 1
    assert any("HERMES-TURN-DEAD:" in line for line in printed)
    failed_hooks = [item for item in hooks if item[0] == "on_turn_failed"]
    assert failed_hooks == [("on_turn_failed", result["turn_dead"])]


def test_invalid_response_exhaustion_uses_same_turn_dead_contract(monkeypatch):
    agent = _runtime_agent()
    hooks = []
    invalid = SimpleNamespace(choices=[], model="test-model", usage=None)

    def _capture_hook(name, **payload):
        hooks.append((name, payload))
        return []

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        _capture_hook,
    )

    with (
        patch.object(agent, "_interruptible_api_call", return_value=invalid),
        patch.object(agent, "_try_activate_fallback", return_value=False),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["failed"] is True
    assert result["turn_dead"]["failure_reason"] == "invalid_response"
    assert "Invalid API response" in result["turn_dead"]["error"]
    failed_hooks = [item for item in hooks if item[0] == "on_turn_failed"]
    assert failed_hooks == [("on_turn_failed", result["turn_dead"])]
