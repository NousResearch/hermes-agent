"""Regression tests for terminal fail-closed live-stream refusal handling."""

from __future__ import annotations

import time

from agent.turn_api_error import handle_api_error
from hermes_cli.middleware import LLMStreamMiddlewareRefusal


class _RecoveryAgent:
    def __init__(self, *, provider="openrouter", api_mode="chat_completions"):
        self.provider = provider
        self.model = "test/model"
        self.api_mode = api_mode
        self._image_rejecting_models = set()
        self._bedrock_converse_fallback_attempted = False
        self._bedrock_region = None
        self.client = object()
        self._client_kwargs = {"sentinel": True}
        self.thinking_callback = None
        self.log_prefix = ""

    def _vprint(self, *args, **kwargs):
        return None


def _handle(agent, error):
    return handle_api_error(
        agent,
        api_error=error,
        _retry=object(),
        thinking_spinner=None,
        messages=[],
        api_messages=[],
        api_kwargs={},
        system_message=None,
        active_system_prompt=None,
        conversation_history=[],
        approx_tokens=0,
        retry_count=0,
        max_retries=3,
        compression_attempts=0,
        max_compression_attempts=1,
        api_call_count=1,
        api_request_id="request-1",
        api_start_time=time.time(),
        effective_task_id="task-1",
        turn_id="turn-1",
    )


def test_refusal_with_image_rejection_text_is_terminal_before_image_repair():
    agent = _RecoveryAgent()
    original_client = agent.client
    original_kwargs = dict(agent._client_kwargs)

    verdict = _handle(
        agent,
        LLMStreamMiddlewareRefusal(
            RuntimeError("image_url is not supported"),
            callback_name="privacy_gate",
        ),
    )

    assert verdict.action == "return"
    assert verdict.result["failure_reason"] == "llm_stream_middleware_refusal"
    assert verdict.result["failure_retryable"] is False
    assert agent._image_rejecting_models == set()
    assert agent.client is original_client
    assert agent._client_kwargs == original_kwargs


def test_refusal_with_bedrock_repair_text_is_terminal_before_runtime_switch():
    agent = _RecoveryAgent(
        provider="bedrock",
        api_mode="anthropic_messages",
    )
    original_client = agent.client
    original_kwargs = dict(agent._client_kwargs)

    verdict = _handle(
        agent,
        LLMStreamMiddlewareRefusal(
            RuntimeError("Unexpected event order"),
            callback_name="privacy_gate",
        ),
    )

    assert verdict.action == "return"
    assert verdict.result["failure_reason"] == "llm_stream_middleware_refusal"
    assert agent.api_mode == "anthropic_messages"
    assert agent._bedrock_converse_fallback_attempted is False
    assert agent.client is original_client
    assert agent._client_kwargs == original_kwargs


def test_genuine_provider_image_rejection_still_uses_existing_repair():
    agent = _RecoveryAgent()

    verdict = _handle(agent, RuntimeError("image_url is not supported"))

    assert verdict.action == "continue"
    assert ("openrouter", "test/model") in agent._image_rejecting_models


def test_genuine_bedrock_event_order_error_still_switches_runtime():
    agent = _RecoveryAgent(
        provider="bedrock",
        api_mode="anthropic_messages",
    )

    verdict = _handle(agent, RuntimeError("Unexpected event order"))

    assert verdict.action == "continue"
    assert agent._bedrock_converse_fallback_attempted is True
    assert agent.api_mode == "bedrock_converse"
    assert agent.client is None
    assert agent._client_kwargs == {}
