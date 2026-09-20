"""Integration tests for ContextEngine.on_turn_complete terminal observation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from agent.context_engine import ContextEngine
from tests.agent.test_turn_finalizer_cleanup_guard import _StubAgent, _run


class _CapturingEngine(ContextEngine):
    last_prompt_tokens = 0

    def __init__(self) -> None:
        self.captured: Dict[str, Any] = {}
        self.calls = 0
        self.events: List[str] = []

    @property
    def name(self) -> str:
        return "capturing"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        pass

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages

    def on_turn_complete(self, messages, usage=None, **kwargs):
        self.events.append("notify")
        self.calls += 1
        self.captured["seen"] = True
        self.captured["messages"] = messages
        self.captured["usage"] = usage
        self.captured["kwargs"] = kwargs


CANONICAL_USAGE = {
    "prompt_tokens": 1200,
    "completion_tokens": 80,
    "total_tokens": 1280,
    "input_tokens": 1200,
    "output_tokens": 80,
    "cache_read_tokens": 1024,
    "cache_write_tokens": 0,
    "reasoning_tokens": 16,
}


def _agent_with_engine() -> _StubAgent:
    agent = _StubAgent(raise_in=())
    agent.context_compressor = _CapturingEngine()
    return agent


def test_finalize_turn_forwards_canonical_usage_when_available():
    agent = _agent_with_engine()
    agent._last_turn_usage = dict(CANONICAL_USAGE)

    _run(agent, final_response="done")

    captured = agent.context_compressor.captured
    assert agent.context_compressor.calls == 1
    assert captured.get("seen") is True
    assert captured["usage"] == CANONICAL_USAGE
    assert captured["kwargs"]["turn_id"] == "turn-1"


def test_finalize_turn_forwards_none_when_no_response_usage():
    agent = _agent_with_engine()
    if hasattr(agent, "_last_turn_usage"):
        delattr(agent, "_last_turn_usage")

    _run(agent, final_response="done")

    captured = agent.context_compressor.captured
    assert agent.context_compressor.calls == 1
    assert captured.get("seen") is True
    assert captured["usage"] is None


def test_finalization_seam_observes_interrupted_turn_with_none_usage():
    from agent.turn_finalizer import finalize_turn

    agent = _agent_with_engine()
    if hasattr(agent, "_last_turn_usage"):
        delattr(agent, "_last_turn_usage")

    finalize_turn(
        agent,
        final_response="interrupted mid-turn",
        api_call_count=1,
        interrupted=True,
        failed=False,
        messages=[
            {"role": "user", "content": "do a thing"},
            {"role": "assistant", "content": "partial"},
        ],
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-int",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="interrupt",
    )

    captured = agent.context_compressor.captured
    assert agent.context_compressor.calls == 1
    assert captured.get("seen") is True
    assert captured["usage"] is None
    assert captured["kwargs"]["interrupted"] is True
    assert captured["kwargs"]["turn_id"] == "turn-int"


def _make_runtime_agent():
    from run_agent import AIAgent

    tool_defs = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "web_search tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.context_compressor = _CapturingEngine()
    return agent


def _run_runtime_turn(agent, response_or_error):
    if isinstance(response_or_error, BaseException):
        agent.client.chat.completions.create.side_effect = response_or_error
    else:
        agent.client.chat.completions.create.return_value = response_or_error
    with (
        patch.object(
            agent,
            "_persist_session",
            side_effect=lambda *args, **kwargs: agent.context_compressor.events.append("persist"),
        ),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.retry_utils.jittered_backoff", lambda *a, **k: 0.0),
    ):
        return agent.run_conversation("hello", task_id="task-1")


def test_policy_terminal_notifies_once_without_changing_result():
    agent = _make_runtime_agent()
    refusal = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=None,
                tool_calls=None,
                reasoning=None,
                reasoning_content=None,
                refusal="I won't help with that.",
            ),
            finish_reason="stop",
        )],
        model="test/model",
        usage=None,
        id="resp-policy",
    )

    result = _run_runtime_turn(agent, refusal)

    assert result["completed"] is False
    assert result["failed"] is True
    assert "content_policy_blocked" in result["error"]
    assert "I won't help with that." in result["final_response"]
    assert agent.client.chat.completions.create.call_count == 1
    captured = agent.context_compressor.captured
    assert agent.context_compressor.calls == 1
    assert captured["kwargs"]["turn_id"]
    assert captured["kwargs"]["task_id"] == "task-1"
    assert captured["kwargs"]["api_call_count"] == 1
    assert captured["kwargs"]["interrupted"] is False
    assert captured["kwargs"]["failed"] is True
    assert captured["kwargs"]["turn_exit_reason"] == "content_policy_blocked"
    assert captured["messages"] == result["messages"]
    assert captured["usage"] is None
    assert captured["messages"][-1]["role"] == "user"
    assert agent.context_compressor.events[-2:] == ["persist", "notify"]


def test_provider_terminal_failure_notifies_once_and_preserves_error_shape():
    agent = _make_runtime_agent()
    agent._api_max_retries = 1
    error = RuntimeError("provider unavailable")
    error.status_code = 500
    error.message = "provider unavailable"

    with (
        patch.object(agent, "_try_recover_primary_transport", return_value=False),
        patch.object(agent, "_has_pending_fallback", return_value=False),
    ):
        result = _run_runtime_turn(agent, error)

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["error"] == "HTTP 500: provider unavailable"
    assert result["final_response"].startswith("API call failed after 1 retries:")
    assert result["failure_retryable"] is True
    assert agent.client.chat.completions.create.call_count == 1
    captured = agent.context_compressor.captured
    assert agent.context_compressor.calls == 1
    assert captured["usage"] is None
    assert captured["kwargs"]["task_id"] == "task-1"
    assert captured["kwargs"]["api_call_count"] == 1
    assert captured["kwargs"]["failed"] is True
    assert captured["kwargs"]["turn_exit_reason"] == "provider_terminal_failure"
    assert captured["messages"] == result["messages"]
    assert agent.context_compressor.events[-2:] == ["persist", "notify"]
