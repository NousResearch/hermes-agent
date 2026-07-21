"""Integration tests for assistant_response middleware inside run_conversation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


ASSISTANT_RESPONSE = "assistant_response"
LLM_REQUEST = "llm_request"


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Mock {name}",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_assistant_msg(content="Hello", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = _mock_assistant_msg(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


@pytest.fixture()
def agent():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("read_file", "search_files", "web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _run(agent: AIAgent, prompt: str = "hello"):
    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(prompt)


def _install_middleware(monkeypatch, *, assistant_decisions, llm_request_decisions=None):
    assistant_decisions = list(assistant_decisions)
    llm_request_decisions = list(llm_request_decisions or [])
    calls: list[tuple[str, dict]] = []

    def has_middleware(kind: str) -> bool:
        return kind in {ASSISTANT_RESPONSE, LLM_REQUEST}

    def invoke_middleware(kind: str, **kwargs):
        calls.append((kind, kwargs))
        if kind == ASSISTANT_RESPONSE:
            if assistant_decisions:
                decision = assistant_decisions.pop(0)
                if callable(decision):
                    decision = decision(**kwargs)
                return [decision]
            return [{"action": "pass", "source": "test-validator"}]
        if kind == LLM_REQUEST:
            if llm_request_decisions:
                decision = llm_request_decisions.pop(0)
                if callable(decision):
                    decision = decision(**kwargs)
                return [decision]
            return []
        return []

    monkeypatch.setattr("hermes_cli.plugins.has_middleware", has_middleware)
    monkeypatch.setattr("hermes_cli.plugins.invoke_middleware", invoke_middleware)
    return calls


def test_assistant_response_rewrite_replaces_final_before_commit(agent, monkeypatch):
    agent.client.chat.completions.create.return_value = _mock_response("unsafe draft")
    _install_middleware(
        monkeypatch,
        assistant_decisions=[
            {
                "action": "rewrite",
                "response_text": "safe replacement",
                "source": "test-validator",
            }
        ],
    )

    result = _run(agent)

    assert result["completed"] is True
    assert result["final_response"] == "safe replacement"
    assert result["messages"][-1]["content"] == "safe replacement"
    assert result["messages"][-1]["_response_validation_decision"] == "rewrite"


def test_assistant_response_block_replaces_final_before_commit(agent, monkeypatch):
    agent.client.chat.completions.create.return_value = _mock_response("unsafe draft")
    _install_middleware(
        monkeypatch,
        assistant_decisions=[
            {
                "action": "block",
                "message": "blocked before delivery",
                "source": "test-validator",
            }
        ],
    )

    result = _run(agent)

    assert result["completed"] is True
    assert result["final_response"] == "blocked before delivery"
    assert result["messages"][-1]["content"] == "blocked before delivery"
    assert result["messages"][-1]["_response_validation_decision"] == "block"


def test_assistant_response_retry_with_feedback_reenters_provider_loop(agent, monkeypatch):
    agent.client.chat.completions.create.side_effect = [
        _mock_response("unsupported reversal"),
        _mock_response("validated answer after retry"),
    ]
    _install_middleware(
        monkeypatch,
        assistant_decisions=[
            {
                "action": "retry_with_feedback",
                "feedback": "Do not reverse without evidence.",
                "max_retries": 1,
                "source": "test-validator",
            },
            {"action": "pass", "source": "test-validator"},
        ],
    )

    result = _run(agent)

    assert result["completed"] is True
    assert result["final_response"] == "validated answer after retry"
    assert result["api_calls"] == 2
    second_request_messages = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert any(
        "Do not reverse without evidence." in msg.get("content", "")
        for msg in second_request_messages
    )
    assert not any(
        "unsupported reversal" in msg.get("content", "")
        for msg in second_request_messages
    )


def test_assistant_response_retry_exhaustion_does_not_return_rejected_draft(agent, monkeypatch):
    agent.max_iterations = 1
    agent.client.chat.completions.create.return_value = _mock_response("unsupported reversal")
    agent._handle_max_iterations = MagicMock(return_value="budget exhausted summary")
    _install_middleware(
        monkeypatch,
        assistant_decisions=[
            {
                "action": "retry_with_feedback",
                "feedback": "Do not reverse without evidence.",
                "max_retries": 1,
                "source": "test-validator",
            }
        ],
    )

    result = _run(agent)

    assert result["completed"] is False
    assert "budget exhausted summary" in result["final_response"]
    assert result["final_response"] != "unsupported reversal"
    assert "unsupported reversal" not in result["final_response"]
    agent._handle_max_iterations.assert_called_once()


def test_assistant_response_require_tool_executes_tool_path_then_reenters(agent, monkeypatch):
    agent.client.chat.completions.create.side_effect = [
        _mock_response("unsupported reversal"),
        _mock_response("validated answer after evidence"),
    ]
    _install_middleware(
        monkeypatch,
        assistant_decisions=[
            {
                "action": "require_tool",
                "feedback": "Read README.md before reversing.",
                "tool_calls": [
                    {
                        "name": "read_file",
                        "args": {"path": "README.md"},
                        "reason": "verify source",
                        "read_only": True,
                    }
                ],
                "max_retries": 1,
                "source": "test-validator",
            },
            {"action": "pass", "source": "test-validator"},
        ],
    )

    with patch("run_agent.handle_function_call", return_value="README evidence") as handle_function_call:
        result = _run(agent)

    assert result["completed"] is True
    assert result["final_response"] == "validated answer after evidence"
    assert result["api_calls"] == 2
    handle_function_call.assert_called_once()
    assert handle_function_call.call_args.args[:2] == (
        "read_file",
        {"path": "README.md"},
    )
    assert isinstance(handle_function_call.call_args.args[2], str)
    assert handle_function_call.call_args.args[2]
    second_request_messages = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert any(
        msg.get("role") == "assistant"
        and msg.get("tool_calls")
        and msg["tool_calls"][0]["function"]["name"] == "read_file"
        for msg in second_request_messages
    )
    assert any(
        msg.get("role") == "tool" and "README evidence" in msg.get("content", "")
        for msg in second_request_messages
    )


def test_llm_request_stream_policy_uses_non_streaming_before_validation(agent, monkeypatch):
    non_streaming_response = _mock_response("validated non-streaming response")
    agent.stream_delta_callback = lambda _delta: None
    agent._interruptible_api_call = MagicMock(return_value=non_streaming_response)
    agent._interruptible_streaming_api_call = MagicMock(
        side_effect=AssertionError("streaming must be disabled before validator-controlled turns")
    )

    def llm_request_decision(**kwargs):
        return {
            "request": kwargs["request"],
            "control": {"stream_policy": "buffer_until_validated"},
            "source": "test-guard",
            "reason": "risky-turn",
        }

    _install_middleware(
        monkeypatch,
        assistant_decisions=[{"action": "pass", "source": "test-validator"}],
        llm_request_decisions=[llm_request_decision],
    )

    result = _run(agent)

    assert result["completed"] is True
    assert result["final_response"] == "validated non-streaming response"
    agent._interruptible_api_call.assert_called_once()
    agent._interruptible_streaming_api_call.assert_not_called()
