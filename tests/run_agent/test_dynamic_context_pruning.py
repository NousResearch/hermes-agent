"""Tests for prune-only dynamic context pressure handling in AIAgent."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None, usage=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = SimpleNamespace(**usage) if usage else None
    return resp


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("read_file")),
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
        a.compression_enabled = True
        a.save_trajectories = False
        return a


class TestDynamicContextPruning:
    def test_preflight_prune_only_avoids_full_compression_when_pruning_suffices(self, agent):
        agent.context_compressor = MagicMock()
        agent.context_compressor.protect_first_n = 1
        agent.context_compressor.protect_last_n = 1
        agent.context_compressor.threshold_tokens = 80000
        agent.context_compressor.context_length = 100000
        agent.context_compressor.last_prompt_tokens = 0
        agent.context_compressor.last_completion_tokens = 0
        agent.context_compressor.should_compress.side_effect = [False]
        pruned_messages = [
            {"role": "user", "content": "older user"},
            {"role": "assistant", "content": "older assistant"},
            {"role": "user", "content": "hello"},
        ]
        agent.context_compressor.prune_only.return_value = (pruned_messages, True)
        agent.client.chat.completions.create.return_value = _mock_response(content="done")

        history = [
            {"role": "user", "content": "older user"},
            {"role": "assistant", "content": "older assistant"},
            {"role": "user", "content": "older user 2"},
            {"role": "assistant", "content": "older assistant 2"},
        ]

        with (
            patch("run_agent.estimate_request_tokens_rough", side_effect=[90000, 1000]),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=history)

        agent.context_compressor.prune_only.assert_called_once()
        mock_compress.assert_not_called()
        assert result["completed"] is True
        assert result["messages"][-1]["role"] == "assistant"
        assert result["messages"][-1]["content"] == "done"

    def test_post_tool_prune_only_avoids_full_compression_when_pruning_suffices(self, agent):
        agent.context_compressor = MagicMock()
        agent.context_compressor.protect_first_n = 1
        agent.context_compressor.protect_last_n = 1
        agent.context_compressor.threshold_tokens = 80000
        agent.context_compressor.context_length = 100000
        agent.context_compressor.last_prompt_tokens = 0
        agent.context_compressor.last_completion_tokens = 0
        agent.context_compressor.should_compress.side_effect = [True, False]

        tool_call = SimpleNamespace(
            id="call-1",
            type="function",
            function=SimpleNamespace(name="read_file", arguments='{"path":"/tmp/f.txt"}'),
        )
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            _mock_response(content="done"),
        ]

        def _fake_execute(_assistant_message, messages, _task_id, _api_call_count=0):
            messages.append({"role": "tool", "tool_call_id": "call-1", "content": "large tool output"})

        pruned_messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"/tmp/f.txt"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "[read_file] read /tmp/f.txt from line 1 (10 chars)"},
        ]
        agent.context_compressor.prune_only.return_value = (pruned_messages, True)

        with (
            patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute),
            patch("run_agent.estimate_messages_tokens_rough", side_effect=[90000, 90000, 1000, 1000, 1000]),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        agent.context_compressor.prune_only.assert_called_once()
        mock_compress.assert_not_called()
        assert result["completed"] is True
        assert result["messages"][-1]["role"] == "assistant"
        assert result["messages"][-1]["content"] == "done"

    def test_post_tool_prune_only_falls_back_to_full_compression_when_request_still_over_threshold(self, agent):
        agent.context_compressor = MagicMock()
        agent.context_compressor.protect_first_n = 1
        agent.context_compressor.protect_last_n = 1
        agent.context_compressor.threshold_tokens = 80000
        agent.context_compressor.context_length = 100000
        agent.context_compressor.last_prompt_tokens = 0
        agent.context_compressor.last_completion_tokens = 0
        agent.context_compressor.should_compress.side_effect = [True, True, False]

        tool_call = SimpleNamespace(
            id="call-1",
            type="function",
            function=SimpleNamespace(name="read_file", arguments='{"path":"/tmp/f.txt"}'),
        )
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            _mock_response(content="done"),
        ]

        def _fake_execute(_assistant_message, messages, _task_id, _api_call_count=0):
            messages.append({"role": "tool", "tool_call_id": "call-1", "content": "large tool output"})

        pruned_messages = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "call-1", "content": "[read_file] summarized"},
        ]
        compressed_messages = [{"role": "user", "content": "hello after compression"}]
        agent.context_compressor.prune_only.return_value = (pruned_messages, True)

        with (
            patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute),
            patch("run_agent.estimate_messages_tokens_rough", side_effect=[90000, 90000, 1000, 1000]),
            patch("run_agent.estimate_request_tokens_rough", return_value=90000),
            patch.object(agent, "_compress_context", return_value=(compressed_messages, "You are helpful.")) as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        agent.context_compressor.prune_only.assert_called_once()
        mock_compress.assert_called_once()
        assert result["completed"] is True

