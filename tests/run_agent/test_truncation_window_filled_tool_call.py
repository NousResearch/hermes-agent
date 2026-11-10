"""Regression tests for #106830: a truncated TOOL call must not burn the 4×
max_tokens-boost retries when the prompt already filled the context window —
no output fits regardless of the cap, so the turn ends on the context-overflow
terminal instead of the misleading incomplete-tool-arguments refusal.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_constants import FINISH_REASON_LENGTH

from agent.turn_truncation import (
    _CONTEXT_OVERFLOW_PARTIAL_FINAL,
    recover_from_truncation,
)


def _make_agent(context_length=8000, max_tokens=2048):
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent.provider = "openai"
    agent.max_tokens = max_tokens
    agent.log_prefix = ""
    agent.context_compressor = SimpleNamespace(context_length=context_length)
    agent._vprint = MagicMock()
    agent._buffer_vprint = MagicMock()
    agent._flush_status_buffer = MagicMock()
    agent._cleanup_task_resources = MagicMock()
    agent._persist_session = MagicMock()
    agent._requested_output_cap_from_api_kwargs = MagicMock(return_value=None)
    normalized = SimpleNamespace(content=None, tool_calls=[{"id": "c1"}])
    transport = MagicMock()
    transport.normalize_response = MagicMock(return_value=normalized)
    agent._get_transport = MagicMock(return_value=transport)
    return agent


def _truncated_tool_call_response(prompt_tokens):
    return SimpleNamespace(
        id="chatcmpl-trunc",
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=12),
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[{"id": "c1"}],
                    reasoning_content=None,
                ),
                finish_reason=FINISH_REASON_LENGTH,
            )
        ],
    )


_TOOL_TAIL_MESSAGES = [
    {"role": "user", "content": "go"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "big file"},
]


def _recover(agent, response, messages):
    return recover_from_truncation(
        agent,
        response,
        FINISH_REASON_LENGTH,
        MagicMock(),
        messages=messages,
        conversation_history=None,
        api_kwargs={},
        api_call_count=0,
        effective_task_id=None,
        current_turn_user_idx=None,
        length_continue_retries=0,
        truncated_response_parts=[],
        truncated_tool_call_retries=0,
        retry_count=0,
        compression_attempts=0,
    )


class TestWindowFilledToolCall:
    def test_full_window_skips_retries_and_ends_on_overflow_terminal(self):
        """Prompt 7900/8000 tokens (< 512 headroom): the 4× max_tokens boosts are
        skipped and the turn ends naming the context window, not the tool args."""
        agent = _make_agent(context_length=8000)
        messages = [dict(m) for m in _TOOL_TAIL_MESSAGES]
        verdict = _recover(agent, _truncated_tool_call_response(7900), messages)

        assert verdict.action == "return"
        result = verdict.result or {}
        assert result.get("failed") is True
        assert result.get("final_response") == _CONTEXT_OVERFLOW_PARTIAL_FINAL
        # Typed bit so the gateway moves future input off the bloated session.
        assert result.get("compression_exhausted") is True
        # No retry was attempted...
        assert verdict.truncated_tool_call_retries == 0
        agent._requested_output_cap_from_api_kwargs.assert_not_called()
        # ...and the interrupted tool tail was closed for the next user turn.
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == _CONTEXT_OVERFLOW_PARTIAL_FINAL

    def test_room_in_window_still_retries_with_boosted_max_tokens(self):
        """Prompt 1000/8000 tokens: the boost retry loop is untouched."""
        agent = _make_agent(context_length=8000, max_tokens=2048)
        messages = [dict(m) for m in _TOOL_TAIL_MESSAGES]
        verdict = _recover(agent, _truncated_tool_call_response(1000), messages)

        assert verdict.action == "continue"
        assert verdict.truncated_tool_call_retries == 1
        assert agent._ephemeral_max_output_tokens == 4096  # 2048 * 2**1
