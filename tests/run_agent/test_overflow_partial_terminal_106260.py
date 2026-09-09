"""Regression tests for #106260: a context-overflow error after partial stream
delivery must NOT seed a continuation stub with the recovered text.

Seeding tens of KB of partial content as a length-continuation stub makes every
later request larger, so a session whose transcript cannot fit (compression
failed / protect_last_n covers it) loops forever growing the context. The stub
is instead marked terminal (content empty) and the loop ends the turn via the
recovery contract.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import PARTIAL_STREAM_STUB_ID, FINISH_REASON_LENGTH


def _make_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


def _make_stream_chunk(content=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=None,
                            reasoning_content=None, reasoning=None)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=None, usage=None)


class TestOverflowTerminalStub:
    def test_build_stub_carries_marker_and_empty_content(self):
        from agent.chat_completion_helpers import _build_partial_stream_stub

        stub = _build_partial_stream_stub(
            "assistant", None, None, "test/model", None, overflow_terminal=True)
        assert stub._overflow_terminal is True
        assert stub.id == PARTIAL_STREAM_STUB_ID
        assert stub.choices[0].finish_reason == FINISH_REASON_LENGTH
        assert stub.choices[0].message.content is None

        normal = _build_partial_stream_stub(
            "assistant", "some text", None, "test/model", None)
        assert getattr(normal, "_overflow_terminal", False) is False
        assert normal.choices[0].message.content == "some text"

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_partial_stream_overflow_error_returns_terminal_stub(
        self, _mock_close, mock_create, monkeypatch,
    ):
        """A stream that delivered text then hit the provider's
        maximum-context-length error must return an EMPTY stub marked terminal,
        not a continuation stub carrying the recovered text (#106260)."""
        def _overflowing_stream():
            yield _make_stream_chunk(content="Here's my long partial answer ...")
            raise RuntimeError(
                "This model's maximum context length is 128000 tokens. "
                "However, you requested 140000 tokens."
            )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _overflowing_stream()
        mock_create.return_value = mock_client

        agent = _make_agent()
        agent._current_streamed_assistant_text = "Here's my long partial answer ..."

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        response = agent._interruptible_streaming_api_call({})

        assert response.id == PARTIAL_STREAM_STUB_ID
        assert getattr(response, "_overflow_terminal", False) is True
        # The recovered text must not be seeded for continuation.
        assert response.choices[0].message.content is None


class TestRecoverFromTruncationOverflowTerminal:
    def _mock_agent(self):
        agent = MagicMock()
        agent._vprint = MagicMock()
        agent._flush_status_buffer = MagicMock()
        agent._cleanup_task_resources = MagicMock()
        agent._persist_session = MagicMock()
        return agent

    def _response(self, overflow_terminal=True, content="recovered text"):
        return SimpleNamespace(
            id=PARTIAL_STREAM_STUB_ID,
            _overflow_terminal=overflow_terminal,
            _dropped_tool_names=None,
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content=content,
                                        tool_calls=None, reasoning_content=None),
                finish_reason=FINISH_REASON_LENGTH,
            )],
        )

    def test_overflow_terminal_stub_ends_turn_without_continuation(self):
        from agent.turn_truncation import (
            _CONTEXT_OVERFLOW_PARTIAL_FINAL,
            recover_from_truncation,
        )

        agent = self._mock_agent()
        verdict = recover_from_truncation(
            agent, self._response(), FINISH_REASON_LENGTH, MagicMock(),
            messages=[], conversation_history=None, api_kwargs={},
            api_call_count=0, effective_task_id=None, current_turn_user_idx=None,
            length_continue_retries=0, truncated_response_parts=[],
            truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
        )

        assert verdict.action == "return"
        result = verdict.result or {}
        assert result.get("failed") is True
        assert result.get("final_response") == _CONTEXT_OVERFLOW_PARTIAL_FINAL
        # No fragment or nudge was appended to the transcript.
        assert result.get("messages") == []

    def test_normal_stub_is_not_treated_as_terminal(self):
        from agent.turn_truncation import (
            _CONTEXT_OVERFLOW_PARTIAL_FINAL,
            recover_from_truncation,
        )

        agent = self._mock_agent()
        verdict = recover_from_truncation(
            agent, self._response(overflow_terminal=False), FINISH_REASON_LENGTH,
            MagicMock(),
            messages=[], conversation_history=None, api_kwargs={},
            api_call_count=0, effective_task_id=None, current_turn_user_idx=None,
            length_continue_retries=0, truncated_response_parts=["recovered text"],
            truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
        )
        # Not the overflow-terminal final; the normal continuation path runs
        # (no early return with the overflow message).
        result = (verdict.result or {}) if verdict.action == "return" else None
        assert result is None or result.get("final_response") != _CONTEXT_OVERFLOW_PARTIAL_FINAL
