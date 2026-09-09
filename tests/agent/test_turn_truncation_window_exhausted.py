"""Regression tests for #106120: length-continuation retries make the prompt LONGER
when the prompt is what filled the window.

When ``finish_reason='length'`` is caused by the *prompt* consuming the context
window (not by a long answer hitting ``max_tokens``), continuing is counterproductive:
each retry appends the partial fragment + a nudge — strictly more prompt — so every
attempt is worse than the last (the death spiral observed live on Ollama /v1, where a
32,638-token prompt against a 32,768 window left ~130 tokens of room and the 4 retries
each shrank it toward zero).

``_window_exhausted_abort`` detects this via ``prompt_tokens + max_tokens >=
context_length`` (the output cap is unreachable) and aborts continuation with a clear,
actionable message instead of looping.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_truncation import (
    _WINDOW_EXHAUSTED,
    _window_exhausted_abort,
    recover_from_truncation,
)
from agent.turn_retry_state import TurnRetryState


def _agent(*, context_length=32768, max_tokens=8192, last_real=0, provider="openai",
           api_mode="chat_completions"):
    """Minimal agent with the attributes ``_window_exhausted_abort`` reads."""
    return SimpleNamespace(
        context_compressor=SimpleNamespace(
            context_length=context_length,
            last_real_prompt_tokens=last_real,
        ),
        max_tokens=max_tokens,
        provider=provider,
        api_mode=api_mode,
    )


def _response(*, prompt_tokens=0):
    """A response carrying provider usage (OpenAI Chat Completions shape)."""
    usage = SimpleNamespace(prompt_tokens=prompt_tokens) if prompt_tokens else None
    return SimpleNamespace(id="resp_1", usage=usage)


class TestWindowExhaustedAbort:
    """``_window_exhausted_abort`` — the pure discriminator."""

    def test_aborts_when_prompt_plus_max_tokens_reaches_context_length(self):
        # Reporter's live case: 32,638 prompt + 8,192 max_tokens >= 32,768 context.
        agent = _agent(context_length=32768, max_tokens=8192)
        response = _response(prompt_tokens=32638)
        assert _window_exhausted_abort(agent, response) is _WINDOW_EXHAUSTED

    def test_no_abort_when_headroom_available(self):
        # 1,000 prompt + 8,192 max_tokens = 9,192 < 32,768 → output cap reachable.
        agent = _agent(context_length=32768, max_tokens=8192)
        response = _response(prompt_tokens=1000)
        assert _window_exhausted_abort(agent, response) is None

    def test_treats_exact_boundary_as_exhausted(self):
        # prompt_tokens + max_tokens == context_length (inclusive >=) → abort.
        agent = _agent(context_length=10000, max_tokens=2000)
        response = _response(prompt_tokens=8000)
        assert _window_exhausted_abort(agent, response) is _WINDOW_EXHAUSTED

    def test_one_short_of_boundary_is_not_exhausted(self):
        # prompt_tokens + max_tokens == context_length - 1 → just enough room, continue.
        agent = _agent(context_length=10000, max_tokens=2000)
        response = _response(prompt_tokens=7999)
        assert _window_exhausted_abort(agent, response) is None

    def test_uses_last_real_prompt_tokens_when_response_usage_missing(self):
        # Silent-clip providers (Ollama /v1) sometimes omit usage; fall back to the
        # compressor's last real count.
        agent = _agent(context_length=32768, max_tokens=8192, last_real=32638)
        response = _response(prompt_tokens=0)  # no usage on this response
        assert _window_exhausted_abort(agent, response) is _WINDOW_EXHAUSTED

    def test_no_abort_when_usage_missing_and_no_fallback(self):
        agent = _agent(context_length=32768, max_tokens=8192, last_real=0)
        response = _response(prompt_tokens=0)
        assert _window_exhausted_abort(agent, response) is None

    def test_no_abort_when_no_context_length(self):
        agent = _agent(context_length=0, max_tokens=8192)
        response = _response(prompt_tokens=32638)
        assert _window_exhausted_abort(agent, response) is None

    def test_no_abort_when_no_max_tokens(self):
        agent = _agent(context_length=32768, max_tokens=0)
        response = _response(prompt_tokens=32638)
        assert _window_exhausted_abort(agent, response) is None

    def test_no_abort_when_no_context_compressor(self):
        agent = SimpleNamespace(context_compressor=None, max_tokens=8192,
                                provider="openai", api_mode="chat_completions")
        response = _response(prompt_tokens=32638)
        assert _window_exhausted_abort(agent, response) is None

    def test_aborts_even_when_prompt_alone_exceeds_context(self):
        # Pathological: prompt bigger than the window (provider accepted it anyway).
        agent = _agent(context_length=32768, max_tokens=4096)
        response = _response(prompt_tokens=40000)
        assert _window_exhausted_abort(agent, response) is _WINDOW_EXHAUSTED

    def test_falls_back_to_last_real_when_normalize_returns_zero(self):
        # usage present but prompt_tokens resolves to 0 (malformed) → fallback fires.
        agent = _agent(context_length=32768, max_tokens=8192, last_real=32638)
        response = SimpleNamespace(id="resp_1", usage=SimpleNamespace(prompt_tokens=0))
        assert _window_exhausted_abort(agent, response) is _WINDOW_EXHAUSTED


class TestRecoverFromTruncationAbortsOnWindowExhaustion:
    """End-to-end: ``recover_from_truncation`` returns the window-exhausted verdict
    (action='return') instead of arming a continuation retry (action='break')."""

    @pytest.fixture
    def agent(self, monkeypatch):
        # Bypass the transport-normalization layer (tested elsewhere) so the stub agent
        # only needs the attributes recover_from_truncation actually touches.
        monkeypatch.setattr(
            "agent.turn_truncation.normalize_response_for_agent",
            lambda agent, response: SimpleNamespace(content="partial", tool_calls=None),
        )
        ag = SimpleNamespace(
            log_prefix="test: ",
            api_mode="chat_completions",
            provider="openai",
            max_tokens=8192,
            context_compressor=SimpleNamespace(
                context_length=32768, last_real_prompt_tokens=32638,
            ),
            # _abort_reason helpers — return content unchanged so thinking/repetition
            # aborts do NOT fire (we want to reach the window-exhaustion check).
            _has_content_after_think_block=lambda content: False,
            _strip_think_blocks=lambda content: content,
            # end_turn persistence/cleanup stubs.
            _cleanup_task_resources=lambda task_id: None,
            _persist_session=lambda messages, history: None,
            _session_messages=[],
            _vprint=MagicMock(),
            _emit_status=MagicMock(),
            _flush_status_buffer=MagicMock(),
            _get_messages_up_to_last_assistant=lambda messages: messages,
            # _continue_text stubs (exercised by the headroom-available test).
            _build_assistant_message=lambda msg, fr: {"role": "assistant", "content": getattr(msg, "content", ""), "finish_reason": fr},
            _ephemeral_reasoning_off=False,
        )
        return ag

    def _retry(self):
        return TurnRetryState()

    def test_returns_window_exhausted_verdict_not_continuation_retry(self, agent):
        response = SimpleNamespace(
            id="resp_1",
            usage=SimpleNamespace(prompt_tokens=32638),
        )
        verdict = recover_from_truncation(
            agent, response, "length", self._retry(),
            messages=[{"role": "user", "content": "hi"}],
            conversation_history=None, api_kwargs={}, api_call_count=1,
            effective_task_id="t1", current_turn_user_idx=0,
            length_continue_retries=0, truncated_response_parts=[],
            truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
        )
        # Aborted (return) — NOT armed for a continuation retry (break).
        assert verdict.action == "return"
        assert verdict.result is not None
        assert verdict.result["error"] == _WINDOW_EXHAUSTED[2]
        assert verdict.result["final_response"] == _WINDOW_EXHAUSTED[1]
        # No continuation retry was armed.
        assert not getattr(self._retry(), "restart_with_length_continuation", False)

    def test_proceeds_to_continuation_when_headroom_available(self, agent):
        # 1,000 prompt + 8,192 max_tokens < 32,768 → continuation path (break) fires.
        agent.context_compressor.last_real_prompt_tokens = 1000
        response = SimpleNamespace(
            id="resp_1",
            usage=SimpleNamespace(prompt_tokens=1000),
        )
        retry = self._retry()
        verdict = recover_from_truncation(
            agent, response, "length", retry,
            messages=[{"role": "user", "content": "hi"}],
            conversation_history=None, api_kwargs={}, api_call_count=1,
            effective_task_id="t1", current_turn_user_idx=0,
            length_continue_retries=0, truncated_response_parts=[],
            truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
        )
        # Continuation arms a retry (break), not an abort (return).
        assert verdict.action == "break"
        assert retry.restart_with_length_continuation is True
