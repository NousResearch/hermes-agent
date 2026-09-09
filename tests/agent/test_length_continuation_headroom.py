"""#106120: abort length continuation only when usage proves context-window exhaustion."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import (
    _agent_context_length,
    _context_ceiling_epsilon,
    _context_window_exhausted,
    _continue_text,
    recover_from_truncation,
)


def _agent(
    *,
    context_length: Optional[int] = 32768,
    api_mode: str = "chat_completions",
    provider: str = "custom",
) -> Any:
    compressor = None
    if context_length is not None:
        compressor = SimpleNamespace(context_length=context_length)

    def _build_assistant_message(msg: Any, finish_reason: str) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": getattr(msg, "content", None),
            "finish_reason": finish_reason,
        }

    return SimpleNamespace(
        api_mode=api_mode,
        provider=provider,
        log_prefix="",
        _ephemeral_reasoning_off=False,
        _session_messages=[],
        context_compressor=compressor,
        _build_assistant_message=_build_assistant_message,
        _strip_think_blocks=lambda text: text or "",
        _has_content_after_think_block=lambda _content: True,
        _vprint=lambda *_a, **_k: None,
        _emit_status=lambda *_a, **_k: None,
        _cleanup_task_resources=lambda *_a, **_k: None,
        _persist_session=lambda *_a, **_k: None,
        _fallback_index=0,
        _fallback_chain=[],
        _try_activate_fallback=lambda: False,
        _get_messages_up_to_last_assistant=lambda msgs: msgs,
        _flush_status_buffer=lambda: None,
        _buffer_vprint=lambda *_a, **_k: None,
        _get_transport=lambda: SimpleNamespace(
            normalize_response=lambda response, **_kwargs: response.choices[0].message
        ),
    )


def _response(
    *,
    content: str | None,
    usage: Any = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    finish_reason: str = "length",
):
    if usage is None and (prompt_tokens is not None or completion_tokens is not None):
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens if completion_tokens is not None else 0,
        )
    message = SimpleNamespace(content=content, tool_calls=None, finish_reason=finish_reason)
    return SimpleNamespace(
        id="chatcmpl-test",
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage,
    )


def _trunc(agent: Any, response: Any, messages: List[Dict[str, Any]], *, retries: int = 0):
    from agent.turn_truncation import _Trunc

    return _Trunc(
        agent=agent,
        response=response,
        finish_reason="length",
        conversation_history=None,
        api_call_count=1,
        effective_task_id=None,
        current_turn_user_idx=0,
        messages=messages,
        length_continue_retries=retries,
        truncated_response_parts=[],
        truncated_tool_call_retries=0,
        retry_count=0,
        compression_attempts=0,
    )


class TestContextCeilingHelpers:
    def test_epsilon_is_tiny_absolute_slack(self):
        assert _context_ceiling_epsilon(32768) == 8
        assert _context_ceiling_epsilon(4096) == 1
        assert _context_ceiling_epsilon(8192) == 2
        assert _context_ceiling_epsilon(100000) == 8

    def test_agent_context_length_from_compressor(self):
        assert _agent_context_length(_agent(context_length=32768)) == 32768
        assert _agent_context_length(_agent(context_length=None)) is None


class TestContextWindowExhaustedPredicate:
    def test_ceiling_confirmed(self):
        agent = _agent(context_length=32768)
        response = _response(content="partial", prompt_tokens=32638, completion_tokens=129)
        assert _context_window_exhausted(agent, response) == (32638, 129, 32768)

    def test_real_output_cap_fail_open(self):
        agent = _agent(context_length=32768)
        response = _response(content="partial", prompt_tokens=10000, completion_tokens=512)
        assert _context_window_exhausted(agent, response) is None

    def test_near_ceiling_ambiguous_fail_open(self):
        agent = _agent(context_length=32768)
        response = _response(content="partial", prompt_tokens=32600, completion_tokens=32)
        assert _context_window_exhausted(agent, response) is None

    def test_missing_usage_fail_open(self):
        agent = _agent(context_length=32768)
        response = _response(content="partial", usage=None)
        assert _context_window_exhausted(agent, response) is None

    def test_missing_context_length_fail_open(self):
        agent = _agent(context_length=None)
        response = _response(content="partial", prompt_tokens=32638, completion_tokens=129)
        assert _context_window_exhausted(agent, response) is None

    def test_zero_output_tokens_fail_open(self):
        agent = _agent(context_length=32768)
        response = _response(content="partial", prompt_tokens=32638, completion_tokens=0)
        assert _context_window_exhausted(agent, response) is None

    def test_anthropic_shaped_usage_via_normalize(self):
        agent = _agent(context_length=32768, api_mode="anthropic_messages", provider="anthropic")
        usage = SimpleNamespace(
            input_tokens=30638,
            cache_read_input_tokens=2000,
            cache_creation_input_tokens=0,
            output_tokens=129,
        )
        response = _response(content="partial", usage=usage)
        # prompt = 30638+2000 = 32638; +129 = 32767 >= 32760
        assert _context_window_exhausted(agent, response) == (32638, 129, 32768)

    def test_dict_shaped_usage_via_normalize(self):
        agent = _agent(context_length=32768)
        response = _response(
            content="partial",
            usage={"prompt_tokens": 32638, "completion_tokens": 129},
        )
        assert _context_window_exhausted(agent, response) == (32638, 129, 32768)

    def test_boundary_exactly_at_epsilon_aborts(self):
        # epsilon(32768)=8 → abort when prompt+output >= 32760
        agent = _agent(context_length=32768)
        response = _response(content="x", prompt_tokens=32631, completion_tokens=129)
        assert 32631 + 129 == 32760
        assert _context_window_exhausted(agent, response) == (32631, 129, 32768)

    def test_boundary_just_below_epsilon_continues(self):
        agent = _agent(context_length=32768)
        response = _response(content="x", prompt_tokens=32630, completion_tokens=129)
        assert 32630 + 129 == 32759
        assert _context_window_exhausted(agent, response) is None


class TestContinueTextCeilingGuard:
    def test_ceiling_confirmed_no_nudge_no_retry(self):
        agent = _agent(context_length=32768)
        messages = [{"role": "user", "content": "huge prompt"}]
        response = _response(
            content="partial answer...", prompt_tokens=32638, completion_tokens=129,
        )
        st = _trunc(agent, response, messages)
        retry = TurnRetryState()
        verdict = _continue_text(st, retry, response.choices[0].message)

        assert verdict.action == "return"
        assert retry.restart_with_length_continuation is False
        assert st.length_continue_retries == 0
        result = verdict.result or {}
        assert result.get("completed") is False
        assert result.get("partial") is True
        final = result.get("final_response", "")
        err = (result.get("error") or "").lower()
        assert "partial answer..." in final
        assert "context window" in final.lower()
        assert "max_tokens" not in final.lower()
        assert "output-length" not in final.lower()
        assert "context window" in err
        assert not any(
            m.get("_length_continuation_nudge") for m in messages if isinstance(m, dict)
        )
        assert not any(
            m.get("_length_continuation_fragment") for m in messages if isinstance(m, dict)
        )

    def test_real_output_cap_still_continues(self):
        agent = _agent(context_length=32768)
        messages = [{"role": "user", "content": "short"}]
        response = _response(
            content="partial answer cut off", prompt_tokens=10000, completion_tokens=512,
        )
        st = _trunc(agent, response, messages)
        retry = TurnRetryState()
        verdict = _continue_text(st, retry, response.choices[0].message)

        assert verdict.action == "break"
        assert retry.restart_with_length_continuation is True
        assert st.length_continue_retries == 1
        assert any(m.get("_length_continuation_nudge") for m in messages if isinstance(m, dict))

    def test_near_ceiling_ambiguous_still_continues(self):
        agent = _agent(context_length=32768)
        messages = [{"role": "user", "content": "near"}]
        response = _response(content="partial", prompt_tokens=32600, completion_tokens=32)
        st = _trunc(agent, response, messages)
        retry = TurnRetryState()
        verdict = _continue_text(st, retry, response.choices[0].message)
        assert verdict.action == "break"
        assert retry.restart_with_length_continuation is True

    def test_missing_usage_keeps_legacy(self):
        agent = _agent(context_length=32768)
        messages = [{"role": "user", "content": "unknown"}]
        response = _response(content="partial", usage=None)
        st = _trunc(agent, response, messages)
        retry = TurnRetryState()
        verdict = _continue_text(st, retry, response.choices[0].message)
        assert verdict.action == "break"
        assert retry.restart_with_length_continuation is True

    def test_missing_context_length_keeps_legacy(self):
        agent = _agent(context_length=None)
        messages = [{"role": "user", "content": "unknown ctx"}]
        response = _response(content="partial", prompt_tokens=32638, completion_tokens=129)
        st = _trunc(agent, response, messages)
        retry = TurnRetryState()
        verdict = _continue_text(st, retry, response.choices[0].message)
        assert verdict.action == "break"
        assert retry.restart_with_length_continuation is True

    def test_trail_cleanup_after_abort_with_prior_nudge(self):
        agent = _agent(context_length=32768)
        messages = [
            {"role": "user", "content": "huge"},
            {
                "role": "assistant",
                "content": "earlier fragment",
                "_length_continuation_fragment": True,
            },
            {
                "role": "user",
                "content": "Please continue.",
                "_length_continuation_nudge": True,
            },
        ]
        response = _response(
            content="partial answer...", prompt_tokens=32638, completion_tokens=129,
        )
        st = _trunc(agent, response, messages, retries=1)
        # length_continue_retries already 1 from a prior attempt — abort must not bump further
        retry = TurnRetryState()
        verdict = _continue_text(st, retry, response.choices[0].message)
        assert verdict.action == "return"
        assert retry.restart_with_length_continuation is False
        # Abort path must not increment retries when ceiling is proven
        assert st.length_continue_retries == 1
        assert not any(
            isinstance(m, dict) and (
                m.get("_length_continuation_nudge") or m.get("_length_continuation_fragment")
            )
            for m in messages
        )
        assert "partial answer..." in (verdict.result or {}).get("final_response", "")

    def test_legacy_ceiling_exit_still_nudges_then_stops(self):
        """Normal output truncation still nudges/retries; 4th attempt still ceiling-exits."""
        agent = _agent(context_length=32768)
        messages = [{"role": "user", "content": "short"}]
        response = _response(
            content="chunk", prompt_tokens=10000, completion_tokens=512,
        )
        st = _trunc(agent, response, messages, retries=3)
        retry = TurnRetryState()
        verdict = _continue_text(st, retry, response.choices[0].message)
        assert verdict.action == "return"
        assert retry.restart_with_length_continuation is False
        assert "chunk" in (verdict.result or {}).get("final_response", "")
        assert not any(
            m.get("_length_continuation_nudge") for m in messages if isinstance(m, dict)
        )


class TestRecoverFromTruncationIntegration:
    def test_recover_path_aborts_on_proven_ceiling(self):
        agent = _agent(context_length=32768)
        messages = [{"role": "user", "content": "huge"}]
        response = _response(
            content="tiny", prompt_tokens=32638, completion_tokens=129,
        )
        response.choices[0].message.tool_calls = None
        retry = TurnRetryState()
        verdict = recover_from_truncation(
            agent, response, "length", retry,
            messages=messages, conversation_history=None, api_kwargs={},
            api_call_count=1, effective_task_id=None, current_turn_user_idx=0,
            length_continue_retries=0, truncated_response_parts=[],
            truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
        )
        assert verdict.action == "return"
        assert retry.restart_with_length_continuation is False
        final = (verdict.result or {}).get("final_response", "")
        assert "context window" in final.lower()
        assert "tiny" in final

    def test_recover_path_continues_on_ambiguous_near_ceiling(self):
        """Old <256 headroom guard would abort here; usage-backed must fail open."""
        agent = _agent(context_length=32768)
        messages = [{"role": "user", "content": "near"}]
        response = _response(content="tiny", prompt_tokens=32600, completion_tokens=32)
        response.choices[0].message.tool_calls = None
        retry = TurnRetryState()
        verdict = recover_from_truncation(
            agent, response, "length", retry,
            messages=messages, conversation_history=None, api_kwargs={},
            api_call_count=1, effective_task_id=None, current_turn_user_idx=0,
            length_continue_retries=0, truncated_response_parts=[],
            truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
        )
        assert verdict.action == "break"
        assert retry.restart_with_length_continuation is True
