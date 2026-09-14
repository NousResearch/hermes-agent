from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.turn_final_response import finish_text_response
from agent.turn_stop_gates import StopGateVerdict


def test_stop_gate_replacement_is_the_returned_final_response():
    messages = []
    agent = SimpleNamespace(
        _has_content_after_think_block=lambda value: bool(value),
        _strip_think_blocks=lambda value: value,
        _emit_pending_fallback_notice=lambda: None,
        _clear_status_buffer=lambda: None,
        _build_assistant_message=lambda _message, reason: {
            "role": "assistant",
            "content": "unsafe clinical candidate",
            "finish_reason": reason,
        },
        _flush_messages_to_session_db=lambda *_args: None,
        _looks_like_codex_intermediate_ack=lambda **_kwargs: False,
        _mute_post_response=False,
        _empty_content_retries=0,
        _thinking_prefill_retries=0,
        _dropped_toolcall_retries=0,
        _stall_guards=False,
        valid_tool_names=[],
        quiet_mode=True,
        session_id="session",
        api_mode="chat_completions",
    )

    def gate(_agent, final_msg, **_kwargs):
        final_msg["content"] = "safe degradation"
        return StopGateVerdict(
            continue_turn=False,
            final_response="safe degradation",
            pending_verification_response=None,
            pending_verification_response_previewed=False,
        )

    with patch("agent.turn_final_response.apply_stop_gates", side_effect=gate):
        verdict = finish_text_response(
            agent,
            assistant_message=SimpleNamespace(content="unsafe clinical candidate", tool_calls=None),
            response=SimpleNamespace(),
            finish_reason="stop",
            messages=messages,
            api_messages=[],
            conversation_history=None,
            api_call_count=1,
            user_message="clinical question",
            active_system_prompt="system",
            final_response=None,
            _turn_exit_reason=None,
            _preflight_compression_blocked=False,
            codex_ack_continuations=0,
            truncated_response_parts=[],
            length_continue_retries=0,
            _pending_verification_response=None,
            _pending_verification_response_previewed=False,
        )

    assert verdict.action == "break"
    assert verdict.final_response == "safe degradation"
    assert messages[-1]["content"] == "safe degradation"
