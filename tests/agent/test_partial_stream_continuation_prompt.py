"""After a mid-stream transport cut the continuation must keep tools on the table (#74990)."""
import pytest

from agent import conversation_loop as cl
from agent.context_compressor import ContextCompressor


@pytest.mark.parametrize("dropped", [None, ["write_file"]])
def test_partial_stub_continuation_says_tools_remain_available(dropped):
    prompt = cl._get_continuation_prompt(True, dropped).lower()
    assert "tools" in prompt and "available" in prompt
    assert "finish the answer directly" not in prompt


@pytest.mark.parametrize("dropped", [None, ["write_file"]])
def test_new_and_legacy_stub_nudges_are_recognized_as_synthetic(dropped):
    for text in (cl._get_continuation_prompt(True, dropped), cl._LEGACY_LENGTH_CONTINUATION_NETWORK_STUB):
        assert ContextCompressor._is_synthetic_compression_user_turn({"role": "user", "content": text})


def test_stage_tool_call_message_absorbs_continuation_trail():
    from types import SimpleNamespace
    from agent.turn_tool_round import stage_tool_call_message

    agent = SimpleNamespace(
        _build_assistant_message=lambda msg, fr: {"role": "assistant", "content": msg.content, "tool_calls": []},
        _has_content_after_think_block=lambda c: bool(c),
        _has_stream_consumers=lambda: False,
        _should_emit_quiet_tool_messages=lambda: False,
        _interim_assistant_visible_text=lambda m: m.get("content", ""),
        _thinking_prefill_retries=0,
        _empty_content_retries=0,
        _post_tool_empty_retried=False,
        _dropped_toolcall_retries=0,
    )
    tc = SimpleNamespace(function=SimpleNamespace(name="write_file", arguments="{}"))
    assistant_message = SimpleNamespace(content="Writing file now.", tool_calls=[tc])
    messages = [
        {"role": "user", "content": "do task"},
        {"role": "assistant", "content": "Partial preamble", "_length_continuation_fragment": True},
        {"role": "user", "content": cl._get_continuation_prompt(True, None), "_length_continuation_nudge": True},
    ]
    stage_tool_call_message(agent, assistant_message=assistant_message, finish_reason="tool_calls", messages=messages)
    assert "_length_continuation_fragment" not in messages[1]
    assert messages[1].get("_absorbed_continuation_fragment") is True
    assert messages[1].get("display_kind") == "hidden"
    assert "_length_continuation_nudge" not in messages[2]
    assert messages[2].get("_absorbed_continuation_nudge") is True
