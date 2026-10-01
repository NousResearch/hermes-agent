"""Regression coverage for #123923: restored Anthropic tool turns lose cross-block order.

The live transport preserves signed thinking/tool_use order in ``anthropic_content_blocks``.
That carrier is intentionally in-memory only. After state.db restore, only parallel
``reasoning_details`` + ``tool_calls`` remain, so a signed thinking block\'s original position
relative to tools is unknowable and must not be replayed as if that order were still proven.
"""

from types import SimpleNamespace

from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.transports import get_transport


def _normalize(blocks):
    return get_transport("anthropic_messages").normalize_response(
        SimpleNamespace(content=blocks, stop_reason="tool_use", usage=None)
    )


def _persisted_shape(normalized):
    provider_data = normalized.provider_data or {}
    return {
        "role": "assistant",
        "content": normalized.content or "",
        "reasoning_details": provider_data.get("reasoning_details"),
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in (normalized.tool_calls or [])
        ],
    }


def _convert_turn(assistant):
    _system, converted = convert_messages_to_anthropic(
        [
            {"role": "user", "content": "continue"},
            assistant,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "one"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": "two"},
        ],
        base_url=None,
        model="claude-opus-5-5",
    )
    return next(message for message in converted if message.get("role") == "assistant")


def _signed_blocks(message):
    return [
        block
        for block in message["content"]
        if isinstance(block, dict)
        and block.get("type") in {"thinking", "redacted_thinking"}
        and (block.get("signature") or block.get("data"))
    ]


def test_reported_interleaved_restore_does_not_replay_stale_signatures():
    normalized = _normalize(
        [
            SimpleNamespace(type="thinking", thinking="first plan", signature="sig-a" * 40),
            SimpleNamespace(type="tool_use", id="toolu_1", name="read_file", input={"path": "a.py"}),
            SimpleNamespace(type="thinking", thinking="second plan", signature="sig-b" * 40),
            SimpleNamespace(type="tool_use", id="toolu_2", name="read_file", input={"path": "b.py"}),
        ]
    )

    assistant = _convert_turn(_persisted_shape(normalized))

    assert not _signed_blocks(assistant)
    assert {b.get("id") for b in assistant["content"] if isinstance(b, dict) and b.get("type") == "tool_use"} == {
        "toolu_1",
        "toolu_2",
    }


def test_single_post_tool_thinking_signature_is_also_demoted():
    """Counterexample left open by #123955: one thinking block can still follow a tool_use."""
    normalized = _normalize(
        [
            SimpleNamespace(type="tool_use", id="toolu_1", name="read_file", input={"path": "a.py"}),
            SimpleNamespace(type="thinking", thinking="plan after first tool", signature="sig-b" * 40),
            SimpleNamespace(type="tool_use", id="toolu_2", name="read_file", input={"path": "b.py"}),
        ]
    )

    assistant = _convert_turn(_persisted_shape(normalized))

    assert not _signed_blocks(assistant)
    reasoning_text = " ".join(
        block.get("text", "")
        for block in assistant["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    )
    assert "plan after first tool" in reasoning_text


def test_single_pre_tool_signature_is_conservatively_demoted_after_restore():
    """Its stored shape is indistinguishable from the unsafe post-tool case once order is lost."""
    normalized = _normalize(
        [
            SimpleNamespace(type="thinking", thinking="plan before tool", signature="sig-a" * 40),
            SimpleNamespace(type="tool_use", id="toolu_1", name="read_file", input={"path": "a.py"}),
            SimpleNamespace(type="tool_use", id="toolu_2", name="read_file", input={"path": "b.py"}),
        ]
    )

    assistant = _convert_turn(_persisted_shape(normalized))

    assert not _signed_blocks(assistant)


def test_live_ordered_carrier_keeps_valid_interleaved_signatures():
    normalized = _normalize(
        [
            SimpleNamespace(type="thinking", thinking="first plan", signature="sig-a" * 40),
            SimpleNamespace(type="tool_use", id="toolu_1", name="read_file", input={"path": "a.py"}),
            SimpleNamespace(type="thinking", thinking="second plan", signature="sig-b" * 40),
            SimpleNamespace(type="tool_use", id="toolu_2", name="read_file", input={"path": "b.py"}),
        ]
    )
    stored = _persisted_shape(normalized)
    stored["anthropic_content_blocks"] = (normalized.provider_data or {})["anthropic_content_blocks"]

    assistant = _convert_turn(stored)

    assert len(_signed_blocks(assistant)) == 2


def test_signed_thinking_without_tool_calls_keeps_existing_replay_policy():
    assistant = {
        "role": "assistant",
        "content": "answer",
        "reasoning_details": [
            {"type": "thinking", "thinking": "private plan", "signature": "sig-a" * 40}
        ],
    }
    _system, converted = convert_messages_to_anthropic(
        [{"role": "user", "content": "question"}, assistant],
        base_url=None,
        model="claude-opus-5-5",
    )
    converted_assistant = next(message for message in converted if message.get("role") == "assistant")

    assert len(_signed_blocks(converted_assistant)) == 1
