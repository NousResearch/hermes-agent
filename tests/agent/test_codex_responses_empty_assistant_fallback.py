"""Thinking-only assistant turns must not emit a synthetic empty-content assistant
item on the Responses wire.

Volcengine ARK api/plan rejects {"role": "assistant", "content": ""} with
400 MissingParameter: missing input.content. The "" fallback exists to satisfy
OpenAI's "every reasoning item needs a following item" rule, but a trailing
function_call IS a following item — so when the turn has tool_calls, the empty
message is redundant and toxic on strict backends (#volc-vision, reproduced
2026-09-07 on v0.21 with doubao-seed-evolving / glm-5.3-flash).
"""
from agent.codex_responses_adapter import _chat_messages_to_responses_input


def _thinking_only_turn_messages():
    return [
        {"role": "user", "content": "看看这张图"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_abc", "type": "function",
                "function": {"name": "vision_analyze", "arguments": '{"image": "x.png"}'},
            }],
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "ENC123", "summary": []},
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "图片内容是..."},
    ]


def test_thinking_only_tool_turn_has_no_empty_assistant_item():
    items = _chat_messages_to_responses_input(_thinking_only_turn_messages())
    types = [it.get("type") for it in items]
    assert "reasoning" in types and "function_call" in types
    assert not [
        it for it in items
        if it.get("role") == "assistant" and it.get("content") == ""
    ]


def test_reasoning_with_no_following_item_still_gets_empty_fallback():
    """OpenAI requires a following item after reasoning; with no tool_calls and
    no text there is nothing to follow, so the "" fallback must remain."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "ENC123", "summary": []},
            ],
        },
    ]
    items = _chat_messages_to_responses_input(messages)
    assert any(
        it.get("role") == "assistant" and it.get("content") == "" for it in items
    )


def test_text_reply_turn_keeps_content_after_reasoning():
    """A text reply replays its own content after the reasoning items — the
    fallback path must not change it."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "你好！",
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "ENC123", "summary": []},
            ],
        },
    ]
    items = _chat_messages_to_responses_input(messages)
    assistant_texts = [
        it.get("content") for it in items if it.get("role") == "assistant"
    ]
    assert "你好！" in assistant_texts
