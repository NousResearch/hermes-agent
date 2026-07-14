"""Request-build-time screenshot eviction for the chat.completions path.

Mirrors the Anthropic adapter's ``_evict_old_screenshots``
(agent/anthropic_adapter.py) for OpenAI-format payloads: keep only the
most recent N image-bearing tool results in the outbound request,
placeholder the rest, and never touch the stored conversation history
(prompt-caching invariant — only the per-call payload copy changes).

See the Anthropic-side coverage in
tests/tools/test_computer_use.py::TestAnthropicAdapterMultimodal.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from agent.conversation_loop import _evict_old_screenshots_openai


FAKE_PNG = "iVBORw0KGgo="


def _image_part(tag: str = "") -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{FAKE_PNG}{tag}"},
    }


def _tool_msg(call_id: str, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "role": "tool",
        "name": "computer_use",
        "tool_call_id": call_id,
        "content": parts,
    }


def _assistant_call(call_id: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "computer_use", "arguments": "{}"},
        }],
    }


def _screenshot_conversation(n_screenshots: int) -> List[Dict[str, Any]]:
    """User turn followed by ``n_screenshots`` assistant/tool pairs."""
    messages: List[Dict[str, Any]] = [{"role": "user", "content": "start"}]
    for i in range(n_screenshots):
        messages.append(_assistant_call(f"call_{i}"))
        messages.append(_tool_msg(
            f"call_{i}",
            [{"type": "text", "text": f"cap {i}"}, _image_part(str(i))],
        ))
    messages.append({"role": "assistant", "content": "done"})
    return messages


def _image_bearing_tool_msgs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        m for m in messages
        if m.get("role") == "tool"
        and isinstance(m.get("content"), list)
        and any(
            isinstance(p, dict) and p.get("type") == "image_url"
            for p in m["content"]
        )
    ]


def _placeholder_tool_msgs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        m for m in messages
        if m.get("role") == "tool"
        and isinstance(m.get("content"), list)
        and any(
            isinstance(p, dict)
            and p.get("type") == "text"
            and "screenshot removed" in p.get("text", "")
            for p in m["content"]
        )
    ]


class TestEviction:
    def test_keeps_most_recent_three_placeholders_older(self):
        messages = _screenshot_conversation(5)
        out = _evict_old_screenshots_openai(messages)

        assert len(_image_bearing_tool_msgs(out)) == 3
        assert len(_placeholder_tool_msgs(out)) == 2

        # It is specifically the OLDEST two that were evicted.
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert _placeholder_tool_msgs(tool_msgs[:2]) == tool_msgs[:2]
        assert _image_bearing_tool_msgs(tool_msgs[2:]) == tool_msgs[2:]

    def test_at_or_below_keep_window_is_a_passthrough(self):
        messages = _screenshot_conversation(3)
        out = _evict_old_screenshots_openai(messages)
        # Same message objects, by identity — nothing was rebuilt.
        assert all(a is b for a, b in zip(out, messages))
        assert len(out) == len(messages)

    def test_text_parts_survive_eviction(self):
        messages = _screenshot_conversation(5)
        out = _evict_old_screenshots_openai(messages)
        evicted = _placeholder_tool_msgs(out)
        assert len(evicted) == 2
        for msg in evicted:
            texts = [p["text"] for p in msg["content"] if p.get("type") == "text"]
            # Original caption preserved alongside the placeholder.
            assert any(t.startswith("cap ") for t in texts)
            # No image parts remain.
            assert all(p.get("type") == "text" for p in msg["content"])

    def test_counts_per_tool_message_not_per_image_part(self):
        # One tool result carrying two images counts once — same semantics
        # as the Anthropic adapter, which counts image-bearing tool_result
        # blocks, not individual image blocks.
        messages: List[Dict[str, Any]] = [{"role": "user", "content": "go"}]
        messages.append(_assistant_call("call_multi"))
        messages.append(_tool_msg(
            "call_multi",
            [_image_part("a"), _image_part("b")],
        ))
        for i in range(2):
            messages.append(_assistant_call(f"call_{i}"))
            messages.append(_tool_msg(f"call_{i}", [_image_part(str(i))]))

        out = _evict_old_screenshots_openai(messages)
        # 3 image-bearing tool messages total — all within the keep window.
        assert len(_image_bearing_tool_msgs(out)) == 3
        assert not _placeholder_tool_msgs(out)

    def test_every_image_part_in_an_evicted_message_is_replaced(self):
        messages: List[Dict[str, Any]] = [{"role": "user", "content": "go"}]
        messages.append(_assistant_call("call_old"))
        messages.append(_tool_msg("call_old", [_image_part("a"), _image_part("b")]))
        for i in range(3):
            messages.append(_assistant_call(f"call_{i}"))
            messages.append(_tool_msg(f"call_{i}", [_image_part(str(i))]))

        out = _evict_old_screenshots_openai(messages)
        evicted = _placeholder_tool_msgs(out)
        assert len(evicted) == 1
        assert [p["text"] for p in evicted[0]["content"]] == [
            "[screenshot removed to save context]",
            "[screenshot removed to save context]",
        ]

    def test_input_image_shape_is_also_evicted(self):
        # OpenAI Responses-style parts stored on tool messages count too —
        # detection is shared with the compressor's _is_image_part.
        input_image = {"type": "input_image", "image_url": f"data:image/png;base64,{FAKE_PNG}"}
        messages: List[Dict[str, Any]] = [{"role": "user", "content": "go"}]
        messages.append(_assistant_call("call_old"))
        messages.append(_tool_msg("call_old", [dict(input_image)]))
        for i in range(3):
            messages.append(_assistant_call(f"call_{i}"))
            messages.append(_tool_msg(f"call_{i}", [_image_part(str(i))]))

        out = _evict_old_screenshots_openai(messages)
        evicted = _placeholder_tool_msgs(out)
        assert len(evicted) == 1
        assert evicted[0]["tool_call_id"] == "call_old"


class TestScope:
    def test_user_uploaded_images_are_untouched(self):
        # A user-role image older than every screenshot must survive —
        # scope matches the Anthropic adapter (tool results only).
        user_upload = {
            "role": "user",
            "content": [{"type": "text", "text": "look at this"}, _image_part("upload")],
        }
        messages = [user_upload] + _screenshot_conversation(5)[1:]
        out = _evict_old_screenshots_openai(messages)

        assert out[0] is user_upload
        assert any(p.get("type") == "image_url" for p in out[0]["content"])
        # Tool-side eviction still happened.
        assert len(_placeholder_tool_msgs(out)) == 2

    def test_string_tool_content_is_ignored(self):
        messages: List[Dict[str, Any]] = [{"role": "user", "content": "go"}]
        messages.append(_assistant_call("call_s"))
        messages.append(_tool_msg("call_s", []))
        messages[-1]["content"] = "plain text result"
        for i in range(4):
            messages.append(_assistant_call(f"call_{i}"))
            messages.append(_tool_msg(f"call_{i}", [_image_part(str(i))]))

        out = _evict_old_screenshots_openai(messages)
        assert len(_image_bearing_tool_msgs(out)) == 3
        assert len(_placeholder_tool_msgs(out)) == 1
        # The string tool message passed through by identity.
        assert any(m.get("content") == "plain text result" for m in out)

    def test_text_only_tool_results_do_not_consume_the_window(self):
        messages: List[Dict[str, Any]] = [{"role": "user", "content": "go"}]
        for i in range(4):
            messages.append(_assistant_call(f"img_{i}"))
            messages.append(_tool_msg(f"img_{i}", [_image_part(str(i))]))
            messages.append(_assistant_call(f"txt_{i}"))
            messages.append(_tool_msg(f"txt_{i}", [{"type": "text", "text": "ok"}]))

        out = _evict_old_screenshots_openai(messages)
        # 4 image results: newest 3 kept, oldest placeholdered; the text-only
        # results neither count against the window nor get rewritten.
        assert len(_image_bearing_tool_msgs(out)) == 3
        assert len(_placeholder_tool_msgs(out)) == 1
        for a, b in zip(messages, out):
            if a.get("tool_call_id", "").startswith("txt_"):
                assert a is b


class TestHistoryNotMutated:
    def test_input_messages_and_content_lists_are_untouched(self):
        messages = _screenshot_conversation(6)
        snapshot = copy.deepcopy(messages)

        out = _evict_old_screenshots_openai(messages)

        # Stored history is byte-identical to what it was before the call.
        assert messages == snapshot
        assert out is not messages

        # Evicted slots hold NEW dicts with NEW content lists; the original
        # dicts still carry their images for compression/UI use.
        original_by_id = {m.get("tool_call_id"): m for m in messages if m.get("role") == "tool"}
        for msg in _placeholder_tool_msgs(out):
            original = original_by_id[msg["tool_call_id"]]
            assert msg is not original
            assert msg["content"] is not original["content"]
            assert any(p.get("type") == "image_url" for p in original["content"])

    def test_unaffected_messages_pass_through_by_reference(self):
        messages = _screenshot_conversation(6)
        out = _evict_old_screenshots_openai(messages)
        rebuilt = 0
        for a, b in zip(messages, out):
            if a is not b:
                rebuilt += 1
        # Exactly the evicted tool messages were rebuilt (6 - 3 kept).
        assert rebuilt == 3


class TestParityWithAnthropicAdapter:
    def test_placeholder_text_matches_anthropic_adapter(self):
        """Both request-build-time evictors must speak the same placeholder.

        The wording is shared context the model sees across providers;
        drifting apart would make cross-provider session handoff
        inconsistent. Derive the expected text from the Anthropic adapter
        instead of hardcoding it twice.
        """
        from agent.anthropic_adapter import _evict_old_screenshots

        # Anthropic-format: 4 tool_result blocks with image blocks — the
        # oldest gets placeholdered by the adapter's evictor.
        anthropic_msgs = [
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": f"call_{i}",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "data": FAKE_PNG}},
                    ],
                }],
            }
            for i in range(4)
        ]
        _evict_old_screenshots(anthropic_msgs)
        anthropic_placeholders = [
            part["text"]
            for msg in anthropic_msgs
            for block in msg["content"]
            for part in block["content"]
            if part.get("type") == "text"
        ]
        assert len(anthropic_placeholders) == 1

        openai_msgs: List[Dict[str, Any]] = [{"role": "user", "content": "go"}]
        for i in range(4):
            openai_msgs.append(_assistant_call(f"call_{i}"))
            openai_msgs.append(_tool_msg(f"call_{i}", [_image_part(str(i))]))
        out = _evict_old_screenshots_openai(openai_msgs)
        openai_placeholders = [
            p["text"]
            for m in _placeholder_tool_msgs(out)
            for p in m["content"]
            if p.get("type") == "text"
        ]
        assert len(openai_placeholders) == 1

        assert openai_placeholders == anthropic_placeholders

    def test_keep_window_matches_anthropic_adapter(self):
        """Both paths keep the same number of recent screenshots.

        Relationship invariant, not a literal: run both evictors over
        equivalently sized conversations and compare survivor counts.
        """
        from agent.anthropic_adapter import _evict_old_screenshots

        n = 7
        anthropic_msgs = [
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": f"call_{i}",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "data": FAKE_PNG}},
                    ],
                }],
            }
            for i in range(n)
        ]
        _evict_old_screenshots(anthropic_msgs)
        anthropic_survivors = sum(
            1
            for msg in anthropic_msgs
            for block in msg["content"]
            if any(p.get("type") == "image" for p in block["content"])
        )

        openai_msgs: List[Dict[str, Any]] = [{"role": "user", "content": "go"}]
        for i in range(n):
            openai_msgs.append(_assistant_call(f"call_{i}"))
            openai_msgs.append(_tool_msg(f"call_{i}", [_image_part(str(i))]))
        out = _evict_old_screenshots_openai(openai_msgs)
        openai_survivors = len(_image_bearing_tool_msgs(out))

        assert openai_survivors == anthropic_survivors
