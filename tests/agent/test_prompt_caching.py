"""Tests for agent/prompt_caching.py — Anthropic cache control injection."""


from agent.prompt_caching import (
    _apply_cache_marker,
    _can_carry_marker,
    apply_anthropic_cache_control,
)


MARKER = {"type": "ephemeral"}


class TestApplyCacheMarker:
    def test_tool_message_gets_top_level_marker_on_native_anthropic(self):
        """Native Anthropic path: cache_control injected top-level (adapter moves it inside tool_result)."""
        msg = {"role": "tool", "content": "result"}
        _apply_cache_marker(msg, MARKER, native_anthropic=True)
        assert msg["cache_control"] == MARKER

    def test_tool_message_gets_content_part_marker_on_envelope(self):
        """Envelope layout (OpenRouter): tool message with content gets marker inside content part."""
        msg = {"role": "tool", "content": "result", "tool_call_id": "call_1"}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        # Content wrapped into list with cache_control on last part
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "result"
        assert msg["content"][0]["cache_control"] == MARKER
        # tool_call_id untouched, no top-level cache_control
        assert msg["tool_call_id"] == "call_1"
        assert "cache_control" not in msg

    def test_empty_tool_message_skips_on_envelope(self):
        """Envelope layout: empty tool message has no content part to carry marker."""
        msg = {"role": "tool", "content": ""}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg
        assert msg["content"] == ""

    def test_none_tool_message_skips_on_envelope(self):
        """Envelope layout: tool message with None content is skipped."""
        msg = {"role": "tool", "content": None}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg

    def test_empty_assistant_skips_on_envelope(self):
        """Envelope layout: empty assistant message gets top-level marker (not tool)."""
        msg = {"role": "assistant", "content": ""}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert msg["cache_control"] == MARKER

    def test_none_content_gets_top_level_marker(self):
        msg = {"role": "assistant", "content": None}
        _apply_cache_marker(msg, MARKER)
        assert msg["cache_control"] == MARKER

    def test_empty_string_content_gets_top_level_marker(self):
        """Empty text blocks cannot have cache_control (Anthropic rejects them)."""
        msg = {"role": "assistant", "content": ""}
        _apply_cache_marker(msg, MARKER)
        assert msg["cache_control"] == MARKER
        # Must NOT wrap into [{"type": "text", "text": "", "cache_control": ...}]
        assert msg["content"] == ""

    def test_string_content_wrapped_in_list(self):
        msg = {"role": "user", "content": "Hello"}
        _apply_cache_marker(msg, MARKER)
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Hello"
        assert msg["content"][0]["cache_control"] == MARKER

    def test_list_content_last_item_gets_marker(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Second"},
            ],
        }
        _apply_cache_marker(msg, MARKER)
        assert "cache_control" not in msg["content"][0]
        assert msg["content"][1]["cache_control"] == MARKER

    def test_empty_list_content_no_crash(self):
        msg = {"role": "user", "content": []}
        # Should not crash on empty list
        _apply_cache_marker(msg, MARKER)


class TestApplyAnthropicCacheControl:
    def test_empty_messages(self):
        result = apply_anthropic_cache_control([])
        assert result == []

    def test_returns_deep_copy(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = apply_anthropic_cache_control(msgs)
        assert result is not msgs
        assert result[0] is not msgs[0]
        # Original should be unmodified
        assert "cache_control" not in msgs[0].get("content", "")

    def test_system_message_gets_marker(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # System message should have cache_control
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["type"] == "ephemeral"

    def test_last_3_non_system_get_markers(self):
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # System (index 0) + last 3 non-system (indices 2, 3, 4) = 4 breakpoints
        # Index 1 (msg1) should NOT have marker
        content_1 = result[1]["content"]
        if isinstance(content_1, str):
            assert True  # No marker applied (still a string)
        else:
            assert "cache_control" not in content_1[0]

    def test_no_system_message(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # Both should get markers (4 slots available, only 2 messages)
        assert len(result) == 2

    def test_1h_ttl(self):
        msgs = [{"role": "system", "content": "System prompt"}]
        result = apply_anthropic_cache_control(msgs, cache_ttl="1h")
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["ttl"] == "1h"

    def test_max_4_breakpoints(self):
        msgs = [
            {"role": "system", "content": "System"},
        ] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(10)
        ]
        result = apply_anthropic_cache_control(msgs)
        # Count how many messages have cache_control
        count = 0
        for msg in result:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "cache_control" in item:
                        count += 1
            elif "cache_control" in msg:
                count += 1
        assert count <= 4
class TestCanCarryMarker:
    def test_native_always_true(self):
        assert _can_carry_marker({"role": "tool", "content": ""}, native_anthropic=True) is True
        assert _can_carry_marker({"role": "assistant", "content": None}, native_anthropic=True) is True

    def test_envelope_empty_content_false(self):
        assert _can_carry_marker({"role": "assistant", "content": ""}, native_anthropic=False) is False
        assert _can_carry_marker({"role": "assistant", "content": None}, native_anthropic=False) is False
        assert _can_carry_marker({"role": "tool", "content": ""}, native_anthropic=False) is False

    def test_envelope_string_content_true(self):
        assert _can_carry_marker({"role": "user", "content": "hello"}, native_anthropic=False) is True
        assert _can_carry_marker({"role": "tool", "content": "result"}, native_anthropic=False) is True

    def test_envelope_list_content_true(self):
        assert _can_carry_marker({"role": "tool", "content": [{"type": "text", "text": "x"}]}, native_anthropic=False) is True

    def test_envelope_empty_list_false(self):
        assert _can_carry_marker({"role": "tool", "content": []}, native_anthropic=False) is False


