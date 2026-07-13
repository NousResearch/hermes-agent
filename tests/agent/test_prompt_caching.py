"""Tests for agent/prompt_caching.py — Anthropic cache control injection."""


from agent.prompt_caching import (
    _apply_cache_marker,
    _can_carry_marker,
    apply_anthropic_cache_control,
    apply_tool_cache_control,
    prepare_tools_for_api,
)


MARKER = {"type": "ephemeral"}


class TestApplyCacheMarker:
    def test_tool_message_gets_top_level_marker_on_native_anthropic(self):
        """Native Anthropic path: cache_control injected top-level (adapter moves it inside tool_result)."""
        msg = {"role": "tool", "content": "result"}
        _apply_cache_marker(msg, MARKER, native_anthropic=True)
        assert msg["cache_control"] == MARKER

    def test_tool_message_skips_marker_on_openrouter(self):
        """OpenRouter path: top-level cache_control on role:tool is invalid and causes silent hang."""
        msg = {"role": "tool", "content": "result"}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg

    def test_tool_message_wraps_non_empty_content_on_openrouter(self):
        """Non-empty tool content should be wrapped so the marker lands on a content part."""
        msg = {"role": "tool", "content": "tool result bytes"}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["cache_control"] == MARKER

    def test_empty_assistant_message_skips_marker_on_openrouter(self):
        """OpenRouter path: empty assistant turns are pure tool_calls, marker would be ignored."""
        msg = {"role": "assistant", "content": ""}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg

    def test_native_anthropic_empty_assistant_gets_top_level_marker(self):
        """Native Anthropic layout can still carry top-level marker on empty content."""
        msg = {"role": "assistant", "content": ""}
        _apply_cache_marker(msg, MARKER, native_anthropic=True)
        assert msg["cache_control"] == MARKER

    def test_none_content_skips_marker_on_openrouter(self):
        """OpenRouter path: None-content assistant turns are ignored."""
        msg = {"role": "assistant", "content": None}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg

    def test_none_content_gets_top_level_marker_on_native_anthropic(self):
        """Native Anthropic path: None content still gets top-level marker."""
        msg = {"role": "assistant", "content": None}
        _apply_cache_marker(msg, MARKER, native_anthropic=True)
        assert msg["cache_control"] == MARKER

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


class TestCanCarryMarker:
    def test_native_anthropic_always_true(self):
        assert _can_carry_marker({"role": "assistant", "content": ""}, native_anthropic=True) is True
        assert _can_carry_marker({"role": "tool", "content": ""}, native_anthropic=True) is True

    def test_openrouter_content_parts_carry_marker(self):
        assert _can_carry_marker({"role": "user", "content": "text"}, native_anthropic=False) is True
        assert _can_carry_marker({"role": "user", "content": [{"type": "text", "text": "a"}]}, native_anthropic=False) is True

    def test_openrouter_empty_or_none_does_not_carry_marker(self):
        assert _can_carry_marker({"role": "assistant", "content": ""}, native_anthropic=False) is False
        assert _can_carry_marker({"role": "assistant", "content": None}, native_anthropic=False) is False
        assert _can_carry_marker({"role": "tool", "content": "result"}, native_anthropic=False) is True
        assert _can_carry_marker({"role": "tool", "content": ""}, native_anthropic=False) is False

    def test_openrouter_list_carrier_requires_last_part_dict(self):
        """Carrier predicate must agree with _apply_cache_marker, which only marks
        the LAST content part. A list whose last element isn't a dict cannot carry
        a marker and must not consume a breakpoint."""
        # Last part is a dict -> carrier.
        assert _can_carry_marker(
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            native_anthropic=False,
        ) is True
        # Last part is a non-dict (stray raw string) -> NOT a carrier, even though
        # an earlier part is a dict. Previously this passed the gate but got no
        # marker, wasting a breakpoint.
        assert _can_carry_marker(
            {"role": "user", "content": [{"type": "text", "text": "a"}, "trailing raw"]},
            native_anthropic=False,
        ) is False
        # Empty list -> not a carrier.
        assert _can_carry_marker({"role": "user", "content": []}, native_anthropic=False) is False


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

    def test_reserved_breakpoint_reduces_message_markers(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        full = apply_anthropic_cache_control(msgs, reserved_breakpoints=0)
        reserved = apply_anthropic_cache_control(msgs, reserved_breakpoints=1)

        def _count_markers(items):
            count = 0
            for msg in items:
                content = msg.get("content")
                if isinstance(content, list):
                    count += sum(
                        1 for part in content
                        if isinstance(part, dict) and "cache_control" in part
                    )
                elif "cache_control" in msg:
                    count += 1
            return count

        assert _count_markers(reserved) == _count_markers(full) - 1


class TestToolCacheControl:
    def test_apply_tool_cache_control_marks_last_tool_only(self):
        tools = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]
        result = apply_tool_cache_control(tools)
        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in tools[1]

    def test_prepare_tools_for_api_returns_original_list_when_caching_off(self):
        tools = [{"type": "function", "function": {"name": "only"}}]
        result, reserved = prepare_tools_for_api(
            tools,
            use_prompt_caching=False,
        )
        assert result is tools
        assert reserved == 0

    def test_prepare_tools_for_api_marks_request_local_copy_when_caching_on(self):
        tools = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]
        result, reserved = prepare_tools_for_api(
            tools,
            use_prompt_caching=True,
            cache_ttl="1h",
        )
        assert result is not tools
        assert reserved == 1
        assert "cache_control" not in tools[1]
        assert result[1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_tool_loop_empty_assistant_and_tool_messages_do_not_consume_breakpoints(self):
        """Tool loops should keep breakpoints on messages that can carry markers."""
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "run tool 1", "cache_control": MARKER},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function"}]},
            {"role": "tool", "content": "tool result 1"},
            {"role": "user", "content": "run tool 2", "cache_control": MARKER},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function"}]},
            {"role": "tool", "content": "tool result 2"},
        ]
        result = apply_anthropic_cache_control(msgs, native_anthropic=False)
        # Empty assistant/tool turns should not get broken markers
        assert "cache_control" not in result[2]
        assert "cache_control" not in result[3]
        assert "cache_control" not in result[5]
        assert "cache_control" not in result[6]

    def test_tool_message_marker_lands_on_content_part_on_openrouter(self):
        """Non-empty tool content should be wrapped so the marker lands on a content part."""
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "tool output"},
        ]
        result = apply_anthropic_cache_control(msgs, native_anthropic=False)
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[1]
