"""Tests for agent/prompt_caching.py — Anthropic cache control injection."""

import copy
import pytest

from agent.prompt_caching import (
    _apply_cache_marker,
    apply_anthropic_cache_control,
    apply_anthropic_cache_control_long_lived,
    mark_tools_for_long_lived_cache,
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


class TestMarkToolsForLongLivedCache:
    def test_returns_unchanged_for_empty_tools(self):
        assert mark_tools_for_long_lived_cache(None) is None
        assert mark_tools_for_long_lived_cache([]) == []

    def test_marks_only_last_tool(self):
        tools = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
            {"type": "function", "function": {"name": "c"}},
        ]
        out = mark_tools_for_long_lived_cache(tools)
        assert "cache_control" not in out[0]
        assert "cache_control" not in out[1]
        assert out[2]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_does_not_mutate_input(self):
        tools = [{"type": "function", "function": {"name": "a"}}]
        mark_tools_for_long_lived_cache(tools)
        assert "cache_control" not in tools[0]

    def test_5m_ttl_drops_ttl_field(self):
        tools = [{"type": "function", "function": {"name": "a"}}]
        out = mark_tools_for_long_lived_cache(tools, long_lived_ttl="5m")
        assert out[0]["cache_control"] == {"type": "ephemeral"}


class TestApplyAnthropicCacheControlLongLived:
    def test_empty_messages(self):
        assert apply_anthropic_cache_control_long_lived([]) == []

    def test_marks_first_block_of_split_system(self):
        msgs = [
            {"role": "system", "content": [
                {"type": "text", "text": "STABLE"},
                {"type": "text", "text": "CONTEXT"},
                {"type": "text", "text": "VOLATILE"},
            ]},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
        ]
        out = apply_anthropic_cache_control_long_lived(msgs)
        sys_blocks = out[0]["content"]
        assert sys_blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert "cache_control" not in sys_blocks[1]
        assert "cache_control" not in sys_blocks[2]

    def test_rolling_marker_on_last_2_messages(self):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "S"}]},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        out = apply_anthropic_cache_control_long_lived(msgs)

        def has_marker(m):
            c = m.get("content")
            if isinstance(c, list) and c and isinstance(c[-1], dict):
                return "cache_control" in c[-1]
            return "cache_control" in m

        # u1 and a1 (older messages) should NOT be marked
        assert not has_marker(out[1])
        assert not has_marker(out[2])
        # u2 and a2 (last 2) SHOULD be marked
        assert has_marker(out[3])
        assert has_marker(out[4])

    def test_rolling_marker_uses_5m_ttl(self):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "S"}]},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        out = apply_anthropic_cache_control_long_lived(
            msgs, long_lived_ttl="1h", rolling_ttl="5m",
        )
        # Last user message: cache_control on the wrapped text part should be 5m
        last = out[-1]
        c = last["content"]
        assert isinstance(c, list)
        assert c[-1]["cache_control"] == {"type": "ephemeral"}  # 5m has no ttl key

    def test_string_system_falls_back_to_envelope_marker(self):
        """When the caller didn't split the system message, we still place a marker."""
        msgs = [
            {"role": "system", "content": "Single string system"},
            {"role": "user", "content": "u1"},
        ]
        out = apply_anthropic_cache_control_long_lived(msgs)
        sys_content = out[0]["content"]
        # Wrapped into a list and the (now sole) block gets the 1h marker
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_does_not_mutate_input(self):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "S"}]},
            {"role": "user", "content": "u1"},
        ]
        before = copy.deepcopy(msgs)
        apply_anthropic_cache_control_long_lived(msgs)
        assert msgs == before

    def test_max_4_breakpoints_with_split_system(self):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "S"}, {"type": "text", "text": "V"}]},
        ] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(10)
        ]
        out = apply_anthropic_cache_control_long_lived(msgs)
        count = 0
        for m in out:
            c = m.get("content")
            if isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and "cache_control" in item:
                        count += 1
            elif "cache_control" in m:
                count += 1
        # 1 system block + last 2 messages = 3 breakpoints from this function.
        # tools[-1] is marked separately (not via this function), so a 4th
        # breakpoint can be added at API-call time.
        assert count == 3
