"""Tests for agent/prompt_caching.py — Anthropic cache control injection."""

import copy
import pytest

from agent.prompt_caching import (
    _apply_cache_marker,
    apply_anthropic_cache_control,
    apply_anthropic_cache_control_v2,
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


# ---------------------------------------------------------------------------
# Tests for apply_anthropic_cache_control_v2
# ---------------------------------------------------------------------------

class TestApplyAnthropicCacheControlV2:
    """Tests for the new v2 function with strategy support."""

    def test_empty_messages(self):
        result = apply_anthropic_cache_control_v2([])
        assert result == []

    def test_returns_deep_copy(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = apply_anthropic_cache_control_v2(msgs)
        assert result is not msgs
        assert result[0] is not msgs[0]

    def test_default_strategy_is_system_and_3(self):
        """Without specifying strategy, behaviour matches legacy function."""
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ]
        result = apply_anthropic_cache_control_v2(msgs)
        # Same as legacy: system + last 3 non-system = all marked
        for msg in result:
            content = msg.get("content")
            if isinstance(content, list):
                assert "cache_control" in content[-1]

    def test_unknown_strategy_raises(self):
        msgs = [{"role": "user", "content": "Hello"}]
        with pytest.raises(ValueError, match="Unknown prompt caching strategy"):
            apply_anthropic_cache_control_v2(msgs, strategy="nonexistent")

    # --- system_and_3 tests (parity with legacy) ---

    def test_system_and_3_parity_with_legacy(self):
        """system_and_3 strategy produces identical results to apply_anthropic_cache_control."""
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        legacy = apply_anthropic_cache_control(msgs)
        v2 = apply_anthropic_cache_control_v2(msgs, strategy="system_and_3")
        assert legacy == v2

    def test_system_and_3_max_4_breakpoints(self):
        msgs = [
            {"role": "system", "content": "System"},
        ] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(10)
        ]
        result = apply_anthropic_cache_control_v2(msgs, strategy="system_and_3")
        count = _count_cache_markers(result)
        assert count <= 4

    def test_system_and_3_no_system_message(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control_v2(msgs, strategy="system_and_3")
        # 4 slots available, 2 messages — both get markers
        for msg in result:
            assert _has_cache_marker(msg)

    # --- system_and_3_double_tail tests ---

    def test_double_tail_last_message_has_two_markers_native(self):
        """On native Anthropic, the last non-system message gets BOTH a
        content-block marker AND a top-level cache_control."""
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        result = apply_anthropic_cache_control_v2(
            msgs,
            strategy="system_and_3_double_tail",
            native_anthropic=True,
        )
        last_msg = result[-1]  # "msg4"
        # Content-block marker (from _apply_cache_marker)
        assert isinstance(last_msg["content"], list)
        assert "cache_control" in last_msg["content"][0]
        # Top-level marker (from double-tail)
        assert "cache_control" in last_msg
        assert last_msg["cache_control"]["type"] == "ephemeral"

    def test_double_tail_non_last_messages_get_one_marker(self):
        """System and non-last tail messages should only have one marker each."""
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        result = apply_anthropic_cache_control_v2(
            msgs,
            strategy="system_and_3_double_tail",
            native_anthropic=True,
        )
        # System (index 0): content-block marker only, NO top-level
        sys_msg = result[0]
        assert isinstance(sys_msg["content"], list)
        assert "cache_control" in sys_msg["content"][0]
        assert "cache_control" not in {k: v for k, v in sys_msg.items() if k != "content"}

        # msg2 (index 2): should have content-block marker but NOT top-level
        msg2 = result[2]
        assert isinstance(msg2["content"], list)
        assert "cache_control" in msg2["content"][0]
        assert "cache_control" not in {k: v for k, v in msg2.items() if k != "content"}

        # msg1 (index 1): should NOT have any marker (not in last 3)
        msg1 = result[1]
        assert isinstance(msg1["content"], str)  # still a string, no marker

    def test_double_tail_single_non_system(self):
        """With only one non-system message, it gets both markers."""
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Only message"},
        ]
        result = apply_anthropic_cache_control_v2(
            msgs,
            strategy="system_and_3_double_tail",
            native_anthropic=True,
        )
        user_msg = result[1]
        assert "cache_control" in user_msg  # top-level marker
        assert isinstance(user_msg["content"], list)
        assert "cache_control" in user_msg["content"][0]  # content-block marker

    def test_double_tail_no_system_message(self):
        """Without a system message, double-tail still works on the last message."""
        msgs = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ]
        result = apply_anthropic_cache_control_v2(
            msgs,
            strategy="system_and_3_double_tail",
            native_anthropic=True,
        )
        last_msg = result[-1]
        assert "cache_control" in last_msg  # top-level
        assert isinstance(last_msg["content"], list)
        assert "cache_control" in last_msg["content"][0]  # content-block

    def test_double_tail_1h_ttl(self):
        """Double-tail markers respect the 1h TTL."""
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
        ]
        result = apply_anthropic_cache_control_v2(
            msgs,
            cache_ttl="1h",
            strategy="system_and_3_double_tail",
            native_anthropic=True,
        )
        user_msg = result[1]
        assert user_msg["cache_control"]["ttl"] == "1h"
        assert user_msg["content"][0]["cache_control"]["ttl"] == "1h"

    def test_double_tail_tool_message_native(self):
        """Double-tail on a tool message in native Anthropic format."""
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Run tool"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
        ]
        result = apply_anthropic_cache_control_v2(
            msgs,
            strategy="system_and_3_double_tail",
            native_anthropic=True,
        )
        tool_msg = result[-1]
        # Tool messages on native get top-level from _apply_cache_marker
        # AND double-tail also sets top-level. They should be the same value.
        assert tool_msg["cache_control"]["type"] == "ephemeral"

    def test_double_tail_max_breakpoints_still_respected(self):
        """Even with double-tail, the base limit of 4 breakpoints applies
        before the extra top-level marker is added."""
        msgs = [
            {"role": "system", "content": "System"},
        ] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(10)
        ]
        result = apply_anthropic_cache_control_v2(
            msgs,
            strategy="system_and_3_double_tail",
            native_anthropic=True,
        )
        # Base breakpoints (from _apply_cache_marker) should still be <= 4
        base_count = _count_content_block_markers(result)
        assert base_count <= 4
        # The last non-system message should have an EXTRA top-level marker
        last_non_sys_idx = None
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") != "system":
                last_non_sys_idx = i
                break
        assert last_non_sys_idx is not None
        assert "cache_control" in result[last_non_sys_idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_cache_marker(msg: dict) -> bool:
    """Check if a message has any cache_control marker."""
    if "cache_control" in msg:
        return True
    content = msg.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and "cache_control" in item:
                return True
    return False


def _count_cache_markers(messages: list) -> int:
    """Count total cache_control markers across all messages."""
    count = 0
    for msg in messages:
        if "cache_control" in msg:
            count += 1
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "cache_control" in item:
                    count += 1
    return count


def _count_content_block_markers(messages: list) -> int:
    """Count only content-block-level cache_control markers (not top-level)."""
    count = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "cache_control" in item:
                    count += 1
    return count
