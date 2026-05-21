"""Tests for pi_dual prompt caching strategy."""

import copy
import pytest

from agent.prompt_caching import apply_anthropic_cache_control


MARKER_5M = {"type": "ephemeral"}
MARKER_1H = {"type": "ephemeral", "ttl": "1h"}


def _has_cache_control(msg_or_block):
    """Check if a message or content block has cache_control."""
    if isinstance(msg_or_block, dict):
        if "cache_control" in msg_or_block:
            return True
        content = msg_or_block.get("content")
        if isinstance(content, list):
            return any("cache_control" in b for b in content if isinstance(b, dict))
    return False


def _get_cache_marker(msg):
    """Extract the cache_control marker from a message."""
    if "cache_control" in msg:
        return msg["cache_control"]
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and "cache_control" in block:
                return block["cache_control"]
    return None


class TestPiDualMarksCorrectMessages:
    def test_marks_system_assistant_tool_use_and_user(self):
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thinking"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "x"},
                    },
                ],
            },
            {"role": "tool", "tool_use_id": "toolu_1", "content": "RESULT"},
            {"role": "user", "content": "next"},
        ]
        result = apply_anthropic_cache_control(
            messages, strategy="pi_dual", native_anthropic=True
        )

        # System marked
        assert _has_cache_control(result[0])
        # Assistant tool_use block marked (not the text block)
        ast_content = result[1]["content"]
        assert "cache_control" not in ast_content[0]  # text block unmarked
        assert ast_content[1].get("cache_control") == MARKER_5M  # tool_use marked
        # Tool message NOT marked
        assert not _has_cache_control(result[2])
        # User marked
        assert _has_cache_control(result[3])

    def test_no_tool_use_still_marks_system_and_user(self):
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "thanks"},
        ]
        result = apply_anthropic_cache_control(
            messages, strategy="pi_dual", native_anthropic=True
        )

        assert _has_cache_control(result[0])  # system
        assert not _has_cache_control(result[1])  # first user
        assert not _has_cache_control(result[2])  # assistant (no tool_use)
        assert _has_cache_control(result[3])  # last user


class TestPiDualNeverMarksToolMessages:
    def test_tool_messages_never_get_cache_control(self):
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "exec", "input": {}},
                    {"type": "tool_use", "id": "t2", "name": "read", "input": {}},
                ],
            },
            {"role": "tool", "tool_use_id": "t1", "content": "output1"},
            {"role": "tool", "tool_use_id": "t2", "content": "output2"},
            {"role": "user", "content": "continue"},
        ]
        result = apply_anthropic_cache_control(
            messages, strategy="pi_dual", native_anthropic=True
        )

        assert not _has_cache_control(result[2])
        assert not _has_cache_control(result[3])


class TestPiDualTTL:
    def test_1h_ttl_marker(self):
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "read", "input": {}},
                ],
            },
            {"role": "tool", "tool_use_id": "t1", "content": "data"},
            {"role": "user", "content": "next"},
        ]
        result = apply_anthropic_cache_control(
            messages, cache_ttl="1h", native_anthropic=True, strategy="pi_dual"
        )

        sys_marker = _get_cache_marker(result[0])
        assert sys_marker == MARKER_1H
        tool_use_marker = result[1]["content"][0].get("cache_control")
        assert tool_use_marker == MARKER_1H
        user_marker = _get_cache_marker(result[3])
        assert user_marker == MARKER_1H


class TestPiDualFallback:
    def test_non_native_falls_back_to_system_and_3(self):
        """When native_anthropic=False, pi_dual falls back to system_and_3."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        pi_result = apply_anthropic_cache_control(
            messages, native_anthropic=False, strategy="pi_dual"
        )
        sys3_result = apply_anthropic_cache_control(
            messages, native_anthropic=False, strategy="system_and_3"
        )
        assert pi_result == sys3_result

    def test_invalid_strategy_defaults_to_system_and_3(self):
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "hello"},
        ]
        bad_result = apply_anthropic_cache_control(messages, strategy="nonexistent")
        default_result = apply_anthropic_cache_control(
            messages, strategy="system_and_3"
        )
        assert bad_result == default_result


class TestPiDualEdgeCases:
    def test_empty_messages(self):
        result = apply_anthropic_cache_control(
            [], strategy="pi_dual", native_anthropic=True
        )
        assert result == []

    def test_deep_copy_preserves_original(self):
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "hello"},
        ]
        original = copy.deepcopy(messages)
        apply_anthropic_cache_control(
            messages, strategy="pi_dual", native_anthropic=True
        )
        assert messages == original

    def test_multiple_assistant_tool_use_marks_last_one(self):
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "first", "input": {}},
                ],
            },
            {"role": "tool", "tool_use_id": "t1", "content": "r1"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t2", "name": "second", "input": {}},
                ],
            },
            {"role": "tool", "tool_use_id": "t2", "content": "r2"},
            {"role": "user", "content": "done"},
        ]
        result = apply_anthropic_cache_control(
            messages, strategy="pi_dual", native_anthropic=True
        )

        # First assistant tool_use should NOT be marked
        assert "cache_control" not in result[1]["content"][0]
        # Second (last) assistant tool_use SHOULD be marked
        assert result[3]["content"][0].get("cache_control") == MARKER_5M

    def test_assistant_with_multiple_tool_use_marks_last_block(self):
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "first", "input": {}},
                    {"type": "text", "text": "middle"},
                    {"type": "tool_use", "id": "t2", "name": "second", "input": {}},
                ],
            },
            {"role": "tool", "tool_use_id": "t1", "content": "r1"},
            {"role": "tool", "tool_use_id": "t2", "content": "r2"},
            {"role": "user", "content": "next"},
        ]
        result = apply_anthropic_cache_control(
            messages, strategy="pi_dual", native_anthropic=True
        )

        ast_content = result[1]["content"]
        assert "cache_control" not in ast_content[0]  # first tool_use
        assert "cache_control" not in ast_content[1]  # text
        assert ast_content[2].get("cache_control") == MARKER_5M  # last tool_use

    def test_max_3_breakpoints(self):
        """pi_dual should use at most 3 breakpoints: system + tool_use + user."""
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "read", "input": {}},
                ],
            },
            {"role": "tool", "tool_use_id": "t1", "content": "data"},
            {"role": "user", "content": "next"},
        ]
        result = apply_anthropic_cache_control(
            messages, strategy="pi_dual", native_anthropic=True
        )

        count = 0
        for msg in result:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "cache_control" in item:
                        count += 1
            if "cache_control" in msg:
                count += 1
        assert count == 3


class TestSystemAnd3Unchanged:
    """Verify system_and_3 behavior is identical to the original implementation."""

    def test_system_and_3_still_works(self):
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        result = apply_anthropic_cache_control(messages, strategy="system_and_3")
        # System + last 3 non-system = 4 breakpoints
        count = 0
        for msg in result:
            if _has_cache_control(msg):
                count += 1
        assert count == 4

    def test_default_strategy_is_system_and_3(self):
        messages = [{"role": "system", "content": "System"}]
        explicit = apply_anthropic_cache_control(messages, strategy="system_and_3")
        default = apply_anthropic_cache_control(messages)
        assert explicit == default
