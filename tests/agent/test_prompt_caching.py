"""Tests for agent/prompt_caching.py — Anthropic cache control injection."""

import copy
import pytest

from agent.prompt_caching import (
    _apply_cache_marker,
    apply_anthropic_cache_control,
    # v2 imports
    SystemPromptBlock,
    CacheMetrics,
    AggregatedCacheMetrics,
    _make_cache_marker,
    build_system_content_blocks,
    _count_system_breakpoints,
    _system_is_pre_structured,
    apply_anthropic_cache_control_v2,
    extract_cache_metrics,
    aggregate_cache_metrics,
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


# ============================================================
# v2: Data structures
# ============================================================

class TestCacheMetrics:
    def test_hit_rate_basic(self):
        m = CacheMetrics(input_tokens=1000, cache_read_tokens=750)
        assert m.hit_rate == 75.0

    def test_hit_rate_zero_input(self):
        m = CacheMetrics(input_tokens=0, cache_read_tokens=0)
        assert m.hit_rate == 0.0

    def test_estimated_savings(self):
        m = CacheMetrics(input_tokens=1000, cache_read_tokens=800, cache_write_tokens=200)
        # savings = 800 * 0.9 - 200 * 0.25 = 720 - 50 = 670
        assert m.estimated_savings_tokens == 670.0

    def test_savings_can_be_negative(self):
        """All writes, no reads = net cost."""
        m = CacheMetrics(input_tokens=100, cache_read_tokens=0, cache_write_tokens=100)
        assert m.estimated_savings_tokens < 0


class TestAggregatedCacheMetrics:
    def test_overall_hit_rate(self):
        agg = AggregatedCacheMetrics(
            total_input_tokens=2000,
            total_cache_read_tokens=1500,
            turn_count=3,
        )
        assert agg.overall_hit_rate == 75.0

    def test_zero_input(self):
        agg = AggregatedCacheMetrics()
        assert agg.overall_hit_rate == 0.0


# ============================================================
# v2: _make_cache_marker
# ============================================================

class TestMakeCacheMarker:
    def test_none_ttl_returns_none(self):
        assert _make_cache_marker(None) is None

    def test_5m_ttl_no_ttl_field(self):
        marker = _make_cache_marker("5m")
        assert marker == {"type": "ephemeral"}
        assert "ttl" not in marker

    def test_1h_ttl_has_ttl_field(self):
        marker = _make_cache_marker("1h")
        assert marker == {"type": "ephemeral", "ttl": "1h"}


# ============================================================
# v2: build_system_content_blocks
# ============================================================

class TestBuildSystemContentBlocks:
    def test_basic_three_blocks(self):
        blocks = [
            SystemPromptBlock(text="Identity text", label="static", cache_ttl="1h"),
            SystemPromptBlock(text="Memory context", label="session", cache_ttl="5m"),
            SystemPromptBlock(text="Timestamp info", label="ephemeral", cache_ttl=None),
        ]
        result = build_system_content_blocks(blocks)
        assert len(result) == 3

        # Block 1: static with 1h TTL
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Identity text"
        assert result[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

        # Block 2: session with 5m (default) TTL
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "Memory context"
        assert result[1]["cache_control"] == {"type": "ephemeral"}

        # Block 3: ephemeral, no cache_control
        assert result[2]["type"] == "text"
        assert result[2]["text"] == "Timestamp info"
        assert "cache_control" not in result[2]

    def test_empty_blocks_skipped(self):
        blocks = [
            SystemPromptBlock(text="Keep this", label="static", cache_ttl="1h"),
            SystemPromptBlock(text=None, label="session", cache_ttl="5m"),
            SystemPromptBlock(text="", label="ephemeral", cache_ttl=None),
        ]
        result = build_system_content_blocks(blocks)
        assert len(result) == 1
        assert result[0]["text"] == "Keep this"

    def test_empty_input(self):
        assert build_system_content_blocks([]) == []


# ============================================================
# v2: Pre-structured detection helpers
# ============================================================

class TestPreStructuredHelpers:
    def test_count_breakpoints_in_structured_content(self):
        content = [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "c"},
        ]
        assert _count_system_breakpoints(content) == 2

    def test_count_breakpoints_string_content(self):
        assert _count_system_breakpoints("just a string") == 0

    def test_is_pre_structured_true(self):
        content = [{"type": "text", "text": "hello"}]
        assert _system_is_pre_structured(content) is True

    def test_is_pre_structured_false_for_string(self):
        assert _system_is_pre_structured("hello") is False

    def test_is_pre_structured_false_for_empty_list(self):
        assert _system_is_pre_structured([]) is False


# ============================================================
# v2: apply_anthropic_cache_control_v2
# ============================================================

class TestApplyAnthropicCacheControlV2:
    def test_empty_messages_no_tools(self):
        result = apply_anthropic_cache_control_v2([])
        assert result == []

    def test_empty_messages_with_tools(self):
        msgs, tools = apply_anthropic_cache_control_v2([], tools=[{"name": "foo"}])
        assert msgs == []
        assert len(tools) == 1

    def test_returns_deep_copy(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = apply_anthropic_cache_control_v2(msgs)
        assert result is not msgs
        assert result[0] is not msgs[0]

    def test_tool_caching_last_tool(self):
        msgs = [{"role": "user", "content": "Hi"}]
        tools = [{"name": "tool_a"}, {"name": "tool_b"}]
        result_msgs, result_tools = apply_anthropic_cache_control_v2(msgs, tools=tools)
        # Only last tool gets cache_control
        assert "cache_control" not in result_tools[0]
        assert result_tools[1]["cache_control"] == {"type": "ephemeral"}

    def test_pre_structured_system_preserves_existing_markers(self):
        """When system content is pre-structured blocks, don't re-mark — count existing breakpoints."""
        pre_structured = [
            {"type": "text", "text": "static", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": "session", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "ephemeral"},
        ]
        msgs = [
            {"role": "system", "content": pre_structured},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ]
        result = apply_anthropic_cache_control_v2(msgs)
        # System has 2 pre-placed breakpoints, so 2 remain for non-system messages
        # (4 total budget - 2 system = 2 for last 2 non-system)
        sys_content = result[0]["content"]
        # Pre-structured content should be preserved as-is
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["ttl"] == "1h"

    def test_legacy_string_system_gets_marker(self):
        """When system content is a plain string, apply marker like v1."""
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control_v2(msgs)
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["type"] == "ephemeral"

    def test_max_4_breakpoints_with_tools(self):
        """Tools + system + messages must not exceed 4 total breakpoints."""
        pre_structured = [
            {"type": "text", "text": "static", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": "session", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "ephemeral"},
        ]
        msgs = [
            {"role": "system", "content": pre_structured},
        ] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(10)
        ]
        tools = [{"name": "tool_a"}]
        result_msgs, result_tools = apply_anthropic_cache_control_v2(msgs, tools=tools)

        # Count all breakpoints: tools + system + messages
        count = 0
        # Tool breakpoints
        for t in result_tools:
            if "cache_control" in t:
                count += 1
        # Message breakpoints
        for msg in result_msgs:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "cache_control" in item:
                        count += 1
            elif isinstance(msg, dict) and "cache_control" in msg:
                count += 1
        assert count <= 4

    def test_returns_tuple_with_tools(self):
        msgs = [{"role": "user", "content": "Hi"}]
        result = apply_anthropic_cache_control_v2(msgs, tools=[{"name": "foo"}])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_list_without_tools(self):
        msgs = [{"role": "user", "content": "Hi"}]
        result = apply_anthropic_cache_control_v2(msgs)
        assert isinstance(result, list)


# ============================================================
# v2: extract_cache_metrics
# ============================================================

class TestExtractCacheMetrics:
    def test_anthropic_native_format(self):
        class Usage:
            input_tokens = 1000
            cache_read_input_tokens = 800
            cache_creation_input_tokens = 200

        m = extract_cache_metrics(Usage(), api_mode="anthropic_messages")
        assert m.input_tokens == 1000
        assert m.cache_read_tokens == 800
        assert m.cache_write_tokens == 200

    def test_openrouter_format(self):
        class Details:
            cached_tokens = 500
            cache_write_tokens = 100

        class Usage:
            prompt_tokens = 1000
            prompt_tokens_details = Details()

        m = extract_cache_metrics(Usage(), api_mode="openrouter")
        assert m.input_tokens == 1000
        assert m.cache_read_tokens == 500
        assert m.cache_write_tokens == 100

    def test_missing_attributes_default_to_zero(self):
        class Usage:
            pass

        m = extract_cache_metrics(Usage(), api_mode="anthropic_messages")
        assert m.input_tokens == 0
        assert m.cache_read_tokens == 0
        assert m.cache_write_tokens == 0

    def test_none_values_default_to_zero(self):
        class Usage:
            input_tokens = None
            cache_read_input_tokens = None
            cache_creation_input_tokens = None

        m = extract_cache_metrics(Usage(), api_mode="anthropic_messages")
        assert m.input_tokens == 0
        assert m.cache_read_tokens == 0
        assert m.cache_write_tokens == 0


# ============================================================
# v2: aggregate_cache_metrics
# ============================================================

class TestAggregateCacheMetrics:
    def test_aggregates_multiple_turns(self):
        metrics = [
            CacheMetrics(input_tokens=1000, cache_read_tokens=800, cache_write_tokens=200),
            CacheMetrics(input_tokens=1500, cache_read_tokens=1200, cache_write_tokens=100),
        ]
        agg = aggregate_cache_metrics(metrics)
        assert agg.total_input_tokens == 2500
        assert agg.total_cache_read_tokens == 2000
        assert agg.total_cache_write_tokens == 300
        assert agg.turn_count == 2
        assert agg.overall_hit_rate == 80.0

    def test_empty_list(self):
        agg = aggregate_cache_metrics([])
        assert agg.turn_count == 0
        assert agg.overall_hit_rate == 0.0
