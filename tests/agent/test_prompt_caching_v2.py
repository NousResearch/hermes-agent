"""Tests for prompt caching v2 — multi-block system prompt, tool caching, TTL tiers, metrics.

TDD: All tests written BEFORE implementation. They should all FAIL initially.
"""

import copy
import pytest
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Import the module under test — new functions will be added here
from agent.prompt_caching import (
    _apply_cache_marker,
    apply_anthropic_cache_control,
)

# These imports will fail until we implement them — that's the RED phase
from agent.prompt_caching import (
    SystemPromptBlock,
    build_system_content_blocks,
    apply_anthropic_cache_control_v2,
    CacheMetrics,
    extract_cache_metrics,
    aggregate_cache_metrics,
)


MARKER_5M = {"type": "ephemeral"}
MARKER_1H = {"type": "ephemeral", "ttl": "1h"}


# ============================================================
# Phase 1: SystemPromptBlock data structure
# ============================================================

class TestSystemPromptBlock:
    def test_block_has_required_fields(self):
        block = SystemPromptBlock(text="Hello world", label="identity")
        assert block.text == "Hello world"
        assert block.label == "identity"
        assert block.cache_ttl is None  # default: no caching

    def test_block_with_cache_ttl(self):
        block = SystemPromptBlock(text="Skills index", label="static", cache_ttl="1h")
        assert block.cache_ttl == "1h"

    def test_block_with_5m_ttl(self):
        block = SystemPromptBlock(text="Memory", label="session", cache_ttl="5m")
        assert block.cache_ttl == "5m"


# ============================================================
# Phase 1+3: build_system_content_blocks — structured system prompt
# ============================================================

class TestBuildSystemContentBlocks:
    """Tests for converting SystemPromptBlock list to Anthropic content blocks."""

    def test_single_block_no_cache(self):
        """A block with no cache_ttl becomes a plain text block."""
        blocks = [SystemPromptBlock(text="timestamp info", label="ephemeral")]
        result = build_system_content_blocks(blocks)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "timestamp info"
        assert "cache_control" not in result[0]

    def test_block_with_1h_ttl_gets_cache_control(self):
        """Block with 1h TTL gets cache_control with ttl."""
        blocks = [SystemPromptBlock(
            text="Identity + skills + guidance",
            label="static",
            cache_ttl="1h",
        )]
        result = build_system_content_blocks(blocks)
        assert len(result) == 1
        assert result[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_block_with_5m_ttl_gets_cache_control(self):
        """Block with 5m TTL gets cache_control without ttl field (default)."""
        blocks = [SystemPromptBlock(
            text="Memory + context files",
            label="session",
            cache_ttl="5m",
        )]
        result = build_system_content_blocks(blocks)
        assert len(result) == 1
        assert result[0]["cache_control"] == {"type": "ephemeral"}
        assert "ttl" not in result[0]["cache_control"]

    def test_three_block_layout(self):
        """The planned 3-block layout: static(1h) + session(5m) + ephemeral(none)."""
        blocks = [
            SystemPromptBlock(text="Identity + skills", label="static", cache_ttl="1h"),
            SystemPromptBlock(text="Memory + context", label="session", cache_ttl="5m"),
            SystemPromptBlock(text="Timestamp + platform", label="ephemeral"),
        ]
        result = build_system_content_blocks(blocks)
        assert len(result) == 3

        # Block 1: 1h TTL
        assert result[0]["text"] == "Identity + skills"
        assert result[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

        # Block 2: 5m TTL (default ephemeral)
        assert result[1]["text"] == "Memory + context"
        assert result[1]["cache_control"] == {"type": "ephemeral"}

        # Block 3: no caching
        assert result[2]["text"] == "Timestamp + platform"
        assert "cache_control" not in result[2]

    def test_empty_blocks_skipped(self):
        """Empty text blocks should be filtered out."""
        blocks = [
            SystemPromptBlock(text="Identity", label="static", cache_ttl="1h"),
            SystemPromptBlock(text="", label="session", cache_ttl="5m"),
            SystemPromptBlock(text="Timestamp", label="ephemeral"),
        ]
        result = build_system_content_blocks(blocks)
        assert len(result) == 2
        assert result[0]["text"] == "Identity"
        assert result[1]["text"] == "Timestamp"

    def test_none_text_blocks_skipped(self):
        """None text blocks should be filtered out."""
        blocks = [
            SystemPromptBlock(text=None, label="empty", cache_ttl="1h"),
            SystemPromptBlock(text="Real content", label="static", cache_ttl="1h"),
        ]
        result = build_system_content_blocks(blocks)
        assert len(result) == 1
        assert result[0]["text"] == "Real content"

    def test_empty_block_list(self):
        result = build_system_content_blocks([])
        assert result == []


# ============================================================
# Phase 2: apply_anthropic_cache_control_v2 — tool + multi-block system
# ============================================================

class TestApplyAnthropicCacheControlV2:
    """Tests for the v2 cache control that handles tools and pre-structured system."""

    def test_backward_compatible_with_string_system(self):
        """When system has string content, v2 behaves like v1."""
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control_v2(msgs)
        # System should get a cache marker
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["type"] == "ephemeral"

    def test_preserves_pre_structured_system_blocks(self):
        """When system already has list content with cache_control, don't re-wrap."""
        pre_structured = [
            {"type": "text", "text": "Block 1", "cache_control": MARKER_1H},
            {"type": "text", "text": "Block 2", "cache_control": MARKER_5M},
            {"type": "text", "text": "Block 3"},
        ]
        msgs = [
            {"role": "system", "content": pre_structured},
            {"role": "user", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control_v2(msgs)
        sys_content = result[0]["content"]
        # Should preserve the pre-placed cache_control markers exactly
        assert sys_content[0]["cache_control"] == MARKER_1H
        assert sys_content[1]["cache_control"] == MARKER_5M
        assert "cache_control" not in sys_content[2]

    def test_tool_definitions_get_cache_marker(self):
        """Last tool definition should get a cache_control breakpoint."""
        tools = [
            {"type": "function", "function": {"name": "tool_a", "parameters": {}}},
            {"type": "function", "function": {"name": "tool_b", "parameters": {}}},
        ]
        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control_v2(msgs, tools=tools)
        # v2 returns (messages, tools) tuple when tools are provided
        assert isinstance(result, tuple)
        cached_msgs, cached_tools = result
        assert cached_tools[-1].get("cache_control") == {"type": "ephemeral"}
        assert "cache_control" not in cached_tools[0]

    def test_tool_caching_returns_tuple(self):
        """When tools are provided, v2 returns (messages, tools) tuple."""
        tools = [
            {"type": "function", "function": {"name": "tool_a", "parameters": {}}},
        ]
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control_v2(msgs, tools=tools)
        assert isinstance(result, tuple)
        assert len(result) == 2
        cached_msgs, cached_tools = result
        assert isinstance(cached_msgs, list)
        assert isinstance(cached_tools, list)

    def test_no_tools_returns_messages_only(self):
        """Without tools, v2 returns just messages (backward compat)."""
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hi"},
        ]
        result = apply_anthropic_cache_control_v2(msgs)
        assert isinstance(result, list)

    def test_breakpoint_budget_with_tools_and_prestructured(self):
        """With pre-structured system (2 bps) + tools (1 bp) = 3 used, 1 for messages."""
        pre_structured = [
            {"type": "text", "text": "Block 1", "cache_control": MARKER_1H},
            {"type": "text", "text": "Block 2", "cache_control": MARKER_5M},
            {"type": "text", "text": "Block 3"},
        ]
        tools = [
            {"type": "function", "function": {"name": "tool_a", "parameters": {}}},
        ]
        msgs = [
            {"role": "system", "content": pre_structured},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
            {"role": "user", "content": "msg5"},
        ]
        cached_msgs, cached_tools = apply_anthropic_cache_control_v2(msgs, tools=tools)

        # Count message breakpoints (excluding system which has pre-placed ones)
        msg_breakpoints = 0
        for msg in cached_msgs[1:]:  # skip system
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "cache_control" in item:
                        msg_breakpoints += 1
            elif isinstance(content, str):
                pass  # no marker
        # Budget: 4 total - 2 system - 1 tools = 1 message breakpoint
        assert msg_breakpoints == 1

    def test_deep_copy_preserved(self):
        """Input messages and tools should not be mutated."""
        tools = [
            {"type": "function", "function": {"name": "tool_a", "parameters": {}}},
        ]
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hi"},
        ]
        original_msgs = copy.deepcopy(msgs)
        original_tools = copy.deepcopy(tools)
        apply_anthropic_cache_control_v2(msgs, tools=tools)
        assert msgs == original_msgs
        assert tools == original_tools

    def test_max_4_breakpoints_total(self):
        """Total breakpoints across tools + system + messages must never exceed 4."""
        pre_structured = [
            {"type": "text", "text": "Block 1", "cache_control": MARKER_1H},
            {"type": "text", "text": "Block 2", "cache_control": MARKER_5M},
        ]
        tools = [
            {"type": "function", "function": {"name": f"tool_{i}", "parameters": {}}}
            for i in range(5)
        ]
        msgs = [
            {"role": "system", "content": pre_structured},
        ] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(10)
        ]
        cached_msgs, cached_tools = apply_anthropic_cache_control_v2(msgs, tools=tools)

        total_bps = 0
        # Count tool breakpoints
        for t in cached_tools:
            if "cache_control" in t:
                total_bps += 1
        # Count system block breakpoints
        sys_content = cached_msgs[0]["content"]
        if isinstance(sys_content, list):
            for block in sys_content:
                if isinstance(block, dict) and "cache_control" in block:
                    total_bps += 1
        # Count message breakpoints
        for msg in cached_msgs[1:]:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "cache_control" in item:
                        total_bps += 1
            elif "cache_control" in msg:
                total_bps += 1
        assert total_bps <= 4

    def test_empty_messages(self):
        result = apply_anthropic_cache_control_v2([])
        assert result == []

    def test_native_anthropic_tool_messages(self):
        """Native Anthropic mode should mark tool messages correctly."""
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Let me check"},
            {"role": "tool", "content": "result data"},
        ]
        result = apply_anthropic_cache_control_v2(msgs, native_anthropic=True)
        if isinstance(result, tuple):
            result = result[0]
        # tool message should have top-level cache_control in native mode
        tool_msg = result[3]
        assert "cache_control" in tool_msg


# ============================================================
# Phase 4: Cache metrics extraction and aggregation
# ============================================================

class TestCacheMetrics:
    def test_extract_metrics_from_anthropic_response(self):
        """Extract cache metrics from a mock Anthropic response."""
        class MockUsage:
            input_tokens = 5000
            output_tokens = 500
            cache_read_input_tokens = 3500
            cache_creation_input_tokens = 1000

        metrics = extract_cache_metrics(MockUsage(), api_mode="anthropic_messages")
        assert metrics.input_tokens == 5000
        assert metrics.cache_read_tokens == 3500
        assert metrics.cache_write_tokens == 1000
        assert metrics.hit_rate == pytest.approx(70.0, abs=0.1)  # 3500/5000

    def test_extract_metrics_from_openrouter_response(self):
        """Extract cache metrics from OpenRouter prompt_tokens_details."""
        class MockDetails:
            cached_tokens = 2000
            cache_write_tokens = 500

        class MockUsage:
            prompt_tokens = 3000
            completion_tokens = 400
            prompt_tokens_details = MockDetails()

        metrics = extract_cache_metrics(MockUsage(), api_mode="openrouter")
        assert metrics.cache_read_tokens == 2000
        assert metrics.cache_write_tokens == 500

    def test_extract_metrics_no_cache_data(self):
        """When no cache data is present, metrics should be zero."""
        class MockUsage:
            prompt_tokens = 3000
            completion_tokens = 400

        metrics = extract_cache_metrics(MockUsage(), api_mode="anthropic_messages")
        assert metrics.cache_read_tokens == 0
        assert metrics.cache_write_tokens == 0
        assert metrics.hit_rate == 0.0

    def test_aggregate_metrics_across_turns(self):
        """Aggregate metrics should sum totals and compute overall hit rate."""
        m1 = CacheMetrics(input_tokens=5000, cache_read_tokens=0, cache_write_tokens=4000)
        m2 = CacheMetrics(input_tokens=5000, cache_read_tokens=4000, cache_write_tokens=0)
        m3 = CacheMetrics(input_tokens=5000, cache_read_tokens=4000, cache_write_tokens=0)

        agg = aggregate_cache_metrics([m1, m2, m3])
        assert agg.total_input_tokens == 15000
        assert agg.total_cache_read_tokens == 8000
        assert agg.total_cache_write_tokens == 4000
        assert agg.overall_hit_rate == pytest.approx(53.3, abs=0.1)  # 8000/15000

    def test_aggregate_empty_list(self):
        agg = aggregate_cache_metrics([])
        assert agg.total_input_tokens == 0
        assert agg.overall_hit_rate == 0.0

    def test_cache_metrics_estimated_savings(self):
        """Savings = cache_read_tokens * 0.9 (90% discount on reads)."""
        m = CacheMetrics(input_tokens=5000, cache_read_tokens=4000, cache_write_tokens=500)
        # Savings: 4000 tokens at 90% discount = 3600 tokens saved
        # Extra cost: 500 tokens at 25% markup = 125 tokens
        # Net: 3600 - 125 = 3475 tokens equivalent saved
        assert m.estimated_savings_tokens == pytest.approx(3475, abs=1)


# ============================================================
# Phase 5: Ephemeral prompt isolation
# ============================================================

class TestEphemeralIsolation:
    """Verify that timestamp and platform hints end up in the uncached block."""

    def test_timestamp_not_in_cached_blocks(self):
        """Build blocks and verify timestamp is in the ephemeral (uncached) block."""
        blocks = [
            SystemPromptBlock(text="Identity + skills", label="static", cache_ttl="1h"),
            SystemPromptBlock(text="Memory + context", label="session", cache_ttl="5m"),
            SystemPromptBlock(text="Started: Monday, April 05, 2026", label="ephemeral"),
        ]
        result = build_system_content_blocks(blocks)
        # The block containing timestamp should NOT have cache_control
        timestamp_block = [b for b in result if "April" in b["text"]]
        assert len(timestamp_block) == 1
        assert "cache_control" not in timestamp_block[0]

    def test_changing_ephemeral_doesnt_affect_cached_blocks(self):
        """If only Block 3 changes between turns, Blocks 1 and 2 remain identical."""
        blocks_turn1 = [
            SystemPromptBlock(text="Static content", label="static", cache_ttl="1h"),
            SystemPromptBlock(text="Session content", label="session", cache_ttl="5m"),
            SystemPromptBlock(text="Turn 1 ephemeral", label="ephemeral"),
        ]
        blocks_turn2 = [
            SystemPromptBlock(text="Static content", label="static", cache_ttl="1h"),
            SystemPromptBlock(text="Session content", label="session", cache_ttl="5m"),
            SystemPromptBlock(text="Turn 2 ephemeral DIFFERENT", label="ephemeral"),
        ]
        result1 = build_system_content_blocks(blocks_turn1)
        result2 = build_system_content_blocks(blocks_turn2)
        # Cached blocks (0 and 1) should be identical
        assert result1[0] == result2[0]
        assert result1[1] == result2[1]
        # Ephemeral block should differ
        assert result1[2] != result2[2]


# ============================================================
# Backward compatibility: existing v1 tests still pass
# ============================================================

class TestV1BackwardCompatibility:
    """Ensure the original apply_anthropic_cache_control still works unchanged."""

    def test_v1_system_and_3_still_works(self):
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # System should have marker
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert sys_content[0]["cache_control"]["type"] == "ephemeral"

    def test_v1_deep_copy(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = apply_anthropic_cache_control(msgs)
        assert result is not msgs

    def test_v1_1h_ttl(self):
        msgs = [{"role": "system", "content": "System"}]
        result = apply_anthropic_cache_control(msgs, cache_ttl="1h")
        sys_content = result[0]["content"]
        assert sys_content[0]["cache_control"]["ttl"] == "1h"
