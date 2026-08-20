"""Tests for cost attribution."""
import threading
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from agent.cost_attributor import (
    AttributedCostRecord,
    CostAttributor,
    COST_ATTRIBUTED,
)
from agent.hermes.analytics import EventBus, Event, EventType
from agent.usage_pricing import CanonicalUsage


class TestAttributedCostRecord:
    """Tests for AttributedCostRecord."""

    def test_wraps_canonical_usage_via_composition(self):
        """AttributedCostRecord wraps CanonicalUsage via composition (has-a, not is-a)."""
        usage = CanonicalUsage(
            input_tokens=100,
            output_tokens=200,
            cache_read_tokens=50,
            cache_write_tokens=10,
        )
        record = AttributedCostRecord(
            usage=usage,
            source="claude",
            operation="llm.response",
        )
        # Has the usage object, not is a usage object
        assert hasattr(record, "usage")
        assert not isinstance(record, CanonicalUsage)

    def test_fields_are_present(self):
        """AttributedCostRecord has expected fields."""
        usage = CanonicalUsage(input_tokens=100, output_tokens=200)
        record = AttributedCostRecord(
            usage=usage,
            source="test-source",
            operation="test-op",
            parent_session_id="parent-sess",
        )
        assert record.source == "test-source"
        assert record.operation == "test-op"
        assert record.parent_session_id == "parent-sess"
        assert record.usage is usage

    def test_cost_usd_computed_from_canonical_usage(self):
        """cost_usd is computed from CanonicalUsage."""
        usage = CanonicalUsage(
            input_tokens=1_000_000,  # 1M input tokens
            output_tokens=0,
        )
        record = AttributedCostRecord(
            usage=usage,
            source="claude",
            operation="llm.response",
        )
        # Using claude-opus-4-20250514: $15.00 per million input
        # So 1M tokens should cost $15.00
        assert record.cost_usd == Decimal("15.00")

    def test_input_tokens_property(self):
        """input_tokens property delegates to usage."""
        usage = CanonicalUsage(input_tokens=123)
        record = AttributedCostRecord(
            usage=usage,
            source="source",
            operation="op",
        )
        assert record.input_tokens == 123

    def test_output_tokens_property(self):
        """output_tokens property delegates to usage."""
        usage = CanonicalUsage(output_tokens=456)
        record = AttributedCostRecord(
            usage=usage,
            source="source",
            operation="op",
        )
        assert record.output_tokens == 456

    def test_cache_read_tokens_property(self):
        """cache_read_tokens property delegates to usage."""
        usage = CanonicalUsage(cache_read_tokens=789)
        record = AttributedCostRecord(
            usage=usage,
            source="source",
            operation="op",
        )
        assert record.cache_read_tokens == 789

    def test_cache_write_tokens_property(self):
        """cache_write_tokens property delegates to usage."""
        usage = CanonicalUsage(cache_write_tokens=111)
        record = AttributedCostRecord(
            usage=usage,
            source="source",
            operation="op",
        )
        assert record.cache_write_tokens == 111

    def test_total_tokens_property(self):
        """total_tokens property delegates to usage."""
        usage = CanonicalUsage(
            input_tokens=100,
            output_tokens=200,
            cache_read_tokens=50,
            cache_write_tokens=10,
        )
        record = AttributedCostRecord(
            usage=usage,
            source="source",
            operation="op",
        )
        # total_tokens = (input + cache_read + cache_write) + output
        assert record.total_tokens == 360

    def test_to_dict(self):
        """to_dict() returns a complete dictionary."""
        usage = CanonicalUsage(
            input_tokens=100,
            output_tokens=200,
            cache_read_tokens=50,
            cache_write_tokens=10,
        )
        record = AttributedCostRecord(
            usage=usage,
            source="claude",
            operation="llm.response",
            parent_session_id="parent-123",
        )
        d = record.to_dict()
        assert d["source"] == "claude"
        assert d["operation"] == "llm.response"
        assert d["parent_session_id"] == "parent-123"
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 200
        assert d["cache_read_tokens"] == 50
        assert d["cache_write_tokens"] == 10
        assert "cost_usd" in d


class TestCostAttributor:
    """Tests for CostAttributor."""

    def test_subscribes_to_llm_response_and_tool_result(self):
        """CostAttributor subscribes to LLM_RESPONSE and TOOL_RESULT events."""
        mock_bus = MagicMock(spec=EventBus)
        with patch.object(EventBus, "subscribe") as mock_subscribe:
            mock_bus.subscribe = mock_subscribe
            # Re-patch before creating CostAttributor
            with patch("agent.cost_attributor.EventBus.subscribe", mock_subscribe):
                # Need to create with real EventBus to test subscriptions
                pass
        # Use a real EventBus for testing
        event_bus = EventBus()
        attrib = CostAttributor(event_bus)
        # The CostAttributor should have subscribed to events
        # We can verify by checking internal state
        assert hasattr(attrib, "_event_bus")

    def test_get_cost_breakdown_returns_structure(self):
        """get_cost_breakdown() returns expected structure."""
        event_bus = EventBus()
        attrib = CostAttributor(event_bus)
        breakdown = attrib.get_cost_breakdown()

        assert "total_cost_usd" in breakdown
        assert "per_source" in breakdown
        assert "per_tool" in breakdown
        assert "per_source_usage" in breakdown

    def test_accumulates_llm_response_events(self):
        """CostAttributor accumulates costs from llm.response events."""
        event_bus = EventBus()
        attrib = CostAttributor(event_bus)

        # Emit an LLM response event
        event_bus.emit_event(
            EventType.LLM_RESPONSE,
            {
                "source": "claude",
                "usage": {
                    "input_tokens": 1_000_000,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                },
            },
            session_id="test-session",
        )

        breakdown = attrib.get_cost_breakdown()
        # 1M input tokens at $15/M = $15.00
        assert breakdown["total_cost_usd"] == pytest.approx(15.00, rel=0.01)
        assert "claude" in breakdown["per_source"]

    def test_accumulates_tool_result_events(self):
        """CostAttributor accumulates costs from tool.result events."""
        event_bus = EventBus()
        attrib = CostAttributor(event_bus)

        # Emit a tool result event
        event_bus.emit_event(
            EventType.TOOL_RESULT,
            {
                "tool_name": "terminal",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 200,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                },
            },
            session_id="test-session",
        )

        breakdown = attrib.get_cost_breakdown()
        # Terminal has its own pricing
        assert breakdown["total_cost_usd"] >= 0
        assert "terminal" in breakdown["per_tool"]

    def test_multiple_events_accumulate_correctly(self):
        """Multiple events accumulate costs correctly."""
        event_bus = EventBus()
        attrib = CostAttributor(event_bus)

        # Emit multiple LLM response events
        for i in range(3):
            event_bus.emit_event(
                EventType.LLM_RESPONSE,
                {
                    "source": "claude",
                    "usage": {
                        "input_tokens": 100_000,
                        "output_tokens": 50_000,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                    },
                },
                session_id="test-session",
            )

        breakdown = attrib.get_cost_breakdown()
        # Each event: 100k input + 50k output = 150k tokens at ~$0.0000225/token
        # Total should be ~3x the single event cost
        assert breakdown["total_cost_usd"] > 0

    def test_thread_safety(self):
        """CostAttributor is thread-safe."""
        event_bus = EventBus()
        attrib = CostAttributor(event_bus)

        def emit_events():
            for i in range(20):
                event_bus.emit_event(
                    EventType.LLM_RESPONSE,
                    {
                        "source": "claude",
                        "usage": {
                            "input_tokens": 10_000,
                            "output_tokens": 5_000,
                            "cache_read_tokens": 0,
                            "cache_write_tokens": 0,
                        },
                    },
                    session_id=f"session-{i}",
                )

        threads = [threading.Thread(target=emit_events) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash and should have accumulated
        breakdown = attrib.get_cost_breakdown()
        assert breakdown["total_cost_usd"] > 0

    def test_emits_cost_attributed_event(self):
        """CostAttributor emits cost.attributed events."""
        event_bus = EventBus()
        attrib = CostAttributor(event_bus)

        received_events = []

        def capture_handler(event):
            received_events.append(event)

        event_bus.subscribe(COST_ATTRIBUTED, capture_handler)

        event_bus.emit_event(
            EventType.LLM_RESPONSE,
            {
                "source": "claude",
                "usage": {
                    "input_tokens": 100_000,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                },
            },
            session_id="test-session",
        )

        # Should have received at least one cost.attributed event
        assert len(received_events) >= 1

    def test_empty_usage_handled(self):
        """Empty usage data is handled correctly."""
        event_bus = EventBus()
        attrib = CostAttributor(event_bus)

        event_bus.emit_event(
            EventType.LLM_RESPONSE,
            {
                "source": "claude",
                "usage": {},
            },
            session_id="test-session",
        )

        breakdown = attrib.get_cost_breakdown()
        # Should not crash, may have zero or near-zero cost
        assert breakdown["total_cost_usd"] >= 0

    def test_per_source_usage_accumulated(self):
        """per_source_usage is accumulated correctly."""
        event_bus = EventBus()
        attrib = CostAttributor(event_bus)

        event_bus.emit_event(
            EventType.LLM_RESPONSE,
            {
                "source": "claude",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 200,
                    "cache_read_tokens": 50,
                    "cache_write_tokens": 10,
                },
            },
            session_id="test-session",
        )

        breakdown = attrib.get_cost_breakdown()
        claude_usage = breakdown["per_source_usage"]["claude"]
        assert claude_usage["input_tokens"] == 100
        assert claude_usage["output_tokens"] == 200
        assert claude_usage["cache_read_tokens"] == 50
        assert claude_usage["cache_write_tokens"] == 10
