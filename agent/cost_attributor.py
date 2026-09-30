"""
Cost Attribution for Hermes-Agent.

Listens to llm.response and tool.result events from the EventBus and
accumulates per-source and per-tool costs.
"""
import threading
from decimal import Decimal
from typing import Dict, Optional

from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
from agent.hermes.analytics import EventBus, Event, EventType

# Event type for attributed cost events
COST_ATTRIBUTED = "cost.attributed"


class AttributedCostRecord:
    """
    Wraps a CanonicalUsage with attribution metadata.

    Uses composition (has-a) rather than inheritance (is-a) to avoid
    duplicating token fields.
    """

    def __init__(
        self,
        usage: CanonicalUsage,
        source: str,
        operation: str,
        parent_session_id: str = "",
    ):
        self.usage = usage  # composition, NOT inheritance
        self.source = source
        self.operation = operation
        self.parent_session_id = parent_session_id
        self.cost_usd = self._calculate_cost(usage)

    def _calculate_cost(self, usage: CanonicalUsage) -> Decimal:
        """Calculate cost using existing pricing logic."""
        # Use a default model since the CanonicalUsage doesn't carry model info
        # The cost is an approximation based on standard pricing
        result = estimate_usage_cost(
            model_name="claude-opus-4-20250514",
            usage=usage,
            provider="anthropic",
        )
        return result.amount_usd or Decimal("0")

    @property
    def input_tokens(self) -> int:
        return self.usage.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.usage.output_tokens

    @property
    def cache_read_tokens(self) -> int:
        return self.usage.cache_read_tokens

    @property
    def cache_write_tokens(self) -> int:
        return self.usage.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "operation": self.operation,
            "parent_session_id": self.parent_session_id,
            "cost_usd": float(self.cost_usd),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
        }


class CostAttributor:
    """
    Listens to EventBus events and accumulates per-source and per-tool costs.

    Thread-safe using threading.Lock for all accumulators.
    """

    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._lock = threading.Lock()

        # Per-source cost accumulation: {source: Decimal}
        self._per_source_costs: Dict[str, Decimal] = {}

        # Per-tool cost accumulation: {tool_name: Decimal}
        self._per_tool_costs: Dict[str, Decimal] = {}

        # Per-source token accumulation: {source: CanonicalUsage}
        self._per_source_usage: Dict[str, CanonicalUsage] = {}

        # Total cost
        self._total_cost = Decimal("0")

        # Subscribe to events
        self._subscribe_to_events()

    def _subscribe_to_events(self) -> None:
        """Subscribe to llm.response and tool.result events."""
        self._event_bus.subscribe(EventType.LLM_RESPONSE, self._on_llm_response)
        self._event_bus.subscribe(EventType.TOOL_RESULT, self._on_tool_result)

    def _on_llm_response(self, event: Event) -> None:
        """Handle llm.response events."""
        payload = event.payload
        source = payload.get("source", "unknown")
        session_id = event.session_id or payload.get("session_id", "")

        # Extract usage from response
        usage_data = payload.get("usage", {})
        if usage_data:
            usage = CanonicalUsage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                cache_read_tokens=usage_data.get("cache_read_tokens", 0),
                cache_write_tokens=usage_data.get("cache_write_tokens", 0),
                reasoning_tokens=usage_data.get("reasoning_tokens", 0),
            )
        else:
            usage = CanonicalUsage()

        record = AttributedCostRecord(
            usage=usage,
            source=source,
            operation="llm.response",
            parent_session_id=session_id,
        )

        self._accumulate(record)

        # Emit cost.attributed event
        self._event_bus.emit_event(
            COST_ATTRIBUTED,
            {
                "record": record.to_dict(),
                "total_cost_usd": float(self._total_cost),
            },
            session_id=session_id,
        )

    def _on_tool_result(self, event: Event) -> None:
        """Handle tool.result events."""
        payload = event.payload
        tool_name = payload.get("tool_name", "unknown")
        session_id = event.session_id or payload.get("session_id", "")

        # Extract usage from tool result (cost is often negligible for tools)
        usage_data = payload.get("usage", {})
        if usage_data:
            usage = CanonicalUsage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                cache_read_tokens=usage_data.get("cache_read_tokens", 0),
                cache_write_tokens=usage_data.get("cache_write_tokens", 0),
            )
        else:
            usage = CanonicalUsage()

        record = AttributedCostRecord(
            usage=usage,
            source=tool_name,
            operation="tool.result",
            parent_session_id=session_id,
        )

        self._accumulate(record)

        # Emit cost.attributed event
        self._event_bus.emit_event(
            COST_ATTRIBUTED,
            {
                "record": record.to_dict(),
                "total_cost_usd": float(self._total_cost),
            },
            session_id=session_id,
        )

    def _accumulate(self, record: AttributedCostRecord) -> None:
        """Thread-safe accumulation of cost records."""
        with self._lock:
            # Accumulate per-source
            if record.source not in self._per_source_costs:
                self._per_source_costs[record.source] = Decimal("0")
                self._per_source_usage[record.source] = CanonicalUsage()
            self._per_source_costs[record.source] += record.cost_usd

            # Update per-source usage
            existing_usage = self._per_source_usage[record.source]
            self._per_source_usage[record.source] = CanonicalUsage(
                input_tokens=existing_usage.input_tokens + record.input_tokens,
                output_tokens=existing_usage.output_tokens + record.output_tokens,
                cache_read_tokens=existing_usage.cache_read_tokens + record.cache_read_tokens,
                cache_write_tokens=existing_usage.cache_write_tokens + record.cache_write_tokens,
            )

            # Accumulate per-tool
            if record.operation == "tool.result":
                if record.source not in self._per_tool_costs:
                    self._per_tool_costs[record.source] = Decimal("0")
                self._per_tool_costs[record.source] += record.cost_usd

            # Update total
            self._total_cost += record.cost_usd

    def get_cost_breakdown(self) -> Dict:
        """
        Return dict with per-source and per-tool costs.

        Thread-safe snapshot of accumulators.
        """
        with self._lock:
            return {
                "total_cost_usd": float(self._total_cost),
                "per_source": {
                    source: float(cost)
                    for source, cost in self._per_source_costs.items()
                },
                "per_tool": {
                    tool: float(cost)
                    for tool, cost in self._per_tool_costs.items()
                },
                "per_source_usage": {
                    source: {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cache_read_tokens": usage.cache_read_tokens,
                        "cache_write_tokens": usage.cache_write_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                    for source, usage in self._per_source_usage.items()
                },
            }
