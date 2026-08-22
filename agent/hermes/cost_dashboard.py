"""
Cost Dashboard for Hermes Agent.

Unified real-time cost observability layer that aggregates data from
CostAttributor, CostController, and AlertManager into a single observable
snapshot. Complies with OS(状态可观测): all cost state is inspectable.

Design principles (OS: Observability):
- All state is observable: cost, budget, tokens, alerts, and rates
- Backward compatible: existing CostAttributor/CostController/AlertManager unchanged
- EventBus integration: emits cost.dashboard.updated for live consumers
- Zero coupling: CostDashboard is a thin facade over existing systems

Usage:
    from agent.hermes.cost_dashboard import CostDashboard, CostDashboardSnapshot

    dashboard = CostDashboard(
        cost_attributor=attributor,
        cost_controller=controller,
        alert_manager=alert_mgr,
        event_bus=event_bus,
    )

    # Query live state
    snap = dashboard.get_snapshot()
    print(f"Total cost: ${snap.total_cost_usd:.4f}")
    print(dashboard.format_terminal(snap))

    # Register a callback for live updates
    dashboard.on_update(lambda snap: print(f"Cost: ${snap.total_cost_usd:.4f}"))
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Event type for dashboard updates
# ──────────────────────────────────────────────────────────────────────────────

COST_DASHBOARD_UPDATED = "cost.dashboard.updated"


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CostDashboardSnapshot:
    """
    Immutable, comprehensive snapshot of all cost-related metrics.

    Aggregates data from CostAttributor, CostController, and AlertManager
    into a single observable view. All fields are deterministic.

    Attributes
    ----------
    total_cost_usd : float
        Cumulative USD cost across all sources.
    cost_per_minute : float
        Rolling cost rate in USD per minute (0.0 if no elapsed time).
    cost_per_call : float
        Average cost per LLM call in USD (0.0 if no calls).
    budget_used_pct : float
        Budget utilization ratio, 0.0–1.0+ (0.0 if no budget set).
    budget_remaining_usd : float
        Remaining budget in USD (negative if over budget, 0 if no budget).
    is_over_budget : bool
        True when cumulative cost exceeds budget.
    is_over_hard_limit : bool
        True when the HARD_LIMIT alert threshold was crossed.
    budget_label : str
        Human-readable budget display (e.g. "$0.48 / $5.00 (9.6%)").
    per_source : Dict[str, float]
        Per-source cost breakdown (source name -> USD).
    per_tool : Dict[str, float]
        Per-tool cost breakdown (tool name -> USD).
    per_source_usage : Dict[str, Dict[str, int]]
        Per-source token usage breakdown.
    total_tokens : int
        Total tokens consumed (input + output + cache).
    input_tokens : int
    output_tokens : int
    cache_read_tokens : int
    cache_write_tokens : int
    llm_call_count : int
        Number of LLM calls tracked.
    tool_call_count : int
        Number of tool calls tracked.
    output_tokens_per_dollar : float
        Cost-effectiveness: output tokens per USD spent.
    active_alerts : List[Dict]
        List of currently triggered alerts (serialized as dicts).
    alert_count : int
        Number of active alerts.
    has_budget : bool
        True when a cost budget is configured.
    session_elapsed_seconds : float
        Elapsed session time in seconds (0.0 if not set).
    timestamp : float
        Epoch timestamp when this snapshot was captured.
    """
    # Cost totals
    total_cost_usd: float = 0.0
    cost_per_minute: float = 0.0
    cost_per_call: float = 0.0

    # Budget
    budget_used_pct: float = 0.0
    budget_remaining_usd: float = 0.0
    is_over_budget: bool = False
    is_over_hard_limit: bool = False
    budget_label: str = "no budget"

    # Attribution
    per_source: Dict[str, float] = field(default_factory=dict)
    per_tool: Dict[str, float] = field(default_factory=dict)
    per_source_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Tokens
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # Counts
    llm_call_count: int = 0
    tool_call_count: int = 0

    # Efficiency
    output_tokens_per_dollar: float = 0.0

    # Alerts
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    alert_count: int = 0

    # Meta
    has_budget: bool = False
    session_elapsed_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize snapshot to a plain dict for API/EventBus/JSON."""
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "cost_per_minute": round(self.cost_per_minute, 6),
            "cost_per_call": round(self.cost_per_call, 6),
            "budget_used_pct": round(self.budget_used_pct, 4),
            "budget_remaining_usd": round(self.budget_remaining_usd, 4),
            "is_over_budget": self.is_over_budget,
            "is_over_hard_limit": self.is_over_hard_limit,
            "budget_label": self.budget_label,
            "per_source": {k: round(v, 6) for k, v in self.per_source.items()},
            "per_tool": {k: round(v, 6) for k, v in self.per_tool.items()},
            "per_source_usage": dict(self.per_source_usage),
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "llm_call_count": self.llm_call_count,
            "tool_call_count": self.tool_call_count,
            "output_tokens_per_dollar": round(self.output_tokens_per_dollar, 2),
            "active_alerts": self.active_alerts,
            "alert_count": self.alert_count,
            "has_budget": self.has_budget,
            "session_elapsed_seconds": round(self.session_elapsed_seconds, 1),
            "timestamp": round(self.timestamp, 3),
        }


# ──────────────────────────────────────────────────────────────────────────────
# CostDashboard
# ──────────────────────────────────────────────────────────────────────────────

class CostDashboard:
    """
    Unified cost observability facade.

    Composes CostAttributor, CostController, and AlertManager into a single
    observable interface. Thread-safe.

    Parameters
    ----------
    cost_attributor : CostAttributor, optional
        Source for per-source/per-tool cost attribution.
    cost_controller : CostController, optional
        Source for budget tracking and cost-effectiveness metrics.
    alert_manager : AlertManager, optional
        Source for active alerts.
    event_bus : EventBus, optional
        EventBus for emitting cost.dashboard.updated events.
    session_elapsed_seconds : callable, optional
        A no-argument callable returning elapsed session time in seconds.
        If not provided, uses wall-clock time since the first call.
    session_id : str, optional
        Session ID for EventBus correlation.

    Usage::

        dashboard = CostDashboard(
            cost_attributor=attributor,
            cost_controller=controller,
            alert_manager=alert_mgr,
            event_bus=event_bus,
        )

        snap = dashboard.get_snapshot()
        print(dashboard.format_terminal(snap))
    """

    def __init__(
        self,
        cost_attributor: Optional[Any] = None,
        cost_controller: Optional[Any] = None,
        alert_manager: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        session_elapsed_seconds: Optional[Callable[[], float]] = None,
        session_id: str = "",
    ):
        self._cost_attributor = cost_attributor
        self._cost_controller = cost_controller
        self._alert_manager = alert_manager
        self._event_bus = event_bus
        self._session_id = session_id

        # Track session time
        if session_elapsed_seconds is not None:
            self._get_elapsed = session_elapsed_seconds
        else:
            self._start_time: float = time.time()
            self._get_elapsed = lambda: time.time() - self._start_time

        # Live update callbacks
        self._update_callbacks: List[Callable[[CostDashboardSnapshot], None]] = []
        self._update_callbacks_lock = threading.Lock()

        # Track LLM call count for per-call cost
        self._llm_call_count: int = 0
        self._tool_call_count: int = 0

        # Subscribe to EventBus events if an EventBus is available
        self._subscribe_to_events()

    # ── EventBus subscriptions ─────────────────────────────────────────────────

    def _subscribe_to_events(self) -> None:
        """Subscribe to EventBus events for automatic tracking."""
        bus = self._event_bus
        if bus is None:
            return
        try:
            from agent.hermes.analytics import EventType
            bus.subscribe(EventType.LLM_RESPONSE, self._on_llm_response)
            bus.subscribe(EventType.TOOL_RESULT, self._on_tool_result)
        except Exception:
            pass

    def _on_llm_response(self, event) -> None:
        """Track LLM call count for per-call cost."""
        self._llm_call_count += 1

    def _on_tool_result(self, event) -> None:
        """Track tool call count."""
        self._tool_call_count += 1

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_snapshot(self) -> CostDashboardSnapshot:
        """
        Return an immutable snapshot of all cost metrics.

        Aggregates data from all configured sources. Missing sources contribute
        zero values — the snapshot is always complete.
        """
        elapsed = self._get_elapsed()
        now = time.time()

        # ── CostAttributor data ────────────────────────────────────────────────
        total_cost = 0.0
        per_source: Dict[str, float] = {}
        per_tool: Dict[str, float] = {}
        per_source_usage: Dict[str, Dict[str, int]] = {}
        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_write = 0
        total_all_tokens = 0

        ca = self._cost_attributor
        if ca is not None:
            try:
                breakdown = ca.get_cost_breakdown()
                total_cost = breakdown.get("total_cost_usd", 0.0)

                per_source = dict(breakdown.get("per_source", {}))
                per_tool = dict(breakdown.get("per_tool", {}))
                per_source_usage = {}
                for src, usage in breakdown.get("per_source_usage", {}).items():
                    if isinstance(usage, dict):
                        per_source_usage[src] = dict(usage)
                        inp = usage.get("input_tokens", 0)
                        out = usage.get("output_tokens", 0)
                        cr = usage.get("cache_read_tokens", 0)
                        cw = usage.get("cache_write_tokens", 0)
                        total_input += inp
                        total_output += out
                        total_cache_read += cr
                        total_cache_write += cw
                        total_all_tokens += inp + out + cr + cw
            except Exception:
                pass

        # ── CostController data ───────────────────────────────────────────────
        budget_used_pct = 0.0
        budget_remaining = 0.0
        is_over_budget = False
        is_over_hard_limit = False
        budget_label = "no budget"
        has_budget = False
        output_tokens_per_dollar = 0.0

        cc = self._cost_controller
        if cc is not None:
            try:
                snap = cc.snapshot()
                budget_used_pct = snap.spent_ratio
                budget_remaining = snap.budget_remaining_usd
                is_over_budget = snap.is_over_budget
                budget_label = snap.budget_label
                has_budget = snap.budget_usd > 0
                output_tokens_per_dollar = snap.output_tokens_per_dollar

                # Check for HARD_LIMIT threshold
                if snap.thresholds_triggered and max(snap.thresholds_triggered) >= 1.0:
                    is_over_hard_limit = True
            except Exception:
                pass

        # ── AlertManager data ─────────────────────────────────────────────────
        active_alerts: List[Dict[str, Any]] = []
        alert_count = 0

        am = self._alert_manager
        if am is not None:
            try:
                alerts = am.get_active_alerts()
                active_alerts = [a.to_dict() if hasattr(a, "to_dict") else dict(a) for a in alerts]
                alert_count = len(active_alerts)
            except Exception:
                pass

        # ── Derived metrics ────────────────────────────────────────────────────
        # Cost per minute
        cost_per_minute = 0.0
        if elapsed > 0:
            cost_per_minute = (total_cost / elapsed) * 60.0

        # Per-call cost (use tracked count if available)
        llm_count = self._llm_call_count
        if ca is not None:
            raw = getattr(ca, "_llm_event_count", None)
            if isinstance(raw, int) and raw > 0:
                llm_count = raw
        cost_per_call = (total_cost / llm_count) if llm_count > 0 else 0.0

        return CostDashboardSnapshot(
            total_cost_usd=total_cost,
            cost_per_minute=cost_per_minute,
            cost_per_call=cost_per_call,
            budget_used_pct=budget_used_pct,
            budget_remaining_usd=budget_remaining,
            is_over_budget=is_over_budget,
            is_over_hard_limit=is_over_hard_limit,
            budget_label=budget_label,
            per_source=per_source,
            per_tool=per_tool,
            per_source_usage=per_source_usage,
            total_tokens=total_all_tokens,
            input_tokens=total_input,
            output_tokens=total_output,
            cache_read_tokens=total_cache_read,
            cache_write_tokens=total_cache_write,
            llm_call_count=llm_count,
            tool_call_count=self._tool_call_count,
            output_tokens_per_dollar=output_tokens_per_dollar,
            active_alerts=active_alerts,
            alert_count=alert_count,
            has_budget=has_budget,
            session_elapsed_seconds=elapsed,
            timestamp=now,
        )

    def emit_update(self) -> None:
        """
        Capture a snapshot and emit it to EventBus + all registered callbacks.

        Called automatically by EventBus listeners. Can also be called manually
        after a cost-contributing operation.
        """
        snap = self.get_snapshot()
        self._emit_to_callbacks(snap)
        self._emit_to_eventbus(snap)

    def on_update(self, callback: Callable[[CostDashboardSnapshot], None]) -> None:
        """
        Register a callback for live dashboard updates.

        Args:
            callback: A function that receives a CostDashboardSnapshot.
                     Will be called on every emit_update().
        """
        with self._update_callbacks_lock:
            self._update_callbacks.append(callback)

    def clear_callbacks(self) -> None:
        """Remove all registered update callbacks."""
        with self._update_callbacks_lock:
            self._update_callbacks.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit_to_callbacks(self, snap: CostDashboardSnapshot) -> None:
        """Dispatch snapshot to all registered callbacks."""
        with self._update_callbacks_lock:
            callbacks = list(self._update_callbacks)
        for cb in callbacks:
            try:
                cb(snap)
            except Exception:
                pass

    def _emit_to_eventbus(self, snap: CostDashboardSnapshot) -> None:
        """Emit snapshot to EventBus."""
        bus = self._event_bus
        if bus is None:
            return
        try:
            bus.emit_event(
                COST_DASHBOARD_UPDATED,
                snap.to_dict(),
                session_id=self._session_id,
            )
        except Exception:
            pass

    # ── Formatting ───────────────────────────────────────────────────────────

    @staticmethod
    def format_terminal(snap: CostDashboardSnapshot) -> str:
        """
        Format a dashboard snapshot as a compact one-line CLI display.

        Format: ``  💰 cost  ██████░░░░  $X.XXXX / $X.XX (XX%)  XX.XK tok/$  ⏱ Xm  🔔 N``

        Returns an empty string if total_cost_usd is 0 and there are no alerts
        and no budget — avoids noise at session start.
        """
        # Skip rendering if nothing interesting has happened
        if (
            snap.total_cost_usd == 0.0
            and snap.alert_count == 0
            and not snap.has_budget
        ):
            return ""

        # ── Cost bar ──────────────────────────────────────────────────────────
        bar_width = 10
        if snap.has_budget and snap.budget_used_pct > 0:
            filled = min(int(snap.budget_used_pct * bar_width), bar_width)
        else:
            filled = 0
        bar = "█" * filled + "░" * (bar_width - filled)

        # ── Cost string ───────────────────────────────────────────────────────
        cost_str = f"${snap.total_cost_usd:.4f}"
        if snap.has_budget:
            cost_str += f" / ${snap.budget_remaining_usd + snap.total_cost_usd:.2f}"
            pct = int(snap.budget_used_pct * 100)
            cost_str += f" ({pct}%)"

        # ── Token efficiency ───────────────────────────────────────────────────
        if snap.output_tokens_per_dollar > 0:
            tok_per_dollar = snap.output_tokens_per_dollar
            if tok_per_dollar >= 1_000_000:
                efficiency_str = f"{tok_per_dollar / 1_000_000:.1f}M"
            elif tok_per_dollar >= 1_000:
                efficiency_str = f"{tok_per_dollar / 1_000:.1f}K"
            else:
                efficiency_str = f"{tok_per_dollar:.0f}"
            efficiency_str += " tok/$"
        else:
            efficiency_str = ""

        # ── Time ───────────────────────────────────────────────────────────────
        elapsed = snap.session_elapsed_seconds
        if elapsed < 60:
            time_str = f"{elapsed:.0f}s"
        elif elapsed < 3600:
            time_str = f"{elapsed / 60:.1f}m"
        else:
            time_str = f"{elapsed / 3600:.1f}h"

        # ── Alerts ─────────────────────────────────────────────────────────────
        alert_str = ""
        if snap.alert_count > 0:
            # Check for hard limit
            if snap.is_over_hard_limit:
                alert_str = "🔴🔔" + str(snap.alert_count)
            elif snap.is_over_budget:
                alert_str = "🟠🔔" + str(snap.alert_count)
            else:
                alert_str = "🔔" + str(snap.alert_count)

        # ── Assemble ───────────────────────────────────────────────────────────
        parts = [f"  💰 cost  {bar}  {cost_str}"]
        if efficiency_str:
            parts.append(f"  {efficiency_str}")
        parts.append(f"  ⏱ {time_str}")
        if alert_str:
            parts.append(f"  {alert_str}")

        return "".join(parts)

    @staticmethod
    def format_detailed(snap: CostDashboardSnapshot) -> str:
        """
        Format a dashboard snapshot as a multi-line detailed display.

        Shows all metrics including per-source breakdown and active alerts.
        """
        lines = ["─── Cost Dashboard ───"]

        # Summary
        lines.append(f"  Total:  ${snap.total_cost_usd:.6f}")
        if snap.cost_per_minute > 0:
            lines.append(f"  Rate:   ${snap.cost_per_minute:.6f}/min")
        if snap.cost_per_call > 0:
            lines.append(f"  /Call:  ${snap.cost_per_call:.6f}")
        lines.append(f"  Tokens: {snap.total_tokens:,} (in:{snap.input_tokens:,} out:{snap.output_tokens:,} "
                     f"cache_r:{snap.cache_read_tokens:,} cache_w:{snap.cache_write_tokens:,})")
        lines.append(f"  LLM:    {snap.llm_call_count} calls")
        lines.append(f"  Tools:  {snap.tool_call_count} calls")
        if snap.output_tokens_per_dollar > 0:
            lines.append(f"  Eff:    {snap.output_tokens_per_dollar:,.0f} tok/$")

        # Budget
        lines.append(f"  Budget: {snap.budget_label}")
        if snap.is_over_hard_limit:
            lines.append("  ⚠️  HARD LIMIT EXCEEDED")
        elif snap.is_over_budget:
            lines.append("  ⚠️  Over budget")

        # Per-source
        if snap.per_source:
            lines.append("  Per source:")
            for source, cost in sorted(snap.per_source.items(), key=lambda x: -x[1]):
                lines.append(f"    {source}: ${cost:.6f}")

        # Per-tool
        if snap.per_tool:
            lines.append("  Per tool:")
            for tool, cost in sorted(snap.per_tool.items(), key=lambda x: -x[1]):
                lines.append(f"    {tool}: ${cost:.6f}")

        # Alerts
        if snap.active_alerts:
            lines.append(f"  Alerts ({snap.alert_count}):")
            for alert in snap.active_alerts:
                severity = alert.get("severity", "?")
                msg = alert.get("message", "")
                lines.append(f"    [{severity}] {msg}")

        return "\n".join(lines)
