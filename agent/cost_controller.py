"""
Cost Controller - Token budget and cost alerting for Hermes Agent.

Provides configurable budget thresholds, cost-effectiveness tracking,
and tiered alerts (50%, 80%, 100%) to prevent runaway spending.

Design principles (OS/observability):
- All state is observable: budgets, spending, alerts, and ratios are inspectable
- Alerts fire at configurable thresholds with actionable messages
- Cost-effectiveness (output value per USD) is tracked per session
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Optional, List

# Default alert thresholds as fractions of budget (0.0 - 1.0)
DEFAULT_ALERT_THRESHOLDS = (0.5, 0.8, 1.0)


@dataclass
class CostAlert:
    """A cost alert event, emitted when a threshold is crossed."""
    threshold: float          # e.g. 0.5 for 50%
    threshold_label: str      # e.g. "50%", "80%", "100%"
    cumulative_cost_usd: float
    budget_usd: float
    spent_ratio: float        # fraction of budget spent (0.0 - 1.0+)
    is_hard_limit: bool       # True when threshold >= 1.0 (budget exceeded)


@dataclass
class CostBudget:
    """Configuration for a cost budget."""
    max_cost_usd: float = 0.0     # 0.0 = no budget cap
    alert_thresholds: tuple[float, ...] = (0.5, 0.8, 1.0)
    enabled: bool = True

    @property
    def has_limit(self) -> bool:
        return self.max_cost_usd > 0


@dataclass
class CostSnapshot:
    """Observable snapshot of current cost state."""
    cumulative_cost_usd: float
    budget_usd: float
    spent_ratio: float          # 0.0 if no budget
    is_over_budget: bool
    alert_threshold_crossed: Optional[float]   # threshold fraction or None
    thresholds_triggered: tuple[float, ...]    # all thresholds triggered so far
    output_tokens_per_dollar: float   # cost-effectiveness ratio
    tool_call_count: int

    @property
    def budget_label(self) -> str:
        if self.budget_usd <= 0:
            return "no-limit"
        return f"${self.cumulative_cost_usd:.4f} / ${self.budget_usd:.2f}"

    @property
    def spent_percent(self) -> str:
        if self.budget_usd <= 0:
            return f"${self.cumulative_cost_usd:.4f}"
        pct = self.spent_ratio * 100
        return f"{pct:.1f}%"


class CostController:
    """
    Thread-safe cost budget tracker with tiered alerting.

    Tracks cumulative spending against an optional budget and emits
    alerts when configurable thresholds (default: 50%, 80%, 100%) are crossed.
    Cost-effectiveness ratio (output tokens per USD) is maintained for
    session introspection.

    Usage:
        controller = CostController(budget=CostBudget(max_cost_usd=5.0))
        controller.reset()

        # After each API call:
        controller.add_cost(0.0032)  # $0.0032 for this call

        # Check if alerts should fire:
        alert = controller.check_alert()
        if alert:
            print(f"⚠️  Cost alert: {alert.threshold_label}")

        # Inspect state:
        snapshot = controller.snapshot()
        print(f"Spent: {snapshot.spent_percent}")
    """

    def __init__(
        self,
        budget: Optional[CostBudget] = None,
        on_alert: Optional[Callable[[CostAlert], None]] = None,
        alert_manager: Optional[Any] = None,
    ):
        """
        Initialize CostController.

        Args:
            budget: Cost budget configuration (default: no limit)
            on_alert: Legacy callback for cost alerts (CostAlert -> None)
            alert_manager: Optional AlertManager for integrated resource + cost alerts.
                           When provided, cost alerts are forwarded to AlertManager.check_cost_alert()
                           and both systems stay in sync.
        """
        self._budget = budget or CostBudget()
        self._on_alert = on_alert
        self._alert_manager = alert_manager

        # Cumulative state
        self._cumulative_cost_usd: float = 0.0
        self._output_tokens: int = 0
        self._tool_call_count: int = 0
        self._triggered_thresholds: List[float] = []
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def budget(self) -> CostBudget:
        return self._budget

    @budget.setter
    def budget(self, value: CostBudget) -> None:
        self._budget = value

    def reset(self) -> None:
        """Reset all counters. Call at session start."""
        with self._lock:
            self._cumulative_cost_usd = 0.0
            self._output_tokens = 0
            self._tool_call_count = 0
            self._triggered_thresholds = []

    def add_cost(
        self,
        cost_usd: float,
        output_tokens: int = 0,
        tool_calls: int = 0,
    ) -> Optional[CostAlert]:
        """
        Accumulate cost and check alert thresholds.

        Returns a CostAlert if a new threshold was crossed, else None.
        Thread-safe.

        If an AlertManager is configured, also forwards the cost to
        AlertManager.check_cost_alert() for unified resource+cost alerting.
        """
        with self._lock:
            self._cumulative_cost_usd += max(0.0, cost_usd)
            self._output_tokens += max(0, output_tokens)
            self._tool_call_count += max(0, tool_calls)

            if not self._should_check_alerts():
                # Still forward to AlertManager even without budget threshold
                if self._alert_manager is not None and cost_usd > 0:
                    self._forward_to_alert_manager()
                return None

            alert = self._evaluate_alerts()
            if self._alert_manager is not None:
                self._forward_to_alert_manager()
            return alert

    def check_alert(self) -> Optional[CostAlert]:
        """
        Re-check alert thresholds without adding cost.
        Returns a CostAlert if the most-recently crossed threshold should fire, else None.
        """
        with self._lock:
            if not self._should_check_alerts():
                return None
            return self._evaluate_alerts(return_only_new=False)

    def snapshot(self) -> CostSnapshot:
        """Return an immutable snapshot of current cost state."""
        with self._lock:
            budget_usd = self._budget.max_cost_usd
            spent_ratio = (
                self._cumulative_cost_usd / budget_usd
                if budget_usd > 0 else 0.0
            )
            is_over = budget_usd > 0 and self._cumulative_cost_usd >= budget_usd
            cost_effectiveness = (
                self._output_tokens / self._cumulative_cost_usd
                if self._cumulative_cost_usd > 0 else 0.0
            )
            return CostSnapshot(
                cumulative_cost_usd=self._cumulative_cost_usd,
                budget_usd=budget_usd,
                spent_ratio=spent_ratio,
                is_over_budget=is_over,
                alert_threshold_crossed=(
                    self._triggered_thresholds[-1]
                    if self._triggered_thresholds else None
                ),
                thresholds_triggered=tuple(self._triggered_thresholds),
                output_tokens_per_dollar=cost_effectiveness,
                tool_call_count=self._tool_call_count,
            )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _should_check_alerts(self) -> bool:
        return (
            self._budget.enabled
            and self._budget.has_limit
            and self._cumulative_cost_usd > 0
        )

    def _forward_to_alert_manager(self) -> None:
        """Forward current cost to AlertManager. Must hold lock."""
        try:
            am = self._alert_manager
            if am is None:
                return
            # Capture snapshot values while holding lock
            elapsed = 0.0  # CostController doesn't track time; AlertManager uses it if available
            am.check_cost_alert(
                cumulative_cost_usd=self._cumulative_cost_usd,
                elapsed_seconds=elapsed,
            )
        except Exception:
            pass  # fire-and-forget

    def _evaluate_alerts(
        self,
        return_only_new: bool = True,
    ) -> Optional[CostAlert]:
        """Evaluate which thresholds have been crossed. Must hold lock."""
        current = self._cumulative_cost_usd
        budget = self._budget.max_cost_usd
        spent_ratio = current / budget if budget > 0 else 0.0

        alert_to_fire: Optional[CostAlert] = None

        for threshold in sorted(self._budget.alert_thresholds, reverse=True):
            if spent_ratio >= threshold:
                is_hard_limit = threshold >= 1.0
                label = "100% (budget exceeded)" if is_hard_limit else f"{int(threshold * 100)}%"

                alert = CostAlert(
                    threshold=threshold,
                    threshold_label=label,
                    cumulative_cost_usd=current,
                    budget_usd=budget,
                    spent_ratio=spent_ratio,
                    is_hard_limit=is_hard_limit,
                )

                # Track triggered thresholds
                if threshold not in self._triggered_thresholds:
                    self._triggered_thresholds.append(threshold)

                # Return the highest newly crossed threshold
                if return_only_new and threshold == self._triggered_thresholds[-1]:
                    alert_to_fire = alert
                elif not return_only_new:
                    alert_to_fire = alert
                break

        if alert_to_fire:
            try:
                if self._on_alert:
                    self._on_alert(alert_to_fire)
            except Exception:
                pass  # Never let alert callbacks crash the agent loop

        return alert_to_fire

    def format_alert_message(self, alert: CostAlert) -> str:
        """Format an alert into a human-readable warning string."""
        if alert.is_hard_limit:
            return (
                f"[COST BUDGET: HARD LIMIT REACHED — ${alert.cumulative_cost_usd:.4f} "
                f"spent (budget: ${alert.budget_usd:.2f}). "
                "Agent should wrap up immediately.]"
            )
        return (
            f"[COST BUDGET: {alert.threshold_label} — ${alert.cumulative_cost_usd:.4f} "
            f"of ${alert.budget_usd:.2f} budget used. "
            "Consider consolidating output to conserve budget.]"
        )


def build_default_budget(
    max_cost_usd: float = 0.0,
    alert_thresholds: tuple[float, ...] = (0.5, 0.8, 1.0),
    enabled: bool = True,
) -> CostBudget:
    """Factory to build a CostBudget with sensible defaults."""
    return CostBudget(
        max_cost_usd=max_cost_usd,
        alert_thresholds=alert_thresholds,
        enabled=enabled,
    )
