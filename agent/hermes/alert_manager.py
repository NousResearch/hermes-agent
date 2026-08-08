"""
Alert Manager for Hermes-Agent.

Provides configurable alert thresholds for resource and cost metrics,
fires tiered alerts (WARNING / CRITICAL / HARD_LIMIT), and integrates
with the EventBus and structured logging.

Design principles (OS/observability):
- All alert rules and triggered alerts are observable
- Thresholds are fully configurable with sensible defaults
- Alerts carry actionable metadata (current value, threshold, ratio)
- Backward compatible: works without EventBus or structured logging

Usage:
    from agent.hermes.alert_manager import AlertManager, AlertThresholds, ResourceAlert

    manager = AlertManager(event_bus=event_bus, logger=structured_logger)

    # Register cost budget threshold
    manager.set_cost_threshold(max_cost_usd=5.0)

    # Check after each cost addition
    alert = manager.check_cost_alert(cumulative_cost_usd=0.0032)
    if alert:
        logger.warning(manager.format_alert(alert))

    # Or check resource metrics
    alert = manager.check_resource_alert(cpu_percent=95.0, memory_mb=2048.0)
    if alert:
        print(f"ALERT: {alert.category} — {alert.message}")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Any

from agent.hermes.analytics import Event, EventType

# ──────────────────────────────────────────────────────────────────────────────
# Alert types and severity
# ──────────────────────────────────────────────────────────────────────────────

class AlertCategory(str, Enum):
    """Categories of alert."""
    COST = "cost"
    CPU = "cpu"
    MEMORY = "memory"
    TOKEN_RATE = "token_rate"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    SYSTEM_MEMORY = "system_memory"


class AlertSeverity(str, Enum):
    """Alert severity levels (mirrors common logging levels)."""
    WARNING = "WARNING"    # 50%–80% threshold
    CRITICAL = "CRITICAL"  # 80%–100% threshold
    HARD_LIMIT = "HARD_LIMIT"  # 100%+ (budget exceeded)


@dataclass
class AlertThresholds:
    """
    Configurable thresholds for resource and cost alerting.

    All thresholds are optional — None means "no alerting for this metric".
    """
    # Cost thresholds
    max_cost_usd: Optional[float] = None
    cost_warning_ratio: float = 0.5    # fire at 50% of budget
    cost_critical_ratio: float = 0.8   # fire at 80% of budget

    # CPU thresholds (%)
    cpu_warning_percent: Optional[float] = 70.0
    cpu_critical_percent: Optional[float] = 90.0

    # Memory thresholds (MB)
    memory_warning_mb: Optional[float] = 512.0
    memory_critical_mb: Optional[float] = 1024.0

    # System memory threshold (%)
    system_memory_warning_percent: Optional[float] = 80.0
    system_memory_critical_percent: Optional[float] = 95.0

    # Token rate thresholds (tokens/second)
    token_rate_warning: Optional[float] = 1000.0
    token_rate_critical: Optional[float] = 5000.0

    # LLM latency thresholds (seconds)
    latency_warning_seconds: Optional[float] = 30.0
    latency_critical_seconds: Optional[float] = 60.0

    # Error count threshold (errors per minute — checked on snapshot)
    error_count_warning: Optional[int] = 5
    error_count_critical: Optional[int] = 20

    @classmethod
    def relaxed(cls) -> "AlertThresholds":
        """Permissive thresholds for high-throughput workloads."""
        return cls(
            cpu_warning_percent=85.0,
            cpu_critical_percent=95.0,
            memory_warning_mb=1024.0,
            memory_critical_mb=2048.0,
            token_rate_warning=5000.0,
            token_rate_critical=10000.0,
            latency_warning_seconds=60.0,
            latency_critical_seconds=120.0,
        )

    @classmethod
    def strict(cls) -> "AlertThresholds":
        """Strict thresholds for cost-sensitive environments."""
        return cls(
            max_cost_usd=1.0,
            cost_warning_ratio=0.25,
            cost_critical_ratio=0.5,
            cpu_warning_percent=50.0,
            cpu_critical_percent=75.0,
            memory_warning_mb=256.0,
            memory_critical_mb=512.0,
            token_rate_warning=500.0,
            token_rate_critical=2000.0,
            latency_warning_seconds=15.0,
            latency_critical_seconds=30.0,
            error_count_warning=2,
            error_count_critical=10,
        )

    @classmethod
    def disabled(cls) -> "AlertThresholds":
        """All alerts disabled (useful for testing)."""
        return cls(
            max_cost_usd=None,
            cpu_warning_percent=None,
            cpu_critical_percent=None,
            memory_warning_mb=None,
            memory_critical_mb=None,
            system_memory_warning_percent=None,
            system_memory_critical_percent=None,
            token_rate_warning=None,
            token_rate_critical=None,
            latency_warning_seconds=None,
            latency_critical_seconds=None,
            error_count_warning=None,
            error_count_critical=None,
        )


@dataclass
class ResourceAlert:
    """
    A resource alert event, emitted when a threshold is crossed.

    All numeric fields are deterministic for inspection.
    """
    # Identity
    alert_id: str = field(default_factory=lambda: f"alert-{int(time.time() * 1000)}")
    category: AlertCategory = AlertCategory.COST
    severity: AlertSeverity = AlertSeverity.WARNING

    # Threshold context
    threshold_name: str = ""           # e.g. "cpu_warning_percent"
    threshold_value: float = 0.0        # the threshold that was crossed
    current_value: float = 0.0         # the metric value that triggered it
    ratio: float = 0.0                 # current_value / threshold (1.0 = at threshold)

    # Session context
    session_id: str = ""
    elapsed_seconds: float = 0.0

    # Counts
    occurrence_count: int = 1          # how many times this alert has fired

    # Human-readable
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "threshold_name": self.threshold_name,
            "threshold_value": self.threshold_value,
            "current_value": self.current_value,
            "ratio": round(self.ratio, 3),
            "session_id": self.session_id,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "occurrence_count": self.occurrence_count,
            "message": self.message,
        }


# ──────────────────────────────────────────────────────────────────────────────
# AlertManager
# ──────────────────────────────────────────────────────────────────────────────

class AlertManager:
    """
    Central alert manager for Hermes-Agent.

    Manages configurable thresholds, checks metrics against them,
    fires alerts, and routes them to EventBus / structured logging.

    Thread-safe: all threshold state is protected by a lock.

    Usage::

        manager = AlertManager(event_bus=event_bus, logger=structured_logger)

        # After each LLM call / metrics snapshot:
        alert = manager.check_cost_alert(cumulative_cost_usd=0.50)
        if alert:
            logger.warning(manager.format_alert(alert))

        # Or with a ResourceSnapshot:
        alert = manager.check_snapshot_alerts(snapshot)
        if alert:
            print(alert.message)
    """

    def __init__(
        self,
        *,
        thresholds: Optional[AlertThresholds] = None,
        event_bus: Optional[Any] = None,
        logger: Optional[Any] = None,
        session_id: str = "",
        on_alert: Optional[Callable[[ResourceAlert], None]] = None,
    ):
        """
        Initialize AlertManager.

        Args:
            thresholds: Alert thresholds (default: AlertThresholds())
            event_bus: Optional EventBus for alert events
            logger: Optional logger for structured alert output
            session_id: Session ID for alert correlation
            on_alert: Optional callback invoked when an alert fires
        """
        self._thresholds = thresholds or AlertThresholds()
        self._event_bus = event_bus
        self._logger = logger
        self._session_id = session_id
        self._on_alert = on_alert

        self._lock = threading.RLock()

        # Per-category triggered-alert tracking (last alert per category/severity)
        self._triggered_alerts: Dict[str, ResourceAlert] = {}

        # Cooldown tracking: category -> last_fired_time
        self._last_fired: Dict[str, float] = {}

        # Cooldown duration per category (seconds) to prevent alert spam
        self._cooldown_seconds: Dict[str, float] = {
            AlertCategory.COST.value: 30.0,
            AlertCategory.CPU.value: 10.0,
            AlertCategory.MEMORY.value: 10.0,
            AlertCategory.SYSTEM_MEMORY.value: 10.0,
            AlertCategory.TOKEN_RATE.value: 15.0,
            AlertCategory.LATENCY.value: 15.0,
            AlertCategory.ERROR_RATE.value: 30.0,
        }

    # ── Threshold management ─────────────────────────────────────────────────

    @property
    def thresholds(self) -> AlertThresholds:
        """Return current thresholds (read-only view)."""
        return self._thresholds

    def update_thresholds(self, **kwargs) -> None:
        """Update specific threshold values by name."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._thresholds, key):
                    setattr(self._thresholds, key, value)
                else:
                    _get_logger().warning("Unknown threshold: %s", key)

    def set_cost_threshold(self, max_cost_usd: float, **kwargs) -> None:
        """Convenience: set cost budget threshold."""
        with self._lock:
            self._thresholds.max_cost_usd = max_cost_usd
            for key, value in kwargs.items():
                if hasattr(self._thresholds, key):
                    setattr(self._thresholds, key, value)

    def set_cpu_threshold(self, warning: float, critical: float) -> None:
        """Convenience: set CPU thresholds."""
        with self._lock:
            self._thresholds.cpu_warning_percent = warning
            self._thresholds.cpu_critical_percent = critical

    def set_memory_threshold(self, warning_mb: float, critical_mb: float) -> None:
        """Convenience: set memory thresholds (MB)."""
        with self._lock:
            self._thresholds.memory_warning_mb = warning_mb
            self._thresholds.memory_critical_mb = critical_mb

    def set_latency_threshold(self, warning_s: float, critical_s: float) -> None:
        """Convenience: set LLM latency thresholds (seconds)."""
        with self._lock:
            self._thresholds.latency_warning_seconds = warning_s
            self._thresholds.latency_critical_seconds = critical_s

    # ── Alert checking ───────────────────────────────────────────────────────

    def check_cost_alert(
        self,
        cumulative_cost_usd: float,
        elapsed_seconds: float = 0.0,
    ) -> Optional[ResourceAlert]:
        """
        Check cost against configured budget thresholds.

        Returns a ResourceAlert if a new threshold was crossed, else None.
        Uses cooldown to prevent repeated alerts for the same level.
        """
        with self._lock:
            max_cost = self._thresholds.max_cost_usd
            if max_cost is None or max_cost <= 0:
                return None

            ratio = cumulative_cost_usd / max_cost
            now = time.time()

            # Determine severity
            severity: Optional[AlertSeverity] = None
            threshold_value = 0.0
            threshold_name = ""

            if ratio >= 1.0:
                severity = AlertSeverity.HARD_LIMIT
                threshold_value = max_cost
                threshold_name = "hard_limit"
            elif ratio >= self._thresholds.cost_critical_ratio:
                severity = AlertSeverity.CRITICAL
                threshold_value = self._thresholds.cost_critical_ratio * max_cost
                threshold_name = "cost_critical_ratio"
            elif ratio >= self._thresholds.cost_warning_ratio:
                severity = AlertSeverity.WARNING
                threshold_value = self._thresholds.cost_warning_ratio * max_cost
                threshold_name = "cost_warning_ratio"

            if severity is None:
                return None

            key = f"{AlertCategory.COST.value}:{severity.value}"
            cooldown = self._cooldown_seconds.get(AlertCategory.COST.value, 30.0)
            last = self._last_fired.get(key, 0.0)

            if now - last < cooldown:
                return None

            alert = self._build_alert(
                category=AlertCategory.COST,
                severity=severity,
                threshold_name=threshold_name,
                threshold_value=threshold_value,
                current_value=cumulative_cost_usd,
                ratio=ratio,
                elapsed_seconds=elapsed_seconds,
                message=self._format_cost_message(severity, cumulative_cost_usd, max_cost, ratio),
            )

            self._triggered_alerts[key] = alert
            self._last_fired[key] = now

            self._dispatch(alert)
            return alert

    def check_resource_alert(
        self,
        cpu_percent: float = 0.0,
        memory_mb: float = 0.0,
        system_memory_percent: float = 0.0,
        token_rate: float = 0.0,
        latency_seconds: float = 0.0,
        error_count: int = 0,
        elapsed_seconds: float = 0.0,
    ) -> Optional[ResourceAlert]:
        """
        Check resource metrics against configured thresholds.

        Checks all provided metrics in priority order:
        CPU → Memory → System Memory → Token Rate → Latency → Error Rate

        Returns the highest-severity new alert, or None.
        """
        with self._lock:
            now = time.time()
            best_alert: Optional[ResourceAlert] = None

            checks: List[tuple[str, AlertCategory, float, Optional[float], Optional[float], str]] = [
                ("cpu", AlertCategory.CPU, cpu_percent,
                 self._thresholds.cpu_critical_percent, self._thresholds.cpu_warning_percent,
                 f"CPU at {cpu_percent:.1f}%"),
                ("memory_mb", AlertCategory.MEMORY, memory_mb,
                 self._thresholds.memory_critical_mb, self._thresholds.memory_warning_mb,
                 f"Memory at {memory_mb:.1f}MB"),
                ("system_mem", AlertCategory.SYSTEM_MEMORY, system_memory_percent,
                 self._thresholds.system_memory_critical_percent,
                 self._thresholds.system_memory_warning_percent,
                 f"System memory at {system_memory_percent:.1f}%"),
                ("token_rate", AlertCategory.TOKEN_RATE, token_rate,
                 self._thresholds.token_rate_critical, self._thresholds.token_rate_warning,
                 f"Token rate at {token_rate:.1f}/s"),
                ("latency", AlertCategory.LATENCY, latency_seconds,
                 self._thresholds.latency_critical_seconds, self._thresholds.latency_warning_seconds,
                 f"Latency at {latency_seconds:.1f}s"),
            ]

            for key_name, category, value, critical_th, warning_th, msg_template in checks:
                if value <= 0:
                    continue

                if critical_th is not None and value >= critical_th:
                    severity = AlertSeverity.CRITICAL
                    threshold_val = critical_th
                elif warning_th is not None and value >= warning_th:
                    severity = AlertSeverity.WARNING
                    threshold_val = warning_th
                else:
                    continue

                cooldown = self._cooldown_seconds.get(category.value, 10.0)
                last = self._last_fired.get(f"{category.value}:{severity.value}", 0.0)
                if now - last < cooldown:
                    continue

                ratio = threshold_val > 0 and (value / threshold_val) or 1.0
                alert = self._build_alert(
                    category=category,
                    severity=severity,
                    threshold_name=key_name,
                    threshold_value=threshold_val,
                    current_value=value,
                    ratio=ratio,
                    elapsed_seconds=elapsed_seconds,
                    message=f"[{severity.value}] {msg_template} "
                            f"(threshold: {threshold_val})",
                )

                self._last_fired[f"{category.value}:{severity.value}"] = now

                # Prefer CRITICAL over WARNING; only return one alert
                if best_alert is None or severity == AlertSeverity.CRITICAL:
                    best_alert = alert

            # Error count check
            if error_count > 0:
                if self._thresholds.error_count_critical is not None and \
                   error_count >= self._thresholds.error_count_critical:
                    severity = AlertSeverity.CRITICAL
                    threshold_val = float(self._thresholds.error_count_critical)
                elif self._thresholds.error_count_warning is not None and \
                     error_count >= self._thresholds.error_count_warning:
                    severity = AlertSeverity.WARNING
                    threshold_val = float(self._thresholds.error_count_warning)
                else:
                    severity = None

                if severity is not None:
                    key = f"{AlertCategory.ERROR_RATE.value}:{severity.value}"
                    cooldown = self._cooldown_seconds.get(AlertCategory.ERROR_RATE.value, 30.0)
                    last = self._last_fired.get(key, 0.0)
                    if now - last >= cooldown:
                        ratio = threshold_val > 0 and (error_count / threshold_val) or 1.0
                        err_alert = self._build_alert(
                            category=AlertCategory.ERROR_RATE,
                            severity=severity,
                            threshold_name="error_count",
                            threshold_value=threshold_val,
                            current_value=float(error_count),
                            ratio=ratio,
                            elapsed_seconds=elapsed_seconds,
                            message=f"[{severity.value}] Error count: {error_count} "
                                    f"(threshold: {threshold_val})",
                        )
                        self._last_fired[key] = now
                        if best_alert is None or severity == AlertSeverity.CRITICAL:
                            best_alert = err_alert

            if best_alert:
                self._dispatch(best_alert)

            return best_alert

    def check_snapshot_alerts(
        self,
        snapshot: "ResourceSnapshot",
    ) -> Optional[ResourceAlert]:
        """
        Check a ResourceSnapshot against all configured thresholds.

        Convenience method wrapping check_resource_alert() with snapshot fields.
        """
        return self.check_resource_alert(
            cpu_percent=snapshot.cpu_percent,
            memory_mb=snapshot.memory_mb,
            system_memory_percent=snapshot.system_memory_percent,
            token_rate=snapshot.tokens_per_second,
            latency_seconds=snapshot.last_llm_latency_seconds,
            error_count=snapshot.error_count,
            elapsed_seconds=snapshot.elapsed_seconds,
        )

    # ── Alert dispatch ───────────────────────────────────────────────────────

    def _build_alert(
        self,
        category: AlertCategory,
        severity: AlertSeverity,
        threshold_name: str,
        threshold_value: float,
        current_value: float,
        ratio: float,
        elapsed_seconds: float,
        message: str,
    ) -> ResourceAlert:
        """Build a ResourceAlert dataclass."""
        key = f"{category.value}:{severity.value}"
        count = 1
        with self._lock:
            existing = self._triggered_alerts.get(key)
            if existing:
                count = existing.occurrence_count + 1

        return ResourceAlert(
            category=category,
            severity=severity,
            threshold_name=threshold_name,
            threshold_value=threshold_value,
            current_value=current_value,
            ratio=ratio,
            session_id=self._session_id,
            elapsed_seconds=elapsed_seconds,
            occurrence_count=count,
            message=message,
        )

    def _dispatch(self, alert: ResourceAlert) -> None:
        """Dispatch an alert to all configured destinations."""
        # Callback
        if self._on_alert:
            try:
                self._on_alert(alert)
            except Exception:
                pass

        # Structured log
        if self._logger is not None:
            try:
                extra = {"alert": alert.to_dict()}
                level = logging.CRITICAL if alert.severity == AlertSeverity.HARD_LIMIT \
                    else logging.ERROR if alert.severity == AlertSeverity.CRITICAL \
                    else logging.WARNING
                self._logger.log(level, alert.message, extra=extra)
            except Exception:
                pass

        # EventBus
        if self._event_bus is not None:
            try:
                self._event_bus.emit_event(
                    "alert.triggered",
                    alert.to_dict(),
                    session_id=self._session_id,
                )
            except Exception:
                pass

    # ── Formatting ───────────────────────────────────────────────────────────

    @staticmethod
    def _format_cost_message(
        severity: AlertSeverity,
        cost_usd: float,
        budget_usd: float,
        ratio: float,
    ) -> str:
        """Format a cost alert message."""
        pct = ratio * 100
        if severity == AlertSeverity.HARD_LIMIT:
            return (
                f"[COST BUDGET: HARD LIMIT REACHED — ${cost_usd:.4f} spent "
                f"(budget: ${budget_usd:.2f}). Agent should wrap up immediately.]"
            )
        if severity == AlertSeverity.CRITICAL:
            return (
                f"[COST BUDGET: CRITICAL {pct:.0f}% — ${cost_usd:.4f} "
                f"of ${budget_usd:.2f} used. Consolidate output to conserve budget.]"
            )
        return (
            f"[COST BUDGET: WARNING {pct:.0f}% — ${cost_usd:.4f} "
            f"of ${budget_usd:.2f} used. Consider consolidating output.]"
        )

    def format_alert(self, alert: ResourceAlert) -> str:
        """Format an alert into a human-readable string."""
        return (
            f"[{alert.severity.value}] {alert.category.value.upper()} — "
            f"{alert.message} | value={alert.current_value:.4g}, "
            f"threshold={alert.threshold_value:.4g}, ratio={alert.ratio:.2f}, "
            f"session={alert.session_id}, elapsed={alert.elapsed_seconds:.0f}s, "
            f"count={alert.occurrence_count}"
        )

    # ── Observable state ────────────────────────────────────────────────────

    def get_active_alerts(self) -> List[ResourceAlert]:
        """Return all currently tracked alerts."""
        with self._lock:
            return list(self._triggered_alerts.values())

    def clear_alerts(self) -> None:
        """Clear all tracked alerts and cooldown state."""
        with self._lock:
            self._triggered_alerts.clear()
            self._last_fired.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Module logger
# ──────────────────────────────────────────────────────────────────────────────

def _get_logger():
    return logging.getLogger(__name__)
