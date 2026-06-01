"""telemetry_feedback.py — Production Telemetry Feedback Loop (Phase 3+).

Bidirectional integration between the HITL harness and production observability.

What it does:
- Pulls runtime metrics (error rate, latency, incidents) from the observability plugin
- Feeds them back into the GOAL.md as dynamic fitness components
- Enables the harness to learn from production signals, not just pre-deploy tests

Usage:
    from agent.hitl_harness.telemetry_feedback import TelemetryFeedback
    feedback = TelemetryFeedback(agent)
    feedback.enrich_fitness_goal(goal)  # adds production metrics to goal
    feedback.on_deployment(agent.session_id)  # starts post-deploy monitoring
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.hitl_harness.fitness_wizard import FitnessGoal

logger = logging.getLogger(__name__)


@dataclass
class TelemetrySnapshot:
    """A point-in-time capture of production signals."""

    timestamp: str
    error_rate: float = 0.0
    p99_latency_ms: float = 0.0
    incident_count: int = 0
    alert_volume: int = 0
    rollback_flag: bool = False
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class TelemetryFeedback:
    """Connects production observability to the fitness function."""

    def __init__(self, agent: Any, metrics_window_hours: int = 24):
        self.agent = agent
        self.metrics_window_hours = metrics_window_hours
        self._snapshots: List[TelemetrySnapshot] = []
        self._monitoring_sessions: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich_fitness_goal(self, goal: FitnessGoal) -> FitnessGoal:
        """Add production telemetry properties to a FitnessGoal.

        This makes the fitness function aware of runtime signals.
        """
        # Add production properties if not present
        prod_props = [
            {"name": "error_rate", "type": "threshold", "target": 0.01, "unit": "ratio"},
            {"name": "p99_latency_ms", "type": "threshold", "target": 200, "unit": "ms"},
            {"name": "incident_count", "type": "threshold", "target": 0, "unit": "count"},
        ]

        existing_names = {p["name"] for p in goal.properties}
        for prop in prod_props:
            if prop["name"] not in existing_names:
                goal.properties.append(prop)
                # Weight production metrics lower than functional correctness
                goal.composite_weights[prop["name"]] = 0.1

        # Renormalize weights so they sum to ~1.0
        total = sum(goal.composite_weights.values())
        if total > 0:
            for k in goal.composite_weights:
                goal.composite_weights[k] = round(goal.composite_weights[k] / total, 4)

        return goal

    def on_deployment(self, session_id: str, commit_hash: str = "") -> None:
        """Start monitoring a deployment for production signals."""
        self._monitoring_sessions[session_id] = True
        logger.info(
            "Telemetry feedback: started monitoring session %s (commit=%s)",
            session_id, commit_hash or "unknown",
        )

    def on_incident(self, session_id: str, severity: str, description: str) -> None:
        """Record an incident attributed to a specific agent session."""
        snapshot = TelemetrySnapshot(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            incident_count=1,
            rollback_flag=severity in {"critical", "high"},
            custom_metrics={"severity_score": self._severity_to_score(severity)},
        )
        self._snapshots.append(snapshot)
        logger.warning(
            "Telemetry feedback: incident recorded for session %s — %s (%s)",
            session_id, description, severity,
        )

    def get_production_score(self) -> float:
        """Compute a composite production health score from recent snapshots.

        Returns a value in [0.0, 1.0] where 1.0 = perfect health.
        """
        if not self._snapshots:
            return 1.0  # No data = assume healthy

        recent = self._snapshots[-20:]  # last 20 snapshots
        avg_error = sum(s.error_rate for s in recent) / len(recent)
        avg_latency = sum(s.p99_latency_ms for s in recent) / len(recent)
        total_incidents = sum(s.incident_count for s in recent)
        rollback_ratio = sum(1 for s in recent if s.rollback_flag) / len(recent)

        # Score components (lower is worse)
        error_score = max(0.0, 1.0 - (avg_error / 0.05))  # 5% error rate = 0
        latency_score = max(0.0, 1.0 - (avg_latency / 1000))  # 1s p99 = 0
        incident_score = max(0.0, 1.0 - (total_incidents / 5))  # 5 incidents = 0
        rollback_score = 1.0 - rollback_ratio

        return round(
            (error_score * 0.4) + (latency_score * 0.2) +
            (incident_score * 0.3) + (rollback_score * 0.1),
            4,
        )

    def pull_from_observability_plugin(self) -> Optional[TelemetrySnapshot]:
        """Attempt to pull metrics from the observability plugin if available."""
        try:
            # Try to find the observability plugin
            plugin = getattr(self.agent, "_observability_plugin", None)
            if plugin is None:
                # Try common attribute names
                for attr in ["observability", "_observability", "metrics_plugin"]:
                    plugin = getattr(self.agent, attr, None)
                    if plugin:
                        break

            if plugin is None:
                return None

            # Call the plugin's metrics method if it exists
            metrics = {}
            if hasattr(plugin, "get_recent_metrics"):
                metrics = plugin.get_recent_metrics(window_hours=self.metrics_window_hours)
            elif hasattr(plugin, "query"):
                metrics = plugin.query("error_rate,latency_p99,incidents", window="1h")

            snapshot = TelemetrySnapshot(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                error_rate=float(metrics.get("error_rate", 0)),
                p99_latency_ms=float(metrics.get("latency_p99", 0)),
                incident_count=int(metrics.get("incidents", 0)),
                alert_volume=int(metrics.get("alerts", 0)),
            )
            self._snapshots.append(snapshot)
            return snapshot

        except Exception as e:
            logger.debug("Telemetry feedback: could not pull from observability plugin: %s", e)
            return None

    def generate_feedback_report(self) -> str:
        """Generate a markdown report of production feedback for the agent."""
        score = self.get_production_score()
        recent = self._snapshots[-10:]

        lines = [
            "# Production Telemetry Feedback Report",
            "",
            f"**Current Production Health Score:** {score:.2f}/1.00",
            f"**Snapshots Recorded:** {len(self._snapshots)}",
            f"**Monitoring Sessions:** {len(self._monitoring_sessions)}",
            "",
            "## Recent Snapshots (last 10)",
            "",
            "| Time | Error Rate | P99 Latency | Incidents | Rollback |",
            "|------|-----------|-------------|-----------|----------|",
        ]

        for s in recent:
            lines.append(
                f"| {s.timestamp} | {s.error_rate:.4f} | {s.p99_latency_ms:.1f}ms | "
                f"{s.incident_count} | {'Yes' if s.rollback_flag else 'No'} |"
            )

        lines.extend([
            "",
            "## Recommendations",
            "",
        ])

        if score < 0.5:
            lines.append("- **CRITICAL**: Production health is poor. Consider halting autonomous improvements and reviewing recent changes.")
        elif score < 0.8:
            lines.append("- **WARNING**: Production health is degraded. Focus on stability metrics in the next iteration.")
        else:
            lines.append("- **HEALTHY**: Production signals are good. Continue with planned improvements.")

        if any(s.rollback_flag for s in recent):
            lines.append("- Recent rollback detected. Review the change that triggered it before proceeding.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _severity_to_score(severity: str) -> float:
        mapping = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.1}
        return mapping.get(severity.lower(), 0.5)
