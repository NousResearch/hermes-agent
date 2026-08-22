"""ExecutiveIntegrationMetrics — local observability (no network)."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from .types import (
    ExecutiveIntegrationMetrics,
    LaunchStatus,
    ObjectiveGatewayDecision,
    RouteKind,
    _now_iso8601,
)


# ──────────────────────────────────────────────────────────────────────
# ExecutiveIntegrationMetricsCollector
# ──────────────────────────────────────────────────────────────────────


class ExecutiveIntegrationMetricsCollector:
    """Local-only metrics collector. No network, no DB write."""

    SCHEMA_VERSION = "eil.v1"

    def __init__(self) -> None:
        self._metrics = ExecutiveIntegrationMetrics()
        self._created_at = _now_iso8601()
        self._metrics.created_at = self._created_at

    # ── public ────────────────────────────────────────────────

    def record_route(self, decision: ObjectiveGatewayDecision, *, elapsed_ms: float) -> None:
        m = self._metrics
        m.total_routes += 1
        m.avg_route_confidence = _update_avg(
            m.avg_route_confidence, m.total_routes, decision.confidence
        )
        m.avg_routing_latency_ms = _update_avg(
            m.avg_routing_latency_ms, m.total_routes, elapsed_ms
        )
        if decision.route_kind == RouteKind.CHAT:
            m.chat_routes += 1
        elif decision.route_kind == RouteKind.TOOL:
            m.tool_routes += 1
        elif decision.route_kind == RouteKind.EXECUTIVE:
            m.executive_routes += 1
            m.launch_requests_created += 1
        elif decision.route_kind == RouteKind.CLARIFY:
            m.clarify_routes += 1
        elif decision.route_kind == RouteKind.REJECT:
            m.reject_routes += 1
        if decision.intent_routing_strategy:
            key = decision.intent_routing_strategy
            m.per_intent_routing_strategy[key] = m.per_intent_routing_strategy.get(key, 0) + 1
        for kw in decision.matched_keywords:
            m.per_keyword_frequency[kw] = m.per_keyword_frequency.get(kw, 0) + 1
        m.updated_at = _now_iso8601()

    def record_launch_approved(self) -> None:
        self._metrics.launch_requests_approved += 1
        self._metrics.updated_at = _now_iso8601()

    def record_launch_rejected(self) -> None:
        self._metrics.launch_requests_rejected += 1
        self._metrics.updated_at = _now_iso8601()

    def record_launch_executed(self) -> None:
        self._metrics.launch_requests_executed += 1
        self._metrics.updated_at = _now_iso8601()

    def record_launch_failed(self) -> None:
        self._metrics.launch_requests_failed += 1
        self._metrics.updated_at = _now_iso8601()

    def record_launch_cancelled(self) -> None:
        self._metrics.launch_requests_cancelled += 1
        self._metrics.updated_at = _now_iso8601()

    def snapshot(self) -> ExecutiveIntegrationMetrics:
        """Return a copy of the current metrics."""
        return ExecutiveIntegrationMetrics(
            total_routes=self._metrics.total_routes,
            chat_routes=self._metrics.chat_routes,
            tool_routes=self._metrics.tool_routes,
            executive_routes=self._metrics.executive_routes,
            clarify_routes=self._metrics.clarify_routes,
            reject_routes=self._metrics.reject_routes,
            avg_route_confidence=self._metrics.avg_route_confidence,
            avg_routing_latency_ms=self._metrics.avg_routing_latency_ms,
            launch_requests_created=self._metrics.launch_requests_created,
            launch_requests_approved=self._metrics.launch_requests_approved,
            launch_requests_rejected=self._metrics.launch_requests_rejected,
            launch_requests_executed=self._metrics.launch_requests_executed,
            launch_requests_failed=self._metrics.launch_requests_failed,
            launch_requests_cancelled=self._metrics.launch_requests_cancelled,
            per_intent_routing_strategy=dict(self._metrics.per_intent_routing_strategy),
            per_keyword_frequency=dict(self._metrics.per_keyword_frequency),
            created_at=self._metrics.created_at,
            updated_at=self._metrics.updated_at,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the metrics as a JSON-serializable dict."""
        snapshot = self.snapshot()
        d = asdict(snapshot)
        d["schema_version"] = self.SCHEMA_VERSION
        return d


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _update_avg(current_avg: float, n: int, new_value: float) -> float:
    """Update a running average."""
    if n <= 1:
        return new_value
    return ((current_avg * (n - 1)) + new_value) / n


__all__ = [
    "ExecutiveIntegrationMetricsCollector",
]
