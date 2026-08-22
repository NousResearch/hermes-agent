"""ExecutiveIntegrationRouter — top-level router.

Wraps the ObjectiveGateway and provides an even higher-level entry
point. Does NOT execute Executive v2.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from .objective_gateway import ObjectiveGateway, _flags_enabled
from .types import (
    ExecutiveIntegrationMetrics,
    ObjectiveGatewayDecision,
    RouteKind,
    _now_iso8601,
)
from .metrics import ExecutiveIntegrationMetricsCollector


# ──────────────────────────────────────────────────────────────────────
# ExecutiveIntegrationRouter
# ──────────────────────────────────────────────────────────────────────


class ExecutiveIntegrationRouter:
    """Top-level router. Decides the route kind for a user message.

    Cardinal rules:
      * No LLM. No provider. No network. No subprocess.
      * No DB write. No new state. No commit.
    """

    SCHEMA_VERSION = "eil.v1"

    def __init__(
        self,
        *,
        gateway: Optional[ObjectiveGateway] = None,
        metrics: Optional[ExecutiveIntegrationMetricsCollector] = None,
        intent_router: Any = None,
        policy_engine: Any = None,
    ) -> None:
        self._gateway = gateway or ObjectiveGateway(
            intent_router=intent_router,
            policy_engine=policy_engine,
        )
        self._metrics = metrics or ExecutiveIntegrationMetricsCollector()

    # ── public ────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return self._gateway.is_enabled()

    @property
    def metrics(self) -> ExecutiveIntegrationMetricsCollector:
        return self._metrics

    def route(
        self,
        user_message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> ObjectiveGatewayDecision:
        """Decide the route kind. Read-only. Idempotent."""
        t0 = time.monotonic()
        decision = self._gateway.route(user_message, context=context)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._metrics.record_route(decision, elapsed_ms=elapsed_ms)
        return decision


__all__ = [
    "ExecutiveIntegrationRouter",
]
