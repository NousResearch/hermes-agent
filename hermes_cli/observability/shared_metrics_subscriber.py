"""Relay subscriber for the persisted Hermes shared-metrics slice."""

from __future__ import annotations

import logging
import platform
import threading
from typing import Any

from agent.relay_runtime import RUNTIME_INSTANCE_KEY
from hermes_cli.config import detect_install_method

from .shared_metrics import SharedMetricsStore
from .shared_metrics_contract import (
    CLIENT_ACTIVE_METRIC,
    MODEL_ROUTE_METRIC,
    TOOL_CALL_METRIC,
    client_active_counter,
    client_resource,
    model_call_dimensions,
    skill_counter,
    task_counter,
    tool_approval_counter,
    tool_call_dimensions,
)

logger = logging.getLogger(__name__)

# Contract projections in match order; each yields (metric_name, dimensions) or None.
_COUNTERS = (
    client_active_counter,
    lambda event: _named(MODEL_ROUTE_METRIC, model_call_dimensions(event)),
    lambda event: _named(TOOL_CALL_METRIC, tool_call_dimensions(event)),
    task_counter,
    tool_approval_counter,
    skill_counter,
)


def _named(metric_name: str, dimensions: dict | None) -> tuple[str, dict] | None:
    return None if dimensions is None else (metric_name, dimensions)


class SharedMetricsSubscriber:
    """Persist validated Hermes counters from Relay lifecycle events."""

    def __init__(
        self,
        store: SharedMetricsStore,
        hermes_version: str,
        *,
        runtime_id: str | None = None,
    ) -> None:
        self.store = store
        self._client_resource = client_resource(
            hermes_version,
            os_name=platform.system(),
            architecture=platform.machine(),
            install_method=detect_install_method(),
        )
        self._runtime_id = runtime_id
        self._active = True
        self._lock = threading.RLock()

    def deactivate(self) -> None:
        """Stop accepting events before telemetry is disabled or torn down."""
        with self._lock:
            self._active = False

    @staticmethod
    def _classify(event: Any) -> tuple[str, dict] | None:
        """Return ``(metric_name, dimensions)`` for the first matching contract, else None."""
        return next((m for m in (project(event) for project in _COUNTERS) if m is not None), None)

    def __call__(self, event: Any) -> None:
        if self._runtime_id is not None:
            metadata = getattr(event, "metadata", None)
            if (
                not isinstance(metadata, dict)
                or metadata.get(RUNTIME_INSTANCE_KEY) != self._runtime_id
            ):
                return
        metric = self._classify(event)
        if metric is None:
            return
        metric_name, dimensions = metric
        with self._lock:
            if not self._active:
                return
            try:
                if metric_name == CLIENT_ACTIVE_METRIC:
                    self.store.record_client_active(self._client_resource)
                else:
                    self.store.record_counter(metric_name, dimensions, self._client_resource)
            except Exception:
                logger.warning(
                    "Unable to persist the Hermes shared metric: %s", metric_name, exc_info=True
                )
