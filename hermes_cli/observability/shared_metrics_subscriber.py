"""Relay subscriber for the persisted Hermes shared-metrics slice."""

from __future__ import annotations

import logging
import threading
from typing import Any

from agent.relay_runtime import RUNTIME_INSTANCE_KEY

from .shared_metrics import SharedMetricsStore
from .shared_metrics_contract import MODEL_CALL_METRIC, model_call_dimensions, task_counter

logger = logging.getLogger(__name__)


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
        self._hermes_version = hermes_version or "unknown"
        self._runtime_id = runtime_id
        self._active = True
        self._lock = threading.RLock()

    def deactivate(self) -> None:
        """Stop accepting events before telemetry is disabled or torn down."""
        with self._lock:
            self._active = False

    def __call__(self, event: Any) -> None:
        if self._runtime_id is not None:
            metadata = getattr(event, "metadata", None)
            if (
                not isinstance(metadata, dict)
                or metadata.get(RUNTIME_INSTANCE_KEY) != self._runtime_id
            ):
                return
        dimensions = model_call_dimensions(event)
        metric_name = MODEL_CALL_METRIC
        if dimensions is None:
            task_metric = task_counter(event)
            if task_metric is None:
                return
            metric_name, dimensions = task_metric
        with self._lock:
            if not self._active:
                return
            try:
                self.store.record_counter(
                    metric_name,
                    dimensions,
                    self._hermes_version,
                )
            except Exception:
                logger.warning(
                    "Unable to persist the Hermes shared metric: %s",
                    metric_name,
                    exc_info=True,
                )
