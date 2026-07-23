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
    CLIENT_FIRST_USABLE_METRIC,
    MODEL_CALL_METRIC,
    TASK_FINISHED_METRIC,
    TASK_STARTED_METRIC,
    TOOL_CALL_METRIC,
    client_active_counter,
    client_lifecycle_counter,
    client_resource,
    model_call_dimensions,
    skill_counter,
    task_counter,
    tool_approval_counter,
    tool_call_dimensions,
)

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

    def __call__(self, event: Any) -> None:
        if self._runtime_id is not None:
            metadata = getattr(event, "metadata", None)
            if (
                not isinstance(metadata, dict)
                or metadata.get(RUNTIME_INSTANCE_KEY) != self._runtime_id
            ):
                return
        metric = client_active_counter(event)
        dimensions = None
        metric_name = CLIENT_ACTIVE_METRIC
        if metric is not None:
            metric_name, dimensions = metric
        if dimensions is None:
            metric = client_lifecycle_counter(event)
            if metric is not None:
                metric_name, dimensions = metric
        if dimensions is None:
            dimensions = model_call_dimensions(event)
            metric_name = MODEL_CALL_METRIC
        if dimensions is None:
            dimensions = tool_call_dimensions(event)
            metric_name = TOOL_CALL_METRIC
        if dimensions is None:
            metric = (
                task_counter(event)
                or tool_approval_counter(event)
                or skill_counter(event)
            )
            if metric is None:
                return
            metric_name, dimensions = metric
        with self._lock:
            if not self._active:
                return
            try:
                if metric_name == CLIENT_ACTIVE_METRIC:
                    self.store.record_client_active(self._client_resource)
                elif metric_name == CLIENT_FIRST_USABLE_METRIC:
                    self.store.record_first_usable(self._client_resource)
                elif metric_name in {TASK_STARTED_METRIC, TASK_FINISHED_METRIC}:
                    self.store.record_task_counter(
                        metric_name,
                        dimensions,
                        self._client_resource,
                    )
                else:
                    self.store.record_counter(
                        metric_name,
                        dimensions,
                        self._client_resource,
                    )
            except Exception:
                logger.warning(
                    "Unable to persist the Hermes shared metric: %s",
                    metric_name,
                    exc_info=True,
                )
