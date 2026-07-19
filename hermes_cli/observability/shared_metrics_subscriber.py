"""Relay subscriber for the persisted Hermes shared-metrics slice."""

from __future__ import annotations

import logging
from typing import Any

from .shared_metrics import SharedMetricsStore
from .shared_metrics_contract import model_call_dimensions

logger = logging.getLogger(__name__)


class SharedMetricsSubscriber:
    """Persist validated primary model-call counters from Relay events."""

    def __init__(self, store: SharedMetricsStore, hermes_version: str) -> None:
        self.store = store
        self._hermes_version = hermes_version or "unknown"

    def __call__(self, event: Any) -> None:
        dimensions = model_call_dimensions(event)
        if dimensions is None:
            return
        try:
            self.store.record_model_call(dimensions, self._hermes_version)
        except Exception:
            logger.warning(
                "Unable to persist the Hermes model-call metric",
                exc_info=True,
            )
