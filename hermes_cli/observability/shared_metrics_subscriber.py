"""Relay subscriber for the persisted Hermes shared-metrics slice."""

from __future__ import annotations

import logging
import platform
import threading
from typing import Any

from agent.relay_runtime import RUNTIME_INSTANCE_KEY
from hermes_cli.config import detect_install_method
from hermes_constants import get_hermes_home

from .shared_metrics import SharedMetricsStore
from .shared_metrics_fields import milestones_for
from .shared_metrics_contract import (
    CLIENT_ACTIVE_METRIC,
    COMMIT_TICKET_KEY,
    INSTALL_SNAPSHOT_METRIC,
    MODEL_ROUTE_METRIC,
    TOOL_CALL_METRIC,
    TOOL_USAGE_METRIC,
    client_active_counter,
    client_resource,
    decision_counter,
    install_snapshot_counter,
    model_call_dimensions,
    model_token_counters,
    skill_counter,
    task_counter,
    tool_approval_counter,
    tool_call_dimensions,
    tool_usage_dimensions,
)

logger = logging.getLogger(__name__)

# Contract projections; each yields (metric_name, dimensions) or None. One tool end event feeds
# both the category counter and the per-tool counter, so every match is recorded.
_COUNTERS = (
    client_active_counter,
    install_snapshot_counter,
    lambda event: _named(MODEL_ROUTE_METRIC, model_call_dimensions(event)),
    lambda event: _named(TOOL_CALL_METRIC, tool_call_dimensions(event)),
    lambda event: _named(TOOL_USAGE_METRIC, tool_usage_dimensions(event)),
    task_counter,
    tool_approval_counter,
    skill_counter,
    decision_counter,
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
        self._milestones_done: set[str] = set(store.recorded_milestones())
        self._saved_tickets: dict[str, int] = {}
        # Events arrive on the Relay thread, which carries no profile binding.
        self._hermes_home = get_hermes_home()

    def deactivate(self) -> None:
        """Stop accepting events before telemetry is disabled or torn down."""
        with self._lock:
            self._active = False

    @staticmethod
    def _classify(event: Any) -> list[tuple[str, dict, int]]:
        """Return every ``(metric_name, dimensions, amount)`` the event satisfies."""
        counted = [(*m, 1) for m in (project(event) for project in _COUNTERS) if m is not None]
        return counted + model_token_counters(event)

    def _record_milestones(self, metric_name: str, dimensions: dict) -> None:
        """Latch every install milestone this counter reaches (each once per install, ever)."""
        reached = [m for m in milestones_for(metric_name, dimensions) if m not in self._milestones_done]
        if not reached:
            return
        from .shared_metrics_snapshot import install_age_bucket

        age = install_age_bucket(self._hermes_home)
        for milestone in reached:
            self.store.record_milestone(milestone, age, self._client_resource)
            self._milestones_done.add(milestone)

    def take_saved(self, ticket: str) -> int:
        """How many events carrying ``ticket`` settled without a store error (and forget the ticket)."""
        with self._lock:
            return self._saved_tickets.pop(ticket, 0)

    def __call__(self, event: Any) -> None:
        metadata = getattr(event, "metadata", None)
        if self._runtime_id is not None:
            if (
                not isinstance(metadata, dict)
                or metadata.get(RUNTIME_INSTANCE_KEY) != self._runtime_id
            ):
                return
        ticket = metadata.get(COMMIT_TICKET_KEY) if isinstance(metadata, dict) else None
        saved = True  # a row the contract rejects is settled too: no retry can change that
        for metric_name, dimensions, amount in self._classify(event):
            with self._lock:
                if not self._active:
                    return
                try:
                    if metric_name == CLIENT_ACTIVE_METRIC:
                        self.store.record_client_active(self._client_resource)
                    elif metric_name == INSTALL_SNAPSHOT_METRIC:
                        self.store.record_install_snapshot(dimensions, self._client_resource)
                    else:
                        self.store.record_counter(metric_name, dimensions, self._client_resource, amount)
                    self._record_milestones(metric_name, dimensions)
                except Exception:
                    saved = False
                    logger.warning(
                        "Unable to persist the Hermes shared metric: %s", metric_name, exc_info=True
                    )
        if ticket and saved:
            with self._lock:
                if self._active:
                    self._saved_tickets[ticket] = self._saved_tickets.get(ticket, 0) + 1
