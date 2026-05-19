"""Public telemetry observer facade for Hermes integrations.

This module is the plugin-facing surface for consuming host-owned telemetry.
Hermes currently backs it with NeMo-Flow when the experimental NeMo-Flow bridge
is enabled and the stable ``nemo_flow.telemetry_v1`` API is available.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeAlias

HERMES_TELEMETRY_SCHEMA_VERSION = "hermes.observer.v1"
NEMO_FLOW_TELEMETRY_SCHEMA_VERSION = "nemo_flow.telemetry.v1"

ErrorPolicy: TypeAlias = Literal["log", "ignore"]
TelemetryEvent: TypeAlias = dict[str, Any]
TelemetryObserver: TypeAlias = Callable[[TelemetryEvent], None]


class TelemetrySubscription(Protocol):
    """Deregistration handle returned by telemetry observer registration."""

    name: str

    def deregister(self) -> bool:
        """Remove the observer from future event delivery."""
        ...

    def __enter__(self) -> "TelemetrySubscription": ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...


def is_enabled() -> bool:
    """Return whether Hermes is configured to mirror hooks into NeMo-Flow."""
    from hermes_cli.nemo_flow_telemetry import is_enabled as _is_enabled

    return _is_enabled()


def is_available() -> bool:
    """Return whether stable serialized telemetry events can be observed."""
    from hermes_cli.nemo_flow_telemetry import observer_available

    return observer_available()


def register_observer(
    name: str,
    callback: TelemetryObserver,
    *,
    error_policy: ErrorPolicy = "log",
) -> TelemetrySubscription:
    """Register a process-local observer for stable telemetry event dicts.

    If NeMo-Flow telemetry is disabled or the installed NeMo-Flow wheel does not
    yet provide ``nemo_flow.telemetry_v1``, Hermes returns a no-op subscription.
    Observer callback exceptions are isolated by the underlying NeMo-Flow
    telemetry facade according to ``error_policy``.
    """
    from hermes_cli.nemo_flow_telemetry import register_observer as _register

    return _register(name, callback, error_policy=error_policy)


def observer(
    name: str,
    callback: TelemetryObserver,
    *,
    error_policy: ErrorPolicy = "log",
) -> TelemetrySubscription:
    """Context-manager alias for :func:`register_observer`."""
    return register_observer(name, callback, error_policy=error_policy)


__all__ = [
    "ErrorPolicy",
    "HERMES_TELEMETRY_SCHEMA_VERSION",
    "NEMO_FLOW_TELEMETRY_SCHEMA_VERSION",
    "TelemetryEvent",
    "TelemetryObserver",
    "TelemetrySubscription",
    "is_available",
    "is_enabled",
    "observer",
    "register_observer",
]
