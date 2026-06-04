"""Event Emitter — validated event dispatch with allowlist checking.

Wires @agrv/hermes-events EVENT_ALLOWLIST into Python submodules.
Mirrors the TypeScript allowlist: 23 canonical events scoped by subsystem.

Event emission mechanism: injectable sinks (default: stdout JSON lines).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Optional


# Allowlist sourced from the generated agent/events_allowlist.py, which mirrors
# packages/hermes-events/src/allowlist.ts (ATTI AI extended set incl. Baumbad events).
# Falls back to the hardcoded baseline if the generated file is absent.
try:
    from agent.events_allowlist import EVENT_ALLOWLIST as _GENERATED_ALLOWLIST
    EVENT_ALLOWLIST: set = set(_GENERATED_ALLOWLIST)
except ImportError:
    EVENT_ALLOWLIST = {
        "hermes.identity.bootstrap",
        "hermes.context.assembled",
        "hermes.interp.done",
        "hermes.intent.classified",
        "hermes.mission.compiled",
        "hermes.route.dispatched",
        "hermes.response.shaped",
        "hermes.summary.emitted",
        "mission.submitted",
        "specialist.dispatch.started",
        "specialist.dispatch.completed",
        "specialist.dispatch.failed",
        "write.payload.created",
        "write.payload.updated",
        "write.payload.published",
        "write.medusa.created",
        "write.medusa.updated",
        "write.twenty.created",
        "write.twenty.updated",
        "approval.requested",
        "approval.granted",
        "approval.denied",
        "approval.timeout",
    }


class EventEmitter:
    """Validates and dispatches events to multiple sinks.

    Parameters
    ----------
    sinks : list[Callable[[dict], None]], optional
        List of functions that consume event payloads.
        Default: single sink that writes JSON lines to stdout.

    Raises
    ------
    ValueError
        If an event name is not in EVENT_ALLOWLIST.
    """

    def __init__(self, sinks: Optional[list[Callable[[dict], None]]] = None):
        """Initialize emitter with sinks.

        Parameters
        ----------
        sinks : list[Callable[[dict], None]], optional
            Event consumers. If None, uses stdout JSON line sink.
        """
        if sinks is None:
            sinks = [self._stdout_sink]
        self.sinks = sinks

    @staticmethod
    def _stdout_sink(payload: dict) -> None:
        """Write JSON event to stdout.

        Parameters
        ----------
        payload : dict
            Event object with 'event', 'payload', 'timestamp' keys.
        """
        line = json.dumps(payload, default=str)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def emit(self, event_name: str, payload: dict[str, Any]) -> None:
        """Emit a validated event to all sinks.

        Parameters
        ----------
        event_name : str
            Must be in EVENT_ALLOWLIST.
        payload : dict[str, Any]
            Event-specific data.

        Raises
        ------
        ValueError
            If event_name not in EVENT_ALLOWLIST.
        """
        if event_name not in EVENT_ALLOWLIST:
            raise ValueError(
                f"Unknown event '{event_name}'. "
                f"Allowed: {sorted(EVENT_ALLOWLIST)}"
            )

        event_obj = {
            "event": event_name,
            "payload": payload,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        for sink in self.sinks:
            sink(event_obj)
