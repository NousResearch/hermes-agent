"""Hermes gateway monitoring.

Service health monitoring plus redacted operational diagnostics for the
gateway daemon, exported over OTLP to an operator-configured endpoint.

``emitter`` is the in-process event bus: producers (gateway status hooks,
the diagnostic log handler) hand typed events to a fire-and-forget queue,
and subscribers (the OTLP streamers) consume them off the hot path. The
emitter never blocks or raises into gateway code (the hot-path invariant),
and nothing is persisted locally — monitoring is an egress path, not a store.

Deliberately out of scope here: run/model/tool trajectory capture, usage
analytics, and any content-bearing signal. Those planes are served by the
NeMo Relay integration and its Hermes-owned subscribers.
"""

from __future__ import annotations

from . import emitter, events

emit = emitter.emit
get_emitter = emitter.get_emitter

__all__ = [
    "emitter",
    "events",
    "emit",
    "get_emitter",
]
