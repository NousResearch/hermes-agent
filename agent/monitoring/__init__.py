"""Hermes gateway monitoring: service health + redacted diagnostics over OTLP.

``emitter`` is the in-process event bus: producers hand typed events to a
fire-and-forget queue that never blocks or raises into gateway code, and OTLP
subscribers consume them off the hot path. Nothing is persisted locally.
Out of scope: trajectory capture, usage analytics, any content-bearing signal
(those belong to the NeMo Relay integration).
"""

from __future__ import annotations

from . import emitter, events

emit = emitter.emit
get_emitter = emitter.get_emitter

__all__ = ["emitter", "events", "emit", "get_emitter"]
