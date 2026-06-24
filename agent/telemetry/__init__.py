"""Hermes telemetry & observability.

Local-first observability, on by default. The ``telemetry`` plugin registers Hermes
lifecycle hooks and hands typed events to the fire-and-forget ``emitter`` (queue ->
background writer -> JSONL + state.db ``tel_*`` index). The emitter never blocks or
raises into a model/tool call (the hot-path invariant).

Events record the observed model ids, provider names, and tool names. ``metrics``
derives rollups for /usage and /insights; ``rollup`` builds the per-run summaries shown
by ``hermes telemetry preview``. ``redaction`` + ``exporter_bulk`` + ``otlp_exporter``
handle export to an operator-chosen destination. ``policy`` holds the consent state
machine for the opt-in aggregate plane (no uploader ships).
"""

from __future__ import annotations

from . import emitter, events, metrics, policy, spans

emit = emitter.emit
get_emitter = emitter.get_emitter

__all__ = [
    "emitter",
    "events",
    "metrics",
    "policy",
    "spans",
    "emit",
    "get_emitter",
]
