"""Content-free per-attempt compression telemetry (attempt log line + shared metric).

Sibling of ``agent/conversation_compression.py`` (the facade), which imports it at module level; this
module must never import the facade (import cycle). It logs under the facade's logger name so log
consumers keep one source.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("agent.conversation_compression")


# Caller-side abort verdicts that only restate an outcome the compressor already classified more precisely:
# ``no_progress`` ("the transcript came back unchanged") covers the structural no-ops
# (``no_compressible_window``, ``insufficient_messages``, ``empty_post_handoff_window``) and
# ``summary_generation_aborted`` covers the terminal summary failures (``summary_auth_failure``,
# ``summary_overload_failure``, ...). The emitter keeps the compressor's class for these so the attempt log
# says WHY, not just THAT (#131412). Every other caller label names an event the compressor cannot see
# (fence cancelled, superseded, rollback, pool saturated, ...) and still wins.
_GENERIC_ABORT_VERDICTS = frozenset({"no_progress", "summary_generation_aborted"})


def _emit_compression_attempt_telemetry(
    agent: Any, *, started_at: float, commit_status: str, split_status: str, failure_class: str | None = None,
    commit_started_at: float | None = None, include_last_telemetry: bool = True,
) -> None:
    """Emit one content-free JSON log line for a compression attempt.

    ``include_last_telemetry=False`` is for emits that fire without an attempt having begun
    (pool-saturation refusals): they must not hydrate from the previous attempt's numbers."""
    try:
        compressor = agent.context_compressor
        telemetry = getattr(compressor, "_last_compression_telemetry", None) if include_last_telemetry else None
        if not isinstance(telemetry, dict):
            telemetry = {}
        payload = dict(telemetry)
        payload.setdefault("event", "compression_attempt")
        payload.setdefault("attempt_id", getattr(agent, "_compression_attempt_id", "") or uuid.uuid4().hex)
        payload.setdefault("session_id", getattr(agent, "session_id", "") or "")
        payload.update(
            total_duration_ms=int((time.monotonic() - started_at) * 1000), commit_status=commit_status,
            split_status=split_status,
        )
        if commit_started_at is not None:
            telemetry["commit_ms"] = payload["commit_ms"] = max(0, int((time.monotonic() - commit_started_at) * 1000))
        # Defer only to THIS attempt's class: an abort restore can put the previous attempt's telemetry back.
        _own_class = telemetry.get("attempt_id") == getattr(agent, "_compression_attempt_id", None)
        if failure_class and not (failure_class in _GENERIC_ABORT_VERDICTS and _own_class and payload.get("failure_class")):
            payload["failure_class"] = failure_class
        payload.setdefault("chunking", False)
        payload.setdefault("chunk_count", 0)
        payload["fallback_used"] = bool(
            payload.get("fallback_used")
            or getattr(compressor, "_last_summary_fallback_used", False)
            or getattr(compressor, "_last_aux_model_failure_model", None)
        )
        logger.info(
            "context compression attempt telemetry: %s", json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        from hermes_cli.observability.shared_metrics_events import finish_compression_attempt

        finish_compression_attempt(
            commit_status, payload.get("failure_class"), getattr(agent.context_compressor, "context_length", None), agent=agent,
        )
    except Exception as exc:
        logger.debug("failed to emit compression attempt telemetry: %s", exc, exc_info=True)


def _emit_aborted_attempt_telemetry(agent: Any, started_at: float, failure_class: str | None) -> None:
    _emit_compression_attempt_telemetry(
        agent, started_at=started_at, commit_status="aborted", split_status="aborted", failure_class=failure_class
    )
