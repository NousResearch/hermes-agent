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


def _attempt_seed(agent: Any, *, attempt_began: bool = True) -> dict[str, Any]:
    """This attempt's id, the session it started in, and its trigger.

    Read from the agent's copy: a pre-commit restore puts the previous attempt's seed back on the compressor.
    An emit with no attempt begun (pool saturation) gets a fresh id and an unknown trigger."""
    attempt_id = getattr(agent, "_compression_attempt_id", None) if attempt_began else None
    seed = getattr(agent, "_compression_attempt_seed", None)
    if attempt_id and isinstance(seed, dict) and seed.get("attempt_id") == attempt_id:
        return dict(seed)
    return {
        "attempt_id": attempt_id or uuid.uuid4().hex, "session_id": getattr(agent, "session_id", "") or "",
        "trigger_source": "unknown",
    }


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
        own = isinstance(telemetry, dict) and telemetry.get("attempt_id") == getattr(agent, "_compression_attempt_id", None)
        if not own:
            # The attempt-start clear leaves no dict before compress() seeds one, and a pre-commit restore puts
            # the previous attempt's back: describe THIS attempt from its seed, never another's numbers.
            telemetry = {**_attempt_seed(agent, attempt_began=include_last_telemetry), "method": "none"}
        payload = dict(telemetry)
        payload.setdefault("event", "compression_attempt")
        payload.setdefault("route", "hermes")
        payload.setdefault("attempt_id", getattr(agent, "_compression_attempt_id", "") or uuid.uuid4().hex)
        payload.setdefault("session_id", getattr(agent, "session_id", "") or "")
        payload.update(
            total_duration_ms=int((time.monotonic() - started_at) * 1000), commit_status=commit_status,
            split_status=split_status,
        )
        if commit_started_at is not None:
            telemetry["commit_ms"] = payload["commit_ms"] = max(0, int((time.monotonic() - commit_started_at) * 1000))
        # Defer only to THIS attempt's class: an abort restore can put the previous attempt's telemetry back.
        if failure_class and not (failure_class in _GENERIC_ABORT_VERDICTS and own and payload.get("failure_class")):
            payload["failure_class"] = failure_class
        payload.setdefault("chunking", False)
        payload.setdefault("chunk_count", 0)
        # The compressor's fallback flags are this attempt's only when its telemetry is.
        payload["fallback_used"] = bool(
            payload.get("fallback_used")
            or own and (
                getattr(compressor, "_last_summary_fallback_used", False)
                or getattr(compressor, "_last_aux_model_failure_model", None)
            )
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


def _emit_bypassed_attempt_telemetry(
    agent: Any, started_at: float, *, commit_status: str, failure_class: str | None, approx_tokens: Any,
    route: str = "hermes", method: str = "none",
) -> None:
    """Log one attempt that never reached the local compressor: an automatic gate blocked it, Codex owns the
    thread, or the session lease showed another path already owns the work. The compressor's telemetry still
    describes an earlier attempt, so this record starts from the attempt seed. Shared metrics have never
    counted these exits and still do not."""
    try:
        compressor = getattr(agent, "context_compressor", None)
        payload = {
            "event": "compression_attempt", "route": route, "method": method, "failure_class": failure_class,
            **_attempt_seed(agent),
            "main_provider": getattr(agent, "provider", "") or "", "main_model": getattr(agent, "model", "") or "",
            # Private caches only: the public properties can trigger a synchronous context-length probe.
            "main_context_limit": getattr(compressor, "_resolved_context_length", None),
            "effective_threshold": getattr(compressor, "_threshold_tokens", None),
            "current_estimated_tokens": approx_tokens if isinstance(approx_tokens, int) else None,
            "total_duration_ms": int((time.monotonic() - started_at) * 1000), "commit_status": commit_status,
            "split_status": "not_applicable", "fallback_used": False,
        }
        logger.info(
            "context compression attempt telemetry: %s", json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
    except Exception as exc:
        logger.debug("failed to emit compression attempt telemetry: %s", exc, exc_info=True)


def _emit_blocked_attempt_telemetry(agent: Any, started_at: float, approx_tokens: Any) -> None:
    """Record an automatic attempt the breaker gate refused. The class keeps only the guard's name
    (``blocked:cooldown``, ``blocked:structural_backoff``, ``blocked:ineffective``), never its seconds."""
    reason = None
    try:
        reason_fn = getattr(getattr(agent, "context_compressor", None), "_compression_block_reason", None)
        reason = reason_fn() if callable(reason_fn) else None
    except Exception:
        logger.debug("compression block-reason read failed", exc_info=True)
    guard = reason.split(":", 1)[0] if isinstance(reason, str) and reason else "unknown"
    _emit_bypassed_attempt_telemetry(
        agent, started_at, commit_status="blocked", failure_class=f"blocked:{guard}", approx_tokens=approx_tokens
    )
