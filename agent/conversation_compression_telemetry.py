"""Content-free per-attempt compression telemetry (attempt log line, Relay mark, shared metric).

Sibling of ``agent/conversation_compression.py`` (the facade), which imports it at module level; this
module must never import the facade (import cycle). It logs under the facade's logger name so log
consumers keep one source.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from typing import Any

from agent.compaction_events import publish_attempt

logger = logging.getLogger("agent.conversation_compression")


# Caller-side abort verdicts that only restate an outcome the compressor already classified more precisely:
# ``no_progress`` ("the transcript came back unchanged") covers the structural no-ops
# (``no_compressible_window``, ``insufficient_messages``, ``empty_post_handoff_window``) and
# ``summary_generation_aborted`` covers the terminal summary failures (``summary_auth_failure``,
# ``summary_overload_failure``, ...). The emitter keeps the compressor's class for these so the attempt log
# says WHY, not just THAT (#131412). Every other caller label names an event the compressor cannot see
# (fence cancelled, superseded, rollback, pool saturated, ...) and still wins.
_GENERIC_ABORT_VERDICTS = frozenset({"no_progress", "summary_generation_aborted"})


def _snapshot_compression_effect_input(messages: list, verbatim_tail: list | None) -> list | None:
    """Content-only snapshot for final effect accounting; telemetry must never break compression."""
    try:
        if not verbatim_tail:
            return messages
        from hermes_cli.partial_compress import rejoin_compressed_head_and_tail

        return rejoin_compressed_head_and_tail(messages, copy.deepcopy(verbatim_tail))
    except Exception:
        logger.debug("failed to snapshot compression effect input", exc_info=True)
        return None


def _record_committed_attempt_effect(
    agent: Any, messages_before: list | None, messages_after: list, verbatim_tail: list | None, split_status: str,
) -> None:
    """Finalize effect fields from the transcript shape that crossed the commit boundary."""
    if messages_before is None:
        return
    try:
        if verbatim_tail and split_status == "not_applicable":
            from hermes_cli.partial_compress import rejoin_compressed_head_and_tail

            # Without a SessionDB, the manual-compression caller performs this rejoin after compress_context returns.
            messages_after = rejoin_compressed_head_and_tail(messages_after, verbatim_tail)
        recorder = getattr(agent.context_compressor, "_record_committed_compression_effect", None)
        if callable(recorder):
            recorder(messages_before, messages_after)
    except Exception:
        logger.debug("failed to record committed compression effect", exc_info=True)


def _attempt_seed(
    agent: Any, *, attempt_began: bool = True, attempt_seed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """This attempt's id, the session it started in, and its trigger.

    ``attempt_seed`` is the attempt's own copy and wins: the agent's current-attempt fields belong to the newest
    attempt, so a stalled attempt unwinding after its fallback began would otherwise log the fallback's identity.
    An emit with no attempt begun (pool saturation) gets a fresh id and an unknown trigger."""
    if attempt_seed:
        return dict(attempt_seed)
    attempt_id = getattr(agent, "_compression_attempt_id", None) if attempt_began else None
    return {
        "attempt_id": attempt_id or uuid.uuid4().hex, "session_id": getattr(agent, "session_id", "") or "",
        "trigger_source": "unknown",
    }


def _log_attempt_record(payload: dict[str, Any]) -> None:
    """The one attempt log line; ``scripts/micro_compaction_report.py`` parses this prefix."""
    logger.info("context compression attempt telemetry: %s", json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _emit_compression_attempt_telemetry(
    agent: Any, *, started_at: float, commit_status: str, split_status: str, failure_class: str | None = None,
    commit_started_at: float | None = None, include_last_telemetry: bool = True,
    attempt_seed: dict[str, Any] | None = None, history_rewritten: bool | None = None,
) -> None:
    """Emit one content-free JSON log line for a compression attempt.

    ``include_last_telemetry=False`` is for emits that fire without an attempt having begun
    (pool-saturation refusals): they must not hydrate from the previous attempt's numbers.
    ``attempt_seed`` is the emitting attempt's own seed (see ``_attempt_seed``).
    ``history_rewritten`` (commit path only) says whether the returned transcript replaced the input, which
    a split that failed after publishing the compacted history still did."""
    try:
        compressor = agent.context_compressor
        telemetry = getattr(compressor, "_last_compression_telemetry", None) if include_last_telemetry else None
        seed = _attempt_seed(agent, attempt_began=include_last_telemetry, attempt_seed=attempt_seed)
        own = isinstance(telemetry, dict) and telemetry.get("attempt_id") == seed["attempt_id"]
        if not own:
            # The attempt-start clear leaves no dict before compress() seeds one, a pre-commit restore puts
            # the previous attempt's back, and a newer attempt may have seeded its own: describe THIS attempt
            # from its seed, never another's numbers.
            telemetry = {**seed, "method": "none"}
        payload = dict(telemetry)
        payload.setdefault("event", "compression_attempt")
        payload.setdefault("route", "hermes")
        payload.setdefault("attempt_id", seed["attempt_id"])
        payload.setdefault("session_id", seed["session_id"])
        payload.update(
            total_duration_ms=int((time.monotonic() - started_at) * 1000), commit_status=commit_status,
            split_status=split_status,
        )
        if history_rewritten is not None:
            payload["history_rewritten"] = history_rewritten
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
        _log_attempt_record(payload)
        publish_attempt(agent, payload)
        from hermes_cli.observability.shared_metrics_events import finish_compression_attempt

        finish_compression_attempt(
            commit_status, payload.get("failure_class"), getattr(agent.context_compressor, "context_length", None), agent=agent,
        )
    except Exception as exc:
        logger.debug("failed to emit compression attempt telemetry: %s", exc, exc_info=True)


def _emit_aborted_attempt_telemetry(
    agent: Any, started_at: float, failure_class: str | None, attempt_seed: dict[str, Any] | None = None,
) -> None:
    _emit_compression_attempt_telemetry(
        agent, started_at=started_at, commit_status="aborted", split_status="aborted", failure_class=failure_class,
        attempt_seed=attempt_seed,
    )


def _emit_bypassed_attempt_telemetry(
    agent: Any, started_at: float, *, commit_status: str, failure_class: str | None, approx_tokens: Any,
    route: str = "hermes", method: str = "none", attempt_seed: dict[str, Any] | None = None,
) -> None:
    """Log one attempt that never reached the local compressor: an automatic gate blocked it, Codex owns the
    thread, or the session lease showed another path already owns the work. The compressor's telemetry still
    describes an earlier attempt, so this record starts from the attempt seed. Shared metrics have never
    counted these exits and still do not."""
    try:
        compressor = getattr(agent, "context_compressor", None)
        payload = {
            "event": "compression_attempt", "route": route, "method": method, "failure_class": failure_class,
            **_attempt_seed(agent, attempt_seed=attempt_seed),
            "main_provider": getattr(agent, "provider", "") or "", "main_model": getattr(agent, "model", "") or "",
            # Private caches only: the public properties can trigger a synchronous context-length probe.
            "main_context_limit": getattr(compressor, "_resolved_context_length", None),
            "effective_threshold": getattr(compressor, "_threshold_tokens", None),
            "current_estimated_tokens": approx_tokens if isinstance(approx_tokens, int) else None,
            "total_duration_ms": int((time.monotonic() - started_at) * 1000), "commit_status": commit_status,
            "split_status": "not_applicable", "fallback_used": False,
        }
        _log_attempt_record(payload)
        publish_attempt(agent, payload)
    except Exception as exc:
        logger.debug("failed to emit compression attempt telemetry: %s", exc, exc_info=True)
    finally:
        # compress_context opened the thread-local shared-metric lifecycle before reaching this exit.
        # Bypassed attempts are log-only, so retire that lifecycle explicitly; otherwise an unrelated
        # pool-saturation emit on the same thread can consume its trigger and token count.
        try:
            from hermes_cli.observability.shared_metrics_events import discard_compression_attempt

            discard_compression_attempt()
        except Exception:
            logger.debug("failed to discard bypassed compression metric", exc_info=True)


def _emit_blocked_attempt_telemetry(
    agent: Any, started_at: float, approx_tokens: Any, attempt_seed: dict[str, Any] | None = None,
) -> None:
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
        agent, started_at, commit_status="blocked", failure_class=f"blocked:{guard}", approx_tokens=approx_tokens,
        attempt_seed=attempt_seed,
    )
