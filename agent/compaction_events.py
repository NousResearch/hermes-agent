"""Content-free ``hermes.compaction`` v1 records for every history rewrite, and their fan-out.

One record per ``compress_context`` attempt (built from the attempt telemetry seam), per
micro-compaction pass, per committed proactive tool-result prune, and per Codex-native thread
compaction Hermes observes. A persistence-detached fork (background review) shares the live session id
but rewrites only its own throwaway transcript, so it publishes nothing. The payload is flat because OpenTelemetry exporters flatten only top-level
keys; every categorical field comes from a closed set with an ``other`` fallback; no field carries
message, summary, focus-topic, error or path text. Publishing never raises into compaction.
"""

from __future__ import annotations

import functools
import logging
import threading
import uuid
from typing import Any

from agent.model_metadata import estimate_messages_tokens_rough

logger = logging.getLogger(__name__)

SCHEMA_NAME = "hermes.compaction"
SCHEMA_VERSION = "1"
# Relay's reserved mark name: it resets the owning agent scope's LLM-history freshness, so it is used only
# for a committed rewrite of what the model will see next. Everything else is an ordinary attempt mark.
COMPACTION_MARK = "compaction"
ATTEMPT_MARK = "compaction.attempt"

# The v1 sets list only values this module emits. Later versions may add request-scope truncation kinds.
_KINDS = frozenset({"summarize", "micro_summarize", "prune_tool_results", "provider_native"})
_SCOPES = frozenset({"history", "provider"})
_METHODS = frozenset({
    "llm_summary", "aux_fallback_main", "deterministic_fallback", "deterministic_prune", "provider", "none",
})
_TRIGGERS = frozenset({
    "manual", "turn_start_threshold", "pre_api", "post_tool", "idle", "engine_preflight", "overflow",
    "gateway_hygiene", "between_turns", "proactive_prune", "provider", "unknown",
})
# Every other known trigger is automatic; an unknown or unlisted one asserts nothing.
_TRIGGER_CLASSES = {
    "manual": "manual", "overflow": "overflow", "provider": "provider", "unknown": "unknown", "other": "unknown",
}
_TRIGGER_CLASS_VALUES = frozenset({"auto", "manual", "overflow", "provider", "unknown"})
# The error-classifier reasons whose recovery compresses with ``trigger="overflow"``.
_OVERFLOW_REASONS = frozenset({"context_overflow", "payload_too_large", "long_context_tier"})
_OUTCOMES = frozenset({"committed", "aborted", "failed", "skipped", "blocked"})
_SPLIT_STATUSES = frozenset({
    "not_applicable", "in_place_committed", "rotated_committed", "failed_not_indexed", "aborted",
})
_TOKEN_COUNT_METHODS = frozenset({"estimate_rough"})
# Attempt classes beyond the shared-metrics closed set: guards, Codex route exits, lease races, and the
# micro-compaction pass outcomes.
_EXTRA_FAILURE_CLASSES = frozenset({
    "blocked:cooldown", "blocked:structural_backoff", "blocked:ineffective", "blocked:unknown",
    "codex_auto_native", "codex_auto_off", "codex_no_thread", "codex_compaction_failed",
    "session_ownership_lost", "session_ownership_unreadable", "cooldown_state_unreadable", "summary_model_benched",
    "summarize_failed", "exchange_skipped", "defrag_failed", "stale_generation", "watermark_unavailable",
})
# An aborted attempt the compressor ran but deliberately did not commit; other aborts are failures.
_ABORTED_CLASSES = frozenset({"would_grow", "no_progress", "commit_fence_cancelled"})
_COMMIT_MODES = frozenset({"in_place_committed", "rotated_committed"})
_MICRO_OUTCOMES = {
    "absorbed": "committed", "defrag": "committed", "stale_generation": "skipped", "watermark_unavailable": "skipped",
}

# Every v1 key, so consumers see a stable shape; unknown values stay null.
_FIELDS = (
    "harness", "kind", "scope", "official", "method", "trigger", "trigger_class", "overflow_reason", "outcome",
    "failure_class", "attempt_id", "session_id", "tokens_before", "tokens_after", "tokens_reclaimed",
    "token_count_method", "messages_before", "messages_after", "items_dropped", "context_limit", "threshold_tokens",
    "protected_head_tokens", "middle_window_tokens", "protected_tail_tokens", "duration_ms",
    "summary_generation_ms", "aux_call_duration_ms", "queue_wait_ms", "commit_ms", "summarizer_provider",
    "summarizer_model", "tool_results_pruned", "reasoning_items_pruned", "summary_input_omitted_chars",
    "compression_count", "in_place", "session_rotated", "split_status", "cache_break", "has_focus_topic",
)
# Integer fields copied as-is from the attempt record (record key -> payload key).
_ATTEMPT_INTS = {
    "tokens_before": "tokens_before", "tokens_after": "tokens_after", "tokens_reclaimed": "tokens_reclaimed",
    "messages_before": "messages_before", "messages_after": "messages_after", "items_dropped": "items_dropped",
    "main_context_limit": "context_limit", "effective_threshold": "threshold_tokens",
    "protected_head_tokens": "protected_head_tokens", "middle_window_tokens": "middle_window_tokens",
    "protected_tail_tokens": "protected_tail_tokens", "total_duration_ms": "duration_ms",
    "summary_generation_ms": "summary_generation_ms", "aux_call_duration_ms": "aux_call_duration_ms",
    "queue_wait_ms": "queue_wait_ms", "commit_ms": "commit_ms", "tool_results_pruned": "tool_results_pruned",
    "reasoning_items_pruned": "reasoning_items_pruned", "summary_input_omitted_chars": "summary_input_omitted_chars",
}

_warned_lock = threading.Lock()
_warned = False


@functools.cache
def _failure_classes() -> frozenset[str]:
    from hermes_cli.observability.shared_metrics_contract import COMPRESSION_FAILURE_CLASSES

    return COMPRESSION_FAILURE_CLASSES | _EXTRA_FAILURE_CLASSES


@functools.cache
def _skip_classes() -> frozenset[str]:
    from hermes_cli.observability.shared_metrics_contract import COMPRESSION_SKIP_CLASSES

    return COMPRESSION_SKIP_CLASSES


def _closed(value: Any, allowed: frozenset[str], default: str | None = "other") -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value if value in allowed else default


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def failure_class(value: Any) -> str:
    """Closed failure class: ``exception:<Type>`` / ``rollback:<Type>`` keep only the prefix (the type name
    may be a plugin's); a missing class reads ``none``."""
    if not isinstance(value, str) or not value:
        return "none"
    if value.startswith(("exception:", "rollback:")):
        value = value.split(":", 1)[0]
    return value if value in _failure_classes() else "other"


def event_name(payload: dict[str, Any]) -> str:
    """``compaction`` only for a committed history or provider rewrite; ``compaction.attempt`` otherwise."""
    committed = payload.get("outcome") == "committed" and payload.get("scope") in {"history", "provider"}
    return COMPACTION_MARK if committed else ATTEMPT_MARK


def _payload(**values: Any) -> dict[str, Any]:
    payload = dict.fromkeys(_FIELDS)
    payload.update(values, harness="hermes")
    payload["kind"] = _closed(payload["kind"], _KINDS)
    payload["scope"] = _closed(payload["scope"], _SCOPES)
    payload["method"] = _closed(payload["method"], _METHODS) or "none"
    payload["trigger"] = _closed(payload["trigger"], _TRIGGERS) or "unknown"
    payload["trigger_class"] = (
        _closed(payload["trigger_class"], _TRIGGER_CLASS_VALUES, None) or _TRIGGER_CLASSES.get(payload["trigger"], "auto")
    )
    payload["outcome"] = _closed(payload["outcome"], _OUTCOMES) or "other"
    payload["token_count_method"] = _closed(payload["token_count_method"], _TOKEN_COUNT_METHODS)
    payload["attempt_id"] = payload["attempt_id"] or uuid.uuid4().hex
    payload["session_id"] = payload["session_id"] or ""
    if payload["cache_break"] is None and payload["scope"] == "history":
        payload["cache_break"] = payload["outcome"] == "committed"
    return payload


def _attempt_outcome(commit_status: Any, failure: str, history_rewritten: Any) -> str:
    # A split that failed after publishing the compacted history still rewrote what the model sees next: it is
    # a committed compaction whose failure_class records the failure.
    if history_rewritten is True:
        return "committed"
    if commit_status != "aborted":
        return commit_status if commit_status in _OUTCOMES else "other"
    if failure in _skip_classes():
        return "skipped"
    return "aborted" if failure in _ABORTED_CLASSES else "failed"


def _attempt_trigger(trigger_source: Any) -> tuple[Any, str | None]:
    """``(trigger, trigger_class)``. ``auto`` is what an unlabelled automatic caller records: the reason is
    unknown, but the attempt is still known to be automatic."""
    if trigger_source == "auto":
        return "unknown", "auto"
    return trigger_source or "unknown", None


def attempt_payload(record: dict[str, Any], *, compression_count: Any = None) -> dict[str, Any]:
    """Map one attempt telemetry record (the attempt log line's dict) onto the v1 schema."""
    codex = record.get("route") == "codex_app_server"
    failure = failure_class(record.get("failure_class"))
    split_status = _closed(record.get("split_status"), _SPLIT_STATUSES)
    outcome = _attempt_outcome(record.get("commit_status"), failure, record.get("history_rewritten"))
    trigger, trigger_class = _attempt_trigger(record.get("trigger_source"))
    values: dict[str, Any] = {key: _int(record.get(src)) for src, key in _ATTEMPT_INTS.items()}
    values.update(
        kind="provider_native" if codex else "summarize", scope="provider" if codex else "history", official=True,
        method=record.get("method"), trigger=trigger, trigger_class=trigger_class,
        overflow_reason=_closed(record.get("overflow_reason"), _OVERFLOW_REASONS), outcome=outcome,
        failure_class=failure,
        attempt_id=record.get("attempt_id"), session_id=record.get("session_id"),
        token_count_method=record.get("token_count_method"),
        summarizer_provider=record.get("aux_provider") or None, summarizer_model=record.get("aux_model") or None,
        compression_count=_int(compression_count), split_status=split_status,
        # Commit mode is known only for a fully committed split; anything else leaves both null.
        in_place=split_status == "in_place_committed" if split_status in _COMMIT_MODES else None,
        session_rotated=split_status == "rotated_committed" if split_status in _COMMIT_MODES else None,
        has_focus_topic=record.get("has_focus_topic") if isinstance(record.get("has_focus_topic"), bool) else None,
    )
    if codex:
        values["cache_break"] = None  # the Codex thread's provider cache is not Hermes's prompt cache
    return _payload(**values)


def micro_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Map one micro-compaction telemetry record onto the v1 schema."""
    outcome = _MICRO_OUTCOMES.get(record.get("outcome"), "failed")
    before, after = _int(record.get("tokens_before")), _int(record.get("tokens_after"))
    return _payload(
        kind="micro_summarize", scope="history", official=False, trigger="between_turns", outcome=outcome,
        method="llm_summary" if outcome == "committed" else "none",
        failure_class="none" if outcome == "committed" else failure_class(record.get("outcome")),
        session_id=record.get("session_id"), tokens_before=before, tokens_after=after,
        tokens_reclaimed=before - after if before is not None and after is not None else None,
        token_count_method="estimate_rough", messages_before=_int(record.get("messages_before")),
        messages_after=_int(record.get("messages_after")), duration_ms=_int(record.get("duration_ms")),
        threshold_tokens=_int(record.get("threshold_tokens")), context_limit=_int(record.get("context_limit")),
        summarizer_model=record.get("aux_model") or None,
    )


def prune_payload(
    *, session_id: str, tokens_before: int, tokens_after: int, messages: int, tool_results_pruned: int,
) -> dict[str, Any]:
    """A committed proactive tool-result prune: deterministic, no LLM, message count unchanged."""
    return _payload(
        kind="prune_tool_results", scope="history", official=False, trigger="proactive_prune", outcome="committed",
        method="deterministic_prune", failure_class="none", session_id=session_id, tokens_before=tokens_before,
        tokens_after=tokens_after, tokens_reclaimed=tokens_before - tokens_after, token_count_method="estimate_rough",
        messages_before=messages, messages_after=messages, tool_results_pruned=tool_results_pruned,
    )


def provider_native_payload(*, session_id: str, compression_count: Any = None) -> dict[str, Any]:
    """A compaction the provider ran on its own thread and Hermes only observed."""
    return _payload(
        kind="provider_native", scope="provider", official=False, trigger="provider", outcome="committed",
        method="provider", failure_class="none", session_id=session_id, compression_count=_int(compression_count),
    )


def _publish(build: Any, *args: Any, turn_session_id: str | None = None, **kwargs: Any) -> None:
    """Build one record and fan it out to the built-in Relay integration. Never raises into compaction: the
    first failure logs at WARNING with its traceback, later ones at DEBUG."""
    global _warned
    try:
        payload = build(*args, **kwargs)
        from agent.relay_compaction import emit_compaction_mark

        emit_compaction_mark(payload["session_id"], event_name(payload), payload, turn_session_id=turn_session_id)
    except Exception:
        with _warned_lock:
            first, _warned = not _warned, True
        logger.log(logging.WARNING if first else logging.DEBUG, "compaction event publish failed", exc_info=True)


def _publishes_for(agent: Any) -> bool:
    """A persistence-detached fork (background review) rewrites only its own throwaway transcript."""
    return not getattr(agent, "_persist_disabled", False)


def _turn_session_id(agent: Any) -> str | None:
    """The session the agent's current turn started in; a rotating commit moves ``agent.session_id`` off it
    mid-turn. ``None`` outside a turn."""
    value = getattr(agent, "_inflight_turn_session_id", None)
    return value if isinstance(value, str) and value else None


def publish_attempt(agent: Any, record: dict[str, Any]) -> None:
    """Publish one ``compress_context`` attempt record (the attempt telemetry seam calls this)."""
    if _publishes_for(agent):
        _publish(lambda: attempt_payload(
            record, compression_count=getattr(getattr(agent, "context_compressor", None), "compression_count", None),
        ), turn_session_id=_turn_session_id(agent))


def publish_micro(record: dict[str, Any]) -> None:
    """Publish one micro-compaction pass record. The finalizer never runs micro-compaction for a fork."""
    _publish(micro_payload, record)


def publish_prune(agent: Any, messages: list, pruned: list, tool_results_pruned: int) -> None:
    """Publish one committed proactive tool-result prune, measured like the attempt record (rough estimate of
    the message list the model sees next)."""
    if _publishes_for(agent):
        _publish(lambda: prune_payload(
            session_id=getattr(agent, "session_id", None) or "", tokens_before=estimate_messages_tokens_rough(messages),
            tokens_after=estimate_messages_tokens_rough(pruned), messages=len(pruned),
            tool_results_pruned=tool_results_pruned,
        ), turn_session_id=_turn_session_id(agent))


def publish_provider_native(agent: Any) -> None:
    """Publish one provider-native compaction Hermes observed but did not run."""
    if _publishes_for(agent):
        _publish(
            provider_native_payload, session_id=getattr(agent, "session_id", None) or "",
            compression_count=getattr(getattr(agent, "context_compressor", None), "compression_count", None),
            turn_session_id=_turn_session_id(agent),
        )
