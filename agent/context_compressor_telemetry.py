"""Content-free per-attempt compression telemetry for ContextCompressor.

Split from ``agent/context_compressor.py`` (file size cap), which imports it at module level; this module
must never import that module at module level (import cycle).
"""

from __future__ import annotations

import uuid
from typing import Any

from agent.model_metadata import estimate_messages_tokens_rough


def _safe_int(value: Any) -> int | None:
    """Best-effort integer coercion for telemetry fields."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class CompressionTelemetryMixin:
    """Per-attempt telemetry dict: seeded at attempt start, filled by each compression phase."""

    def _begin_compression_telemetry(
        self, *, current_tokens: int | None, attempt_id: str | None = None, session_id: str | None = None,
        trigger_source: str | None = None,
    ) -> dict[str, Any]:
        """Initialize content-free per-attempt compression telemetry."""
        seed = getattr(self, "_compression_telemetry_seed", None)
        seed = seed if isinstance(seed, dict) else {}
        attempt_id = attempt_id or seed.get("attempt_id")
        session_id = session_id or seed.get("session_id")
        trigger_source = trigger_source or seed.get("trigger_source")
        telemetry: dict[str, Any] = {
            "event": "compression_attempt", "attempt_id": attempt_id or uuid.uuid4().hex,
            "session_id": session_id or "", "trigger_source": trigger_source or "unknown",
            "main_provider": self.provider or "", "main_model": self.model or "",
            "main_context_limit": _safe_int(self.context_length),
            "current_estimated_tokens": _safe_int(current_tokens),
            "effective_threshold": _safe_int(self.threshold_tokens), "protected_head_tokens": None,
            "protected_tail_tokens": None, "middle_window_tokens": None, "prellm_skip_count": 0,
            "aux_prompt_tokens": None, "aux_output_reservation": None, "aux_provider": "", "aux_model": "",
            "effective_aux_context": None, "fit_margin": None, "chunking": False, "chunk_count": 0,
            "total_duration_ms": None, "aux_call_duration_ms": None, "queue_wait_ms": None, "prompt_build_ms": None,
            "time_to_first_progress_ms": None, "summary_generation_ms": None, "commit_ms": None,
            "fallback_used": False, "commit_status": "unknown", "split_status": "unknown", "failure_class": None,
            # Lean-sampling coverage (filled by _record_summary_input_coverage; None on the legacy path).
            "summary_input_chars": None, "summary_input_sampled_chars": None, "summary_input_omitted_chars": None,
            "summary_input_record_count": None, "summary_input_sampled_record_count": None,
            "summary_input_elided_record_count": None,
            # Candidate effect is captured by the compressor, then replaced with the exact committed shape.
            "method": "none", "items_dropped": 0, "messages_before": None, "messages_after": None,
            "tokens_before": None, "tokens_after": None, "tokens_reclaimed": None, "token_count_method": None,
            "tool_results_pruned": None, "reasoning_items_pruned": None, "has_focus_topic": False,
        }
        self._active_compression_telemetry = self._last_compression_telemetry = telemetry
        return telemetry

    def _record_compression_regions(
        self, *, head_messages: list[dict[str, Any]], middle_messages: list[dict[str, Any]],
        tail_messages: list[dict[str, Any]],
    ) -> None:
        telemetry = getattr(self, "_active_compression_telemetry", None)
        if isinstance(telemetry, dict):
            telemetry["protected_head_tokens"] = estimate_messages_tokens_rough(head_messages)
            telemetry["middle_window_tokens"] = estimate_messages_tokens_rough(middle_messages)
            telemetry["protected_tail_tokens"] = estimate_messages_tokens_rough(tail_messages)

    def _record_aux_compression_call(
        self, *, prompt_messages: list[dict[str, Any]], max_tokens: int | None, duration_ms: int,
        aux_provider: str | None = None, aux_model: str | None = None,
        effective_aux_context: int | None = None, phase_timings: dict[str, Any] | None = None,
    ) -> None:
        telemetry = getattr(self, "_active_compression_telemetry", None)
        if not isinstance(telemetry, dict):
            return
        telemetry["aux_prompt_tokens"] = estimate_messages_tokens_rough(prompt_messages)
        telemetry["aux_output_reservation"] = _safe_int(max_tokens)
        if aux_provider:
            telemetry["aux_provider"] = aux_provider
        if aux_model:
            telemetry["aux_model"] = aux_model
        if effective_aux_context is not None:
            telemetry["effective_aux_context"] = _safe_int(effective_aux_context)
        if telemetry["effective_aux_context"] is not None and telemetry["aux_prompt_tokens"] is not None:
            telemetry["fit_margin"] = (telemetry["effective_aux_context"] - telemetry["aux_prompt_tokens"]
                                       - (telemetry["aux_output_reservation"] or 0))
        telemetry["aux_call_duration_ms"] = (telemetry.get("aux_call_duration_ms") or 0) + max(0, int(duration_ms))
        for key in ("queue_wait_ms", "prompt_build_ms", "time_to_first_progress_ms", "summary_generation_ms", "commit_ms"):
            if not isinstance(phase_timings, dict) or key not in phase_timings:
                continue
            value = _safe_int(phase_timings[key])
            # Wait and generation phases accumulate across retries; the rest are point readings.
            accumulate = key in {"queue_wait_ms", "summary_generation_ms"} and value is not None
            telemetry[key] = (telemetry.get(key) or 0) + value if accumulate else value

    def _compression_method(self) -> str:
        """How the summary that replaced the middle window was produced."""
        if getattr(self, "_last_summary_fallback_used", False):
            return "deterministic_fallback"
        if getattr(self, "_last_aux_model_failure_error", None):
            return "aux_fallback_main"
        return "llm_summary"

    def _record_compression_effect(
        self, messages_before: int, compressed: list[dict[str, Any]], tokens_before: int, tool_results_pruned: int,
        reasoning_items_pruned: int,
    ) -> None:
        """Record what a finished rewrite did. ``tokens_before`` and the after count are rough message-only
        estimates (system prompt and tool schemas excluded), so they compare like for like."""
        telemetry = getattr(self, "_active_compression_telemetry", None)
        if not isinstance(telemetry, dict):
            return
        tokens_after = estimate_messages_tokens_rough(compressed)
        telemetry.update(
            method=self._compression_method(), items_dropped=getattr(self, "_last_summary_dropped_count", 0) or 0,
            messages_before=messages_before, messages_after=len(compressed), tokens_before=tokens_before,
            tokens_after=tokens_after, tokens_reclaimed=tokens_before - tokens_after,
            token_count_method="estimate_rough", tool_results_pruned=tool_results_pruned,
            reasoning_items_pruned=reasoning_items_pruned,
        )

    def _record_committed_compression_effect(
        self, messages_before: list[dict[str, Any]], messages_after: list[dict[str, Any]],
    ) -> None:
        """Replace candidate counts with the exact transcript shape that crossed the commit boundary."""
        telemetry = getattr(self, "_active_compression_telemetry", None)
        if not isinstance(telemetry, dict):
            return
        tokens_before = estimate_messages_tokens_rough(messages_before)
        tokens_after = estimate_messages_tokens_rough(messages_after)
        telemetry.update(
            messages_before=len(messages_before), messages_after=len(messages_after),
            tokens_before=tokens_before, tokens_after=tokens_after,
            tokens_reclaimed=tokens_before - tokens_after, token_count_method="estimate_rough",
        )
