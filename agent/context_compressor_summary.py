"""Summary-hook dispatch and cancellation rollback for context compression."""

from __future__ import annotations

import inspect
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent.auxiliary_client import AuxiliaryExplicitCancellation

if TYPE_CHECKING:
    from agent.context_compressor import _HandoffScan


def _accepts_keyword_argument(callable_obj: Any, name: str) -> bool:
    """Return whether an inspectable callable accepts ``name`` as a keyword."""
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return True
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class SummaryDispatchMixin:
    def _call_summary_llm(self, prompt: str, prompt_started_at: float) -> str:
        """Issue the single aux call and reject an empty, refused or partial checkpoint."""
        from agent import context_compressor as cc
        from agent.context_compressor_cache import summary_main_runtime, summary_prompt_estimate

        _aux_route: dict[str, str] = {}
        call_kwargs: dict[str, Any] = {
            "task": "compression",
            "main_runtime": summary_main_runtime({
                "model": self.model, "provider": self.provider, "base_url": self.base_url, "api_key": self.api_key,
                "api_mode": self.api_mode,
            }),
            "messages": [{"role": "user", "content": prompt}], "route_info": _aux_route,
            # A hard output cap can truncate reasoning-model summaries. Keep the existing
            # prompt budget and auxiliary timeout instead.
        }
        self._apply_summary_route(call_kwargs, cc._pinned_summary_call_kwargs())
        _aux_call_start = time.monotonic()
        _latency_info: dict[str, int] = {"prompt_build_ms": max(0, int((_aux_call_start - prompt_started_at) * 1000))}
        call_kwargs["latency_info"] = _latency_info
        cc.logger.info(
            "Compression summary call dispatched: model=%s prompt_chars=%s prompt_build_ms=%s",
            self.summary_model or self.model, f"{len(prompt):,}", _latency_info["prompt_build_ms"],
        )
        try:
            with cc.aux_interrupt_protection():
                response = cc.call_llm(**call_kwargs)
        finally:
            route_known = bool(_aux_route.get("provider") and _aux_route.get("model"))
            _aux_model = _aux_route.get("model") or self.summary_model or self.model or ""
            self._last_aux_resolved_model = _aux_model or None
            estimate_kwargs = {}
            estimate = summary_prompt_estimate()
            if estimate is not None:
                estimate_kwargs["prompt_tokens_estimate"] = estimate
            self._record_aux_compression_call(
                prompt_messages=call_kwargs["messages"], max_tokens=call_kwargs.get("max_tokens"),
                duration_ms=int((time.monotonic() - _aux_call_start) * 1000),
                aux_provider=_aux_route.get("provider") or self.provider or "", aux_model=_aux_model,
                effective_aux_context=self.context_length if route_known and _aux_model == self.model else None,
                phase_timings=_latency_info, **estimate_kwargs,
            )
        if self._compression_cancelled():
            raise AuxiliaryExplicitCancellation()
        self._note_aux_route_fallback(_aux_route)
        content = cc.extract_content_or_reasoning(response, max_reasoning_chars=8000)
        where = f"(provider={self.provider or 'auto'} model={self.summary_model or self.model})"
        if not content.strip():
            raise RuntimeError(f"Context compression LLM returned empty content {where}")
        if cc._is_refusal_response(response, content):
            raise RuntimeError(f"Context compression LLM returned refusal content {where}")
        # Never persist a partial checkpoint: it would replace the original middle
        # and contaminate each subsequent iterative summary.
        if cc._response_finish_reason(response) == "length":
            raise RuntimeError(
                f"Context compression summary was truncated ({cc._TRUNCATED_SUMMARY_MARKER}): generation hit the output "
                f"token cap and the summary is incomplete {where}"
            )
        return content

    def _apply_summary_route(self, call_kwargs: dict, pinned: Optional[dict[str, Any]] = None) -> None:
        """Pin the summary route onto ``call_kwargs`` for both compression entry points.

        After a fallback, an omitted route is NOT "use the main model": ``call_llm`` re-resolves
        ``auxiliary.compression`` from config — the model that just failed — and a second fallback
        is refused, so every later attempt aborts (#123362). Name the main runtime explicitly.
        A stall-fallback ``pinned`` route replaces the whole route: merged over the main runtime, the
        main api_key/base_url would ride into the fallback entry's call (to its host, or instead of it).
        The main route is complete even with empty fields: an authoritative task keeps compression
        policy but never fills an unset endpoint/key/wire mode from the failed aux route (#130895).
        """
        if pinned:
            # Clear first: a keyless pin (local server) resolves its own credential, never the main one.
            for key in ("provider", "model", "base_url", "api_key", "api_mode"):
                call_kwargs.pop(key, None)
            call_kwargs.update(pinned)
            return
        if self.summary_model:
            call_kwargs["model"] = self.summary_model
            return
        if not getattr(self, "_summary_model_fallen_back", False):
            return
        from agent.auxiliary_task_config import authoritative_auxiliary_task

        call_kwargs["task"] = authoritative_auxiliary_task(call_kwargs.get("task") or "compression")
        for key, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("base_url", self.base_url),
            ("api_key", self.api_key),
            ("api_mode", getattr(self, "api_mode", "")),
        ):
            if value:
                call_kwargs[key] = value

    def _summarize_window(
        self, messages: list[dict[str, Any]], turns_to_summarize: list[dict[str, Any]], scan: _HandoffScan,
        focus_topic: Optional[str], memory_context: str, bypass_cooldown: bool, summary_prefix: Any = None,
    ) -> Optional[str]:
        """Run the summary LLM; a cancellation rolls back the handoff scan's self-heal mutation first.
        A deterministic pin (repeated stall, #112420) skips the LLM: ``None`` lets Phase 3 insert the static
        fallback summary, or abort under ``abort_on_summary_failure`` exactly like a failed summary call."""
        from agent.context_compressor import take_deterministic_summary_pin
        if take_deterministic_summary_pin():
            # A detached stale attempt must not stamp error state or mutate the shared telemetry dict
            # the fallback owns; unwind as a cancellation before any write lands.
            from agent.conversation_compression import _raise_if_stale_attempt

            _raise_if_stale_attempt(self)
            # Surfaces through the fallback summary's reason line and the host's one-shot user warning.
            self._last_summary_error = (
                "summary model stalled on every route; deterministic fallback summary inserted"
            )
            telemetry = getattr(self, "_active_compression_telemetry", None)
            if isinstance(telemetry, dict):
                telemetry["failure_class"] = "stall_deterministic_fallback"
            return None
        # Focus-topic derivation scans user turns; only pay when a summary is generated.
        summary_kwargs: dict[str, Any] = {
            "focus_topic": focus_topic or self._derive_auto_focus_topic(messages),
            "memory_context": memory_context,
        }
        if _accepts_keyword_argument(self._generate_summary, "bypass_cooldown"):
            summary_kwargs["bypass_cooldown"] = bypass_cooldown
        if summary_prefix is not None and _accepts_keyword_argument(self._generate_summary, "summary_prefix"):
            summary_kwargs["summary_prefix"] = summary_prefix
        try:
            return self._generate_summary(turns_to_summarize, **summary_kwargs)
        except AuxiliaryExplicitCancellation:
            # Cancellation is a true no-op: restore the scan's mutation before the exception escapes.
            # Guard by THIS attempt's ownership: a detached stale attempt (a fallback already claimed
            # and is working the compressor) must not revert summary state the newer attempt advanced.
            # The caller's own generation rides a ContextVar because shared compressor attributes can
            # only name the current owner, never the caller's attempt. Working-attempt comparison (not
            # the entry generation) so a no-op claim does not suppress the owning attempt's rollback.
            from agent.conversation_compression import _caller_attempt_is_current

            if _caller_attempt_is_current(self):
                self._previous_summary = scan.previous_summary_before
                self._summary_has_user_turn = scan.has_user_turn_before
            raise
