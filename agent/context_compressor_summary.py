"""Summary-hook dispatch and cancellation rollback for context compression."""

from __future__ import annotations

import inspect
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.auxiliary_relay_metadata import observe_served_dispatch

if TYPE_CHECKING:
    from agent.context_compressor import _HandoffScan

logger = logging.getLogger(__name__)


class _SummaryRouteUnreproducible(ValueError):
    """A ladder hop answered a chunk but its dispatch was not observed: the plan stops, nothing is adopted."""


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
    def _apply_summary_route(self, call_kwargs: dict, pinned: Optional[dict[str, Any]] = None) -> None:
        """Pin the summary route onto ``call_kwargs`` for both compression entry points.

        After a fallback, an omitted route is NOT "use the main model": ``call_llm`` re-resolves
        ``auxiliary.compression`` from config — the model that just failed — and a second fallback
        is refused, so every later attempt aborts (#123362). Name the main runtime explicitly.
        A stall-fallback ``pinned`` route replaces the whole route: merged over the main runtime, the
        main api_key/base_url would ride into the fallback entry's call (to its host, or instead of it).
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
        for key, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("base_url", self.base_url),
            ("api_key", self.api_key),
            ("api_mode", getattr(self, "api_mode", "")),
        ):
            if value:
                call_kwargs[key] = value

    def _include_demoted_boundary_tools(
        self, source: list[dict[str, Any]], projected: list[dict[str, Any]],
        compress_start: int, tail_start: int, turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Cover bodies pruned outside the middle; never resend unchanged tail output.

        Boundaries are selected on the smaller projection. Head protection and
        tail alignment can retain a stub whose original body lies outside the
        summary window. Include that body and its call metadata, in source order,
        before adopting the projection. Serialization still redacts and chunks
        the complete input, so this does not relax the transport budget.
        """
        extra_tools = {
            i for i, (original, reduced) in enumerate(zip(source, projected))
            if (i < compress_start or i >= tail_start)
            and original.get("role") == "tool"
            and isinstance(original.get("content"), str)
            and original.get("content") != reduced.get("content")
        }
        if not extra_tools:
            return turns
        call_ids = {source[i].get("tool_call_id") for i in extra_tools}
        extra_rows = extra_tools | {
            i for i, original in enumerate(source)
            if (i < compress_start or i >= tail_start)
            and original.get("role") == "assistant"
            and any(call.get("id") in call_ids for call in original.get("tool_calls") or [])
        }
        return (
            [source[i] for i in sorted(extra_rows) if i < compress_start]
            + turns
            + [source[i] for i in sorted(extra_rows) if i >= tail_start]
        )

    def _summarize_window(
        self, messages: list[dict[str, Any]], turns_to_summarize: list[dict[str, Any]], scan: _HandoffScan,
        focus_topic: Optional[str], memory_context: str, bypass_cooldown: bool,
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


    def _call_summary_llm(self, prompt: str, prompt_started_at: float) -> str:
        """Issue the single aux summary call; return validated content text.
        Raises RuntimeError for empty content or a length-truncated (PARTIAL) summary so the failure
        routes through main-model fallback + cooldown instead of wiping the compacted turns."""
        from agent import context_compressor as cc
        # call_llm writes the route it actually selected; never pre-resolve a second, stale pair.
        _aux_route = cc._SummaryRouteRecord()
        call_kwargs: dict[str, Any] = {
            "task": "compression",
            "main_runtime": {
                "model": self.model, "provider": self.provider, "base_url": self.base_url, "api_key": self.api_key,
                "api_mode": self.api_mode,
            },
            "messages": [{"role": "user", "content": prompt}], "route_info": _aux_route,
            # NO max_tokens: Anthropic/NIM wires forward it and a hard cap truncates summaries
            # (thinking models burn it on reasoning). Timeout comes from call_llm config.
        }
        # Preserve upstream's explicit main-model fallback and whole-route pin replacement.
        self._apply_summary_route(call_kwargs, cc._pinned_summary_call_kwargs())
        # Compression is atomic: protect the in-flight summary call from a mid-turn gateway interrupt.
        # Without this, an incoming user message aborts the summary and compression falls back to a degraded
        # static marker, losing the real handoff (#23975). Re-entrant: a main-model retry (_generate_summary
        # recursion) re-enters harmlessly.
        _aux_call_start = time.monotonic()
        _latency_info: dict[str, int] = {"prompt_build_ms": max(0, int((_aux_call_start - prompt_started_at) * 1000))}
        call_kwargs["latency_info"] = _latency_info
        # Per-attempt observable (#114594): with this line a stalled attempt is distinguishable from a slow
        # one — silence before it is prompt build, silence after it is the summary provider.
        cc.logger.info(
            "Compression summary call dispatched: model=%s prompt_chars=%s prompt_build_ms=%s",
            self.summary_model or self.model, f"{len(prompt):,}", _latency_info["prompt_build_ms"],
        )
        try:
            # Compression is atomic: shield the summary call from gateway interrupts. Re-entrant.
            with cc.aux_interrupt_protection(), observe_served_dispatch() as dispatched:
                response = cc.call_llm(**call_kwargs)
        finally:
            route_known = bool(_aux_route.get("provider") and _aux_route.get("model"))
            _aux_model = _aux_route.get("model") or self.summary_model or self.model or ""
            # Remember the resolved model for the failure path: an ``auto`` route picks one per call
            # without setting ``summary_model``, so only this names it in the user warning (#116472).
            self._last_aux_resolved_model = _aux_model or None
            self._record_aux_compression_call(
                prompt_messages=call_kwargs["messages"],
                # max_tokens is intentionally absent; .get() keeps the telemetry hook from breaking the call.
                max_tokens=call_kwargs.get("max_tokens"),
                duration_ms=int((time.monotonic() - _aux_call_start) * 1000),
                aux_provider=_aux_route.get("provider") or self.provider or "",
                aux_model=_aux_model,
                effective_aux_context=self.context_length if route_known and _aux_model == self.model else None,
                phase_timings=_latency_info,
            )
        if self._compression_cancelled():
            raise AuxiliaryExplicitCancellation()
        # Reasoning-field fallback (DeepSeek/Qwen/Kimi put the summary in reasoning_content); capped.
        content = cc.extract_content_or_reasoning(response, max_reasoning_chars=8000)
        where = f"(provider={self.provider or 'auto'} model={self.summary_model or self.model})"
        # Some OpenAI-compatible proxies (e.g. cmkey.cn, one-api channels) return a well-formed HTTP 200
        # with an empty or whitespace-only ``content`` instead of an error or empty ``choices``. That
        # payload passes ``_validate_llm_response`` (a ``message`` exists), so it reaches here and would
        # otherwise be stored as a prefix-only summary with no body — silently wiping the compacted turns
        # and making the model forget the in-progress task (#11978, #11914). Treat empty content as a
        # failure so it routes through the same main-model fallback + cooldown machinery as a transport
        # error, rather than replacing real context with an empty summary.
        if not content.strip():
            raise RuntimeError(f"Context compression LLM returned empty content {where}")
        if cc._is_refusal_response(response, content):
            # Treat a refusal as unusable content. This deliberately reuses the
            # established fallback/cooldown/abort path for an empty body, so it
            # can never be committed as `_previous_summary`.
            raise RuntimeError(f"Context compression LLM returned refusal content {where}")
        # A finish_reason of "length" means the summarizer hit its output token cap mid-generation: the text
        # present is PARTIAL. Persisting a partial summary as the compaction checkpoint silently truncates
        # the conversation's memory — the cut-off text replaces the real middle turns AND is fed back into
        # every subsequent iterative update prompt, compounding the loss across compactions. Treat it as a
        # failure so it routes through the same main-model fallback + abort machinery as other degraded
        # responses instead of becoming a checkpoint. (Ported from earendil-works/pi#7048.)
        # A length stop means the merged rolling summary is partial — persisting it would silently drop the
        # tail of the merge and feed the cut-off text into every later micro-compact pass. Leave the
        # exchange unabsorbed instead; a later pass retries it. (Same class as _generate_summary's guard;
        # pi#7048.)
        if cc._response_finish_reason(response) == "length":
            raise RuntimeError(
                f"Context compression summary was truncated ({cc._TRUNCATED_SUMMARY_MARKER}): generation hit the output "
                f"token cap and the summary is incomplete {where}"
            )
        observer = cc._SUMMARY_SERVED_ROUTE.get()
        if observer is not None:
            observer.update(provider=_aux_route.get("provider"), model=_aux_route.get("model"),
                            ladder_moved=_aux_route.records > 1, lane=dict(dispatched) or None)
        return content

    def _served_lane_route(self: Any, served: dict[str, Any], pin: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Route for the remaining chunks: the dispatch that answered this one, when that must be followed.

        ``auxiliary_relay_metadata`` records it once the response came back: provider and host and
        credential of the ONE client that answered, model and timeout of the request actually sent, its
        API mode. Nothing is looked up again or matched against configuration, so the remaining chunks can
        neither return to the lane that failed nor pair one route's host with another's credential.
        Followed after a ladder hop (pinned plan or not), and under a pin after a same-route recovery
        (refreshed credential, rotated pool key, new host): the pin's explicit fields would otherwise
        re-send what just failed. An unpinned chunk without a hop leaves the next one to re-resolve its
        own route and credential. A hop whose dispatch was not observed completely (provider, model,
        host, credential) fails the plan closed, nothing adopted, instead of letting call_llm pair the
        host with a credential of its own choosing.
        """
        from agent import context_compressor as cc
        hop = bool(served.get("ladder_moved"))
        if not hop and pin is None:
            return None
        lane = served.get("lane") or {}
        if not all(lane.get(key) for key in ("provider", "model", "base_url", "api_key")):
            if hop:
                raise _SummaryRouteUnreproducible(
                    "remaining summary chunks have no observed route (provider, host, credential) after a ladder rescue")
            return pin  # the pin answered and nothing observed says it changed
        if hop:
            cc.logger.info("Compression summary: a recovery lane served one chunk; remaining chunks pinned to it "
                           "(provider=%s model=%s)", lane.get("provider"), lane.get("model"))
        return {"label": f"served lane ({lane['provider']})", **lane}

    def _summary_source_room(self, overhead: str) -> int:
        """Source characters the next summary prompt may carry under the transport ceiling.

        The ceiling prices the WHOLE prompt (instructions, previous summary, focus, memory) and
        holds before dispatch on every route. No advertised window ever widens it: the auxiliary
        ladder can re-send an already-built prompt to a task fallback or to the main model after a
        capacity, payment or auth error, so an uncertain window or a possible fallback is not proven
        capacity. At the rough CHARS_PER_TOKEN estimate 160K chars is ~40K tokens, which with at
        most ``_SUMMARY_TOKENS_CEILING`` output sits under the MINIMUM_CONTEXT_LENGTH floor that the
        startup feasibility check and the ladder's window screening apply to compression routes;
        denser text can exceed that estimate, and a provider refusal then fails closed as before.
        """
        return self._SUMMARY_INPUT_MAX_CHARS - len(overhead)

    def _generate_summary(
        self, turns_to_summarize: list[dict[str, Any]], focus_topic: Optional[str] = None,
        memory_context: str = "", bypass_cooldown: bool = False,
    ) -> Optional[str]:
        """Transport the complete redacted transcript in bounded, chronological summary calls.

        The existing 160K character ceiling covers the *whole* prompt including prior summary and
        instructions. Never mistake sampling coverage for source coverage. This work runs only inside
        an already-triggered major compression; no intermediate transcript is committed.
        """
        from agent import context_compressor as cc
        if self._compression_cancelled():
            raise AuxiliaryExplicitCancellation()
        if time.monotonic() < self._summary_failure_cooldown_until and not bypass_cooldown:
            return None
        records = self._serialize_records_for_summary(turns_to_summarize, full_tool_results=True)
        source = "\n\n".join(records)
        working_summary = cc._redact_compaction_text(self._previous_summary) if self._previous_summary else None
        transmitted = 0
        chunks = 0
        result = None
        # A host pin covers this whole plan: every chunk's first attempt goes to the pinned route,
        # never a main-model retry, and never the chunks left after such a retry moved the plan.
        pinned_route = cc.take_pinned_summary_route()
        try:
            # Even an empty transcript needs one call (existing direct-call contract).
            while transmitted < len(source) or chunks == 0:
                if self._compression_cancelled():
                    raise AuxiliaryExplicitCancellation()
                from agent.conversation_compression import _raise_if_stale_attempt
                _raise_if_stale_attempt(self)
                if chunks >= self._MAX_COMPLETE_SUMMARY_CHUNKS:
                    raise ValueError("complete summary input exceeds bounded chunk count")
                budget = self._compute_summary_budget(turns_to_summarize)
                has_user_turn = getattr(self, "_summary_has_user_turn", None)
                if has_user_turn is None:
                    has_user_turn = self._transcript_has_real_user_turn(turns_to_summarize)
                # Build the actual prompt with an empty source to price instructions, previous
                # summary, focus and provider memory before taking a contiguous redacted span.
                overhead = self._build_summary_prompt(
                    "", budget, cc._redact_compaction_text(focus_topic) if focus_topic else None,
                    memory_context, has_user_turn, previous_summary=working_summary,
                )
                available = self._summary_source_room(overhead)
                if available <= 0:
                    raise ValueError("complete summary input has no room after prompt overhead")
                end = min(transmitted + available, len(source))
                if end < len(source) and available > 16:
                    # Prefer a complete line or word near the end of the budget.
                    # A long unbroken record still advances contiguously without elision.
                    boundary = source.rfind("\n", max(transmitted, end - available // 10), end)
                    if boundary < 0:
                        boundary = source.rfind(" ", max(transmitted, end - available // 10), end)
                    if boundary >= 0:
                        end = boundary + 1
                chunk = source[transmitted:end]
                fell_back_before = bool(getattr(self, "_summary_model_fallen_back", False))
                served: dict[str, Any] = {}
                served_token = cc._SUMMARY_SERVED_ROUTE.set(served)
                try:
                    with cc.pin_summary_route(pinned_route):
                        result = self._generate_summary_once(
                            turns_to_summarize, focus_topic=focus_topic, memory_context=memory_context,
                            bypass_cooldown=bypass_cooldown, serialized_chunk=chunk,
                            previous_summary=working_summary,
                        )
                finally:
                    cc._SUMMARY_SERVED_ROUTE.reset(served_token)
                if not fell_back_before and getattr(self, "_summary_model_fallen_back", False):
                    pinned_route = None
                if result is None:
                    return None
                transmitted += len(chunk)
                chunks += 1
                working_summary = self._strip_summary_prefix(result)
                if transmitted < len(source):
                    # After a ladder hop, or under a pin, the remaining chunks go first to the dispatch
                    # that answered this one, never back to the route or credential that just failed.
                    pinned_route = self._served_lane_route(served, pinned_route)
            # Serialize publication with a new worker's claim: a detached worker cannot
            # publish even if superseded between its last response and commit.
            from agent.conversation_compression import (
                _compressor_attempt_serial_lock, _raise_if_stale_attempt, _summary_adoption_window,
            )
            with _compressor_attempt_serial_lock(self):
                _raise_if_stale_attempt(self)
                # A cancellation that lands after the last answer must not publish it either. The window holds this
                # attempt's fence lock for the adoption only, so a host cancellation lands wholly before this check
                # (nothing is adopted) or after the adoption (and then refuses its commit), and never waits on the
                # durable success writes below.
                with _summary_adoption_window():
                    if self._compression_cancelled():
                        raise AuxiliaryExplicitCancellation()
                    previous_summary = self._previous_summary
                    self._previous_summary = working_summary
                    # Admission revocation and the deadline flip without that lock: one that landed during the
                    # write is undone here instead of adopted.
                    if self._compression_cancelled():
                        self._previous_summary = previous_summary
                        raise AuxiliaryExplicitCancellation()
                self._clear_compression_failure_cooldown()
                self._summary_model_fallen_back = False
                self._last_summary_error = None
                self._clear_terminal_summary_failures()
                self._reset_consecutive_overload_aborts()

            self._record_summary_input_coverage({
                "input_chars": sum(map(len, records)),
                "sampled_chars": sum(map(len, records)), "omitted_chars": 0,
                "record_count": len(records), "sampled_record_count": len(records),
                "elided_record_count": 0,
            })
            telemetry = getattr(self, "_active_compression_telemetry", None)
            if isinstance(telemetry, dict):
                telemetry["chunk_count"] = chunks
            return result
        except ValueError as exc:
            # A budget limit or an unobserved ladder hop is not a successful summary or a degraded
            # static checkpoint.
            route_lost = isinstance(exc, _SummaryRouteUnreproducible)
            cc.logger.warning("Context compression %s: %s", "route lost" if route_lost else "input not covered", exc)
            from agent.conversation_compression import _raise_if_stale_attempt
            _raise_if_stale_attempt(self)
            self._last_summary_error = str(exc)
            # Only a budget limit is an incomplete-summary failure; a lost route is a plain generation failure.
            self._last_summary_truncated_failure = not route_lost
            telemetry = getattr(self, "_active_compression_telemetry", None)
            if isinstance(telemetry, dict):
                telemetry["failure_class"] = "summary_generation_failed" if route_lost else "summary_input_uncovered"
            return None

    def _generate_summary_once(
        self, turns_to_summarize: list[dict[str, Any]], focus_topic: Optional[str] = None,
        memory_context: str = "", bypass_cooldown: bool = False,
        serialized_chunk: Optional[str] = None, previous_summary: Optional[str] = None,
    ) -> Optional[str]:
        """Native call, fallback, validation and provenance for one fully covered source span."""
        from agent import context_compressor as cc
        prompt_started_at = time.monotonic()
        if self._compression_cancelled():
            raise AuxiliaryExplicitCancellation()
        # bypass_cooldown: provider-proven overflow gets ONE real attempt while armed.
        if prompt_started_at < self._summary_failure_cooldown_until and not bypass_cooldown:
            cc.logger.debug(
                # See #100661.
                "Skipping context summary during cooldown (%.0fs remaining)",
                self._summary_failure_cooldown_until - prompt_started_at,
            )
            return None
        # Strict-redact inputs that bypass _serialize_for_summary (focus string, prior summary).
        if focus_topic:
            focus_topic = cc._redact_compaction_text(focus_topic)
        if previous_summary:
            previous_summary = cc._redact_compaction_text(previous_summary)
        summary_budget = self._compute_summary_budget(turns_to_summarize)
        # Ghost-skill defense: LLMs paraphrase [SKILL_PRUNED] markers away; collect the names
        # deterministically BEFORE the call (from the turn LIST, not the bounded text), re-inject after.
        _pruned_skill_names = list(dict.fromkeys(
            cc._collect_ghosted_skill_names(turns_to_summarize) + cc._extract_pruned_skill_names(previous_summary or "")
        ))[:cc._MAX_PRUNED_SKILL_MARKERS]
        # The caller already established full redacted coverage and the actual prompt budget.
        # Keep the native sampling helper available for existing diagnostics, never for an accepted
        # full-coverage checkpoint.
        content_to_summarize = serialized_chunk if serialized_chunk is not None else self._serialize_for_summary(turns_to_summarize)
        has_user_turn = getattr(self, "_summary_has_user_turn", None)
        if has_user_turn is None:
            has_user_turn = self._transcript_has_real_user_turn(turns_to_summarize)
        prompt = self._build_summary_prompt(
            content_to_summarize, summary_budget, focus_topic, memory_context, has_user_turn,
            previous_summary=previous_summary,
        )
        if len(prompt) > self._SUMMARY_INPUT_MAX_CHARS:
            # Transport bound BEFORE dispatch: the auxiliary ladder may re-send this exact prompt to
            # a task fallback or to the main model, so it must never leave the process oversized.
            from agent.conversation_compression import _raise_if_stale_attempt
            _raise_if_stale_attempt(self)
            cc.logger.warning(
                "Context compression summary prompt refused before dispatch: %s chars > %s ceiling",
                f"{len(prompt):,}", f"{self._SUMMARY_INPUT_MAX_CHARS:,}",
            )
            self._last_summary_error = "summary prompt exceeds the transport ceiling"
            self._last_summary_truncated_failure = True
            telemetry = getattr(self, "_active_compression_telemetry", None)
            if isinstance(telemetry, dict):
                telemetry["failure_class"] = "summary_input_uncovered"
            return None
        try:
            content = self._call_summary_llm(prompt, prompt_started_at)
            # Strip <think> blocks: they would be stored, injected, and compounded on every iterative update.
            from agent.agent_runtime_helpers import strip_think_blocks
            content = strip_think_blocks(None, content).strip() or content
            # The summarizer may echo secrets verbatim; redact the output too.
            summary = cc._redact_compaction_text(content.strip())
            # Restore any [SKILL_PRUNED] marker the summarizer paraphrased away.
            # See #32106.
            summary = cc._reinject_pruned_skill_markers(summary, _pruned_skill_names)
            summary = self._ground_historical_task_snapshot(summary, turns_to_summarize)
            summary = self._augment_summary_lean(summary, turns_to_summarize)
            self._validate_summary_user_provenance(summary, has_user_turn)
            # Keep intermediate results private; _generate_summary publishes only a full result.
            from agent.conversation_compression import _raise_if_stale_attempt

            _raise_if_stale_attempt(self)
            return self._with_summary_prefix(summary)
        except Exception as e:
            logger.debug("Context summary attempt failed", exc_info=True)
            return self._on_summary_failure(
                e, turns_to_summarize, focus_topic, memory_context, serialized_chunk=serialized_chunk,
                previous_summary=previous_summary,
            )
