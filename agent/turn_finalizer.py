"""Post-loop turn finalization for ``run_conversation``.

Budget summary, trajectory save, persist, diagnostics, response transforms, result
assembly, steer drain, memory/skill review. Synchronous, single return. ``logger`` is
imported lazily from ``agent.conversation_loop`` (no cycle, same logger name)."""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from typing import Any, Callable, List, Optional, Tuple

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.message_content import flatten_message_text
from agent.message_metadata import append_message, stamp_message_timestamp
from agent.message_sanitization import _sanitize_surrogates

# Verification-continuation nudges (verify-on-stop / pre_verify) must be stripped from
# returned/live history to avoid role-alternation breaks; the assistant response is
# real content and is not flagged. (#65919)
_VERIFICATION_CONTINUATION_FLAGS = ("_verification_stop_synthetic", "_pre_verify_synthetic")

_SENTENCE_END = {".", "!", "?", "。", "！", "？", "`", ")"}

# ``result[key] = agent.session_<key>`` for the per-session usage/cost counters.
_SESSION_TOKEN_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens", "prompt_tokens", "completion_tokens", "total_tokens",
)
_SESSION_COST_KEYS = ("estimated_cost_usd", "cost_status", "cost_source")


def _assistant_row_missing_visible_text(msg: dict) -> bool:
    """True when an assistant row has no visible text (blank final or tool-only)."""
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return False
    return not flatten_message_text(msg.get("content")).strip()


def _record_kanban_budget_exhausted(
    kanban_task: str, api_call_count: int, max_iterations: int, logger: logging.Logger
) -> None:
    """Record a terminal ``timed_out`` outcome for a kanban worker out of budget.

    Routed via ``_record_task_failure`` (not ``kanban_block``) so it counts toward the
    consecutive-failure circuit breaker. Idempotent via the ``_end_run`` CAS
    (``WHERE ended_at IS NULL``), so safe from multiple exit paths.

    This is a bounded fallback (#87096): the CAS invariant in ``_end_run`` (``WHERE ended_at IS NULL``)
    guarantees idempotence — if another path already closed the run this is a no-op — so it is safe to call
    from multiple exit paths.
    """
    try:
        from hermes_cli import kanban_db as _kb
        from hermes_cli import kanban_db_connect as _kbc
        from hermes_cli import kanban_db_dispatch as _kbd
        _conn = _kbc.connect()
        try:
            _kbd._record_task_failure(
                _conn,
                kanban_task,
                error=(
                    f"Iteration budget exhausted ({api_call_count}/{max_iterations}) — "
                    "task could not complete within the allowed iterations"
                ),
                outcome="timed_out",
                release_claim=True,
                end_run=True,
                event_payload_extra={"budget_used": api_call_count, "budget_max": max_iterations},
            )
        finally:
            with suppress(Exception):
                _conn.close()
    except Exception:
        logger.warning(
            "Failed to record budget-exhausted failure for task %s", kanban_task, exc_info=True
        )


def _drop_verification_continuation_scaffolding(messages) -> None:
    """Remove verification-continuation nudges in place; only the synthetic nudges carry
    these flags, so the real attempted final answer persisted to state.db survives."""
    messages[:] = [
        m for m in messages
        if not (isinstance(m, dict) and any(m.get(f) for f in _VERIFICATION_CONTINUATION_FLAGS))
    ]


def _clone_background_review_messages(messages):
    """Copy the review input without aliasing the live transcript."""
    # Lazy: conversation_loop imports this module (cycle).
    from agent.conversation_loop import _clone_message_for_send

    return [_clone_message_for_send(message) for message in messages]


def _invoke_hook_safely(name: str, logger: logging.Logger, **kwargs) -> list:
    """Fire a lifecycle plugin hook; a failing hook is logged, never fatal."""
    try:
        from hermes_cli.lifecycle import invoke_hook
        return invoke_hook(name, **kwargs)
    except Exception as exc:
        logger.warning("%s hook failed: %s", name, exc)
        return []


def _guarded_cleanup(label: str, fn: Callable[[], Any], errors: List[str], logger) -> None:
    """Post-loop cleanup must never lose the response: each step is guarded
    independently and errors surface via ``cleanup_errors`` (#8049)."""
    try:
        fn()
    except Exception as err:
        errors.append(f"{label}: {err}")
        logger.error("finalize_turn: _%s failed: %s", label, err, exc_info=True)


def _resolve_budget_fallback(
    agent, *, final_response, api_call_count, interrupted, failed, messages, _turn_exit_reason,
    _pending_verification_response, _pending_verification_response_previewed, logger,
) -> Tuple[Any, Any, bool]:
    """Iteration-budget exhaustion. Returns ``(final_response, _turn_exit_reason,
    preserved_verification_fallback)``."""
    budget_exhausted = (
        api_call_count >= agent.max_iterations or agent.iteration_budget.remaining <= 0
    )
    preserved_verification_fallback = False
    if (
        final_response is None and budget_exhausted and not interrupted and not failed
        and str(_turn_exit_reason) in {"unknown", "budget_exhausted"}
    ):
        _turn_exit_reason = f"max_iterations_reached({api_call_count}/{agent.max_iterations})"
        if _pending_verification_response:
            # A verification gate withheld a composed answer, then the budget ran out:
            # preserve it rather than make another fallible call. The explicit pending
            # value is the provenance guard; unrelated error exits never enter here.
            # Previewed only if the reused candidate was actually streamed as interim.
            final_response = _pending_verification_response
            if _pending_verification_response_previewed:
                agent._response_was_previewed = True
            preserved_verification_fallback = True
        else:
            # _handle_max_iterations makes one extra toolless request for a summary.
            agent._emit_status(
                f"⚠️ Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
                "— asking model to summarise"
            )
            if not agent.quiet_mode:
                agent._safe_print(
                    f"\n⚠️  Iteration budget exhausted ({api_call_count}/{agent.max_iterations}) "
                    "— requesting summary..."
                )
            final_response = agent._handle_max_iterations(messages, api_call_count)

    # A kanban worker must record a terminal outcome whether or not a fallback path
    # was eligible, so the dispatcher learns the worker could not complete.
    _kanban_task = os.environ.get("HERMES_KANBAN_TASK") if budget_exhausted else None
    # If running as a kanban worker, signal the dispatcher that the worker could not complete (rather than
    # treating it as a protocol violation). This applies whether the user-facing fallback came from the
    # summary call or an explicitly pending continuation; both exhausted the task budget and must advance
    # the failure circuit. We route through ``_record_task_failure(outcome="timed_out")`` rather than
    # ``kanban_block`` so this counts toward the dispatcher's consecutive-failure circuit breaker (#29747
    # gap 2).
    # Bounded fallback (#87096): budget was exhausted but none of the normal fallback paths were eligible
    # (interrupted / failed / anomalous exit_reason). If running as a kanban worker we must still record a
    # terminal outcome so the task does not remain in an ambiguous lifecycle state. The worker's run is
    # closed via ``_record_task_failure`` (compare-and-swap receipt path) which is a no-op if another path
    # closed it — the CAS invariant in ``_end_run`` (``WHERE ended_at IS NULL``) guarantees idempotence.
    if _kanban_task:
        _record_kanban_budget_exhausted(_kanban_task, api_call_count, agent.max_iterations, logger)
    return final_response, _turn_exit_reason, preserved_verification_fallback


def _rollback_interrupted_preflight_display(agent, interrupted) -> None:
    """Roll back the preflight-seeded display count only when an interrupt wins before
    any provider response; compaction state (incl. ``-1``) stays with the real-usage
    path. Type-pinned guards keep MagicMock/SimpleNamespace doubles inert."""
    _preflight_snapshot = getattr(agent, "_turn_preflight_display_snapshot", None)
    if (
        interrupted is True
        and isinstance(_preflight_snapshot, int)
        and not isinstance(_preflight_snapshot, bool)
        and getattr(agent, "_turn_received_provider_response", False) is not True
        and getattr(agent, "context_compressor", None) is not None
    ):
        _rollback_fn = getattr(
            agent.context_compressor, "rollback_interrupted_preflight_display_tokens", None
        )
        if callable(_rollback_fn):
            _rollback_fn(_preflight_snapshot)


def _drop_transcript_scaffolding(agent, messages) -> None:
    """Strip private retry scaffolding first, or a later "continue" replays
    assistant("(empty)") / recovery nudges into the same empty-response loop. Only
    the synthetic verification nudges go; the assistant candidate persists (#65919)."""
    agent._drop_trailing_empty_response_scaffolding(messages)
    _drop_verification_continuation_scaffolding(messages)


def _recover_final_from_stream(agent, final_response, interrupted, failed) -> Tuple[Any, bool]:
    """An empty terminal completion is not authoritative when the stream already
    delivered text; recover before persist so a blank tail isn't frozen (#95514).
    Returns ``(final_response, recovered_from_stream)``. Called by the finalizer BEFORE
    the fallible tail-shaping/persist steps so the recovered text is already bound when
    one of them raises — a persist failure must not lose text the user already saw."""
    if interrupted or failed:
        return final_response, False
    _streamed = getattr(agent, "_current_streamed_assistant_text", "") or ""
    _streamed = _streamed.strip() if isinstance(_streamed, str) else ""
    if not (flatten_message_text(final_response).strip() if final_response else "") and _streamed:
        return _streamed, True
    return final_response, False


def _close_transcript_tail(agent, messages, final_response, interrupted, _recovered_from_stream) -> None:
    """Shape the transcript tail before the durable snapshot (scaffolding already dropped
    and ``final_response`` already stream-recovered by the caller)."""
    # An interrupt can leave a tool result as the tail; close the sequence so strict
    # providers don't see ``tool → user`` (placeholder: final_response is usually empty).
    if interrupted:
        from agent.message_sanitization import close_interrupted_tool_sequence
        close_interrupted_tool_sequence(messages, final_response)

    # Recovery ``break`` sites can return a final_response with no closing assistant
    # row; enforce "delivered final_response ⇒ assistant row" here. Compare content,
    # not role, so a matching verification candidate isn't dup'd.
    if final_response and not interrupted:
        # Some recovery/fallback paths return a real final_response without adding a closing assistant
        # message to the transcript (e.g. the partial-stream and prior-turn-content recovery ``break`` sites
        # in ``conversation_loop``). If persisted as-is, the durable session can end at a tool/user message
        # even though the caller — and the gateway platform — already saw a completed assistant response.
        # The next turn then replays a user-only backlog and the model re-answers every "unanswered"
        # message. Close the durable turn at the source, at the single chokepoint every recovery ``break``
        # flows through, so the invariant "delivered final_response ⇒ assistant row in transcript" holds
        # regardless of which path produced it. (#43849 / #44100) Compare content (not just role) so a
        # verification candidate that matches the final response is not duplicated at budget exhaustion.
        # (#65919 §7)
        _tail = messages[-1] if messages else None
        if not isinstance(_tail, dict) or _tail.get("role") != "assistant":
            append_message(messages, {"role": "assistant", "content": final_response})
        elif (
            _tail.get("content") != final_response
            and _assistant_row_missing_visible_text(_tail)
            and (_tail.get("tool_calls") or _recovered_from_stream)
        ):
            # Pure tool-call turn or stream-recovered blank (#95514): fill the persisted
            # blank row's content rather than append a second row.
            _tail["content"] = final_response
            stamp_message_timestamp(_tail)
            _tail.pop(_DB_PERSISTED_MARKER, None)
            agent._db_flush_scan_prefix = None

    # Request is complete, so replace API-local voice/model/skill guidance with the
    # clean user input before the durable snapshot (earlier flushes still needed them).
    # Earlier turn-start flushes use the DB-only override because their messages are still needed for the
    # API request; this finalizer runs after that request is complete (#48677 / #63766).
    _apply_override = getattr(agent, "_apply_persist_user_message_override", None)
    if callable(_apply_override):
        _apply_override(messages)


def _micro_compact_after_turn(agent, messages, final_response, logger) -> None:
    """Post-turn micro-compaction: absorb the oldest uncompacted exchange into the
    rolling summary before persist, amortizing compression across turns."""
    try:
        _compressor = getattr(agent, "context_compressor", None)
        # Strict `is True` + callable gates: plugin context engines and MagicMock
        # compressors pass duck checks and would wipe the transcript. Never run while
        # compression.checkpoint_required is armed (no checkpoint hook here), nor for
        # persistence-isolated agents (background review fork): that burns an aux-LLM
        # call on a throwaway transcript and could compact the CANONICAL session rows.
        if (
            _compressor
            and getattr(_compressor, '_micro_compact_enabled', False) is True
            and callable(getattr(_compressor, '_micro_compact', None))
            and final_response
            and getattr(agent, "compression_checkpoint_required", False) is not True
            and not getattr(agent, "_persist_disabled", False)
        ):
            _before = len(messages)
            _compacted = _compressor._micro_compact(messages)
            # Defrag rewrites the newest MICRO marker in place and pops _db_persisted;
            # the compressor flags us to invalidate the flush-scan cursor, else the
            # rewritten row is identity-skipped (stale).
            if getattr(_compressor, "_flush_scan_cursor_invalidated", False):
                _compressor._flush_scan_cursor_invalidated = False
                agent._db_flush_scan_prefix = None
            if isinstance(_compacted, list) and _compacted:
                messages[:] = _compacted
            if _before != len(messages):
                logger.info("Micro-compaction: %d -> %d messages", _before, len(messages))
    except Exception as _mc_err:
        logger.info("Micro-compaction failed: %s", _mc_err)


def _log_turn_exit(agent, messages, final_response, api_call_count, _turn_exit_reason, interrupted, logger) -> None:
    """Always INFO so agent.log captures WHY every turn ended; WARNING when the last
    message is a tool result (the "just stops" scenario)."""
    _last_msg_role = messages[-1].get("role") if messages else None
    _last_tool_name = None
    if _last_msg_role == "tool":
        # Walk back to the assistant message with the tool call.
        for _m in reversed(messages):
            if _m.get("role") == "assistant" and _m.get("tool_calls"):
                _tcs = _m["tool_calls"]
                if _tcs and isinstance(_tcs[0], dict):
                    _last_tool_name = _tcs[-1].get("function", {}).get("name")
                break

    _turn_tool_count = sum(
        1 for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
    )
    _diag_msg = (
        "Turn ended: reason=%s model=%s api_calls=%d/%d budget=%d/%d "
        "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
    )
    _diag_args = (
        _turn_exit_reason, agent.model, api_call_count, agent.max_iterations,
        agent.iteration_budget.used if agent.iteration_budget else 0,
        agent.iteration_budget.max_total if agent.iteration_budget else 0,
        _turn_tool_count, _last_msg_role, len(final_response) if final_response else 0,
        agent.session_id or "none",
    )
    if _last_msg_role == "tool" and not interrupted:
        logger.warning(
            "Turn ended with pending tool result (agent may appear stuck). "
            + _diag_msg + " last_tool=%s",
            *_diag_args, _last_tool_name,
        )
    else:
        logger.info(_diag_msg, *_diag_args)


def _append_file_mutation_footer(agent, final_response, logger):
    """Append the verifier advisory when ``write_file`` / ``patch`` calls failed and were
    never superseded by a successful write to the same path (surfaces over-claiming)."""
    try:
        # File-mutation verifier footer. This catches the specific case — reported by Ben Eng
        # (#15524-adjacent) — where a model issues a batch of parallel patches, half of them fail with
        # "Could not find old_string", and the model summarises the turn claiming every file was edited. The
        # user then has to manually run ``git status`` to catch the lie. With this footer the truth is
        # surfaced on every turn, so over-claiming is structurally impossible past the model. Gate: only
        # applied when a real text response exists for this turn and the user didn't interrupt.
        # Empty/interrupted turns already have other surface text that shouldn't be augmented.
        _failed = getattr(agent, "_turn_failed_file_mutations", None) or {}
        if _failed and agent._file_mutation_verifier_enabled():
            footer = agent._format_file_mutation_failure_footer(_failed)
            if footer:
                final_response = final_response.rstrip() + "\n\n" + footer
    except Exception as _ver_err:
        logger.debug("file-mutation verifier footer failed: %s", _ver_err)
    return final_response


def _explain_abnormal_exit(agent, final_response, _turn_exit_reason, preserved_verification_fallback, logger):
    """Turn-completion explainer: on abnormal exits, surface one explanation from
    ``_turn_exit_reason``. Only acts when no usable reply exists (empty, "(empty)",
    or a short unpunctuated fragment); ``text_response(...)`` exits stay silent."""
    try:
        if not agent._turn_completion_explainer_enabled():
            return final_response
        _stripped = (final_response or "").strip()
        _is_empty_terminal = _stripped in ("", "(empty)")
        # A short fragment not from a text_response exit and lacking sentence-ending
        # punctuation is treated as a truncated partial (#34452).
        _is_partial_fragment = (
            not _is_empty_terminal
            and not preserved_verification_fallback
            and not str(_turn_exit_reason).startswith("text_response")
            and len(_stripped) <= 24
            and _stripped[-1:] not in _SENTENCE_END
        )
        if _is_empty_terminal or _is_partial_fragment or str(_turn_exit_reason) == "partial_stream_recovery":
            _explanation = agent._format_turn_completion_explanation(
                _turn_exit_reason, getattr(agent, "_last_persistence_error_cause", None)
            )
            if _explanation:
                # Replace the bare sentinel; keep a partial fragment and append why.
                final_response = _explanation if _is_empty_terminal else _stripped + "\n\n" + _explanation
    except Exception as _exp_err:
        logger.debug("turn-completion explainer failed: %s", _exp_err)
    return final_response


def _last_turn_reasoning(messages) -> Optional[Any]:
    """Reasoning from the CURRENT turn only: stop at this turn's user message (#17055),
    but take the most recent non-empty reasoning since many providers emit it on the
    tool-call step and leave the final step with reasoning=None."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return None  # turn boundary — don't cross into prior turns
        if msg.get("role") == "assistant" and msg.get("reasoning"):
            return msg["reasoning"]
    return None


def _apply_output_hooks(
    agent, final_response, logger, *, platform, effective_task_id, turn_id, original_user_message,
    messages,
) -> Tuple[Any, bool, Optional[Any]]:
    """Fire ``transform_llm_output`` then ``post_llm_call`` once per turn after the tool loop.
    Returns ``(final_response, transformed, pre_transform_response)``."""
    transformed, pre_transform = False, None
    # First hook to return a string wins; None/empty leaves the text unchanged.
    for _hook_result in _invoke_hook_safely(
        "transform_llm_output", logger,
        response_text=final_response,
        session_id=agent.session_id or "",
        model=agent.model,
        platform=platform,
    ):
        if isinstance(_hook_result, str) and _hook_result:
            pre_transform, final_response, transformed = final_response, _hook_result, True
            break
    # post_llm_call (e.g. sync conversation data to an external memory system).
    _invoke_hook_safely(
        "post_llm_call", logger,
        session_id=agent.session_id,
        task_id=effective_task_id,
        turn_id=turn_id,
        user_message=original_user_message,
        assistant_response=final_response,
        conversation_history=list(messages),
        model=agent.model,
        platform=platform,
    )
    return final_response, transformed, pre_transform


def finalize_turn(
    agent, *, final_response, api_call_count, interrupted, failed, messages, conversation_history,
    effective_task_id, turn_id, user_message, original_user_message, _should_review_memory,
    _turn_exit_reason, _pending_verification_response=None,
    _pending_verification_response_previewed=False,
):
    """Run the post-loop finalization and return the turn ``result`` dict."""
    from agent.conversation_loop import logger

    final_response, _turn_exit_reason, preserved_verification_fallback = _resolve_budget_fallback(
        agent, final_response=final_response, api_call_count=api_call_count,
        interrupted=interrupted, failed=failed, messages=messages,
        _turn_exit_reason=_turn_exit_reason,
        _pending_verification_response=_pending_verification_response,
        _pending_verification_response_previewed=_pending_verification_response_previewed,
        logger=logger,
    )

    completed = (
        final_response is not None
        and not failed
        and (api_call_count < agent.max_iterations or str(_turn_exit_reason).startswith("text_response("))
    )

    _rollback_interrupted_preflight_display(agent, interrupted)

    _cleanup_errors: List[str] = []
    # ``user_message`` may be a multimodal list of parts; the trajectory format wants a string.
    _guarded_cleanup(
        "save_trajectory",
        lambda: agent._save_trajectory(messages, _summarize_user_message_for_log(user_message), completed),
        _cleanup_errors, logger,
    )
    _guarded_cleanup(
        "cleanup_task_resources", lambda: agent._cleanup_task_resources(effective_task_id),
        _cleanup_errors, logger,
    )
    # Persist only after the transcript tail is shaped and scaffolding removed. Each
    # sub-step runs in the same order as the original inline block, and the
    # stream-recovered ``final_response`` is rebound the moment it is computed — BEFORE
    # the fallible tail-shaping / override / micro-compaction / persist calls — so a
    # raise in any of them can't drop text the user already saw (#95514, #8049).
    def _persist_step():
        nonlocal final_response
        _drop_transcript_scaffolding(agent, messages)
        final_response, _recovered_from_stream = _recover_final_from_stream(
            agent, final_response, interrupted, failed
        )
        _close_transcript_tail(agent, messages, final_response, interrupted, _recovered_from_stream)
        if not interrupted and not failed:
            _micro_compact_after_turn(agent, messages, final_response, logger)
        agent._persist_session(messages, conversation_history)

    _guarded_cleanup("persist_session", _persist_step, _cleanup_errors, logger)

    # Keep the gateway's separate in-memory history snapshot current even on
    # cleanup error, so a later prompt isn't sent with a pre-turn snapshot.
    with suppress(Exception):
        agent._session_messages = messages

    _log_turn_exit(agent, messages, final_response, api_call_count, _turn_exit_reason, interrupted, logger)

    # Response transforms apply only to real, uninterrupted responses.
    if final_response and not interrupted:
        final_response = _append_file_mutation_footer(agent, final_response, logger)
    if not interrupted:
        final_response = _explain_abnormal_exit(
            agent, final_response, _turn_exit_reason, preserved_verification_fallback, logger,
        )

    _platform = getattr(agent, "platform", None) or ""
    _response_transformed = False
    _pre_transform_response = None
    if final_response and not interrupted:
        final_response, _response_transformed, _pre_transform_response = _apply_output_hooks(
            agent, final_response, logger, platform=_platform, effective_task_id=effective_task_id,
            turn_id=turn_id, original_user_message=original_user_message, messages=messages,
        )

    # Context engine observation hook: the turn finished with the finalized transcript.
    # Fail-open. ``_last_turn_usage`` is the last response's canonical usage dict, or
    # ``None`` on turns that never reached a provider response — by contract.
    try:
        from agent.conversation_loop import _notify_context_engine_turn_complete
        _notify_context_engine_turn_complete(
            agent, messages, usage=getattr(agent, "_last_turn_usage", None), logger=logger,
            turn_id=turn_id, task_id=effective_task_id, api_call_count=api_call_count,
            interrupted=interrupted, failed=failed, turn_exit_reason=_turn_exit_reason,
        )
    except Exception as exc:
        logger.warning("on_turn_complete notification failed: %s", exc)

    # Surrogate chokepoint: RAW SDK text with a lone UTF-16 surrogate crashes downstream
    # consumers (stdout, Telegram ``utf16_len``, JSON); scrub once where it leaves the loop.
    # Class-level surrogate chokepoint (#80366, #55143, #55309, #19819): ``final_response`` is often the RAW
    # SDK content (``assistant_message.content``), not the sanitized copy stored in history by
    # ``build_assistant_message``. Any lone UTF-16 surrogate (U+D800–U+DFFF) in it crashes downstream
    # consumers — oneshot stdout writes, Telegram's ``utf16_len`` length check, Signal formatting, JSON
    # envelope encodes — on every provider (Ollama, NVIDIA NIM, …). Scrub once here, where model text leaves
    # the conversation loop, so every delivery surface receives valid Unicode.
    if isinstance(final_response, str):
        final_response = _sanitize_surrogates(final_response)

    result = {
        "final_response": final_response,
        "last_reasoning": _last_turn_reasoning(messages),
        "messages": messages,
        "api_calls": api_call_count,
        "completed": completed,
        "turn_exit_reason": _turn_exit_reason,
        "failed": failed,
        "partial": False,  # True only when stopped due to invalid tool calls
        "interrupted": interrupted,
        "response_transformed": _response_transformed,
        "pre_transform_response": _pre_transform_response,
        "response_previewed": getattr(agent, "_response_was_previewed", False),
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        **{key: getattr(agent, f"session_{key}") for key in _SESSION_TOKEN_KEYS},
        "last_prompt_tokens": getattr(agent.context_compressor, "last_prompt_tokens", 0) or 0,
        **{key: getattr(agent, f"session_{key}") for key in _SESSION_COST_KEYS},
        # Requested service tier, for billing audits (`hermes -z --usage-file`).
        "service_tier": (
            (getattr(agent, "request_overrides", {}) or {}).get("extra_body") or {}
        ).get("service_tier"),
        "session_id": agent.session_id,
    }
    if agent._tool_guardrail_halt_decision is not None:
        result["guardrail"] = agent._tool_guardrail_halt_decision.to_metadata()
    # Persistence failures already set failed=True; also stamp `error` so the gateway
    # surfaces status="error" (desktop can toast) instead of a quiet complete frame, plus
    # the machine-readable cause 'session_persistence_failed:<locked|compression|...>'.
    if failed and str(_turn_exit_reason) == "session_persistence_failed":
        result["error"] = final_response or (
            "session storage could not be written — check the state database "
            "health (`hermes doctor`), then send your message again"
        )
        _cause = getattr(agent, "_last_persistence_error_cause", None)
        result["failure_reason"] = "session_persistence_failed:" + (_cause or "unknown")
    # Cleanup failures are surfaced, but the response is returned either way (#8049).
    if _cleanup_errors:
        result["cleanup_errors"] = _cleanup_errors
    # A /steer landing after the final assistant turn has no tool batch to drain into;
    # hand it back so it becomes the next user turn instead of being lost.
    _leftover_steer = agent._drain_pending_steer()
    if _leftover_steer:
        result["pending_steer"] = _leftover_steer
    agent._response_was_previewed = False
    if interrupted and agent._interrupt_message:
        result["interrupt_message"] = agent._interrupt_message
    agent.clear_interrupt()
    agent._stream_callback = None  # don't leak into future calls

    # Skill trigger is checked NOW — based on how many tool iterations THIS turn used.
    _should_review_skills = (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    )
    if _should_review_skills:
        agent._iters_since_skill = 0

    # External memory provider: sync the completed turn + queue next prefetch.
    agent._sync_external_memory_for_turn(
        original_user_message=original_user_message, final_response=final_response,
        interrupted=interrupted, messages=messages,
    )

    # Background memory/skill review runs AFTER delivery so it never competes with the
    # user's task. Suppressed by skip_background_review (e.g. cron): the fork costs
    # ~30K tokens / event with no human-in-the-loop benefit. Best-effort; the review
    # clones the snapshot structurally so its sanitizers can't reach the live transcript.
    if (
        final_response
        and not interrupted
        and not getattr(agent, "skip_background_review", False)
        and (_should_review_memory or _should_review_skills)
    ):
        with suppress(Exception):
            agent._spawn_background_review(
                messages_snapshot=list(messages), review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )

    # Memory provider on_session_end()/shutdown_all() are NOT called here:
    # run_conversation() runs once per message; CLI/gateway own session-end cleanup.
    from hermes_cli.session_hook_context import agent_session_identity
    identity = agent_session_identity(agent)
    identity["task_id"] = effective_task_id
    _invoke_hook_safely(
        "on_session_end", logger,
        **identity,
        turn_id=turn_id,
        completed=completed,
        failed=failed,
        interrupted=interrupted,
        turn_exit_reason=_turn_exit_reason,
        model=agent.model,
        platform=_platform,
    )

    agent._turn_preflight_display_snapshot = None
    agent._turn_received_provider_response = False
    return result
