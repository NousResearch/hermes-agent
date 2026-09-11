"""The agent conversation loop — extracted from ``run_agent.AIAgent``.

``run_conversation(agent, ...)`` drives one user turn (model call, tool dispatch,
retries, fallbacks, compression, post-turn hooks). Symbols that callers patch on
``run_agent`` (``handle_function_call``, ``_set_interrupt``, ``OpenAI``) resolve via
``_ra`` so those patches keep working."""

from __future__ import annotations

import inspect
import json
import logging
import re
import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.fast_mode import begin_turn as begin_fast_mode_turn
from agent.message_metadata import append_message
from agent.message_sanitization import _repair_tool_call_arguments, _sanitize_surrogates
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH, _estimate_tools_tokens_rough
from agent.process_bootstrap import _install_safe_stdio
from agent.prompt_builder import RUNTIME_ENVIRONMENT_END, RUNTIME_ENVIRONMENT_HEADING
from agent.prompt_caching import (
    build_prompt_cache_plan,
    effective_cache_ttl,
    strip_anthropic_cache_control,
    strip_anthropic_tool_cache_control,
)
from agent.runtime_cwd import resolve_agent_cwd
from agent.surface_switch import (
    identity_line_value, note_inert_pinned_tools, split_runtime_boundary, stage_surface_switch_note,
)
from agent.turn_context import PreflightCompressionTimedOut, build_turn_context
from agent.turn_retry_state import TurnRetryState
# Phase helpers of the turn loop, bound at import so a source-tree swap cannot load a
# skewed phase mid-turn.
from agent.turn_api_call import handle_api_interrupt, nous_rate_limit_guard, perform_api_call
from agent.turn_api_error import handle_api_error
from agent.turn_api_request import build_api_request
from agent.turn_final_response import finish_text_response
from agent.turn_finalizer import finalize_turn
from agent.turn_iteration_prep import (
    announce_api_call,
    apply_retry_restarts,
    begin_iteration,
    prepare_iteration,
)
from agent.turn_loop_errors import handle_outer_loop_error
from agent.turn_preflight_gate import run_preflight_gate
from agent.turn_request_assembly import assemble_api_request
from agent.turn_response_check import check_api_response
from agent.turn_response_intake import normalize_model_response
from agent.turn_tool_round import run_tool_round
from hermes_logging import set_session_context
from tools.skill_provenance import set_current_write_origin
from utils import base_url_host_matches

logger = logging.getLogger(__name__)

# Must mirror _STALE_TOOL_CALL_MARKER_RE in hermes_state.py; kept local so importing
# hermes_state (module-level DEFAULT_DB_PATH) is not forced at load time.
_STALE_MARKER_RE = re.compile(r"^\[[A-Za-z_][A-Za-z0-9_.-]*\]$")

# Shared by _apply_active_turn_redirect and the api_messages ghost-row filter so both sites cannot drift.
_INTERRUPT_SCAFFOLD_MARKER = "[This response was interrupted by a user correction.]"


# One-time wrap-up notice appended when a wall-clock run budget (--run-budget) crosses 80%.
RUN_BUDGET_WRAPUP_NOTICE = (
    "[SYSTEM NOTICE — run time budget nearly exhausted] Run time budget nearly exhausted. "
    "Stop new discovery/verification work now. Produce the required final deliverable "
    "(answer/JSON/summary) from the state you already have, completing only mandatory writes."
)


def _midturn_request_pressure_tokens(
    agent: Any, api_messages: List[Dict[str, Any]], effective_system: str, approx_tokens: int
) -> int:
    """Token figure the mid-turn pre-API compression guard compares: the pruned
    native-Responses estimate when native compaction eligibility is proven (the generic
    estimate overstates the wire on compacted sessions, #96995), else messages+tools.
    The system prompt is counted exactly once.

    When the upcoming request is eligible for native Responses compaction the transport will
    checkpoint-prune the payload before sending, so the generic durable-history estimate overstates the wire
    by orders of magnitude on a compacted session and fires a 600s local compression the main request never
    needed (#96995).
    """
    try:
        from agent.codex_responses_adapter import estimate_native_responses_preflight_tokens
        native = estimate_native_responses_preflight_tokens(
            agent, api_messages, system_prompt=effective_system or "",
            tools=getattr(agent, "tools", None) or None,
        )
        if isinstance(native, int) and not isinstance(native, bool) and native >= 0:
            return native
    except Exception:
        logger.debug(
            "native Responses mid-turn estimate unavailable; using generic transcript estimate",
            exc_info=True,
        )
    return approx_tokens + (_estimate_tools_tokens_rough(agent.tools) if agent.tools else 0)


def _review_input_budget_exhausted(agent: Any) -> bool:
    """True when a detached review fork has replayed its aggregate input budget.

    Only forks with an explicit ``_review_input_token_budget`` are gated (#93057). Fires
    at the top of the NEXT iteration, so the budget-crossing request completes first."""
    budget = getattr(agent, "_review_input_token_budget", None)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        return False
    used = getattr(agent, "session_input_tokens", 0)
    return isinstance(used, int) and not isinstance(used, bool) and used >= budget


def _maybe_inject_run_budget_wrapup(agent: Any, messages: List[Dict[str, Any]]) -> bool:
    """Inject the one-time wall-clock wrap-up notice when past 80% of budget.

    Appends to the NEWEST ``role:"tool"`` message (cache-safe, like /steer); latches
    ``_run_budget_wrapup_injected`` only on a successful append."""
    budget = getattr(agent, "run_budget_seconds", None)
    started = getattr(agent, "_run_budget_started_at", None)
    if not budget or not started or getattr(agent, "_run_budget_wrapup_injected", False) or (
        (time.time() - started) < 0.8 * float(budget)
    ):
        return False
    from agent.context_compressor import _DB_PERSISTED_MARKER
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            # Only the current tool-result tail is mutable; an older turn may already be
            # cached (same contract as _maybe_inject_iteration_budget_warning).
            if msg.get(_DB_PERSISTED_MARKER):
                return False
            existing = msg.get("content", "")
            if isinstance(existing, str):
                msg["content"] = existing + f"\n\n{RUN_BUDGET_WRAPUP_NOTICE}"
            else:  # multimodal content blocks — append a text block
                try:
                    msg["content"] = [*(existing or []), {"type": "text", "text": RUN_BUDGET_WRAPUP_NOTICE}]
                except Exception:
                    return False
            agent._run_budget_wrapup_injected = True
            logger.info(
                "Run budget wrap-up notice injected (budget=%.0fs, elapsed=%.0fs)",
                float(budget), time.time() - started,
            )
            return True
    return False


def _restore_user_after_reference_handoff(
    messages: List[Dict[str, Any]], user_message: Any
) -> bool:
    """Re-append this turn's real user ask when compaction left only a handoff (#80622).
    Returns True when a restore append happened."""
    if isinstance(user_message, str):
        restorable = bool(user_message.strip())
    else:
        restorable = isinstance(user_message, list) and bool(user_message)
    if not restorable:
        return False
    last = messages[-1] if messages else None
    if isinstance(last, dict) and last.get("role") == "user" and last.get("content") == user_message:
        return False
    append_message(messages, {"role": "user", "content": user_message})
    return True


def _should_skip_model_call_for_reference_handoff(
    messages: List[Dict[str, Any]], user_message: Any
) -> bool:
    """Guard post-compaction continues against sole-handoff active turns (#80622)."""
    from agent.context_compressor import reference_handoff_would_drive_next_model_call
    # A restored ask is an actionable non-synthetic user row appended after the
    # handoff — by construction the handoff no longer drives.
    return reference_handoff_would_drive_next_model_call(messages) and not (
        _restore_user_after_reference_handoff(messages, user_message)
    )


# Fallback final_response for the sole-handoff skip (#80622); finalize_turn appends it as a
# fresh assistant row, so it must not replay the last assistant text.
# Deliberately NOT a replay of the last assistant text: finalize_turn's non-assistant-tail chokepoint
# (#43849) appends final_response as a fresh assistant row, so recovering the previous turn's prose here
# would duplicate it in the durable transcript AND re-deliver it to the user as if it were this turn's
# answer. A short status is honest and idempotent.
_HANDOFF_SKIP_FINAL_RESPONSE = (
    "Context was compacted. The previous response is complete — awaiting your next message."
)

# Terminal final_response when compression timed out while the request was still oversized (#98722).
# Terminal final_response for a turn ended because context compression hit its host progress-aware timeout
# while the request was still oversized (#98722, salvaged from #98741). Sending the unchanged request would
# only bounce off the provider's overflow error and re-enter compression in the same turn.
_COMPRESSION_TIMEOUT_FINAL_RESPONSE = (
    "Context compression timed out without reducing this conversation. No messages were "
    "dropped. Start a fresh session with /new, or check auxiliary.compression before retrying /compress."
)


# Stable prefix ACP/TUI match on to treat the text as cancellation metadata, not assistant prose.
INTERRUPT_WAITING_FOR_MODEL_PREFIX = "Operation interrupted: waiting for model response ("


def _should_rearm_compression_budget(
    compression_attempts: int, *, completed_compaction_pending: bool, prompt_tokens: int, threshold_tokens: int
) -> bool:
    """True once a provider proves a completed compaction worked: rough estimates cannot
    rearm the anti-thrash budget, only the completed-compaction latch plus a positive
    normalized prompt count below the threshold."""
    return bool(
        compression_attempts and completed_compaction_pending and 0 < prompt_tokens < threshold_tokens
    )


# Modules whose presence in a traceback (without any API-call module) marks a
# deterministic local bug not worth retrying. NEVER add "conversation_loop" or
# "run_agent": every exception passes through them; _hit_local would be True (#66267)
_LOCAL_PROCESSING_MODULES = frozenset({
    "agent_runtime_helpers",
    "message_content",
    "message_sanitization",
    "chat_completion_helpers",  # only local when NOT also an API-call module
})
_API_CALL_MODULES = frozenset({"chat_completion_helpers"})

# Max outer-loop exceptions per user turn before giving up; only exceptions that
# ESCAPE the inner retry/fallback machinery count, so this can be small (#92450).
_MAX_OUTER_LOOP_ERRORS = 8


def _is_interpreter_shutdown_error(exc: Exception) -> bool:
    """True for a fatal interpreter-shutdown RuntimeError. The RuntimeError type gate
    stays here: a ValueError carrying similar text must not match (#93269)."""
    if isinstance(exc, RuntimeError):
        # ── Interpreter finalization: abandon immediately ── The process is exiting (TUI quit, SIGTERM,
        # one-shot done) while this turn — typically the post-turn review fork's daemon thread — is
        # mid-flight. Retries, credential rotation, and fallbacks are all futile ("cannot schedule new
        # futures..."), and the buffered ⚠️/❌ retry trace spams the shell after the TUI already exited. End
        # the turn with a single log line: no print, no traceback, no debug dump, no retry. Same class as
        # cron delivery (#55924/#58720) and concurrent tool submission — shared predicate.
        from tools.interpreter_shutdown import interpreter_shutting_down
        return interpreter_shutting_down(exc)
    return False


def _moa_client_consumes_prepared_request(client: Any) -> bool:
    """True when ``client`` is the in-process MoA facade (only ``MoAChatCompletions`` exposes
    ``prepare()``; other clients raise TypeError on ``_moa_prepared_request`` even while
    ``agent.provider`` stays ``"moa"``)."""
    completions = getattr(getattr(client, "chat", None), "completions", None)
    return callable(getattr(completions, "prepare", None))


def _join_truncated_parts(parts: List[str]) -> str:
    """Join continuation fragments, adding a newline where two would glue together (#78577)."""
    joined = ""
    for part in parts:
        if joined and not joined[-1].isspace() and part and not part[0].isspace():
            joined += "\n"
        joined += part
    return joined


def _moa_reference_metrics_for_hook(agent: Any) -> Any:
    """Per-advisor metrics for post_api_request, or None off the MoA path (a plugin only
    sees the aggregator generation; this carries the per-slot advisor spend)."""
    client = getattr(agent, "client", None)
    getter = getattr(client, "last_reference_metrics", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


def _apply_active_turn_redirect(agent: Any, messages: List[Dict[str, Any]], text: str) -> None:
    """Append a provider-safe checkpoint and correction to the live turn so role alternation
    holds and cached messages stay byte-identical. INVARIANTS: raw chain-of-thought never enters
    replayable content (inlined CoT reads as a prefill jailbreak and bricks the session with
    empty-response storms); the interruption scaffold is replay text carried only in the user
    correction's ``api_content``; an on-screen-empty placeholder is ``display_kind=hidden``."""
    visible = agent._strip_think_blocks(getattr(agent, "_current_streamed_assistant_text", "") or "").strip()

    checkpoint_parts = [_INTERRUPT_SCAFFOLD_MARKER]
    if visible:
        checkpoint_parts += ["Visible response before the interruption:", visible]
    checkpoint = "\n\n".join(checkpoint_parts)
    correction = f"[Context from the interrupted assistant response]\n{checkpoint}\n\n{text}"

    # The live tail is normally user or tool, so an assistant placeholder + correction
    # keeps strict alternation; if the tail is already assistant, the checkpoint is folded
    # into the user correction instead of creating assistant→assistant. The placeholder
    # preserves alternation only — scaffold bytes must never land in it, since api_content
    # is substituted back into content on replay (#81841).
    if not (messages and messages[-1].get("role") == "assistant"):
        placeholder: Dict[str, Any] = {"role": "assistant", "content": visible or ""}
        if not visible:
            placeholder["display_kind"] = "hidden"
            # Hidden row, but a non-empty neutral api_content so the pre-call sanitizer
            # does not re-heal it every call (#88955). Never _INTERRUPT_SCAFFOLD_MARKER:
            # as assistant text the model echoes it (#81841).
            from agent.agent_runtime_helpers import _INTERRUPTED_PLACEHOLDER
            placeholder["api_content"] = _INTERRUPTED_PLACEHOLDER
        append_message(messages, placeholder)
    # Transcript shows the user's own words; the provider replays the scaffolded form.
    append_message(messages, {"role": "user", "content": text, "api_content": correction})

    # Stateful scrubber for <memory-context> spans split across stream deltas (#5719).  sanitize_context()
    # alone can't survive chunk boundaries because the block regex needs both tags in one string.
    # Stateful scrubber for reasoning/thinking tags in streamed deltas (#17924). Replaces the per-delta
    # _strip_think_blocks regex that destroyed downstream state (e.g. MiniMax-M2.7 streaming '<think>' as
    # delta1 and 'Let me check' as delta2 — the regex erased delta1, so downstream state machines never
    # learned a block was open and leaked delta2 as content).
    agent._current_streamed_assistant_text = ""
    agent._stream_needs_break = True


def _is_copilot_provider(agent: Any) -> bool:
    """Delegate to ``AIAgent._is_copilot_provider``; the fallback keeps the ``github-copilot`` /
    ``github`` aliases so credential recovery is not skipped for them."""
    try:
        return bool(agent._is_copilot_provider())
    except Exception:
        return (getattr(agent, "provider", "") or "").strip().lower() in {
            "copilot",
            "github-copilot",
            "github",
        }


def _is_stale_copilot_credential_error(status_code: Optional[int], error_message: str) -> bool:
    """Detect a Copilot 400 that is really a STALE / DEGRADED credential (status 400 AND an
    integrator/model-not-supported marker, so a wrong model name never triggers the
    single-shot re-exchange). Caller enforces scoping/guard."""
    lowered = (error_message or "").lower()
    if status_code != 400 and "error code: 400" not in lowered:
        return False
    return any(marker in lowered for marker in (
        "model_not_available_for_integrator",
        "not available for integrator",
        "model_not_supported",
        "the requested model is not supported",
    ))


def _pressure_with_real_floor(compressor: Any, rough_tokens: int) -> int:
    """Floor the ROUGH pre-API pressure estimate at the last REAL prompt size.

    Applied only on the fallback path -- when ``anchored_context_tokens`` has
    no valid anchor (first request, transcript rewritten under the anchor,
    provider never reported usage). A valid anchor is provider-exact and is
    used as-is; in particular on MoA turns the anchor deliberately uses the
    pre-fold aggregator usage while ``last_real_prompt_tokens`` holds the
    folded figure, so flooring an anchored value would re-add fan-out tokens
    the anchor exists to exclude.

    On the rough path, non-ASCII text (Cyrillic, Greek, Polish, ...)
    under-counts by up to ~2x, so a session can sit at the provider's real
    context ceiling while the rough figure stays under the compaction
    threshold -- on silent-clip providers (ollama /v1) that is a truncation
    death spiral the reactive overflow handler never sees (observed live:
    real prompts 64,842->64,995 against a 55,705 threshold). The provider's
    last reported prompt_tokens is authoritative; never let the rough figure
    fall below it. Skipped for exactly one turn after a compaction, when
    last_real_prompt_tokens still holds the stale pre-compression value
    (#36718's awaiting_real_usage_after_compression window).
    """
    last_real = int(getattr(compressor, "last_real_prompt_tokens", 0) or 0)
    if last_real > rough_tokens and not getattr(
        compressor, "awaiting_real_usage_after_compression", False
    ):
        return last_real
    return rough_tokens


def _ollama_context_limit_error(agent: Any, request_tokens: int) -> Optional[str]:
    """Return a user-facing error when Ollama is loaded with too little context."""
    runtime_ctx = getattr(agent, "_ollama_num_ctx", None)
    if (
        not getattr(agent, "tools", None)
        or not isinstance(runtime_ctx, int)
        or not 0 < runtime_ctx < MINIMUM_CONTEXT_LENGTH
    ):
        return None

    model = getattr(agent, "model", "") or "the selected model"
    logger.warning(
        "Ollama runtime context too small for Hermes tool use: model=%s provider=%s base_url=%s "
        "runtime_context=%d minimum_context=%d estimated_request_tokens=%d tool_count=%d session=%s",
        model, getattr(agent, "provider", "") or "unknown",
        getattr(agent, "base_url", "") or "unknown base URL", runtime_ctx, MINIMUM_CONTEXT_LENGTH,
        request_tokens, len(getattr(agent, "tools", None) or []),
        getattr(agent, "session_id", None) or "none",
    )
    return (
        f"Ollama loaded `{model}` with only {runtime_ctx:,} tokens of runtime context, but Hermes "
        f"needs at least {MINIMUM_CONTEXT_LENGTH:,} tokens for reliable tool use.\n\n"
        "Increase the Ollama context for this model and restart/reload the model before trying "
        "again. A known-good starting point is 65,536 tokens. In Hermes config, set "
        "`model.ollama_num_ctx: 65536` (and `model.context_length: 65536` if you also override the "
        "displayed model context). If you manage the model through an Ollama Modelfile, set "
        "`PARAMETER num_ctx 65536` there instead."
    )


def _maybe_grow_local_window(agent: Any, compressor: Any,
                             request_tokens: int) -> Optional[int]:
    """Grow a managed local model's context window before compressing; returns the new
    window when the ladder granted one, else None."""
    provider = (getattr(agent, "provider", "") or "").strip().lower()
    base_url = getattr(agent, "base_url", "") or ""
    if provider not in ("llamacpp", "llama.cpp", "llama-cpp", "custom") or not (
        "127.0.0.1" in base_url or "localhost" in base_url
    ):
        return None
    try:
        from hermes_cli.local_runtime.growth import maybe_grow_window
        current_window = int(getattr(compressor, "context_length", 0) or 0)
        if current_window <= 0:
            return None
        return maybe_grow_window(
            getattr(agent, "model", "") or "", base_url=base_url,
            session_tokens=int(request_tokens), current_window=current_window,
        )
    except Exception as exc:  # noqa: BLE001 — growth must never break a turn
        logger.debug("local window growth check failed: %s", exc)
        return None


def _ra():
    """Lazy ``run_agent`` reference so patches on ``run_agent.*`` reach this code path."""
    import run_agent
    return run_agent


def _nous_entitlement_message(capability: str) -> str:
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )
        account_info = get_nous_portal_account_info(force_fresh=True)
        return format_nous_portal_entitlement_message(
            account_info, capability=capability, in_chat=True
        ) or ""
    except Exception:
        return ""


def _print_guidance(agent, message: str) -> bool:
    """Print each line of ``message`` as a 💡 hint; False when there is nothing to print."""
    if not message:
        return False
    for line in message.splitlines():
        agent._vprint(f"{agent.log_prefix}   💡 {line}", force=True)
    return True


def _print_nous_entitlement_guidance(agent, capability: str) -> bool:
    return _print_guidance(agent, _nous_entitlement_message(capability))


def _system_prompt_for_hooks(api_kwargs: Any, request_messages: Any) -> Any:
    """System prompt as sent to the provider (``system`` / ``instructions`` / ``messages[0]``)
    for observability hooks; None when the request carries none."""
    system_prompt = api_kwargs.get("system")
    if system_prompt is None:
        system_prompt = api_kwargs.get("instructions")
    if system_prompt is None and isinstance(request_messages, list) and request_messages:
        first = request_messages[0]
        if isinstance(first, dict) and first.get("role") == "system":
            system_prompt = first.get("content")
    return system_prompt


def _is_nous_inference_route(provider: str, base_url: str) -> bool:
    return (provider or "").strip().lower() == "nous" or base_url_host_matches(
        str(base_url or ""), "inference-api.nousresearch.com"
    )


def _billing_or_entitlement_message(
    *, capability: str, provider: str, base_url: str, model: str, unverified: bool = False
) -> str:
    if _is_nous_inference_route(provider, base_url):
        return _nous_entitlement_message(capability)

    provider_label = (provider or "").strip() or "the selected provider"
    model_label = (model or "").strip() or "the selected model"

    # Anthropic Pro/Max OAuth surfaces "extra usage" exhaustion as a hard 400 — "add credits"
    # does not apply. ``unverified`` (#82154): the same 400 is returned for a server-side
    # content-filter rejection, so hedge and name the other cause.
    if (provider or "").strip().lower() == "anthropic":
        switch = (
            "You can also switch to an Anthropic API key or another provider with "
            "/model <model> --provider <provider>."
        )
        if unverified:
            return "\n".join([
                f"{provider_label} reported that your Claude subscription usage may be exhausted for "
                f"{model_label} (included quota + extra-usage credits) — but this specific error is "
                "not proof of a billing problem.",
                "If https://claude.ai/settings/usage still shows quota remaining, this is probably NOT "
                "a billing problem: on a Claude subscription (OAuth) token Anthropic returns this same "
                "message when its content filter rejects part of the request — typically a phrase in "
                "the system prompt.",
                "If usage really is exhausted: wait for the billing cycle to reset, or add extra usage "
                "at https://claude.ai/settings/usage",
                switch,
                # The exhaustion latch replays the stored error without a request.
                "Retry with a fresh credential state: `hermes auth reset anthropic`. Until that "
                "cooldown clears, this error can be replayed from cache without contacting the API.",
            ])
        return "\n".join([
            f"{provider_label} reported that your Claude subscription usage is exhausted for "
            f"{model_label} (included quota + extra-usage credits).",
            "Options: wait for the billing cycle to reset, or add extra usage at https://claude.ai/settings/usage",
            switch,
        ])

    # Provider-agnostic billing URL so every text surface shows the same actionable link.
    try:
        from agent.billing_links import build_billing_block
        _link = build_billing_block(provider=provider, base_url=base_url, model=model)
        provider_label = _link.provider_label or provider_label
        billing_url = _link.billing_url
    except Exception:
        billing_url = None
    return "\n".join([
        f"{provider_label} reported that billing, credits, or account entitlement is exhausted for {model_label}.",
        "Add credits or update billing with that provider, then retry.",
        *([f"{provider_label} billing: {billing_url}"] if billing_url else []),
        "You can switch providers temporarily with /model <model> --provider <provider>.",
    ])


def _billing_block_dict(provider, base_url, model, message="", *, unverified: bool = False) -> Optional[dict]:
    """Best-effort structured billing descriptor (None if billing_links is unavailable)."""
    try:
        from agent.billing_links import build_billing_block
        block = build_billing_block(
            provider=provider, base_url=str(base_url), model=model, message=message
        ).to_dict()
    except Exception:
        return None
    if block is not None and unverified:
        block["unverified"] = True  # every surface rendering the block can hedge too (#82154)
    return block


def _billing_terminal_label(summary: str, unverified: bool) -> str:
    """Terminal-failure prefix for a billing-classified error; ``unverified`` (#82154) must
    not assert exhaustion as fact."""
    if unverified:
        return (
            "Provider reported usage/credit exhaustion (unverified — the same "
            f"error can be a content-filter rejection, not billing): {summary}"
        )
    return f"Billing or credits exhausted: {summary}"


def _billing_failure_result(
    *, classified, summary: str, messages, api_call_count: int, provider: str, base_url, model: str,
    guidance: Optional[str] = None,
) -> dict:
    """Structured terminal result for a billing-classified failure — the single construction
    point for the non-retryable abort and max-retries paths (#82154)."""
    unverified = bool(getattr(classified, "billing_unverified", False))
    if guidance is None:
        guidance = _billing_or_entitlement_message(
            capability="model access", provider=provider, base_url=str(base_url), model=model,
            unverified=unverified,
        )
    final = _billing_terminal_label(summary, unverified) + (f"\n\n{guidance}" if guidance else "")
    return {
        "final_response": final, "messages": messages, "api_calls": api_call_count,
        "completed": False, "failed": True, "error": summary,
        "failure_reason": classified.reason.value,
        # Classifier's own retry verdict so the UI shows Retry only when a re-run can differ.
        "failure_retryable": bool(classified.retryable),
        "billing_unverified": unverified,
        "billing_block": _billing_block_dict(provider, base_url, model, guidance, unverified=unverified),
    }


def _print_billing_or_entitlement_guidance(
    agent, *, capability: str, provider: str, base_url: str, model: str, unverified: bool = False
) -> bool:
    return _print_guidance(agent, _billing_or_entitlement_message(
        capability=capability, provider=provider, base_url=base_url, model=model,
        unverified=unverified,
    ))


def _bot_chat_prompt_stale(agent, stored_prompt: str) -> bool:
    """Bot Chat capability epoch check for a stored prompt.

    The stored prompt embeds a capability fingerprint; a mismatch is a deliberate
    once-per-change rebuild. Unstamped prompts never match; probe failures fail closed
    to "reuse" so the cache is kept. Legacy upgrade: a Bot Chat prompt predating the
    epoch mechanism gets ONE title-gated migration rebuild; the stamped result cannot
    re-fire."""
    try:
        from tools.bot_mode_probe import (
            BOT_CHAT_TITLE,
            stored_bot_chat_prompt_needs_upgrade,
            stored_prompt_capability_stale,
        )
        home = None
        try:
            from agent.system_prompt import _agent_home
            home = _agent_home(agent)
        except Exception:
            pass
        if stored_prompt_capability_stale(stored_prompt, home):
            return True
        if not getattr(agent, "_bot_mode_protocol", True):
            return False
        title = str(getattr(agent, "_session_title_hint", "") or "").strip()
        if not title and agent._session_db and agent.session_id:
            try:
                title = str(agent._session_db.get_session_title(agent.session_id) or "").strip()
            except Exception:
                title = ""
        return title == BOT_CHAT_TITLE and bool(stored_bot_chat_prompt_needs_upgrade(stored_prompt, home))
    except Exception:
        return False


def _persist_system_prompt(agent, failure_message: str, *, persist_tools: bool = False) -> None:
    """Persist ``agent._cached_system_prompt`` to the session row; failures log at WARNING
    (with ``failure_message``) because the gateway path (fresh AIAgent per turn) reads
    this row every turn, so a silent failure breaks prefix-cache reuse."""
    if not agent._session_db:
        return
    try:
        agent._session_db.update_system_prompt(agent.session_id, agent._cached_system_prompt)
        if persist_tools:
            from tools.mcp_tool_agent import persist_agent_tool_names
            persist_agent_tool_names(agent)
    except Exception as exc:
        logger.warning(failure_message, agent.session_id, exc)


def _restore_or_build_system_prompt(agent, system_message, conversation_history):
    """Restore the cached system prompt from the session DB or build it fresh.

    Mutates ``agent._cached_system_prompt`` and persists a freshly-built prompt on first
    build. Row states ``missing``/``null``/``empty``/``present`` are logged and DB
    failures log at WARNING so silent prefix-cache misses show in ``agent.log``."""
    stored_prompt = None
    stored_state = "missing"
    session_row = None
    if conversation_history and agent._session_db:
        try:
            session_row = agent._session_db.get_session(agent.session_id)
            if session_row is not None:
                raw_prompt = session_row.get("system_prompt")
                stored_state = "null" if raw_prompt is None else ("empty" if raw_prompt == "" else "present")
                stored_prompt = raw_prompt or None
        except Exception as exc:
            logger.warning(
                "Session DB get_session failed for system-prompt restore (session=%s): %s. "
                "Falling back to fresh build — prefix cache will miss for this turn.",
                agent.session_id, exc,
            )

    if stored_prompt and _stored_prompt_matches_runtime(agent, stored_prompt):
        if _bot_chat_prompt_stale(agent, stored_prompt):
            logger.info(
                "Bot Chat capability epoch changed for session %s; rebuilding system prompt to "
                "adopt the new capability surface (one-time prefix-cache break).",
                agent.session_id,
            )
            agent._session_title_hint = "Bot Chat"
            # The skills index cache (LRU + disk snapshot) does not watch the skills
            # dir; a capability refresh must rebuild THROUGH it or new skills are lost.
            try:
                from agent.prompt_builder import clear_skills_system_prompt_cache
                clear_skills_system_prompt_cache(clear_snapshot=True)
            except Exception:
                pass
            agent._cached_system_prompt = agent._build_system_prompt(system_message)
            stage_surface_switch_note(agent, agent._cached_system_prompt, conversation_history)
            # Persist so the NEXT turn restores the new bytes verbatim (cache break is
            # once per capability change). on_session_start not re-fired: continuation.
            _persist_system_prompt(
                agent,
                "Session DB update_system_prompt failed after Bot Chat capability refresh "
                "(session=%s): %s. The refresh will re-fire next turn.",
            )
            return
        # Continuing session — reuse the exact system prompt from the
        # previous turn so the Anthropic cache prefix matches.
        agent._cached_system_prompt = stored_prompt
        # The reused bytes may describe the surface this conversation STARTED on; correct that
        # at the tail of the request instead of rebuilding the prompt in front of it (#104414).
        announced_switch = stage_surface_switch_note(agent, stored_prompt, conversation_history)
        # Same contract for tools[]: pin the array to the order this session already
        # sent (tools freeze) instead of re-probing every check_fn on a fresh AIAgent.
        # The pin holds ON the announcing turn too.  tools[] is serialized AHEAD of the system
        # prompt this branch just preserved, so dropping the previous surface's toolset would
        # change the request at token 0 and re-prefill everything behind it — the exact cost
        # #104414 is about, paid on the exact turn we are here to make cheap.  The merge still
        # ADDS what the new surface brought (a tui -> desktop switch pays a break no freeze can
        # avoid), and what it carries FORWARD is named in the note instead, so a tool that can
        # only answer ``tool_error("desktop only")`` here does not read as a live capability.
        try:
            saved_tools = session_row.get("tool_names") if session_row else None
            if saved_tools:
                from tools.mcp_tool_agent import agent_tool_names, restore_agent_tool_prefix
                # Captured BEFORE the pin merges the previous surface's tools back in.
                built_for_this_surface = agent_tool_names(agent) if announced_switch else []
                restore_agent_tool_prefix(agent, json.loads(saved_tools))
                if announced_switch:
                    note_inert_pinned_tools(agent, built_for_this_surface)
        except Exception:
            logger.debug("tool prefix restore skipped", exc_info=True)
        # Prompt-section callbacks are new-session-only; recover their frozen bytes
        # from the persisted prompt so a compression rebuild keeps them. The static
        # prefix is not persisted either; rebuild it for the early cache breakpoint or
        # fresh-per-turn gateway agents fall back to the single-breakpoint layout
        # (reconstruct_static_prefix gates on _use_prompt_caching, fails open to legacy).
        from agent.system_prompt import reconstruct_static_prefix, restore_plugin_prompt_sections
        restore_plugin_prompt_sections(agent, stored_prompt)
        reconstruct_static_prefix(agent, system_message=system_message)
        return
    if stored_prompt:
        stored_state = "stale_runtime"
        logger.info(
            "Stored system prompt for session %s has stale runtime identity; "
            "rebuilding for model=%s provider=%s.",
            agent.session_id, getattr(agent, "model", "") or "", getattr(agent, "provider", "") or "",
        )

    if conversation_history and stored_state in ("null", "empty"):
        # Continuing session with an unusable stored prompt: every turn now rebuilds
        # and the prefix cache misses every time.
        logger.warning(
            "Stored system prompt for session %s is %s; rebuilding from scratch this turn. Prefix "
            "cache will miss until the rebuild persists. Investigate the previous turn's "
            "update_system_prompt write path.",
            agent.session_id, stored_state,
        )

    # First turn of a new session (or recovering from a broken stored prompt).
    agent._cached_system_prompt = agent._build_system_prompt(system_message)

    # The rebuilt prompt describes the CURRENT surface, but a surface note left in the
    # transcript by an earlier switch does not — retire it here too, or a rebuild for an
    # unrelated reason (a model switch) would leave the newest interface statement in the
    # request naming a surface the conversation has left (#104414).
    stage_surface_switch_note(agent, agent._cached_system_prompt, conversation_history)

    # Persistence-disabled forks share their parent's session ID and are not real sessions.
    if not getattr(agent, "_persist_disabled", False):
        try:
            from hermes_cli.lifecycle import invoke_hook as _invoke_hook
            _invoke_hook(
                "on_session_start", session_id=agent.session_id, model=agent.model,
                platform=getattr(agent, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("on_session_start hook failed: %s", exc)

    # Cold-start credits seed (L3) fallback for the first-turn path; TUI/desktop seed at
    # session open, so this is idempotent (skips when _credits_state exists). Fail-open.
    try:
        from agent.credits_tracker import seed_credits_at_session_start
        seed_credits_at_session_start(agent)
    except Exception:
        logger.debug("cold-start credits seed failed (fail-open)", exc_info=True)

    _persist_system_prompt(
        agent,
        "Session DB update_system_prompt failed for session %s: %s. Subsequent turns will "
        "rebuild the system prompt and miss the prefix cache.",
        persist_tools=True,
    )


def _stored_prompt_matches_runtime(agent, prompt: str) -> bool:
    """Return False when the persisted runtime-identity lines are stale."""

    _identity, runtime_marker, runtime = split_runtime_boundary(prompt)

    def host_info_value(label: str) -> str:
        """New prompts delimit runtime hints; legacy prompts put them before context."""
        prefix = f"{label}:"
        host_lines = (runtime.split("\n\n", 1)[0] if runtime_marker else prompt).splitlines()
        for idx, line in enumerate(host_lines):
            if line.startswith("User home directory:"):
                for candidate in host_lines[idx + 1: idx + 4]:
                    if candidate.startswith(prefix):
                        return candidate[len(prefix):].strip()
        return ""

    # Model/provider identity, then cwd drift.  A cwd change is a real content change (context
    # files, the workspace snapshot and the coding posture are all resolved from it), so it
    # still rebuilds; the runtime surface does not (agent/surface_switch.py).
    for label, attr in (("Model", "model"), ("Provider", "provider")):
        stored = identity_line_value(prompt, label)
        current = str(getattr(agent, attr, "") or "").strip()
        if stored and current and stored != current:
            return False
    # Compare against resolve_agent_cwd() — the SAME resolver used to build the
    # prompt — so TERMINAL_CWD sessions are not falsely rejected.
    stored_cwd = host_info_value("Current working directory")
    if stored_cwd and stored_cwd != str(resolve_agent_cwd()):
        return False
    # Platform is deliberately NOT an identity field: a surface switch does not invalidate the
    # stored bytes, it only makes their interface section out of date, and that is corrected by
    # agent.surface_switch.stage_surface_switch_note without touching the cached prefix (#104414).
    return True


# Named so _is_synthetic_compression_user_turn can recognize a crash-persisted nudge by
# content (SessionDB projection strips the _length_continuation_nudge tag).
_LENGTH_CONTINUATION_NETWORK_STUB = (
    "[System: The previous response was cut off by a network error mid-stream. Continue exactly "
    "where you left off. Do not restart or repeat prior text. Finish the answer directly.]"
)
_LENGTH_CONTINUATION_OUTPUT_LIMIT = (
    "[System: Your previous response was truncated by the output length limit. Continue exactly "
    "where you left off. Do not restart or repeat prior text. Finish the answer directly.]"
)
# The dropped-tools variant interpolates tool names; matched by prefix.
_LENGTH_CONTINUATION_DROPPED_TOOLS_PREFIX = "[System: Your previous tool call "


def _get_continuation_prompt(is_partial_stub: bool, dropped_tools: Optional[List[str]] = None) -> str:
    if is_partial_stub and dropped_tools:
        tool_list = ", ".join(dropped_tools[:3])
        return (
            f"{_LENGTH_CONTINUATION_DROPPED_TOOLS_PREFIX}({tool_list}) was too large and "
            "the stream timed out before it could be delivered. Do NOT retry the same tool call "
            "with the same large content. Instead, break the content into multiple smaller tool "
            "calls (e.g. use multiple patch calls or write smaller files). Each tool call's "
            "arguments must be under ~8K tokens to avoid stream timeouts.]"
        )
    return _LENGTH_CONTINUATION_NETWORK_STUB if is_partial_stub else _LENGTH_CONTINUATION_OUTPUT_LIMIT


# Codex/Responses turns that returned only internal reasoning: a bare retry would be
# byte-identical, so the model repeats it.
_CODEX_INCOMPLETE_NUDGE = (
    "[System: Your previous response contained only internal reasoning and never produced a "
    "visible answer or tool call. Do not keep thinking. Produce your final answer as plain text "
    "now (or make the tool call you were planning).]"
)


# Re-prompt after an acknowledgment-only Codex/Responses reply.
_CODEX_ACK_CONTINUATION_NUDGE = (
    "[System: Continue now. Execute the required tool calls and only send your final answer "
    "after completing the task.]"
)

# Re-prompt for finish_reason="tool_calls" with empty tool_calls (an interrupt mid-retry can persist it).
_DROPPED_TOOLCALL_NUDGE_CONTENT = (
    "Your previous turn indicated a tool call but none was included. Do not narrate a plan or "
    "restate intent — issue the actual tool call now to continue the task."
)

# Re-prompt for an empty response after tool calls (#9400); the metadata flag does not
# survive SessionDB projection, so it is matched by content.
_EMPTY_TOOL_RESPONSE_NUDGE = (
    "You just executed tool calls but returned an empty response. Please process the tool "
    "results above and continue with the task."
)


# Shared trailer for both content-policy refusal paths so guidance cannot drift.
_CONTENT_POLICY_RECOVERY_HINT = (
    "Try rephrasing the request, narrowing the context, or adding a fallback provider with "
    "`hermes fallback add`."
)


# Memo for send-path tool-call argument canonicalization (re-run on every historical call
# each iteration). Sound because canonicalization is pure; malformed strings raise before
# being stored, so the repair fallback is never memoized. The byte budget exists because
# argument strings can run 100KB+, so a count bound alone does not bound memory.
_CANON_ARGS_CACHE: Dict[str, str] = {}
_CANON_ARGS_CACHE_MAX = 4096
_CANON_ARGS_CACHE_MAX_BYTES = 32 * 1024 * 1024
_canon_args_cache_bytes = 0


def _canonicalize_tool_call_arguments(arg_str: str) -> str:
    """Canonical wire form of a tool-call arguments JSON string; raises on malformed input
    (the caller falls back to ``_repair_tool_call_arguments``)."""
    global _canon_args_cache_bytes
    cached = _CANON_ARGS_CACHE.get(arg_str)
    if cached is not None:
        return cached
    canonical = json.dumps(json.loads(arg_str), separators=(",", ":"), sort_keys=True)
    _CANON_ARGS_CACHE[arg_str] = canonical
    _canon_args_cache_bytes += len(arg_str) + len(canonical)
    while len(_CANON_ARGS_CACHE) > _CANON_ARGS_CACHE_MAX or (
        _canon_args_cache_bytes > _CANON_ARGS_CACHE_MAX_BYTES and len(_CANON_ARGS_CACHE) > 1
    ):
        try:
            evicted_key = next(iter(_CANON_ARGS_CACHE))
            _canon_args_cache_bytes -= len(evicted_key) + len(_CANON_ARGS_CACHE.pop(evicted_key))
        except (StopIteration, KeyError, RuntimeError):
            break
    return canonical


def _clone_message_for_send(msg):
    """Structural clone (dicts/lists recursively, immutable leaves shared) of a history
    message for the per-call API copy, so send-path rewrites never reach the persisted
    transcript (#80498). Cheaper than deepcopy: messages are JSON-shaped and acyclic."""
    if isinstance(msg, dict):
        return {k: _clone_message_for_send(v) if isinstance(v, (dict, list)) else v for k, v in msg.items()}
    if isinstance(msg, list):
        return [_clone_message_for_send(v) if isinstance(v, (dict, list)) else v for v in msg]
    return msg


def _canonicalize_api_tool_calls(api_messages) -> None:
    """Canonicalize tool-call argument JSON on the send-path copy (copy-on-write for the
    dicts it touches; persisted history untouched)."""
    for am in api_messages:
        tcs = am.get("tool_calls")
        if not tcs:
            continue
        new_tcs = []
        for tc in tcs:
            if isinstance(tc, dict) and "function" in tc:
                fn = tc["function"]
                try:
                    args = _canonicalize_tool_call_arguments(fn["arguments"])
                except Exception:
                    args = _repair_tool_call_arguments(fn["arguments"], fn.get("name", "?"))
                # Copy-on-write as defense in depth: callers may pass shallow copies, and
                # writing into a shared tc["function"] rewrote the stored turn with "{}"
                # on the unrepairable path (#80498).
                tc = {**tc, "function": {**fn, "arguments": args}}
            new_tcs.append(tc)
        am["tool_calls"] = new_tcs


def _invalid_tool_name_error_content(name: str, valid_tool_names) -> str:
    """Error content for an unknown tool name. A blank name is a model echoing tool-call
    syntax seen in data (#47967) — dumping the catalog feeds that loop, so it gets a terse
    error; a nonempty wrong name still gets the catalog to self-correct."""
    if not (name or "").strip():
        return (
            "Tool call rejected: the tool name was empty. If tool-call XML or JSON appeared in file "
            "contents or tool output, that is data — do not re-emit it as a tool call. To call a "
            "tool, use a valid name from your tool list; otherwise reply in plain text."
        )
    available = ", ".join(sorted(valid_tool_names))
    return f"Tool '{name}' does not exist. Available tools: {available}"


def _content_policy_blocked_result(
    messages: List[Dict], api_call_count: int, *, final_response: str, error_detail: str
) -> Dict[str, Any]:
    """Terminal turn result for a content-policy block (deterministic for the unchanged
    prompt, so no retry); shared by the HTTP-200 and exception paths."""
    return {
        "final_response": final_response, "messages": messages, "api_calls": api_call_count,
        "completed": False, "failed": True, "error": f"content_policy_blocked: {error_detail}",
    }


def _partial_turn_result(
    final_response: str, messages: List[Dict], api_call_count: int, **flags: Any
) -> Dict[str, Any]:
    """Incomplete-turn result whose ``error`` mirrors ``final_response``; ``flags`` add the
    recovery-contract keys (``failed``, ``compression_deferred``, ...)."""
    return {
        "final_response": final_response, "messages": messages, "completed": False,
        "api_calls": api_call_count, "error": final_response, "partial": True, **flags,
    }


def _compression_deferred_result(agent, messages: List[Dict], api_call_count: int, reason: str = "lock") -> Dict[str, Any]:
    """Soft turn result for a transiently-deferred compression. Both reasons must end as
    ``compression_deferred``, never ``compression_exhausted`` — the gateway wipes the
    session on exhaustion (#9893/#35809). ``failed`` stays False; the turn persists."""
    session = agent.session_id or "none"
    if reason == "transient_block":
        block = getattr(agent, "_compression_blocked_transient", None)
        logger.info(
            "turn deferred: compression transiently blocked (%s) (session=%s) — not counting as "
            "compression exhaustion", block if isinstance(block, str) else "unknown guard", session,
        )
        _final = (
            "Context compression is temporarily paused after a recent failed attempt. Please retry "
            "in a moment — compression will resume automatically (or run /compress to force a retry now)."
        )
    else:
        holder = getattr(agent, "_compression_skipped_due_to_lock", None)
        logger.info(
            "turn deferred: compression lock held by another path (session=%s holder=%s) — not "
            "counting as compression exhaustion", session, holder if isinstance(holder, str) else "unconfirmed",
        )
        _final = (
            "Context compression is already running for this session. Please retry in a moment — "
            "your next message will be processed once the concurrent compression finishes."
        )
    try:
        agent._flush_status_buffer()
    except Exception:
        pass
    return _partial_turn_result(
        _final, messages, api_call_count,
        failed=False, compression_deferred=True, session_id=agent.session_id,
    )


def _provider_overflow_exhausted_result(
    agent, messages: List[Dict], conversation_history, api_call_count: int,
    request_pressure_tokens: int, max_compression_attempts: int,
) -> Dict[str, Any]:
    """Fail closed when a rebuilt request is still too large after recovery."""
    agent._flush_status_buffer()
    logger.error(
        "%sContext compression failed after %d attempts; rebuilt request "
        "remains over threshold at ~%s tokens.",
        agent.log_prefix, max_compression_attempts, f"{request_pressure_tokens:,}",
    )
    # Host progress-aware timeout (#98722, salvaged from #98741): the provider proved the request does not
    # fit, but this recovery pass spent the full wait budget without a committed summary. Re-sending the
    # unchanged request would bounce off the same overflow error and re-enter compression in the same turn.
    # End the turn with the typed recovery contract instead — transcript intact, no further doomed provider
    # sends.
    # Prior <3 retries (or an earlier successful tool batch) leave a tool-result tail. Closing it here
    # matches interrupt aborts (#48879 / #52592) so the next user turn is not tool→user for strict
    # providers.
    agent._persist_session(messages, conversation_history)
    return _partial_turn_result(
        "Context length exceeded: compression could not reduce the rebuilt request below the safe threshold.",
        messages, api_call_count, failed=True, compression_exhausted=True,
        turn_exit_reason="context_compression_exhausted",
    )


def _rewrite_system_content_blocks(system_message: dict, effective: str) -> bool:
    """Rewrite a cache-decorated system message in place, keeping its blocks (a bare string
    over the ``[static prefix, volatile tail]`` list would drop both cache_control
    breakpoints). Returns False when the shape cannot be safely patched."""
    content = system_message.get("content")
    if not isinstance(content, list) or not content or not all(
        isinstance(part, dict) and part.get("type") == "text" for part in content
    ):
        return False
    if len(content) == 1:
        content[0]["text"] = effective
        return True
    if len(content) == 2:
        head = content[0].get("text") or ""
        if head and effective.startswith(head) and effective[len(head):]:
            content[1]["text"] = effective[len(head):]
            return True
    return False


def _sync_failover_system_message(agent, api_messages, active_system_prompt):
    """Refresh the in-flight system message after a provider failover: ``api_messages`` were
    built pre-failover and are reused each retry. Returns the new ``active_system_prompt``."""
    sp = getattr(agent, "_cached_system_prompt", None)
    if not isinstance(sp, str) or not sp:
        return active_system_prompt
    if api_messages and api_messages[0].get("role") == "system":
        effective = (sp + "\n\n" + agent.ephemeral_system_prompt).strip() if agent.ephemeral_system_prompt else sp
        if not _rewrite_system_content_blocks(api_messages[0], effective):
            api_messages[0]["content"] = effective
    return sp


def _arm_fallback_restart(agent, api_messages, active_system_prompt, _retry):
    """After a successful fallback activation: sync the system message and arm
    ``restart_with_rebuilt_messages``. Callers also zero ``retry_count`` /
    ``compression_attempts`` and ``break`` the retry loop."""
    active_system_prompt = _sync_failover_system_message(
        agent, api_messages, active_system_prompt)
    _retry.primary_recovery_attempted = False
    _retry.restart_with_rebuilt_messages = True
    return active_system_prompt


def _ensure_cached_system_prompt_static(agent, system_message=None) -> None:
    """Rebuild ``_cached_system_prompt_static`` when caching becomes active (#72626): sessions
    restored under a cache-off primary would otherwise fall back to the legacy layout after
    failover to a cache-on provider."""
    from agent.system_prompt import reconstruct_static_prefix
    reconstruct_static_prefix(agent, system_message=system_message, log_label="failover redecoration")


def _peel_moa_guidance(messages: List[Dict[str, Any]], guidance: Any) -> List[Dict[str, Any]]:
    """Remove MoA reference guidance attached by ``_attach_reference_guidance``."""
    from agent.moa_loop import peel_reference_guidance
    return peel_reference_guidance(messages, guidance)


def _redecorate_prompt_cache_for_provider(
    agent, api_messages: List[Dict[str, Any]], *, system_message=None,
    moa_prepared: Optional[Dict[str, Any]] = None, tools_for_api: Optional[List[Dict[str, Any]]] = None,
) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]] | tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Strip and re-apply cache_control for the *current* provider policy — failover
    ``continue`` paths reuse ``api_messages`` (#72626). MoA guidance is peeled and rebased."""
    messages: List[Dict[str, Any]] = [dict(m) if isinstance(m, dict) else m for m in (api_messages or [])]
    prepared = moa_prepared
    guidance = prepared.get("guidance") if isinstance(prepared, dict) else None
    if guidance:
        messages = _peel_moa_guidance(messages, guidance)

    strip_anthropic_cache_control(messages)
    planned_tools = strip_anthropic_tool_cache_control(
        tools_for_api if tools_for_api is not None else getattr(agent, "tools", [])
    )
    if prepared is not None and getattr(agent, "provider", None) == "moa":
        # Prepared MoA state is canonical: the synchronous acting-aggregator
        # sender owns its destination-local cache plan after it resolves the slot.
        completions = getattr(getattr(agent.client, "chat", None), "completions", None)
        rebase = getattr(completions, "rebase_prepared_request", None)
        if callable(rebase):
            prepared = rebase(prepared, messages)
            messages = prepared["messages"]
    # Direct attribute access, not getattr: the flags are always initialized on
    # AIAgent, and a default would mask a real init bug as silent cache-off.
    elif agent._use_prompt_caching:
        _ensure_cached_system_prompt_static(agent, system_message=system_message)
        static = getattr(agent, "_cached_system_prompt_static", None)
        from agent.prompt_caching import envelope_tool_part_cache_markers_supported
        plan = build_prompt_cache_plan(
            messages,
            planned_tools,
            # Clamp per-destination: a configured 1h regresses to 5m on
            # Qwen/Alibaba routes, whose context cache is 5m-only (#84733).
            cache_ttl=effective_cache_ttl(agent._cache_ttl, provider=agent.provider, model=agent.model),
            native_anthropic=agent._use_native_cache_layout,
            static_system_prefix=static if isinstance(static, str) else None,
            direct_native_tool_cache=getattr(
                agent, "_direct_native_anthropic_tool_cache_capability", lambda: False
            )(),
            # LiteLLM-style envelope routes forward part-level markers into
            # tool_result.content[] → non-retryable 400 (#89886).
            tool_part_markers=envelope_tool_part_cache_markers_supported(
                getattr(agent, "provider", ""), getattr(agent, "base_url", "")
            ),
        )
        messages, planned_tools = plan.messages, plan.tools

    if tools_for_api is None:
        return messages, prepared
    return messages, prepared, planned_tools


def _engine_overrides_hook(engine: Any, name: str) -> bool:
    """True when ``engine`` implements ContextEngine hook ``name`` itself.

    Non-implementing engines must pay nothing per turn; ``hasattr`` is not enough because
    the ABC defines a no-op default. Lazy import avoids a cycle with agent.context_engine."""
    hook = getattr(engine, name, None)
    if engine is None or not callable(hook):
        return False
    try:
        from agent.context_engine import ContextEngine as _CE
        return getattr(hook, "__func__", None) is not getattr(_CE, name)
    except Exception:
        return True


def _apply_context_engine_selection(
    agent: Any, api_messages: List[Dict[str, Any]], conversation_messages: List[Dict[str, Any]],
    incoming_message: Optional[Dict[str, Any]], *, logger: Any,
) -> List[Dict[str, Any]]:
    """Run the optional per-turn ``ContextEngine.select_context()`` hook, fail-open: any
    exception or invalid return yields ``api_messages`` unchanged; history is never mutated."""
    engine = getattr(agent, "context_compressor", None)
    if not _engine_overrides_hook(engine, "select_context"):
        return api_messages

    session_label = getattr(agent, "session_id", None) or "-"
    # Structural clones: the engine must not be able to write through nested
    # containers into persisted history; only the request list is acted on (#80498).
    try:
        selected = engine.select_context(
            api_messages,
            conversation_messages=(
                [_clone_message_for_send(m) for m in conversation_messages]
                if conversation_messages is not None else None
            ),
            incoming_message=(
                _clone_message_for_send(incoming_message)
                if isinstance(incoming_message, dict) else incoming_message
            ),
            budget_tokens=getattr(engine, "context_length", 0) or 0,
        )
    except Exception:
        logger.warning(
            "Context engine select_context hook failed; using unmodified request messages (session=%s)",
            session_label, exc_info=True,
        )
        return api_messages

    if selected is None:
        return api_messages
    # Require a NON-EMPTY list of dicts: ``all([])`` is ``True``, so a ``[]`` from a
    # buggy engine would otherwise replace the request instead of failing open.
    if isinstance(selected, list) and selected and all(isinstance(m, dict) for m in selected):
        return selected
    logger.warning(
        "Context engine select_context returned an invalid value "
        "(not a non-empty list of dicts); ignoring (session=%s)", session_label,
    )
    return api_messages


def _notify_context_engine_turn_complete(
    agent: Any, messages: List[Dict[str, Any]], *, usage: Optional[Dict[str, Any]] = None, logger: Any, **meta: Any
) -> None:
    """Notify the active context engine that a user turn has finished (fail-open; the engine
    gets a copy so it cannot mutate the persisted transcript)."""
    engine = getattr(agent, "context_compressor", None)
    if not _engine_overrides_hook(engine, "on_turn_complete"):
        return
    try:
        # Structural clones: dict(m) would let a hook write into nested containers of the
        # persisted transcript (#80498).
        engine.on_turn_complete([_clone_message_for_send(m) for m in messages], usage=usage, **meta)
    except Exception:
        logger.warning(
            "Context engine on_turn_complete hook failed (session=%s)",
            getattr(agent, "session_id", None) or "-", exc_info=True,
        )


def _decode_inline_moa_turn(user_message, persist_user_message):
    """Decode a MoA preset encoded into ``user_message``; returns ``(user_message,
    moa_config, persist_user_message)``, unchanged with ``moa_config=None`` otherwise."""
    try:
        from hermes_cli.moa_config import decode_moa_turn
        _decoded_message, _decoded_moa_config = decode_moa_turn(user_message)
        if _decoded_moa_config is not None:
            if persist_user_message is None:
                persist_user_message = _decoded_message
            return _decoded_message, _decoded_moa_config, persist_user_message
    except Exception:
        pass
    return user_message, None, persist_user_message


def _preflight_timeout_result(agent, exc, conversation_history) -> Dict[str, Any]:
    """Typed recovery result when turn-start preflight compression timed out (#98424): no
    provider call was sent, and surfaces would otherwise hide the actionable guidance."""
    logger.warning(
        "Turn-start preflight compression timed out — ending turn with typed recovery result: %s", exc,
    )
    # Clear the tripwire slot note_turn_start registered (the early return skips the persist
    # funnel). The user row is deliberately NOT persisted (#7100).
    from agent.agent_runtime_helpers import note_turn_persisted
    note_turn_persisted(agent)
    # Not _COMPRESSION_TIMEOUT_FINAL_RESPONSE — that describes a different state
    # (compression ran, could not reduce); the exception text carries the guidance.
    return _partial_turn_result(
        str(exc), list(conversation_history or []), 0,
        failed=True, compression_exhausted=True, turn_exit_reason="context_compression_timeout",
    )


@dataclass
class _LoopState:
    """Every local the turn loop threads through the phase helpers in ``agent/turn_*.py``.

    Helpers take the loop locals they need as keyword arguments named like these fields and
    return a verdict whose non-``action``/``result`` fields carry the same names;
    :func:`_run_phase` passes and copies them back by name, so a new helper input/output
    needs a field here and nothing else. Per-iteration slots are rebound by the phases
    before any later phase reads them, exactly as the former inline locals were."""

    # Fixed for the turn.
    user_message: Any
    system_message: Any
    moa_config: Any
    original_user_message: Any
    conversation_history: Any
    effective_task_id: Any
    turn_id: Any
    _should_review_memory: Any
    _plugin_user_context: Any
    _ext_prefetch_cache: Any
    # Turn-scoped state (rebound by the phases).
    messages: Any
    active_system_prompt: Any
    current_turn_user_idx: Any
    _preflight_compression_blocked: Any
    # Compression attempt cap shared by the pre-API gate, 413 handlers and post-tool compaction:
    # a consecutive-ineffective-attempt backstop, rearmed only after a provider response
    # reports a prompt below threshold.
    max_compression_attempts: Any
    api_call_count: int = 0
    final_response: Any = None
    interrupted: bool = False
    failed: bool = False
    codex_ack_continuations: int = 0
    length_continue_retries: int = 0
    # Per-turn backstop for the refunding restarts (redirect / rebuilt-for-fallback).
    # Unlike ``retry_count`` (rebound to 0 each iteration) this accumulates for the whole
    # turn so a runaway interrupt/redirect that keeps re-arming a restart flag cannot
    # refund the iteration budget forever and hold the turn lease indefinitely.
    restart_count: int = 0
    _outer_error_count: int = 0  # outer-loop exceptions this turn (#92450), see _MAX_OUTER_LOOP_ERRORS
    truncated_tool_call_retries: int = 0
    truncated_response_parts: List[str] = field(default_factory=list)
    compression_attempts: int = 0
    _last_preflight_pressure: Optional[int] = None
    # A provider overflow outweighs the rough-estimate calibration that defers preflight after
    # compaction: stays armed until the rebuilt request is below the threshold.
    _provider_overflow_recovery_pending: bool = False
    # A compression host-timeout ended the turn; finalize reuses the gateway context-recovery
    # contract (error/partial/compression_exhausted) (#98722).
    _compression_timeout_exhausted: bool = False
    _turn_exit_reason: str = "unknown"  # diagnostic: why the loop ended
    # Answer held back by a verification gate (best user-facing result if the continuation
    # exhausts the budget) and whether it was streamed as interim; ``_response_was_previewed``
    # is set ONLY if it becomes the final response (#65919).
    _pending_verification_response: Any = None
    _pending_verification_response_previewed: bool = False
    # MoA guidance retained across a pre-API compression, rebased next iteration (no second fan-out).
    pending_moa_prepared_request: Any = None
    # Per-iteration slots.
    request_logger: Any = None
    api_messages: Any = None
    tools_for_api: Any = None
    _moa_prepared_request: Any = None
    approx_tokens: Any = None
    request_pressure_tokens: Any = None
    total_chars: Any = None
    thinking_spinner: Any = None
    api_start_time: Any = None
    retry_count: int = 0
    max_retries: Any = None
    _retry: Any = None
    finish_reason: str = "stop"
    response: Any = None  # None when every retry failed
    api_kwargs: Any = None  # None until built; read by the except handlers
    api_request_id: Any = None
    _original_api_kwargs: Any = None
    _llm_middleware_trace: Any = None
    api_duration: Any = None
    assistant_message: Any = None


# _LoopState fields seeded from TurnContext (same name minus the leading underscore).
_CTX_FIELDS = frozenset({
    "user_message", "original_user_message", "conversation_history", "effective_task_id", "turn_id",
    "_should_review_memory", "_plugin_user_context", "_ext_prefetch_cache", "messages",
    "active_system_prompt", "current_turn_user_idx", "_preflight_compression_blocked",
})
# Keyword names each phase helper takes (minus ``agent``), cached per function object.
_PHASE_PARAMS: Dict[Any, tuple] = {}
# Verdict fields the loop latches (only ever sets True) instead of copying back:
# ``handle_api_error`` reports overflow recovery per call and must not clear an earlier arm.
_LATCHED_VERDICT_FIELDS = {"handle_api_error": frozenset({"_provider_overflow_recovery_pending"})}


def _run_phase(fn, agent, state: _LoopState, **extra):
    """Call phase helper ``fn`` with the loop locals it names, copy its verdict fields back.

    ``extra`` supplies non-state arguments (the caught exception). Returns the verdict so
    the caller can act on ``.action`` / ``.result``."""
    params = _PHASE_PARAMS.get(fn)
    if params is None:
        params = _PHASE_PARAMS[fn] = tuple(p for p in inspect.signature(fn).parameters if p != "agent")
    verdict = fn(agent, **{n: extra[n] if n in extra else getattr(state, n) for n in params})
    latched = _LATCHED_VERDICT_FIELDS.get(getattr(fn, "__name__", ""), ())
    for f in fields(verdict):
        if f.name in ("action", "result"):
            continue
        value = getattr(verdict, f.name)
        if f.name not in latched:
            setattr(state, f.name, value)
        elif value:
            setattr(state, f.name, True)
    return verdict


def _run_api_retry_loop(agent, s: _LoopState) -> Optional[Dict[str, Any]]:
    """One API call with its retry/recovery loop (guard → build → call → check, error handlers).

    Returns a turn result dict when a phase ends the turn, else None once the loop is left
    (success, a restart armed on ``s._retry``, interrupt, or retries exhausted)."""
    while s.retry_count < s.max_retries:
        _ng = _run_phase(nous_rate_limit_guard, agent, s)
        if _ng.action == "return":
            return _ng.result
        if _ng.action == "break":
            return None
        try:
            _run_phase(build_api_request, agent, s)
            if _run_phase(perform_api_call, agent, s).action == "break":
                return None
            _rc = _run_phase(check_api_response, agent, s)
            if _rc.action == "return":
                return _rc.result
            if _rc.action == "break":
                return None
        except InterruptedError:
            if _run_phase(handle_api_interrupt, agent, s).action == "break":
                return None
        except Exception as api_error:
            _ae = _run_phase(handle_api_error, agent, s, api_error=api_error)
            if _ae.action == "return":
                return _ae.result
            if _ae.action == "break":
                return None
    return None


def _run_conversation_turn(
    agent,
    user_message: Any,
    system_message: str = None,
    conversation_history: List[Dict[str, Any]] = None,
    task_id: str = None,
    stream_callback: Optional[callable] = None,
    persist_user_message: Optional[Any] = None,
    persist_user_timestamp: Optional[float] = None,
    persist_user_display_kind: Optional[str] = None,
    persist_user_display_metadata: Optional[Dict[str, Any]] = None,
    persist_user_platform_id: Optional[str] = None,
    turn_author: Optional[Dict[str, Any]] = None,
    moa_config: Optional[dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a complete conversation with tool calling until completion; returns the result dict.

    ``stream_callback``: per-text-delta callback (TTS). ``persist_user_message``: clean text to
    store when ``user_message`` carries API-only synthetic prefixes; timestamp / platform id are
    stored as metadata (platform id lets restart drain recovery dedup). ``persist_user_display_*``:
    display-only event rendering; the model still receives the message unchanged."""
    if moa_config is None:
        user_message, moa_config, persist_user_message = _decode_inline_moa_turn(
            user_message, persist_user_message
        )

    # The gateway caches agents across turns; compression state is per-turn, or a stale
    # in-place boundary would make a later uncompressed result look compacted.
    agent._last_compaction_in_place = agent._last_compression_attempt_recorded = False
    agent._last_compression_attempt_in_place = None
    begin_fast_mode_turn(agent, conversation_history)

    # Adopt ~/.hermes/.env credential/base-url edits made since the last turn — a
    # Settings save updates .env, not this worker's client (#67821). No-op if unchanged.
    try:
        agent._try_refresh_env_client_credentials()
    except Exception:
        logger.debug("per-turn env credential refresh failed", exc_info=True)

    # Per-turn setup: build_turn_context mutates ``agent`` and returns the locals the loop reads.
    try:
        _ctx = build_turn_context(
            agent, user_message, system_message, conversation_history, task_id,
            stream_callback, persist_user_message, persist_user_timestamp,
            persist_user_display_kind=persist_user_display_kind,
            persist_user_display_metadata=persist_user_display_metadata,
            persist_user_platform_id=persist_user_platform_id,
            turn_author=turn_author,
            restore_or_build_system_prompt=_restore_or_build_system_prompt,
            install_safe_stdio=_install_safe_stdio,
            sanitize_surrogates=_sanitize_surrogates,
            summarize_user_message_for_log=_summarize_user_message_for_log,
            set_session_context=set_session_context,
            set_current_write_origin=set_current_write_origin,
            ra=_ra,
            # MoA turns append per-call aggregated context to the API copy of the
            # user message, so no byte-stable api_content sidecar can be stamped.
            moa_active=bool(moa_config),
        )
    except PreflightCompressionTimedOut as _preflight_timeout_exc:
        return _preflight_timeout_result(agent, _preflight_timeout_exc, conversation_history)

    # Per-turn agent state (the gateway caches agents across turns, so none of this may
    # leak into the next message): interim-commentary dedup spans the whole turn but not
    # the next; a SessionDB append failure (and its classified cause) halts only this turn;
    # a failed compression-tip adoption is reported only against its own turn; the
    # thinking-only-truncation one-shot must not survive an interrupted turn; credential-
    # pool refresh tallies cap same-entry refreshes on a persistent 401 (#26080); usage
    # for on_turn_complete() stays None on turns that never reach a response.
    agent._delivered_interim_texts = set()
    agent._incremental_persistence_failed = False
    agent._last_persistence_error_cause = None
    agent._compression_adoption_failed = False
    agent._ephemeral_reasoning_off = False
    agent._auth_pool_refresh_counts = {}
    agent._last_turn_usage = None

    s = _LoopState(
        system_message=system_message, moa_config=moa_config,
        max_compression_attempts=getattr(agent, "max_compression_attempts", 3),
        **{f.name: getattr(_ctx, f.name.lstrip("_")) for f in fields(_LoopState) if f.name in _CTX_FIELDS},
    )
    # Opt-in runtime: api_mode == codex_app_server hands the whole turn to the codex
    # app-server subprocess (see agent/transports/codex_app_server_session.py).
    if agent.api_mode == "codex_app_server":
        return agent._run_codex_app_server_turn(
            user_message=s.user_message, original_user_message=s.original_user_message,
            messages=s.messages, effective_task_id=s.effective_task_id,
            should_review_memory=s._should_review_memory,
        )

    while (s.api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0) or agent._budget_grace_call:
        if _run_phase(begin_iteration, agent, s).action == "break":
            break

        # Pre-API pressure check. The turn-prologue preflight only saw the
        # incoming user message; a single turn can then grow by many large
        # tool results and leave no output budget before the NEXT call (the
        # live 271k/272k Codex failure). The post-response should_compress
        # gate at the tool-loop tail uses API-reported last_prompt_tokens,
        # which LAGS a just-appended huge tool result — so it misses this
        # case. Re-check here against the current request estimate.
        #
        # Mirror the turn-prologue preflight's guard chain exactly (see
        # turn_context.py): (1) defer when the rough estimate is known-noisy
        # relative to a recent real provider prompt that fit under threshold
        # (schema overhead / post-compaction over-count, #36718); (2) skip
        # while a same-session compression-failure cooldown is active; (3) then
        # should_compress() — reusing the canonical threshold_tokens (output
        # room already reserved by _compute_threshold_tokens) and its summary-
        # LLM cooldown + anti-thrash guards (#11529). compression_attempts is a
        # hard per-turn backstop shared with the overflow error handlers.
        _compressor = agent.context_compressor
        _preflight_threshold = int(
            getattr(_compressor, "threshold_tokens", 0) or 0
        )
        # A previous mid-turn preflight pass deliberately continued the loop so
        # API-only context and all sanitization could be rebuilt. Compare that
        # fully assembled request with the fully assembled request that caused
        # the pass. Raw ``messages`` are not equivalent here: they omit
        # api_content/plugin injections, prefills, MoA context, and ephemeral
        # system text.
        _previous_preflight_pressure = _last_preflight_pressure
        _last_preflight_pressure = None
        if (
            _previous_preflight_pressure is not None
            and request_pressure_tokens >= _preflight_threshold
            and not _compression_warrants_another_preflight_pass(
                _previous_preflight_pressure,
                request_pressure_tokens,
                _preflight_threshold,
            )
        ):
            # Stop proactive retries for this turn without consuming the
            # shared overflow-recovery budget. If the provider proves the
            # request truly does not fit, its error handler may still compact
            # with that stronger signal.
            _preflight_compression_blocked = True
            logger.warning(
                "Pre-API compression made insufficient progress: ~%s -> "
                "~%s request tokens; skipping additional preflight passes",
                f"{_previous_preflight_pressure:,}",
                f"{request_pressure_tokens:,}",
            )
        _defer_preflight = getattr(
            _compressor, "should_defer_preflight_to_real_usage", lambda _t: False
        )
        _compression_cooldown = getattr(
            _compressor, "get_active_compression_failure_cooldown", lambda: None
        )()
        if (
            agent.compression_enabled
            and len(messages) > 1
            and compression_attempts < max_compression_attempts
            and not _preflight_compression_blocked
            and not _defer_preflight(request_pressure_tokens)
            and not _compression_cooldown
            and _compressor.should_compress(request_pressure_tokens)
        ):
            if _moa_prepared_request is not None:
                pending_moa_prepared_request = _moa_prepared_request
            compression_attempts += 1
            # Compression is actually running (block cleared / was never
            # blocked) — reset the blocked-overflow warning dedup so a future
            # blocked-over-threshold turn can warn again. Mirrors the
            # turn-context preflight reset (silent-overflow fix #62625).
            # getattr guard: test doubles built via object.__new__ lack the
            # method (gateway test-double pitfall) — treat absence as no-op.
            _clear_warn = getattr(agent, "_clear_context_overflow_warn", None)
            if callable(_clear_warn):
                _clear_warn()
            logger.info(
                "Pre-API compression: ~%s request tokens >= %s threshold "
                "(context=%s, attempt=%s/%s)",
                f"{request_pressure_tokens:,}",
                f"{int(getattr(_compressor, 'threshold_tokens', 0) or 0):,}",
                f"{int(getattr(_compressor, 'context_length', 0) or 0):,}"
                if getattr(_compressor, "context_length", 0) else "unknown",
                compression_attempts,
                max_compression_attempts,
            )
            _pre_api_status = automatic_compaction_status_message(
                _compressor,
                phase="pre_api",
                default_message=PRE_API_COMPRESSION_STATUS_TEMPLATE.format(
                    tokens=request_pressure_tokens
                ),
                approx_tokens=request_pressure_tokens,
                threshold_tokens=int(
                    getattr(_compressor, "threshold_tokens", 0) or 0
                ),
                context_length=int(
                    getattr(_compressor, "context_length", 0) or 0
                ),
                model=agent.model,
                attempt=compression_attempts,
                max_attempts=max_compression_attempts,
            )
            if _pre_api_status:
                agent._emit_status(_pre_api_status)
            _last_preflight_pressure = request_pressure_tokens
            _pre_api_input = messages
            messages, active_system_prompt = agent._compress_context(
                messages,
                system_message,
                approx_tokens=request_pressure_tokens,
                task_id=effective_task_id,
            )
            if messages is _pre_api_input and compression_skipped_due_to_lock(agent):
                # #69870 lock-skip: another path holds this session's
                # compression lock, so this pass no-oped. That is a temporary
                # DEFER, not evidence about compressibility — refund the
                # attempt (it must not burn the shared overflow-recovery
                # budget toward compression_exhausted → gateway auto-reset,
                # #9893/#35809) and leave the insufficient-progress blocker
                # unarmed. Proceed with the current request: if it truly does
                # not fit, the provider's 413/overflow handler returns the
                # soft compression_deferred result with that stronger signal.
                compression_attempts -= 1
                _last_preflight_pressure = None
                if pending_moa_prepared_request is _moa_prepared_request:
                    pending_moa_prepared_request = None
            else:
                # Reset retry/empty-response state so the compacted request
                # gets a fresh chance instead of inheriting stale recovery
                # counters from the pre-compaction history.
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                agent._last_content_with_tools = None
                agent._last_content_tools_all_housekeeping = False
                agent._mute_post_response = False
                # Re-baseline the flush cursor for the compaction mode that just
                # ran. Legacy session-rotation returns None (the child session has
                # not seen the compacted transcript, so the next flush writes it
                # whole); in-place compaction returns list(messages) because the
                # compacted rows are already persisted under the same session id —
                # leaving None there would re-append them, doubling the active
                # context and retriggering compression. Mirrors the post-response
                # and preflight compaction sites; see
                # conversation_history_after_compression().
                conversation_history = conversation_history_after_compression(
                    agent, messages, conversation_history
                )
                # This preflight iteration never reaches the provider whether
                # we skip the turn (handoff guard below) or re-run the loop —
                # refund the consumed call/budget in BOTH cases, mirroring the
                # ollama_runtime_context_too_small early-exit above. Without
                # the refund on the break path, every skipped turn leaked one
                # iteration-budget unit for the agent's lifetime and
                # finalize_turn logged an api_call_count including a call that
                # was never made.
                api_call_count -= 1
                agent._api_call_count = api_call_count
                agent.iteration_budget.refund()
                if _should_skip_model_call_for_reference_handoff(
                    messages, user_message
                ):
                    # Reference-only handoff must not become the active turn
                    # after a completed assistant response (#80622).
                    logger.info(
                        "Skipping post-compaction model call: reference-only "
                        "handoff would be the sole active user turn (#80622)"
                    )
                    if not final_response:
                        final_response = _HANDOFF_SKIP_FINAL_RESPONSE
                    _turn_exit_reason = "compaction_handoff_not_actionable"
                    break
                continue
        elif (
            agent.compression_enabled
            and len(messages) > 1
            and compression_attempts < max_compression_attempts
            and not _defer_preflight(request_pressure_tokens)
            and _compression_cooldown
        ):
            # Blocked by the summary-LLM cooldown. Surface a deduped warning
            # (only when actually over threshold — should_compress_info
            # returns a None reason below threshold) so the user isn't left
            # with a silently growing context. Mirrors the turn-context
            # preflight and the loop-compaction guards (silent-overflow fix
            # #62625).
            _block_reason = None
            try:
                _block_reason = _compressor.should_compress_info(
                    request_pressure_tokens
                )[1]
            except Exception:
                _block_reason = None
            if _block_reason:
                agent._warn_context_overflow_blocked(
                    _block_reason,
                    request_pressure_tokens,
                    int(getattr(_compressor, "threshold_tokens", 0) or 0),
                )
        
        # Thinking spinner for quiet mode (animated during API call)
        thinking_spinner = None
        
        if not agent.quiet_mode:
            agent._vprint(f"\n{agent.log_prefix}🔄 Making API call #{api_call_count}/{agent.max_iterations}...")
            agent._vprint(f"{agent.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
            agent._vprint(f"{agent.log_prefix}   🔧 Available tools: {len(agent.tools) if agent.tools else 0}")
        else:
            # Animated thinking spinner in quiet mode
            face = random.choice(KawaiiSpinner.get_thinking_faces())
            verb = random.choice(KawaiiSpinner.get_thinking_verbs())
            if agent.thinking_callback:
                # CLI TUI mode: use prompt_toolkit widget instead of raw spinner
                # (works in both streaming and non-streaming modes)
                agent.thinking_callback(f"{face} {verb}...")
            elif not agent._has_stream_consumers() and agent._should_start_quiet_spinner():
                # Raw KawaiiSpinner only when no streaming consumers and the
                # spinner output has a safe sink.
                spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=agent._print_fn)
                thinking_spinner.start()
        
        # Log request details if verbose
        if agent.verbose_logging:
            logging.debug(f"API Request - Model: {agent.model}, Messages: {len(messages)}, Tools: {len(agent.tools) if agent.tools else 0}")
            logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
            logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
        
        api_start_time = time.time()
        retry_count = 0
        max_retries = agent._api_max_retries
        _retry = TurnRetryState()

        finish_reason = "stop"
        response = None  # Guard against UnboundLocalError if all retries fail
        api_kwargs = None  # Guard against UnboundLocalError in except handler
        api_request_id = f"{turn_id}:api:{api_call_count}"
        agent._current_api_request_id = api_request_id

        while retry_count < max_retries:
            # ── Nous Portal rate limit guard ──────────────────────
            # If another session already recorded that Nous is rate-
            # limited, skip the API call entirely.  Each attempt
            # (including SDK-level retries) counts against RPH and
            # deepens the rate limit hole.
            if agent.provider == "nous":
                try:
                    from agent.nous_rate_guard import (
                        nous_rate_limit_remaining,
                        format_remaining as _fmt_nous_remaining,
                    )
                    _nous_remaining = nous_rate_limit_remaining()
                    if _nous_remaining is not None and _nous_remaining > 0:
                        _nous_msg = (
                            f"Nous Portal rate limit active — "
                            f"resets in {_fmt_nous_remaining(_nous_remaining)}."
                        )
                        agent._buffer_vprint(
                            f"⏳ {_nous_msg} Trying fallback..."
                        )
                        agent._buffer_status(f"⏳ {_nous_msg}")
                        if agent._try_activate_fallback():
                            active_system_prompt = _sync_failover_system_message(
                                agent, api_messages, active_system_prompt)
                            retry_count = 0
                            compression_attempts = 0
                            _retry.primary_recovery_attempted = False
                            continue
                        # No fallback available — surface buffered context
                        # so user sees the rate-limit message that led here.
                        agent._flush_status_buffer()
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": (
                                f"⏳ {_nous_msg}\n\n"
                                "No fallback provider available. "
                                "Try again after the reset, or add a "
                                "fallback provider in config.yaml."
                            ),
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _nous_msg,
                        }
                except ImportError:
                    pass
                except Exception:
                    pass  # Never let rate guard break the agent loop

            try:
                agent._reset_stream_delivery_tracking()
                # api_messages is built once, before this retry loop, while the
                # primary provider is active.  A mid-conversation fallback can
                # switch to a require-side provider (DeepSeek / Kimi / MiMo) that
                # rejects assistant turns lacking reasoning_content.  Re-apply the
                # echo-back pad for the *current* provider here (idempotent no-op
                # unless the active provider needs it) so the fallback request
                # isn't sent with stale, primary-shaped reasoning fields.
                agent._reapply_reasoning_echo_for_provider(api_messages)
                # Same story for prompt-cache decoration (#72626): try_activate_
                # fallback refreshes the policy flags, but the decorated list
                # still carries the primary's breakpoints (or none). Strip and
                # re-render for the current provider before building kwargs.
                api_messages, _moa_prepared_request, tools_for_api = (
                    _redecorate_prompt_cache_for_provider(
                        agent,
                        api_messages,
                        system_message=system_message,
                        moa_prepared=_moa_prepared_request,
                        tools_for_api=tools_for_api,
                    )
                )
                if tools_for_api == agent.tools:
                    api_kwargs = agent._build_api_kwargs(api_messages)
                else:
                    api_kwargs = agent._build_api_kwargs(
                        api_messages,
                        tools_for_api=tools_for_api,
                    )
                if agent._force_ascii_payload:
                    _sanitize_structure_non_ascii(api_kwargs)
                if agent.api_mode == "codex_responses":
                    api_kwargs = agent._get_transport().preflight_kwargs(
                        api_kwargs,
                        allow_stream=False,
                        is_github_responses=agent._is_copilot_url(),
                        sanitize_harmony_tokens=agent._is_codex_backend(),
                    )
                # Copilot x-initiator: the first API call of a user turn is
                # marked "user" so Copilot bills a premium request; tool-loop
                # follow-ups keep the default "agent" header (#3040).
                if getattr(agent, "_is_user_initiated_turn", False) and agent._is_copilot_url():
                    _xh = dict(api_kwargs.get("extra_headers") or {})
                    _xh["x-initiator"] = "user"
                    api_kwargs["extra_headers"] = _xh
                    agent._is_user_initiated_turn = False
                try:
                    from hermes_cli.middleware import apply_llm_request_middleware

                    _llm_request_mw = apply_llm_request_middleware(
                        api_kwargs,
                        task_id=effective_task_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        session_id=agent.session_id or "",
                        platform=agent.platform or "",
                        model=agent.model,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_mode=agent.api_mode,
                        api_call_count=api_call_count,
                    )
                    api_kwargs = _llm_request_mw.payload
                    _original_api_kwargs = _llm_request_mw.original_payload
                    _llm_middleware_trace = _llm_request_mw.trace
                except Exception:
                    _original_api_kwargs = dict(api_kwargs)
                    _llm_middleware_trace = []

                try:
                    from hermes_cli.lifecycle import (
                        has_hook,
                        invoke_hook as _invoke_hook,
                    )
                    if has_hook("pre_api_request"):
                        request_messages = api_kwargs.get("messages")
                        if not isinstance(request_messages, list):
                            request_messages = api_kwargs.get("input")
                        if not isinstance(request_messages, list):
                            request_messages = api_messages
                        # Shallow-copy the outer list so plugins that retain the
                        # reference for async snapshotting don't observe later
                        # mutations of api_messages.  The inner dicts are not
                        # mutated by the agent loop, so a shallow copy is
                        # sufficient; a deepcopy would walk every tool result
                        # and base64 image on every API call.
                        #
                        # The ``request_messages`` and ``conversation_history``
                        # kwargs below are pre-existing raw passthroughs
                        # consumed by the bundled langfuse plugin
                        # (``plugins/observability/langfuse/__init__.py:_coerce_request_messages``).
                        # They predate ``request`` and are intentionally NOT
                        # sanitised — secrets are not expected here because
                        # ``api_kwargs`` is the same object passed to the
                        # provider client.  New consumers should read the
                        # sanitised view from ``request["body"]["messages"]``.
                        _request_payload = agent._api_request_payload_for_hook(api_kwargs)
                        _invoke_hook(
                            "pre_api_request",
                            task_id=effective_task_id,
                            turn_id=turn_id,
                            api_request_id=api_request_id,
                            session_id=agent.session_id or "",
                            user_message=original_user_message,
                            conversation_history=list(messages),
                            platform=agent.platform or "",
                            model=agent.model,
                            provider=agent.provider,
                            base_url=agent.base_url,
                            api_mode=agent.api_mode,
                            api_call_count=api_call_count,
                            retry_count=retry_count,
                            request_messages=list(request_messages)
                            if isinstance(request_messages, list)
                            else [],
                            message_count=len(api_messages),
                            tool_count=len(agent.tools or []),
                            approx_input_tokens=approx_tokens,
                            request_char_count=total_chars,
                            max_tokens=agent.max_tokens,
                            started_at=api_start_time,
                            middleware_trace=list(_llm_middleware_trace),
                            request=_request_payload,
                        )
                except Exception:
                    pass

                if env_var_enabled("HERMES_DUMP_REQUESTS"):
                    agent._dump_api_request_debug(api_kwargs, reason="preflight")

                # This object is private to the in-process MoA facade.  Add it
                # only after middleware, hooks, and debug dumps so none of them
                # attempts to serialize it as part of the provider payload.
                if _moa_prepared_request is not None and agent.provider == "moa":
                    api_kwargs["_moa_prepared_request"] = _moa_prepared_request

                # Always prefer the streaming path — even without stream
                # consumers.  Streaming gives us fine-grained health
                # checking (90s stale-stream detection, 60s read timeout)
                # that the non-streaming path lacks.  Without this,
                # subagents and other quiet-mode callers can hang
                # indefinitely when the provider keeps the connection
                # alive with SSE pings but never delivers a response.
                # The streaming path is a no-op for callbacks when no
                # consumers are registered, and falls back to non-
                # streaming automatically if the provider doesn't
                # support it.
                def _stop_spinner():
                    nonlocal thinking_spinner
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")

                _use_streaming = True
                # Provider signaled "stream not supported" on a previous
                # attempt — switch to non-streaming for the rest of this
                # session instead of re-failing every retry.
                if getattr(agent, "_disable_streaming", False):
                    _use_streaming = False
                # CopilotACPClient communicates via subprocess stdio and
                # returns a plain SimpleNamespace — not an iterable
                # stream.  Mirror the ACP exclusion used for Responses
                # API upgrade (lines ~1083-1085).
                elif (
                    agent.provider in {"copilot-acp"}
                    or str(agent.base_url or "").lower().startswith("acp://copilot")
                    or str(agent.base_url or "").lower().startswith("acp+tcp://")
                ):
                    _use_streaming = False
                # MoA streams only when a display/TTS consumer is present to
                # receive the deltas. MoAChatCompletions.create() honors
                # stream=True (runs the references, then returns the aggregator's
                # raw token stream) and is reached here because, for provider
                # "moa", _create_request_openai_client returns the MoA facade
                # itself. Without consumers (quiet mode, subagents, health-check
                # probes) we keep the complete-response path: the facade returns a
                # whole response when stream is not requested, preserving the
                # prior behavior for those callers.
                elif agent.provider == "moa" and not agent._has_stream_consumers():
                    _use_streaming = False
                elif not agent._has_stream_consumers():
                    # No display/TTS consumer. Still prefer streaming for
                    # health checking, but skip for Mock clients in tests
                    # (mocks return SimpleNamespace, not stream iterators).
                    from unittest.mock import Mock
                    if isinstance(getattr(agent, "client", None), Mock):
                        _use_streaming = False

                def _perform_api_call(next_api_kwargs):
                    if agent.api_mode == "codex_responses":
                        next_api_kwargs = agent._get_transport().preflight_kwargs(
                            next_api_kwargs,
                            allow_stream=False,
                            is_github_responses=agent._is_copilot_url(),
                            sanitize_harmony_tokens=agent._is_codex_backend(),
                        )
                    if _use_streaming:
                        return agent._interruptible_streaming_api_call(
                            next_api_kwargs, on_first_delta=_stop_spinner
                        )
                    from agent import relay_llm

                    return relay_llm.execute(
                        next_api_kwargs,
                        agent._interruptible_api_call,
                        session_id=str(agent.session_id or ""),
                        name=str(agent.provider or "provider"),
                        model_name=str(agent.model or ""),
                        metadata={
                            "api_mode": agent.api_mode,
                            "api_request_id": api_request_id,
                            "call_role": (
                                "delegated"
                                if getattr(agent, "is_subagent", False)
                                else "fallback"
                                if int(getattr(agent, "_fallback_index", 0) or 0) > 0
                                else "primary"
                            ),
                            "retry_count": retry_count,
                        },
                        defer_logical_completion=True,
                    )

                from hermes_cli.middleware import run_llm_execution_middleware

                _model_request_active = getattr(agent, "_model_request_active", None)
                _redirect_lock = getattr(agent, "_pending_redirect_lock", None)
                if _redirect_lock is not None:
                    with _redirect_lock:
                        if _model_request_active is not None:
                            _model_request_active.set()
                elif _model_request_active is not None:
                    _model_request_active.set()
                _redirect_crossed_response = False
                try:
                    response = run_llm_execution_middleware(
                        api_kwargs,
                        _perform_api_call,
                        original_request=_original_api_kwargs,
                        task_id=effective_task_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        session_id=agent.session_id or "",
                        platform=agent.platform or "",
                        model=agent.model,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_mode=agent.api_mode,
                        api_call_count=api_call_count,
                        middleware_trace=list(_llm_middleware_trace),
                    )
                finally:
                    if _redirect_lock is not None:
                        with _redirect_lock:
                            if _model_request_active is not None:
                                _model_request_active.clear()
                            _redirect_crossed_response = bool(
                                agent._pending_redirect
                            )
                    else:
                        if _model_request_active is not None:
                            _model_request_active.clear()
                        _redirect_crossed_response = agent._has_pending_redirect()
                if _redirect_crossed_response:
                    # The response and redirect can cross on different threads:
                    # redirect() observed the request as active just before this
                    # call returned. Discard that now-stale response and rebuild
                    # from the correction rather than silently losing it.
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")
                    if agent.clear_interrupt(preserve_redirect=True):
                        _retry.restart_with_redirected_messages = True
                    else:
                        interrupted = True
                    break
                
                api_duration = time.time() - api_start_time
                
                # Stop thinking spinner silently -- the response box or tool
                # execution messages that follow are more informative.
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")
                
                if not agent.quiet_mode:
                    agent._vprint(f"{agent.log_prefix}⏱️  API call completed in {api_duration:.2f}s")
                
                if agent.verbose_logging:
                    # Log response with provider info if available
                    resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                    logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                
                # Validate response shape before proceeding
                response_invalid = False
                error_details = []
                if agent.api_mode == "codex_responses":
                    _ct_v = agent._get_transport()
                    if not _ct_v.validate_response(response):
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        else:
                            # Provider returned a terminal failure (e.g. quota exhaustion).
                            # Treat as invalid so the fallback chain is triggered instead of
                            # letting the error bubble up outside the retry/fallback loop.
                            _codex_resp_status = str(getattr(response, "status", "") or "").strip().lower()
                            if _codex_resp_status in {"failed", "cancelled"}:
                                _codex_error_obj = getattr(response, "error", None)
                                _codex_error_msg = (
                                    _codex_error_obj.get("message") if isinstance(_codex_error_obj, dict)
                                    else str(_codex_error_obj) if _codex_error_obj
                                    else f"Responses API returned status '{_codex_resp_status}'"
                                )
                                logger.warning(
                                    "Codex response status='%s' (error=%s). Routing to fallback. %s",
                                    _codex_resp_status, _codex_error_msg,
                                    agent._client_log_context(),
                                )
                                response_invalid = True
                                error_details.append(f"response.status={_codex_resp_status}: {_codex_error_msg}")
                            else:
                                # output_text fallback: stream backfill may have failed
                                # but normalize can still recover from output_text
                                _out_text = getattr(response, "output_text", None)
                                _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
                                if _out_text_stripped:
                                    logger.debug(
                                        "Codex response.output is empty but output_text is present "
                                        "(%d chars); deferring to normalization.",
                                        len(_out_text_stripped),
                                    )
                                else:
                                    _resp_status = getattr(response, "status", None)
                                    _resp_incomplete = getattr(response, "incomplete_details", None)
                                    logger.warning(
                                        "Codex response.output is empty after stream backfill "
                                        "(status=%s, incomplete_details=%s, model=%s). %s",
                                        _resp_status, _resp_incomplete,
                                        getattr(response, "model", None),
                                        f"api_mode={agent.api_mode} provider={agent.provider}",
                                    )
                                    response_invalid = True
                                    error_details.append("response.output is empty")
                elif agent.api_mode == "anthropic_messages":
                    _tv = agent._get_transport()
                    if not _tv.validate_response(response):
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        else:
                            error_details.append("response.content invalid (not a non-empty list)")
                elif agent.api_mode == "bedrock_converse":
                    _btv = agent._get_transport()
                    if not _btv.validate_response(response):
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        else:
                            error_details.append("Bedrock response invalid (no output or choices)")
                else:
                    _ctv = agent._get_transport()
                    if not _ctv.validate_response(response):
                        response_invalid = True
                        if response is None:
                            error_details.append("response is None")
                        elif not hasattr(response, 'choices'):
                            error_details.append("response has no 'choices' attribute")
                        elif response.choices is None:
                            error_details.append("response.choices is None")
                        else:
                            error_details.append("response.choices is empty")

                if response_invalid:
                    agent._invoke_api_request_error_hook(
                        task_id=effective_task_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        api_call_count=api_call_count,
                        api_start_time=api_start_time,
                        api_kwargs=api_kwargs,
                        error_type="InvalidAPIResponse",
                        error_message=", ".join(error_details) or "Invalid API response",
                        status_code=getattr(getattr(response, "error", None), "code", None),
                        retry_count=retry_count,
                        max_retries=max_retries,
                        retryable=True,
                        reason="invalid_response",
                    )
                    # Stop spinner silently — retry status is now buffered
                    # and only surfaced if every retry+fallback exhausts.
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")
                    
                    # Invalid response — could be rate limiting, provider timeout,
                    # upstream server error, or malformed response.
                    retry_count += 1
                    
                    # Eager fallback: empty/malformed responses are a common
                    # rate-limit symptom.  Switch to fallback immediately
                    # rather than retrying with extended backoff.
                    if agent._fallback_index < len(agent._fallback_chain):
                        agent._buffer_status("⚠️ Empty/malformed response — switching to fallback...")
                    if agent._try_activate_fallback():
                        active_system_prompt = _sync_failover_system_message(
                            agent, api_messages, active_system_prompt)
                        retry_count = 0
                        compression_attempts = 0
                        _retry.primary_recovery_attempted = False
                        continue

                    # Check for error field in response (some providers include this)
                    error_msg = "Unknown"
                    provider_name = "Unknown"
                    if response and hasattr(response, 'error') and response.error:
                        error_msg = str(response.error)
                        # Try to extract provider from error metadata
                        if hasattr(response.error, 'metadata') and response.error.metadata:
                            provider_name = response.error.metadata.get('provider_name', 'Unknown')
                    elif response and hasattr(response, 'message') and response.message:
                        error_msg = str(response.message)
                    
                    # Try to get provider from model field (OpenRouter often returns actual model used)
                    if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                        provider_name = f"model={response.model}"
                    
                    # Check for x-openrouter-provider or similar metadata
                    if provider_name == "Unknown" and response:
                        # Log all response attributes for debugging
                        resp_attrs = {k: str(v)[:100] for k, v in (vars(response) if hasattr(response, "__dict__") else response.items() if isinstance(response, dict) else {}).items()}
                        if agent.verbose_logging:
                            logging.debug(f"Response attributes for invalid response: {resp_attrs}")
                    
                    # Extract error code from response for contextual diagnostics
                    _resp_error_code = None
                    if response and hasattr(response, 'error') and response.error:
                        _code_raw = getattr(response.error, 'code', None)
                        if _code_raw is None and isinstance(response.error, dict):
                            _code_raw = response.error.get('code')
                        if _code_raw is not None:
                            try:
                                _resp_error_code = int(_code_raw)
                            except (TypeError, ValueError):
                                pass

                    # Build a human-readable failure hint from the error code
                    # and response time, instead of always assuming rate limiting.
                    if _resp_error_code == 524:
                        _failure_hint = f"upstream provider timed out (Cloudflare 524, {api_duration:.0f}s)"
                    elif _resp_error_code == 504:
                        _failure_hint = f"upstream gateway timeout (504, {api_duration:.0f}s)"
                    elif _resp_error_code == 429:
                        _failure_hint = "rate limited by upstream provider (429)"
                    elif _resp_error_code in {500, 502}:
                        _failure_hint = f"upstream server error ({_resp_error_code}, {api_duration:.0f}s)"
                    elif _resp_error_code in {503, 529}:
                        _failure_hint = f"upstream provider overloaded ({_resp_error_code})"
                    elif _resp_error_code is not None:
                        _failure_hint = f"upstream error (code {_resp_error_code}, {api_duration:.0f}s)"
                    elif api_duration < 10:
                        _failure_hint = f"fast response ({api_duration:.1f}s) — likely rate limited"
                    elif api_duration > 60:
                        _failure_hint = f"slow response ({api_duration:.0f}s) — likely upstream timeout"
                    else:
                        _failure_hint = f"response time {api_duration:.1f}s"

                    agent._buffer_vprint(f"⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}")
                    agent._buffer_vprint(f"   🏢 Provider: {provider_name}")
                    cleaned_provider_error = agent._clean_error_message(error_msg)
                    agent._buffer_vprint(f"   📝 Provider message: {cleaned_provider_error}")
                    agent._buffer_vprint(f"   ⏱️  {_failure_hint}")
                    
                    if retry_count >= max_retries:
                        # Try fallback before giving up
                        if agent._has_pending_fallback():
                            agent._buffer_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                        if agent._try_activate_fallback():
                            active_system_prompt = _sync_failover_system_message(
                                agent, api_messages, active_system_prompt)
                            retry_count = 0
                            compression_attempts = 0
                            _retry.primary_recovery_attempted = False
                            continue
                        # Terminal — flush buffered retry trace so user sees what happened.
                        agent._flush_status_buffer()
                        agent._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                        logger.error("%sInvalid API response after %d retries.", agent.log_prefix, max_retries)
                        agent._persist_session(messages, conversation_history)
                        _final_response = f"Invalid API response after {max_retries} retries: {_failure_hint}"
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": _final_response,
                            "failed": True  # Mark as failure for filtering
                        }
                    
                    # Backoff before retry — jittered exponential: 5s base, 120s cap
                    wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                    agent._buffer_vprint(f"⏳ Retrying in {wait_time:.1f}s ({_failure_hint})...")
                    logger.warning("Invalid API response (retry %d/%d): %s | Provider: %s", retry_count, max_retries, ', '.join(error_details), provider_name)
                    
                    # Sleep in small increments to stay responsive to interrupts
                    sleep_end = time.time() + wait_time
                    _backoff_touch_counter = 0
                    while time.time() < sleep_end:
                        if agent._interrupt_requested:
                            # A redirect uses the interrupt machinery to cancel
                            # only the live request. Aborting the retry here
                            # with clear_interrupt() would DESTROY the pending
                            # correction and kill the turn with "Operation
                            # interrupted" — the exact mid-stream steer loss
                            # users hit when a redirect lands during provider
                            # backoff. Rebuild from the correction instead,
                            # mirroring the InterruptedError handler.
                            if agent.clear_interrupt(preserve_redirect=True):
                                _retry.restart_with_redirected_messages = True
                                break
                            agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                            _interrupt_text = f"Operation interrupted during retry ({_failure_hint}, attempt {retry_count}/{max_retries})."
                            close_interrupted_tool_sequence(messages, _interrupt_text)
                            agent._persist_session(messages, conversation_history)
                            agent.clear_interrupt()
                            return {
                                "final_response": _interrupt_text,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)
                        # Touch activity every ~30s so the gateway's inactivity
                        # monitor knows we're alive during backoff waits.
                        _backoff_touch_counter += 1
                        if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                            agent._touch_activity(
                                f"retry backoff ({retry_count}/{max_retries}), "
                                f"{int(sleep_end - time.time())}s remaining"
                            )
                    if _retry.restart_with_redirected_messages:
                        break  # rebuild this iteration from the correction
                    continue  # Retry the API call

                agent._turn_received_provider_response = True

                # Check finish_reason before proceeding
                if agent.api_mode == "codex_responses":
                    status = getattr(response, "status", None)
                    if isinstance(status, str):
                        status = status.strip().lower()
                    incomplete_details = getattr(response, "incomplete_details", None)
                    incomplete_reason = None
                    if isinstance(incomplete_details, dict):
                        incomplete_reason = incomplete_details.get("reason")
                    else:
                        incomplete_reason = getattr(incomplete_details, "reason", None)
                    if incomplete_reason is not None:
                        incomplete_reason = str(incomplete_reason).strip().lower()
                    if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                        # Responses API max-output exhaustion is a normal
                        # Codex incomplete turn.  Let the Codex-specific
                        # continuation path below append the incomplete
                        # assistant state and retry, instead of routing to
                        # the generic chat-completions length rollback that
                        # emits "Response truncated due to output length
                        # limit" and stops gateway turns.
                        finish_reason = "incomplete"
                    elif status == "incomplete" and incomplete_reason == "content_filter":
                        finish_reason = "content_filter"
                    else:
                        finish_reason = "stop"
                elif agent.api_mode == "anthropic_messages":
                    _tfr = agent._get_transport()
                    finish_reason = _tfr.map_finish_reason(response.stop_reason)
                elif agent.api_mode == "bedrock_converse":
                    # Bedrock response already normalized at dispatch — use transport
                    _bt_fr = agent._get_transport()
                    _bedrock_result = _bt_fr.normalize_response(response)
                    finish_reason = _bedrock_result.finish_reason
                else:
                    _cc_fr = agent._get_transport()
                    _finish_result = _cc_fr.normalize_response(response)
                    finish_reason = _finish_result.finish_reason
                    assistant_message = _finish_result
                    if agent._should_treat_stop_as_truncated(
                        finish_reason,
                        assistant_message,
                        messages,
                    ):
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Treating suspicious Ollama/GLM stop response as truncated",
                            force=True,
                        )
                        finish_reason = "length"

                # ── Content-policy refusal (HTTP 200) ──────────────────
                # The model — or the provider's safety system — returned a
                # *successful* response whose stop/finish reason is a refusal:
                # Anthropic ``stop_reason="refusal"`` → ``content_filter``;
                # OpenAI / portal ``finish_reason="content_filter"`` or a
                # populated ``message.refusal`` (mapped in the chat_completions
                # transport); Bedrock ``guardrail_intervened``. The content is
                # typically empty, so without this branch the response falls
                # through to the empty-response / invalid-response retry loops
                # and is mis-surfaced as "rate limited" / "no content after
                # retries" — burning paid attempts reproducing a deterministic
                # refusal. Surface it clearly and stop. Mirrors the
                # exception-based ``content_policy_blocked`` recovery: try a
                # configured fallback once, otherwise return the refusal.
                if finish_reason == "content_filter":
                    _refusal_transport = agent._get_transport()
                    if agent.api_mode == "anthropic_messages":
                        _refusal_result = _refusal_transport.normalize_response(
                            response, strip_tool_prefix=agent._is_anthropic_oauth
                        )
                    else:
                        _refusal_result = _refusal_transport.normalize_response(response)
                    _refusal_text = (getattr(_refusal_result, "content", None) or "").strip()
                    # Some refusals carry the explanation only in the reasoning
                    # channel; fall back to it so the user sees *something*.
                    if not _refusal_text:
                        _refusal_text = (agent._extract_reasoning(_refusal_result) or "").strip()

                    agent._invoke_api_request_error_hook(
                        task_id=effective_task_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        api_call_count=api_call_count,
                        api_start_time=api_start_time,
                        api_kwargs=api_kwargs,
                        error_type="ContentPolicyBlocked",
                        error_message=_refusal_text or "model declined to respond (content_filter)",
                        status_code=None,
                        retry_count=retry_count,
                        max_retries=max_retries,
                        retryable=False,
                        reason=FailoverReason.content_policy_blocked.value,
                    )

                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if agent.thinking_callback:
                        agent.thinking_callback("")

                    # Deterministic for the unchanged prompt — never retry.
                    # Try a configured fallback once (a different model may not
                    # refuse); otherwise surface the refusal terminally.
                    if agent._has_pending_fallback():
                        agent._buffer_status(
                            "⚠️ Model declined to respond (safety refusal) — trying fallback..."
                        )
                    if agent._try_activate_fallback():
                        active_system_prompt = _sync_failover_system_message(
                            agent, api_messages, active_system_prompt)
                        retry_count = 0
                        compression_attempts = 0
                        _retry.primary_recovery_attempted = False
                        continue

                    agent._flush_status_buffer()
                    _refusal_log = (
                        _refusal_text[:500] + "..."
                        if len(_refusal_text) > 500
                        else _refusal_text
                    )
                    logger.warning(
                        "%sModel declined to respond (finish_reason=content_filter). "
                        "model=%s provider=%s refusal=%s",
                        agent.log_prefix, agent.model, agent.provider,
                        _refusal_log or "(no text)",
                    )
                    agent._emit_status(
                        "⚠️ The model declined to respond to this request (safety refusal)."
                    )

                    _refusal_detail = (
                        f"Model's explanation: {_refusal_text}"
                        if _refusal_text
                        else "The model returned no explanation."
                    )
                    _refusal_response = (
                        "⚠️  The model declined to respond to this request "
                        "(safety refusal — not a Hermes/gateway failure).\n\n"
                        f"{_refusal_detail}\n\n"
                        f"{_CONTENT_POLICY_RECOVERY_HINT}"
                    )

                    agent._cleanup_task_resources(effective_task_id)
                    agent._persist_session(messages, conversation_history)
                    return _content_policy_blocked_result(
                        messages,
                        api_call_count,
                        final_response=_refusal_response,
                        error_detail=_refusal_text or "model declined (content_filter)",
                    )

                if finish_reason == "length":
                    if getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Stream interrupted by network error "
                            f"(finish_reason='length' on partial-stream-stub)",
                            force=True,
                        )
                    else:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Response truncated "
                            f"(finish_reason='length') - model hit max output tokens",
                            force=True,
                        )

                    # Normalize the truncated response to a single OpenAI-style
                    # message shape so text-continuation and tool-call retry
                    # work uniformly across chat_completions, bedrock_converse,
                    # and anthropic_messages.  For Anthropic we use the same
                    # adapter the agent loop already relies on so the rebuilt
                    # interim assistant message is byte-identical to what
                    # would have been appended in the non-truncated path.
                    _trunc_msg = None
                    _trunc_transport = agent._get_transport()
                    if agent.api_mode == "anthropic_messages":
                        _trunc_result = _trunc_transport.normalize_response(
                            response, strip_tool_prefix=agent._is_anthropic_oauth
                        )
                    else:
                        _trunc_result = _trunc_transport.normalize_response(response)
                    _trunc_msg = _trunc_result

                    _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                    _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False

                    # ── Detect thinking-budget exhaustion ──────────────
                    # When the model spends ALL output tokens on reasoning
                    # and has none left for the response, continuation
                    # retries are pointless.  Detect this early and give a
                    # targeted error instead of wasting 3 API calls.
                    # A response is "thinking exhausted" only when the model
                    # actually produced reasoning blocks but no visible text after
                    # them.  Models that do not use <think> tags (e.g. GLM-4.7 on
                    # NVIDIA Build, minimax) may return content=None or an empty
                    # string for unrelated reasons — treat those as normal
                    # truncations that deserve continuation retries, not as
                    # thinking-budget exhaustion.
                    _has_think_tags = bool(
                        _trunc_content and re.search(
                            r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>',
                            _trunc_content,
                            re.IGNORECASE,
                        )
                    )
                    _thinking_exhausted = (
                        not _trunc_has_tool_calls
                        and _has_think_tags
                        and (
                            (_trunc_content is not None and not agent._has_content_after_think_block(_trunc_content))
                            or _trunc_content is None
                        )
                    )

                    if _thinking_exhausted:
                        _exhaust_error = (
                            "Model used all output tokens on reasoning with none left "
                            "for the response. Try lowering reasoning effort or "
                            "increasing max_tokens."
                        )
                        agent._vprint(
                            f"{agent.log_prefix}💭 Reasoning exhausted the output token budget — "
                            f"no visible response was produced.",
                            force=True,
                        )
                        # Return a user-friendly message as the response so
                        # CLI (response box) and gateway (chat message) both
                        # display it naturally instead of a suppressed error.
                        _exhaust_response = (
                            "⚠️ **Thinking Budget Exhausted**\n\n"
                            "The model used all its output tokens on reasoning "
                            "and had none left for the actual response.\n\n"
                            "To fix this:\n"
                            "→ Lower reasoning effort: `/thinkon low` or `/thinkon minimal`\n"
                            "→ Or switch to a larger/non-reasoning model with `/model`"
                        )
                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": _exhaust_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": _exhaust_error,
                        }

                    if agent.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
                        assistant_message = _trunc_msg
                        # ── Content-filter stream stall → fallback (#32421) ──
                        # When the provider's output-layer safety filter (e.g.
                        # MiniMax "output new_sensitive (1027)", Azure
                        # content_filter) kills the stream mid-delivery, the
                        # raw error was classified at the swallow point and the
                        # stub tagged ``_content_filter_terminated``.  This
                        # filter is content-deterministic — continuation
                        # retries against the SAME primary just re-hit it and
                        # burn paid attempts (the loop used to give up with
                        # "Response remained truncated after 3 continuation
                        # attempts" and never consult the fallback chain).
                        # Escalate to the configured fallback BEFORE retrying.
                        _cf_terminated = getattr(
                            response, "_content_filter_terminated", False
                        )
                        if (
                            _cf_terminated
                            and agent._fallback_index < len(agent._fallback_chain)
                        ):
                            agent._vprint(
                                f"{agent.log_prefix}🛡️  Content filter terminated "
                                f"stream — activating fallback provider...",
                                force=True,
                            )
                            agent._emit_status(
                                "Content filter terminated stream; switching to fallback..."
                            )
                            if agent._try_activate_fallback():
                                # Roll the partial content (if any was already
                                # appended in a prior continuation pass) back to
                                # the last clean turn so the fallback provider
                                # gets a coherent continuation point.
                                if truncated_response_parts:
                                    messages = agent._get_messages_up_to_last_assistant(messages)
                                agent._session_messages = messages
                                length_continue_retries = 0
                                truncated_response_parts = []
                                retry_count = 0
                                compression_attempts = 0
                                _retry.primary_recovery_attempted = False
                                _retry.restart_with_rebuilt_messages = True
                                break
                            # No fallback available — fall through to normal
                            # continuation (best-effort, may loop).
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  No fallback provider "
                                f"configured — retrying with same provider "
                                f"(may re-hit filter)...",
                                force=True,
                            )
                        if assistant_message is not None and not _trunc_has_tool_calls:
                            length_continue_retries += 1
                            # An EMPTY partial-stream stub (stream dropped
                            # mid tool-call before any text was delivered)
                            # must not be appended as an interim assistant
                            # message: it would serialize as
                            # {"role": "assistant", "content": ""}, and
                            # strict providers (Moonshot/Kimi via OpenRouter)
                            # reject empty assistant content with HTTP 400
                            # ("message ... with role 'assistant' must not be
                            # empty") on the very next replay — permanently
                            # poisoning the session history.  There is no
                            # partial text to continue from anyway, so only
                            # the continuation user-message is appended.
                            _is_empty_partial_stub = (
                                getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
                                and not getattr(assistant_message, "content", None)
                            )
                            if not _is_empty_partial_stub:
                                interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
                                messages.append(interim_msg)
                                if assistant_message.content:
                                    truncated_response_parts.append(assistant_message.content)

                            if length_continue_retries < 4:
                                _is_partial_stream_stub = (
                                    getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
                                )
                                _dropped_tools = getattr(
                                    response, "_dropped_tool_names", None
                                )

                                if _is_partial_stream_stub and _dropped_tools:
                                    _tool_list = ", ".join(_dropped_tools[:3])
                                    agent._vprint(
                                        f"{agent.log_prefix}↻ Stream interrupted mid "
                                        f"tool-call ({_tool_list}) — requesting "
                                        f"chunked retry "
                                        f"({length_continue_retries}/4)..."
                                    )
                                elif _is_partial_stream_stub:
                                    agent._vprint(
                                        f"{agent.log_prefix}↻ Stream interrupted — "
                                        f"requesting continuation "
                                        f"({length_continue_retries}/4)..."
                                    )
                                else:
                                    agent._vprint(
                                        f"{agent.log_prefix}↻ Requesting continuation "
                                        f"({length_continue_retries}/4)..."
                                    )

                                _continue_content = _get_continuation_prompt(
                                    _is_partial_stream_stub, _dropped_tools
                                )
                                continue_msg = {
                                    "role": "user",
                                    "content": _continue_content,
                                }
                                messages.append(continue_msg)
                                agent._session_messages = messages
                                _retry.restart_with_length_continuation = True
                                break

                            partial_response = agent._strip_think_blocks("".join(truncated_response_parts)).strip()
                            agent._cleanup_task_resources(effective_task_id)
                            agent._persist_session(messages, conversation_history)
                            return {
                                "final_response": partial_response or None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response remained truncated after 4 continuation attempts",
                            }

                    if agent.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
                        assistant_message = _trunc_msg
                        if assistant_message is not None and _trunc_has_tool_calls:
                            _is_stub_stall = (
                                getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
                            )
                            if truncated_tool_call_retries < 4:
                                truncated_tool_call_retries += 1
                                if _is_stub_stall:
                                    # The stream broke mid tool-call (network /
                                    # peer-closed connection), not a real output
                                    # cap — say so instead of "max output tokens".
                                    agent._buffer_vprint(
                                        f"⚠️  Stream interrupted mid tool-call — "
                                        f"retrying ({truncated_tool_call_retries}/4)..."
                                    )
                                else:
                                    agent._buffer_vprint(
                                        f"⚠️  Truncated tool call detected — "
                                        f"retrying API call "
                                        f"({truncated_tool_call_retries}/4)..."
                                    )
                                # Boost max_tokens on each retry so the model has
                                # more room to complete the tool-call JSON. A
                                # network stall doesn't need a bigger budget, but
                                # a genuine output-cap truncation does, and the
                                # boost is harmless for the stall case.
                                _tc_boost_base = agent.max_tokens if agent.max_tokens else 4096
                                _tc_boost = _tc_boost_base * (2 ** truncated_tool_call_retries)
                                _tc_requested_cap = agent._requested_output_cap_from_api_kwargs(api_kwargs)
                                if _tc_requested_cap is not None:
                                    _tc_boost = max(_tc_boost, _tc_requested_cap)
                                _tc_boost_cap = max(32768, _tc_requested_cap or 0)
                                agent._ephemeral_max_output_tokens = min(_tc_boost, _tc_boost_cap)
                                # Don't append the broken response to messages;
                                # just re-run the same API call from the current
                                # message state, giving the model another chance.
                                continue
                            agent._flush_status_buffer()
                            if _is_stub_stall:
                                agent._vprint(
                                    f"{agent.log_prefix}⚠️  Stream kept dropping mid tool-call after 4 retries — the action was not executed.",
                                    force=True,
                                )
                            else:
                                agent._vprint(
                                    f"{agent.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
                                    force=True,
                                )
                            agent._cleanup_task_resources(effective_task_id)
                            _final_response = (
                                "Stream repeatedly dropped mid tool-call (network); "
                                "the tool was not executed"
                                if _is_stub_stall
                                else "Response truncated due to output length limit"
                            )
                            # Prior successful tool batches (or injected tool
                            # errors) can leave a tool-result tail; this path
                            # never reaches finalize_turn (#48879 class).
                            close_interrupted_tool_sequence(messages, _final_response)
                            agent._persist_session(messages, conversation_history)
                            return {
                                "final_response": _final_response,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": _final_response,
                            }

                    # If we have prior messages, roll back to last complete state
                    if len(messages) > 1:
                        agent._vprint(f"{agent.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                        rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)

                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)

                        return {
                            "final_response": "Response truncated due to output length limit",
                            "messages": rolled_back_messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Response truncated due to output length limit"
                        }
                    else:
                        # First message was truncated - mark as failed
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ First response truncated - cannot recover", force=True)
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": "First response truncated due to output length limit",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": "First response truncated due to output length limit"
                        }
                
                # Track actual token usage from response for context management
                if hasattr(response, 'usage') and response.usage:
                    canonical_usage = normalize_usage(
                        response.usage,
                        provider=agent.provider,
                        api_mode=agent.api_mode,
                    )
                    # Aggregator-only usage is retained for cost pricing: MoA
                    # advisor tokens must be priced at each advisor's OWN model
                    # rate, not the aggregator's, so they are added as dollars
                    # (below) rather than folded into the priced usage.
                    aggregator_usage = canonical_usage
                    # MoA: fold the reference (advisor) fan-out's token usage
                    # into this turn's REPORTED token counts. MoA runs advisors
                    # before the aggregator and returns only the aggregator's
                    # usage, so without this the entire advisor spend — usually
                    # the bulk of a MoA turn — is invisible in token counts.
                    _moa_ref_cost = None
                    _moa_client = getattr(agent, "client", None)
                    if _moa_client is not None and hasattr(_moa_client, "consume_reference_usage"):
                        try:
                            _ref_usage, _moa_ref_cost = _moa_client.consume_reference_usage()
                            if _ref_usage is not None:
                                canonical_usage = canonical_usage + _ref_usage
                        except Exception as _moa_acct_exc:  # pragma: no cover - defensive
                            logger.debug("MoA reference usage accounting failed: %s", _moa_acct_exc)
                    # Flush the full-turn MoA trace (references + aggregator I/O)
                    # to disk when moa.save_traces is on. No-op otherwise and
                    # for non-MoA clients. Uses the live session_id so traces
                    # land in the right per-session file. On the streaming path
                    # the aggregator's output wasn't captured inline (its raw
                    # token stream went to the live consumer), so pass the
                    # resolved streamed acting text as a fallback — makes the
                    # trace self-contained instead of only pointing at state.db.
                    if _moa_client is not None and hasattr(_moa_client, "consume_and_save_trace"):
                        try:
                            _agg_streamed_text = (
                                getattr(agent, "_current_streamed_assistant_text", "") or ""
                            )
                            _moa_client.consume_and_save_trace(
                                agent.session_id,
                                aggregator_output_fallback=_agg_streamed_text or None,
                            )
                        except Exception as _moa_trace_exc:  # pragma: no cover - defensive
                            logger.debug("MoA trace flush failed: %s", _moa_trace_exc)
                    prompt_tokens = canonical_usage.prompt_tokens
                    completion_tokens = canonical_usage.output_tokens
                    total_tokens = canonical_usage.total_tokens
                    # Forward canonical token + cache buckets so context engines
                    # can make decisions on cache hit ratios / reasoning costs,
                    # not just legacy aggregate tokens. Legacy keys stay for
                    # back-compat with engines that only read prompt/completion/total.
                    usage_dict = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "input_tokens": canonical_usage.input_tokens,
                        "output_tokens": canonical_usage.output_tokens,
                        "cache_read_tokens": canonical_usage.cache_read_tokens,
                        "cache_write_tokens": canonical_usage.cache_write_tokens,
                        "reasoning_tokens": canonical_usage.reasoning_tokens,
                    }
                    agent.context_compressor.update_from_response(usage_dict)

                    # Stash this response's canonical usage so the post-turn
                    # on_turn_complete() observation hook can forward it (the
                    # same dict shape passed to update_from_response). A turn
                    # may make several API calls; the engine's per-turn signal
                    # of interest is the cost/size of the latest assembled
                    # request, so we keep the most recent call's usage.
                    agent._last_turn_usage = dict(usage_dict)
                elif getattr(
                    agent.context_compressor,
                    "awaiting_real_usage_after_compression",
                    False,
                ):
                    # A response with no usage cannot adjudicate whether the
                    # prior compaction cleared the threshold. Consume the pending
                    # verdict now so a much later, unrelated reading is not
                    # charged to that old compaction, and so preflight deferral
                    # does not remain latched indefinitely.
                    agent.context_compressor.update_from_response({})

                if hasattr(response, 'usage') and response.usage:
                    # Cache discovered context length after successful call.
                    # Only persist limits confirmed by the provider (parsed
                    # from the error message), not guessed probe tiers.
                    if getattr(agent.context_compressor, "_context_probed", False):
                        ctx = agent.context_compressor.context_length
                        if getattr(agent.context_compressor, "_context_probe_persistable", False):
                            save_context_length(agent.model, agent.base_url, ctx)
                            agent._safe_print(f"{agent.log_prefix}💾 Cached context length: {ctx:,} tokens for {agent.model}")
                        agent.context_compressor._context_probed = False
                        agent.context_compressor._context_probe_persistable = False

                    agent.session_prompt_tokens += prompt_tokens
                    agent.session_completion_tokens += completion_tokens
                    agent.session_total_tokens += total_tokens
                    agent.session_api_calls += 1
                    agent.session_input_tokens += canonical_usage.input_tokens
                    agent.session_output_tokens += canonical_usage.output_tokens
                    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
                    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
                    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens

                    # Log API call details for debugging/observability
                    _cache_pct = ""
                    if canonical_usage.cache_read_tokens and prompt_tokens:
                        _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
                    logger.info(
                        "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                        agent.session_api_calls, agent.model, agent.provider or "unknown",
                        prompt_tokens, completion_tokens, total_tokens,
                        api_duration, _cache_pct,
                    )

                    # On the MoA path, agent.model/provider are the virtual
                    # preset name ("closed") and "moa", which have no pricing
                    # entry — estimating against them returns None and silently
                    # drops the aggregator's own spend, leaving the session cost
                    # as advisor-fan-out only (a ~50% undercount when the
                    # aggregator does the full acting loop). Price the aggregator
                    # turn at its REAL model/provider, read from the MoA client's
                    # resolved aggregator slot.
                    _agg_cost_model = agent.model
                    _agg_cost_provider = agent.provider
                    _agg_cost_base_url = agent.base_url
                    _agg_slot = getattr(_moa_client, "last_aggregator_slot", None) if _moa_client is not None else None
                    if _agg_slot and _agg_slot.get("model"):
                        _agg_cost_model = _agg_slot["model"]
                        _agg_cost_provider = _agg_slot.get("provider") or agent.provider
                        _agg_cost_base_url = _agg_slot.get("base_url") or agent.base_url
                    cost_result = estimate_usage_cost(
                        _agg_cost_model,
                        aggregator_usage,
                        provider=_agg_cost_provider,
                        base_url=_agg_cost_base_url,
                        api_key=getattr(agent, "api_key", ""),
                    )
                    if cost_result.amount_usd is not None:
                        agent.session_estimated_cost_usd += float(cost_result.amount_usd)
                    # Add MoA advisor cost (already priced per-advisor at each
                    # advisor's own model rate) on top of the aggregator cost.
                    if _moa_ref_cost is not None:
                        try:
                            agent.session_estimated_cost_usd += float(_moa_ref_cost)
                        except (TypeError, ValueError):  # pragma: no cover - defensive
                            pass
                    agent.session_cost_status = cost_result.status
                    agent.session_cost_source = cost_result.source

                    # Persist token counts to session DB for /insights.
                    # Do this for every platform with a session_id so non-CLI
                    # sessions (gateway, cron, delegated runs) cannot lose
                    # token/accounting data if a higher-level persistence path
                    # is skipped or fails. Gateway/session-store writes use
                    # absolute totals, so they safely overwrite these per-call
                    # deltas instead of double-counting them.
                    if agent._session_db and agent.session_id:
                        try:
                            # Ensure the session row exists before attempting UPDATE.
                            # Under concurrent load (cron/kanban), the initial
                            # _ensure_db_session() may have failed due to SQLite
                            # locking.  Retry here so per-call token deltas are
                            # not silently lost (UPDATE on a non-existent row
                            # affects 0 rows without error).
                            if not agent._session_db_created:
                                agent._ensure_db_session()
                            # Per-call cost delta = aggregator cost + MoA
                            # advisor cost (each priced at its own rate). Folded
                            # here so state.db's estimated_cost_usd includes the
                            # full MoA spend, matching the folded token counts.
                            _cost_delta = None
                            if cost_result.amount_usd is not None:
                                _cost_delta = float(cost_result.amount_usd)
                            if _moa_ref_cost is not None:
                                try:
                                    _cost_delta = (_cost_delta or 0.0) + float(_moa_ref_cost)
                                except (TypeError, ValueError):  # pragma: no cover
                                    pass
                            # Enqueued, not written: the background writer
                            # applies the delta off the turn thread (a cold
                            # state.db UPDATE here stalled the tool loop for
                            # up to hundreds of ms per API call). Drained at
                            # turn finalize via _persist_session.
                            agent._session_db.queue_token_counts(
                                agent.session_id,
                                input_tokens=canonical_usage.input_tokens,
                                output_tokens=canonical_usage.output_tokens,
                                cache_read_tokens=canonical_usage.cache_read_tokens,
                                cache_write_tokens=canonical_usage.cache_write_tokens,
                                reasoning_tokens=canonical_usage.reasoning_tokens,
                                estimated_cost_usd=_cost_delta,
                                cost_status=cost_result.status,
                                cost_source=cost_result.source,
                                billing_provider=agent.provider,
                                billing_base_url=agent.base_url,
                                billing_mode="subscription_included"
                                if cost_result.status == "included" else None,
                                model=agent.model,
                                api_call_count=1,
                            )
                        except Exception as e:
                            # Log token persistence failures so they're
                            # visible in agent.log — silent loss here is
                            # the root cause of undercounted analytics.
                            logger.debug(
                                "Token persistence failed (session=%s, tokens=%d): %s",
                                agent.session_id, total_tokens, e,
                            )
                    
                    if agent.verbose_logging:
                        logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")
                    
                    # Surface cache hit stats for any provider that reports
                    # them — not just those where we inject cache_control
                    # markers.  OpenAI/Kimi/DeepSeek/Qwen all do automatic
                    # server-side prefix caching and return
                    # ``prompt_tokens_details.cached_tokens``; users
                    # previously could not see their cache % because this
                    # line was gated on ``_use_prompt_caching``, which is
                    # only True for Anthropic-style marker injection.
                    # ``canonical_usage`` is already normalised from all
                    # three API shapes (Anthropic / Codex / OpenAI-chat)
                    # so we can rely on its values directly.
                    cached = canonical_usage.cache_read_tokens
                    written = canonical_usage.cache_write_tokens
                    prompt = usage_dict["prompt_tokens"]
                    if (cached or written) and not agent.quiet_mode:
                        hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                        agent._vprint(
                            f"{agent.log_prefix}   💾 Cache: "
                            f"{cached:,}/{prompt:,} tokens "
                            f"({hit_pct:.0f}% hit, {written:,} written)"
                        )
                
                _retry.has_retried_429 = False  # Reset on success
                # Note: don't clear the retry buffer here — an "API call
                # success" only means we got bytes back, not that we got
                # usable content. Empty responses still loop through the
                # empty-retry path below; the buffer is cleared when
                # genuinely successful content is detected later (~L4127).
                # Clear Nous rate limit state on successful request —
                # proves the limit has reset and other sessions can
                # resume hitting Nous.
                if agent.provider == "nous":
                    try:
                        from agent.nous_rate_guard import clear_nous_rate_limit
                        clear_nous_rate_limit()
                    except Exception:
                        pass
                from agent import relay_llm

                relay_llm.complete_logical_call(
                    api_request_id,
                    outcome="success",
                )
                agent._touch_activity(f"API call #{api_call_count} completed")
                break  # Success, exit retry loop

            except InterruptedError:
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")
                if agent._has_pending_redirect():
                    # redirect() deliberately used the interrupt machinery to
                    # cancel only this provider request. Keep its correction
                    # queued, clear the cancellation bit, and let the outer
                    # loop rebuild a clean request tail. Never materialize
                    # incomplete signed/encrypted reasoning items.
                    if agent.clear_interrupt(preserve_redirect=True):
                        _retry.restart_with_redirected_messages = True
                        break
                api_elapsed = time.time() - api_start_time
                agent._vprint(f"{agent.log_prefix}⚡ Interrupted during API call.", force=True)
                interrupted = True
                # Preserve any assistant text already streamed to the user
                # before the stop landed. Dropping it leaves history with no
                # record of the half-finished reply on screen, so the next turn
                # the model "forgets" what it just said — exactly what users hit
                # when they stop to redirect mid-response.
                _partial = agent._strip_think_blocks(
                    getattr(agent, "_current_streamed_assistant_text", "") or ""
                ).strip()
                if _partial:
                    messages.append({"role": "assistant", "content": _partial})
                    final_response = _partial
                else:
                    final_response = f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}{api_elapsed:.1f}s elapsed)."
                agent._persist_session(messages, conversation_history)
                break

            except Exception as api_error:
                # Stop spinner silently — retry status is buffered and
                # only flushed when every retry+fallback is exhausted.
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")

                # -----------------------------------------------------------
                # UnicodeEncodeError recovery.  Two common causes:
                #   1. Lone surrogates (U+D800..U+DFFF) from clipboard paste
                #      (Google Docs, rich-text editors) — sanitize and retry.
                #   2. ASCII codec on systems with LANG=C or non-UTF-8 locale
                #      (e.g. Chromebooks) — any non-ASCII character fails.
                #      Detect via the error message mentioning 'ascii' codec.
                # We sanitize messages in-place and may retry twice:
                # first to strip surrogates, then once more for pure
                # ASCII-only locale sanitization if needed.
                # -----------------------------------------------------------
                if isinstance(api_error, UnicodeEncodeError) and getattr(agent, '_unicode_sanitization_passes', 0) < 2:
                    _err_str = str(api_error).lower()
                    _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
                    # Detect surrogate errors — utf-8 codec refusing to
                    # encode U+D800..U+DFFF.  The error text is:
                    #   "'utf-8' codec can't encode characters in position
                    #    N-M: surrogates not allowed"
                    _is_surrogate_error = (
                        "surrogate" in _err_str
                        or ("'utf-8'" in _err_str and not _is_ascii_codec)
                    )
                    # Sanitize surrogates from both the canonical `messages`
                    # list AND `api_messages` (the API-copy, which may carry
                    # `reasoning_content`/`reasoning_details` transformed
                    # from `reasoning` — fields the canonical list doesn't
                    # have directly).  Also clean `api_kwargs` if built and
                    # `prefill_messages` if present.  Mirrors the ASCII
                    # codec recovery below.
                    _surrogates_found = _sanitize_messages_surrogates(messages)
                    if isinstance(api_messages, list):
                        if _sanitize_messages_surrogates(api_messages):
                            _surrogates_found = True
                    if isinstance(api_kwargs, dict):
                        if _sanitize_structure_surrogates(api_kwargs):
                            _surrogates_found = True
                    if isinstance(getattr(agent, "prefill_messages", None), list):
                        if _sanitize_messages_surrogates(agent.prefill_messages):
                            _surrogates_found = True
                    # Gate the retry on the error type, not on whether we
                    # found anything — _force_ascii_payload / the extended
                    # surrogate walker above cover all known paths, but a
                    # new transformed field could still slip through.  If
                    # the error was a surrogate encode failure, always let
                    # the retry run; the proactive sanitizer at line ~8781
                    # runs again on the next iteration.  Bounded by
                    # _unicode_sanitization_passes < 2 (outer guard).
                    if _surrogates_found or _is_surrogate_error:
                        agent._unicode_sanitization_passes += 1
                        if _surrogates_found:
                            agent._buffer_vprint(
                                "⚠️  Stripped invalid surrogate characters from messages. Retrying..."
                            )
                        else:
                            agent._buffer_vprint(
                                "⚠️  Surrogate encoding error — retrying after full-payload sanitization..."
                            )
                        continue
                    if _is_ascii_codec:
                        agent._force_ascii_payload = True
                        # ASCII codec: the system encoding can't handle
                        # non-ASCII characters at all. Sanitize all
                        # non-ASCII content from messages/tool schemas and retry.
                        # Sanitize both the canonical `messages` list and
                        # `api_messages` (the API-copy built before the retry
                        # loop, which may contain extra fields like
                        # reasoning_content that are not in `messages`).
                        _messages_sanitized = _sanitize_messages_non_ascii(messages)
                        if isinstance(api_messages, list):
                            _sanitize_messages_non_ascii(api_messages)
                        # Also sanitize the last api_kwargs if already built,
                        # so a leftover non-ASCII value in a transformed field
                        # (e.g. extra_body, reasoning_content) doesn't survive
                        # into the next attempt via _build_api_kwargs cache paths.
                        if isinstance(api_kwargs, dict):
                            _sanitize_structure_non_ascii(api_kwargs)
                        _prefill_sanitized = False
                        if isinstance(getattr(agent, "prefill_messages", None), list):
                            _prefill_sanitized = _sanitize_messages_non_ascii(agent.prefill_messages)

                        _tools_sanitized = False
                        if isinstance(getattr(agent, "tools", None), list):
                            _tools_sanitized = _sanitize_tools_non_ascii(agent.tools)

                        _system_sanitized = False
                        if isinstance(active_system_prompt, str):
                            _sanitized_system = _strip_non_ascii(active_system_prompt)
                            if _sanitized_system != active_system_prompt:
                                active_system_prompt = _sanitized_system
                                agent._cached_system_prompt = _sanitized_system
                                _system_sanitized = True
                        if isinstance(getattr(agent, "ephemeral_system_prompt", None), str):
                            _sanitized_ephemeral = _strip_non_ascii(agent.ephemeral_system_prompt)
                            if _sanitized_ephemeral != agent.ephemeral_system_prompt:
                                agent.ephemeral_system_prompt = _sanitized_ephemeral
                                _system_sanitized = True

                        _headers_sanitized = False
                        _default_headers = (
                            agent._client_kwargs.get("default_headers")
                            if isinstance(getattr(agent, "_client_kwargs", None), dict)
                            else None
                        )
                        if isinstance(_default_headers, dict):
                            _headers_sanitized = _sanitize_structure_non_ascii(_default_headers)

                        # Sanitize the API key — non-ASCII characters in
                        # credentials (e.g. ʋ instead of v from a bad
                        # copy-paste) cause httpx to fail when encoding
                        # the Authorization header as ASCII.  This is the
                        # most common cause of persistent UnicodeEncodeError
                        # that survives message/tool sanitization (#6843).
                        _credential_sanitized = False
                        _raw_key = getattr(agent, "api_key", None) or ""
                        # Entra ID bearer providers are callables — their
                        # minted JWTs are always ASCII, so no sanitization
                        # is needed (and ``_strip_non_ascii`` would crash
                        # on a callable input).
                        if _raw_key and isinstance(_raw_key, str):
                            _clean_key = _strip_non_ascii(_raw_key)
                            if _clean_key != _raw_key:
                                agent.api_key = _clean_key
                                if isinstance(getattr(agent, "_client_kwargs", None), dict):
                                    agent._client_kwargs["api_key"] = _clean_key
                                # Also update the live client — it holds its
                                # own copy of api_key which auth_headers reads
                                # dynamically on every request.
                                if getattr(agent, "client", None) is not None and hasattr(agent.client, "api_key"):
                                    agent.client.api_key = _clean_key
                                _credential_sanitized = True
                                agent._vprint(
                                    f"{agent.log_prefix}⚠️  API key contained non-ASCII characters "
                                    f"(bad copy-paste?) — stripped them. If auth fails, "
                                    f"re-copy the key from your provider's dashboard.",
                                    force=True,
                                )

                        # Always retry on ASCII codec detection —
                        # _force_ascii_payload guarantees the full
                        # api_kwargs payload is sanitized on the
                        # next iteration (line ~8475).  Even when
                        # per-component checks above find nothing
                        # (e.g. non-ASCII only in api_messages'
                        # reasoning_content), the flag catches it.
                        # Bounded by _unicode_sanitization_passes < 2.
                        agent._unicode_sanitization_passes += 1
                        _any_sanitized = (
                            _messages_sanitized
                            or _prefill_sanitized
                            or _tools_sanitized
                            or _system_sanitized
                            or _headers_sanitized
                            or _credential_sanitized
                        )
                        if _any_sanitized:
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  System encoding is ASCII — stripped non-ASCII characters from request payload. Retrying...",
                                force=True,
                            )
                        else:
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  System encoding is ASCII — enabling full-payload sanitization for retry...",
                                force=True,
                            )
                        continue

                # ── Image-rejection recovery ──────────────────────────────
                # Some providers (mlx-lm, text-only endpoints, text-only
                # fallbacks on multimodal models) reject any message that
                # contains image_url content with a 4xx error like
                # "Only 'text' content type is supported."  On first hit,
                # strip all images from the message list, mark the session
                # as vision-unsupported, and retry with text only.
                #
                # Detection is best-effort English phrase matching — a
                # locale-translated or heavily-reworded upstream error
                # will bypass this guard and fall through to the normal
                # error handler.  Expand the phrase list when new
                # provider wordings are observed in the wild.
                _err_body = ""
                try:
                    _err_body = str(getattr(api_error, "body", None) or
                                    getattr(api_error, "message", None) or
                                    str(api_error))
                except Exception:
                    pass
                _err_status = getattr(api_error, "status_code", None)
                _IMAGE_REJECTION_PHRASES = (
                    "only 'text' content type is supported",
                    "only text content type is supported",
                    "image_url is not supported",
                    "image content is not supported",
                    "multimodal is not supported",
                    "multimodal content is not supported",
                    "multimodal input is not supported",
                    "vision is not supported",
                    "vision input is not supported",
                    "does not support images",
                    "does not support image input",
                    "does not support multimodal",
                    "does not support vision",
                    "model does not support image",
                    # ChatGPT-account Codex backend
                    # (https://chatgpt.com/backend-api/codex) rejects
                    # data:image/...base64 URLs in input_image fields
                    # with HTTP 400 "Invalid 'input[N].content[K].image_url'.
                    # Expected a valid URL, but got a value with an
                    # invalid format." The OpenAI Responses API on the
                    # public endpoint accepts data URLs, but the
                    # ChatGPT-account variant does not. Without this
                    # phrase the agent cascaded into compression /
                    # context-too-large recovery instead of just
                    # stripping the images. Match is narrow on
                    # purpose — keyed on the field-path apostrophe so
                    # we don't false-trip on other URL validation
                    # errors. (issue #23570)
                    "image_url'. expected",
                    # DeepSeek's OpenAI-compatible API reports text-only
                    # request-body variants as:
                    # "unknown variant `image_url`, expected `text`".
                    "unknown variant `image_url`, expected `text`",
                    "unknown variant image_url, expected text",
                    # OpenRouter routes a request to upstream endpoints and,
                    # when none of the candidate endpoints for the model accept
                    # image input, returns HTTP 404 "No endpoints found that
                    # support image input". Without this phrase the agent never
                    # strips the images, the retry loop re-sends the same
                    # rejected request until exhaustion, and the gateway leaves
                    # every subsequent message queued behind the stuck turn —
                    # the P1 in issue #21160. The 404 passes the 4xx gate below.
                    "no endpoints found that support image input",
                )
                _err_lower = _err_body.lower()
                _looks_like_image_rejection = any(
                    p in _err_lower for p in _IMAGE_REJECTION_PHRASES
                )
                # 4xx-only gate: never interpret 5xx/timeout as "server
                # said no to images" — those are transient and must
                # route to the normal retry path.
                _status_ok = _err_status is None or (400 <= int(_err_status) < 500)
                if (
                    getattr(agent, "_vision_supported", True)
                    and _looks_like_image_rejection
                    and _status_ok
                ):
                    agent._vision_supported = False
                    _imgs_removed = _strip_images_from_messages(messages)
                    if isinstance(api_messages, list):
                        _strip_images_from_messages(api_messages)
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Server rejected image content — "
                        f"switching to text-only mode for this session"
                        + (". Stripped images from history and retrying." if _imgs_removed else "."),
                        force=True,
                    )
                    continue

                # ── Bedrock AnthropicBedrock SDK streaming failure ──
                # The Anthropic SDK's stream accumulator raises RuntimeError
                # "Unexpected event order" when Bedrock returns an error event
                # before message_start (throttling, overload, validation).
                # Fall back to the native Converse API path for the rest of
                # this session — it handles these errors gracefully.  Ref: #28156.
                if (
                    isinstance(api_error, RuntimeError)
                    and "unexpected event order" in str(api_error).lower()
                    and getattr(agent, "provider", "") == "bedrock"
                    and agent.api_mode == "anthropic_messages"
                    and not getattr(agent, "_bedrock_converse_fallback_attempted", False)
                ):
                    agent._bedrock_converse_fallback_attempted = True
                    agent.api_mode = "bedrock_converse"
                    agent._bedrock_region = getattr(agent, "_bedrock_region", None) or "us-east-1"
                    agent.client = None  # Drop the AnthropicBedrock client
                    agent._client_kwargs = {}
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  AnthropicBedrock SDK streaming failed — "
                        f"falling back to native Converse API for this session.",
                        force=True,
                    )
                    continue

                status_code = getattr(api_error, "status_code", None)
                error_context = agent._extract_api_error_context(api_error)

                # ── Classify the error for structured recovery decisions ──
                _compressor = getattr(agent, "context_compressor", None)
                _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
                classified = classify_api_error(
                    api_error,
                    provider=getattr(agent, "provider", "") or "",
                    model=getattr(agent, "model", "") or "",
                    approx_tokens=approx_tokens,
                    context_length=_ctx_len,
                    num_messages=len(api_messages) if api_messages else 0,
                )
                logger.debug(
                    "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
                    classified.reason.value, classified.status_code,
                    classified.retryable, classified.should_compress,
                    classified.should_rotate_credential, classified.should_fallback,
                )
                agent._invoke_api_request_error_hook(
                    task_id=effective_task_id,
                    turn_id=turn_id,
                    api_request_id=api_request_id,
                    api_call_count=api_call_count,
                    api_start_time=api_start_time,
                    api_kwargs=api_kwargs,
                    error_type=type(api_error).__name__,
                    error_message=str(api_error),
                    status_code=status_code,
                    retry_count=retry_count,
                    max_retries=max_retries,
                    retryable=classified.retryable,
                    reason=classified.reason.value,
                )

                if (
                    classified.reason == FailoverReason.billing
                    and _is_nous_inference_route(
                        getattr(agent, "provider", "") or "",
                        getattr(agent, "base_url", "") or "",
                    )
                    and not _retry.nous_paid_entitlement_refresh_attempted
                ):
                    _retry.nous_paid_entitlement_refresh_attempted = True
                    if _try_refresh_nous_paid_entitlement_credentials(agent):
                        agent._vprint(
                            f"{agent.log_prefix}🔐 Nous paid access verified — "
                            "refreshed runtime credentials and retrying request...",
                            force=True,
                        )
                        continue

                recovered_with_pool, _retry.has_retried_429 = agent._recover_with_credential_pool(
                    status_code=status_code,
                    has_retried_429=_retry.has_retried_429,
                    classified_reason=classified.reason,
                    error_context=error_context,
                )
                if recovered_with_pool:
                    continue

                # Image-too-large recovery: shrink oversized native image
                # parts in-place and retry once.  Triggered by Anthropic's
                # per-image 5 MB ceiling (400 with "image exceeds 5 MB
                # maximum") or any other provider that complains about
                # image size.  If shrink fails or a second attempt still
                # fails, fall through to normal error handling.
                if (
                    classified.reason == FailoverReason.image_too_large
                    and not _retry.image_shrink_retry_attempted
                ):
                    _retry.image_shrink_retry_attempted = True
                    image_max_dimension = _image_error_max_dimension(api_error) or 8000
                    if agent._try_shrink_image_parts_in_messages(
                        api_messages,
                        max_dimension=image_max_dimension,
                    ):
                        agent._vprint(
                            f"{agent.log_prefix}📐 Image(s) exceeded provider size limit — "
                            f"shrank and retrying...",
                            force=True,
                        )
                        continue
                    else:
                        logger.info(
                            "image-shrink recovery: no data-URL image parts found "
                            "or shrink didn't reduce size; surfacing original error."
                        )

                # Multimodal-tool-content recovery: providers that follow
                # the OpenAI spec strictly (tool message content must be a
                # string) reject our list-type content with a 400.  Strip
                # image parts from any list-type tool messages, mark the
                # (provider, model) as no-list-tool-content for the rest
                # of this session so future tool results preemptively
                # downgrade, and retry once.  See issue #27344.
                if (
                    classified.reason == FailoverReason.multimodal_tool_content_unsupported
                    and not _retry.multimodal_tool_content_retry_attempted
                ):
                    _retry.multimodal_tool_content_retry_attempted = True
                    if agent._try_strip_image_parts_from_tool_messages(api_messages):
                        agent._vprint(
                            f"{agent.log_prefix}📐 Provider rejected list-type tool content — "
                            f"downgraded screenshots to text and retrying...",
                            force=True,
                        )
                        continue
                    else:
                        logger.info(
                            "multimodal-tool-content recovery: no list-type tool "
                            "messages with image parts found; surfacing original error."
                        )

                # Anthropic OAuth subscription rejected the 1M-context beta
                # header ("long context beta is not yet available for this
                # subscription"). Disable the beta for the rest of this
                # session, rebuild the client, and retry once.  1M-capable
                # subscriptions never hit this branch — they accept the
                # beta and keep full 1M context.  See PR #17680 for the
                # original report (we chose reactive recovery over the
                # proposed unconditional omit so capable subscriptions
                # don't silently lose the capability).
                if (
                    classified.reason == FailoverReason.oauth_long_context_beta_forbidden
                    and agent.api_mode == "anthropic_messages"
                    and agent._is_anthropic_oauth
                    and not _retry.oauth_1m_beta_retry_attempted
                ):
                    _retry.oauth_1m_beta_retry_attempted = True
                    if not getattr(agent, "_oauth_1m_beta_disabled", False):
                        agent._oauth_1m_beta_disabled = True
                        try:
                            agent._anthropic_client.close()
                        except Exception:
                            pass
                        agent._rebuild_anthropic_client()
                        agent._vprint(
                            f"{agent.log_prefix}🔕 OAuth subscription doesn't support "
                            f"the 1M-context beta — disabled for this session and retrying...",
                            force=True,
                        )
                        continue

                if (
                    agent.api_mode == "codex_responses"
                    and agent.provider in {"openai-codex", "xai-oauth"}
                    and status_code == 401
                    and not _retry.codex_auth_retry_attempted
                ):
                    _retry.codex_auth_retry_attempted = True
                    if agent._try_refresh_codex_client_credentials(force=True):
                        _label = "xAI OAuth" if agent.provider == "xai-oauth" else "Codex"
                        agent._buffer_vprint(f"🔐 {_label} auth refreshed after 401. Retrying request...")
                        continue
                if (
                    agent.api_mode == "chat_completions"
                    and agent.provider == "vertex"
                    and status_code == 401
                    and not _retry.vertex_auth_retry_attempted
                ):
                    _retry.vertex_auth_retry_attempted = True
                    if agent._try_refresh_vertex_client_credentials():
                        agent._buffer_vprint("🔐 Vertex AI token refreshed after 401. Retrying request...")
                        continue
                if (
                    agent.api_mode in ("chat_completions", "anthropic_messages")
                    and agent.provider == "nous"
                    and status_code == 401
                    and not _retry.nous_auth_retry_attempted
                ):
                    _retry.nous_auth_retry_attempted = True
                    if agent._try_refresh_nous_client_credentials(force=True):
                        print(f"{agent.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request...")
                        continue
                    # Credential refresh didn't help — show diagnostic info.
                    # Most common causes: Portal OAuth expired/revoked,
                    # account out of credits, or agent key blocked.
                    from hermes_constants import display_hermes_home as _dhh_fn
                    _dhh = _dhh_fn()
                    _body_text = ""
                    try:
                        _body = getattr(api_error, "body", None) or getattr(api_error, "response", None)
                        if _body is not None:
                            _body_text = str(_body)[:200]
                    except Exception:
                        pass
                    print(f"{agent.log_prefix}🔐 Nous 401 — Portal authentication failed.")
                    if _body_text:
                        print(f"{agent.log_prefix}   Response: {_body_text}")
                    if not _print_nous_entitlement_guidance(agent, "Nous model access"):
                        print(f"{agent.log_prefix}   Most likely: Portal OAuth expired, account out of credits, or agent key revoked.")
                    print(f"{agent.log_prefix}   Troubleshooting:")
                    print(f"{agent.log_prefix}     • Re-authenticate: hermes auth add nous")
                    print(f"{agent.log_prefix}     • Check credits / billing: https://portal.nousresearch.com")
                    print(f"{agent.log_prefix}     • Verify stored credentials: {_dhh}/auth.json")
                    print(f"{agent.log_prefix}     • Switch providers temporarily: /model <model> --provider openrouter")
                if (
                    _is_copilot_provider(agent)
                    and status_code == 401
                    and not _retry.copilot_auth_retry_attempted
                ):
                    _retry.copilot_auth_retry_attempted = True
                    if agent._try_refresh_copilot_client_credentials():
                        agent._buffer_vprint("🔐 Copilot credentials refreshed after 401. Retrying request...")
                        continue
                if (
                    agent.api_mode == "anthropic_messages"
                    and status_code == 401
                    and hasattr(agent, '_anthropic_api_key')
                    and not _retry.anthropic_auth_retry_attempted
                ):
                    _retry.anthropic_auth_retry_attempted = True
                    from agent.anthropic_adapter import _is_oauth_token
                    from agent.azure_identity_adapter import is_token_provider
                    if agent._try_refresh_anthropic_client_credentials():
                        print(f"{agent.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
                        continue
                    # Credential refresh didn't help — show diagnostic info
                    key = agent._anthropic_api_key
                    print(f"{agent.log_prefix}🔐 Anthropic 401 — authentication failed.")
                    if is_token_provider(key):
                        # Azure Foundry Entra ID — the bearer token is
                        # minted per-request by an httpx event hook on a
                        # custom http_client passed to the SDK. The 401
                        # means Azure rejected the JWT (RBAC role missing,
                        # az login expired, IMDS unreachable, etc.).
                        print(f"{agent.log_prefix}   Auth method: Microsoft Entra ID (httpx event hook)")
                        print(f"{agent.log_prefix}   Run `hermes doctor` for credential-chain diagnostics, or")
                        print(f"{agent.log_prefix}   `az login` if your developer session expired.")
                    else:
                        auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
                        print(f"{agent.log_prefix}   Auth method: {auth_method}")
                        print(f"{agent.log_prefix}   Token prefix: {key[:12]}..." if isinstance(key, str) and len(key) > 12 else f"{agent.log_prefix}   Token: (empty or short)")
                    print(f"{agent.log_prefix}   Troubleshooting:")
                    from hermes_constants import display_hermes_home as _dhh_fn
                    _dhh = _dhh_fn()
                    print(f"{agent.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
                    print(f"{agent.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
                    print(f"{agent.log_prefix}     • For API keys: verify at https://platform.claude.com/settings/keys")
                    print(f"{agent.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
                    print(f"{agent.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"")
                    print(f"{agent.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"")

                # Thinking block signature recovery.
                #
                # Anthropic signs thinking blocks against the full turn
                # content. Any upstream mutation (context compression,
                # session truncation, message merging) invalidates the
                # signature and the API replies HTTP 400 ("invalid
                # signature" or "cannot be modified"). Recovery strips
                # ``reasoning_details`` so the retry sends no thinking
                # blocks at all. One-shot per outer loop.
                #
                # The strip targets ``api_messages``, which is the
                # API-call-time list that ``_build_api_kwargs`` consumes
                # on every retry. ``api_messages`` was populated once at
                # the start of the turn from shallow copies of
                # ``messages``, so mutating it does not touch the
                # canonical store. The previous implementation popped
                # ``reasoning_details`` from ``messages`` instead, which
                # had two problems: ``api_messages`` carried its own
                # reference to the field through the shallow copy, so the
                # retry's wire payload still included thinking blocks and
                # the recovery never reached the API; and the mutation
                # persisted into ``state.db`` through any subsequent
                # ``_persist_session`` call, permanently corrupting the
                # conversation. Future turns would replay the stripped
                # state, hit the same 400, and the agent would terminate
                # with ``max_retries_exhausted``, often spawning
                # cascading compaction-ended sessions chained off the
                # corrupted parent.
                if (
                    classified.reason == FailoverReason.thinking_signature
                    and not _retry.thinking_sig_retry_attempted
                ):
                    _retry.thinking_sig_retry_attempted = True
                    _api_stripped = 0
                    for _m in api_messages:
                        if isinstance(_m, dict) and "reasoning_details" in _m:
                            _m.pop("reasoning_details", None)
                            _api_stripped += 1
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Thinking block signature invalid, "
                        f"stripped reasoning_details from api_messages for retry...",
                        force=True,
                    )
                    logger.warning(
                        "%sThinking block signature recovery: stripped "
                        "reasoning_details from %d api_messages "
                        "(canonical messages unchanged)",
                        agent.log_prefix, _api_stripped,
                    )
                    continue

                # ── Invalid encrypted reasoning replay recovery ───────
                # OpenAI Responses API surfaces (and some compatible relays)
                # return HTTP 400 ``invalid_encrypted_content`` when a
                # replayed ``codex_reasoning_items`` blob from a previous
                # turn fails verification (provider rotated the encryption
                # key, the route doesn't actually persist reasoning state,
                # etc.).  Recovery: disable replay for the rest of the
                # session, strip cached items from history, retry once.
                # One-shot — if a second 400 fires we fall through to the
                # normal retry/backoff path.  Only fires for codex_responses
                # mode with at least one assistant message that has cached
                # ``codex_reasoning_items``; without replay state, the
                # error is unrelated to our cache so the normal retry path
                # handles it (the provider is rejecting something else).
                if (
                    classified.reason == FailoverReason.invalid_encrypted_content
                    and not _retry.invalid_encrypted_content_retry_attempted
                    and agent.api_mode == "codex_responses"
                    and bool(getattr(agent, "_codex_reasoning_replay_enabled", True))
                    and any(
                        isinstance(_m, dict)
                        and _m.get("role") == "assistant"
                        and isinstance(_m.get("codex_reasoning_items"), list)
                        and _m.get("codex_reasoning_items")
                        for _m in messages
                    )
                ):
                    _retry.invalid_encrypted_content_retry_attempted = True
                    replay_stats = agent._disable_codex_reasoning_replay(messages)
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Encrypted reasoning replay was rejected by the provider — "
                        f"disabled replay and stripped {replay_stats['items']} item(s) from "
                        f"{replay_stats['messages']} message(s), retrying...",
                        force=True,
                    )
                    logger.warning(
                        "%sInvalid encrypted reasoning recovery: disabled replay and stripped %d items from %d messages",
                        agent.log_prefix,
                        replay_stats["items"],
                        replay_stats["messages"],
                    )
                    continue

                # ── llama.cpp grammar-parse recovery ──────────────────
                # llama.cpp's ``json-schema-to-grammar`` converter rejects
                # regex escape classes (``\d``, ``\w``, ``\s``) and most
                # ``format`` values in tool schemas.  MCP servers emit
                # these routinely for date/phone/email params.  Recovery:
                # strip ``pattern``/``format`` from ``agent.tools`` and
                # retry once.  We keep the keywords by default so cloud
                # providers get the full prompting hints; this branch
                # fires only for users on llama.cpp's OAI server.
                if (
                    classified.reason == FailoverReason.llama_cpp_grammar_pattern
                    and not _retry.llama_cpp_grammar_retry_attempted
                ):
                    _retry.llama_cpp_grammar_retry_attempted = True
                    try:
                        from tools.schema_sanitizer import strip_pattern_and_format
                        _, _stripped = strip_pattern_and_format(agent.tools)
                    except Exception as _strip_exc:  # pragma: no cover — defensive
                        logger.warning(
                            "%sllama.cpp grammar recovery: strip helper failed: %s",
                            agent.log_prefix, _strip_exc,
                        )
                        _stripped = 0
                    if _stripped:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  llama.cpp rejected tool schema grammar — "
                            f"stripped {_stripped} pattern/format keyword(s), retrying...",
                            force=True,
                        )
                        logger.warning(
                            "%sllama.cpp grammar recovery: stripped %d "
                            "pattern/format keyword(s) from tool schemas",
                            agent.log_prefix, _stripped,
                        )
                        continue
                    # No keywords found to strip — fall through to normal
                    # retry path rather than loop forever on the same error.
                    logger.warning(
                        "%sllama.cpp grammar error but no pattern/format "
                        "keywords to strip — falling through to normal retry",
                        agent.log_prefix,
                    )

                retry_count += 1
                elapsed_time = time.time() - api_start_time
                agent._touch_activity(
                    f"API error recovery (attempt {retry_count}/{max_retries})"
                )
                
                error_type = type(api_error).__name__
                error_msg = str(api_error).lower()
                _error_summary = agent._summarize_api_error(api_error)
                logger.warning(
                    "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
                    retry_count,
                    max_retries,
                    error_type,
                    agent._client_log_context(),
                    _error_summary,
                )

                _provider = getattr(agent, "provider", "unknown")
                _base = getattr(agent, "base_url", "unknown")
                _model = getattr(agent, "model", "unknown")
                _status_code_str = f" [HTTP {status_code}]" if status_code else ""
                agent._buffer_vprint(f"⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}")
                agent._buffer_vprint(f"   🔌 Provider: {_provider}  Model: {_model}")
                agent._buffer_vprint(f"   🌐 Endpoint: {_base}")
                agent._buffer_vprint(f"   📝 Error: {_error_summary}")
                if status_code and status_code < 500:
                    _err_body = getattr(api_error, "body", None)
                    _err_body_str = str(_err_body)[:300] if _err_body else None
                    if _err_body_str:
                        agent._buffer_vprint(f"   📋 Details: {_err_body_str}")
                agent._buffer_vprint(f"   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

                # Actionable hint for OpenRouter "no tool endpoints" error.
                # Buffered like the rest of the retry trace — surfaced only
                # if every retry+fallback exhausts.  Avoids spamming users
                # who recover automatically via fallback.
                if (
                    agent._is_openrouter_url()
                    and "support tool use" in error_msg
                ):
                    agent._buffer_vprint(
                        f"   💡 No OpenRouter providers for {_model} support tool calling with your current settings."
                    )
                    if agent.providers_allowed:
                        agent._buffer_vprint(
                            "      Your provider_routing.only restriction is filtering out tool-capable providers."
                        )
                        agent._buffer_vprint(
                            "      Try removing the restriction or adding providers that support tools for this model."
                        )
                    agent._buffer_vprint(
                        f"      Check which providers support tools: https://openrouter.ai/models/{_model}"
                    )

                # Actionable hint for a bare 404 on a provider whose catalogue
                # uses ``vendor/model`` ids.  A model id that lost its prefix
                # (e.g. ``nemotron-…`` instead of ``nvidia/nemotron-…``) gets
                # a content-free "404 page not found" from the provider that
                # never names the model, so it reads like an outage or an auth
                # failure.  Name the real cause and the exact id to use (#78796).
                if getattr(api_error, "status_code", None) == 404:
                    try:
                        from hermes_cli.model_normalize import suggest_prefixed_model_id

                        _suggestion = suggest_prefixed_model_id(_provider, _model)
                    except Exception:
                        _suggestion = None
                    if _suggestion:
                        agent._buffer_vprint(
                            f"   💡 Model '{_model}' is not a valid id for provider {_provider} — "
                            f"it is missing its vendor prefix."
                        )
                        agent._buffer_vprint(
                            f"      Did you mean '{_suggestion}'?  Re-pick it with `hermes model`."
                        )

                # Check for interrupt before deciding to retry
                if agent._interrupt_requested:
                    # Preserve a pending redirect (mid-stream correction): the
                    # user is steering, not stopping. Rebuild the turn from the
                    # correction instead of aborting with a dead-end interrupt.
                    if agent.clear_interrupt(preserve_redirect=True):
                        _retry.restart_with_redirected_messages = True
                        break
                    agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                    _interrupt_text = f"Operation interrupted: handling API error ({error_type}: {agent._clean_error_message(str(api_error))})."
                    close_interrupted_tool_sequence(messages, _interrupt_text)
                    agent._persist_session(messages, conversation_history)
                    agent.clear_interrupt()
                    return {
                        "final_response": _interrupt_text,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "interrupted": True,
                    }
                
                # Check for 413 payload-too-large BEFORE generic 4xx handler.
                # A 413 is a payload-size error — the correct response is to
                # compress history and retry, not abort immediately.
                status_code = getattr(api_error, "status_code", None)

                # ── Respect disabled auto-compaction on overflow ──────
                # Ported from anomalyco/opencode#30749.  When the user has
                # turned auto-compaction off (``compression.enabled: false``),
                # NO automatic compaction trigger may fire — including the
                # provider/request-size overflow recovery paths below
                # (long-context-tier 429, 413 payload-too-large, and
                # context-overflow).  Without this guard the proactive
                # threshold path correctly honours the setting (see the
                # preflight check and the post-response ``should_compress``
                # gate) but a provider overflow error would still silently
                # compress + rotate the session, bypassing the user's
                # explicit choice.  Surface a terminal error instead so the
                # user can compact manually (``/compress``), start fresh
                # (``/new``), switch to a larger-context model, or reduce
                # attachments.  Forced compaction via ``/compress``
                # (``force=True``) is unaffected — it never reaches this loop.
                #
                # Output-cap errors (max_tokens too large) are NOT input
                # overflow — the recovery is a max_tokens-only retry that
                # does not require compression.  Exempt them from this guard
                # so the retry still fires even when compression is disabled.
                _overflow_reasons = {
                    FailoverReason.long_context_tier,
                    FailoverReason.payload_too_large,
                    FailoverReason.context_overflow,
                }
                _is_output_cap_error = (
                    is_output_cap_error(error_msg)
                    or parse_available_output_tokens_from_error(error_msg) is not None
                )
                if (
                    classified.reason in _overflow_reasons
                    and not getattr(agent, "compression_enabled", True)
                    and not _is_output_cap_error
                ):
                    agent._flush_status_buffer()
                    agent._vprint(
                        f"{agent.log_prefix}❌ Context overflow, but auto-compaction is disabled "
                        f"(compression.enabled: false).",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}   💡 Run /compress to compact manually, /new to start fresh, "
                        f"switch to a larger-context model, or reduce attachments.",
                        force=True,
                    )
                    logger.error(
                        f"{agent.log_prefix}Context overflow ({classified.reason.value}) with "
                        f"auto-compaction disabled — not compressing."
                    )
                    agent._persist_session(messages, conversation_history)
                    _final_response = (
                        "Context overflow and auto-compaction is disabled "
                        "(compression.enabled: false). Run /compress to compact manually, "
                        "/new to start fresh, or switch to a larger-context model."
                    )
                    return {
                        "final_response": _final_response,
                        "messages": messages,
                        "completed": False,
                        "api_calls": api_call_count,
                        "error": _final_response,
                        "partial": True,
                        "failed": True,
                        "compaction_disabled": True,
                    }

                # ── Anthropic Sonnet long-context tier gate ───────────
                # Anthropic returns HTTP 429 "Extra usage is required for
                # long context requests" when a Claude Max (or similar)
                # subscription doesn't include the 1M-context tier.  This
                # is NOT a transient rate limit — retrying or switching
                # credentials won't help.  Reduce context to 200k (the
                # standard tier) and compress.
                if classified.reason == FailoverReason.long_context_tier:
                    _reduced_ctx = 200000
                    compressor = agent.context_compressor
                    old_ctx = compressor.context_length
                    if old_ctx > _reduced_ctx:
                        compressor.update_model(
                            model=agent.model,
                            context_length=_reduced_ctx,
                            base_url=agent.base_url,
                            api_key=getattr(agent, "api_key", ""),
                            provider=agent.provider,
                            api_mode=agent.api_mode,
                        )
                        # Context probing flags — only set on built-in
                        # compressor (plugin engines manage their own).
                        if hasattr(compressor, "_context_probed"):
                            compressor._context_probed = True
                            # Don't persist — this is a subscription-tier
                            # limitation, not a model capability.  If the
                            # user later enables extra usage the 1M limit
                            # should come back automatically.
                            compressor._context_probe_persistable = False
                        agent._buffer_vprint(
                            f"⚠️  Anthropic long-context tier "
                            f"requires extra usage — reducing context: "
                            f"{old_ctx:,} → {_reduced_ctx:,} tokens"
                        )

                    compression_attempts += 1
                    if compression_attempts <= max_compression_attempts:
                        original_len = len(messages)
                        # Option A (LCM issue 441): overhead-aware request size so recovery arms on
                        # the true request (msgs + tools + system), not the tool-blind message count.
                        messages, active_system_prompt = agent._compress_context(
                            messages, system_message,
                            approx_tokens=estimate_request_tokens_rough(api_messages, tools=agent.tools or None),
                            task_id=effective_task_id,
                        )
                        conversation_history = conversation_history_after_compression(
                            agent, messages, conversation_history
                        )
                        if len(messages) < original_len or old_ctx > _reduced_ctx:
                            agent._buffer_status(
                                COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE.format(
                                    new_ctx=_reduced_ctx, old_ctx=old_ctx
                                )
                            )
                            time.sleep(2)
                            _retry.restart_with_compressed_messages = True
                            break
                    # Fall through to normal error handling if compression
                    # is exhausted or didn't help.

                # Eager fallback for rate-limit errors (429 or quota exhaustion)
                # and transport errors (connection failure / timeout / provider
                # overloaded).  Rate limits and billing: switch immediately —
                # the primary provider won't recover within the retry window.
                # Transport errors: allow 1 retry first (transient hiccups
                # recover), then fall back if the provider is truly unreachable.
                is_rate_limited = classified.reason in {
                    FailoverReason.rate_limit,
                    FailoverReason.billing,
                    FailoverReason.upstream_rate_limit,
                }
                _is_transport_failure = classified.reason in {
                    FailoverReason.timeout,
                    FailoverReason.overloaded,
                }
                # Z.AI Coding Plan GLM-5.2 overload 429s classify as
                # `overloaded` (to spare the credential pool), but `overloaded`
                # is excluded from `is_rate_limited` — the gate for the adaptive
                # Z.AI backoff below. Detect the overload directly so its
                # long-backoff schedule runs, and raise the retry ceiling so the
                # long tier (30/60/90/120s) is reachable. See
                # zai_coding_overload_retry_ceiling() for the ceiling rationale.
                _is_zai_coding_overload = is_zai_coding_overload_error(
                    base_url=str(_base), model=_model, error=api_error
                )
                if _is_zai_coding_overload:
                    max_retries = max(max_retries, zai_coding_overload_retry_ceiling())
                _should_fallback = (
                    is_rate_limited
                    or (_is_transport_failure and retry_count >= 2)
                )
                if _should_fallback and agent._fallback_index < len(agent._fallback_chain):
                    # Don't eagerly fallback if credential pool rotation may
                    # still recover.  See _pool_may_recover_from_rate_limit
                    # for the single-credential-pool exception.  Fixes #11314.
                    #
                    # Exception: an upstream-aggregator 429 — the credential
                    # pool can't help when the *upstream* model (DeepSeek,
                    # etc.) is throttling OpenRouter, so always fall back to a
                    # different model regardless of pool state.
                    _is_upstream = classified.reason == FailoverReason.upstream_rate_limit
                    pool_may_recover = (
                        False if _is_upstream
                        else _ra()._pool_may_recover_from_rate_limit(
                            agent._credential_pool,
                        )
                    )
                    if not pool_may_recover:
                        if _is_upstream:
                            _upstream_name = (classified.error_context or {}).get(
                                "upstream_provider", "aggregator"
                            )
                            agent._buffer_status(
                                f"⚠️ Upstream {_upstream_name} rate-limited — "
                                "switching to fallback model..."
                            )
                        elif classified.reason == FailoverReason.billing:
                            agent._buffer_status(
                                "⚠️ Billing or credits exhausted — switching to fallback provider..."
                            )
                        elif _is_transport_failure:
                            agent._buffer_status(
                                "⚠️ Provider unreachable — switching to fallback provider..."
                            )
                        else:
                            agent._buffer_status("⚠️ Rate limited — switching to fallback provider...")
                        if agent._try_activate_fallback(reason=classified.reason):
                            active_system_prompt = _sync_failover_system_message(
                                agent, api_messages, active_system_prompt)
                            retry_count = 0
                            compression_attempts = 0
                            _retry.primary_recovery_attempted = False
                            continue

                # ── Auth-failure provider failover ───────────────────────
                # A 401/403 that survives the per-provider credential-refresh
                # attempt above (each guarded by its own
                # ``*_auth_retry_attempted`` flag) means the active provider's
                # credential or endpoint is broken in a way refreshing can't
                # fix (revoked OAuth, blocked/expired key, an account pinned to
                # a dead/staging endpoint). Previously the loop only printed
                # "switch providers manually" advice and fell through, so a
                # user with a configured fallback chain kept thrashing on the
                # same dead credential every turn instead of failing over.
                # Escalate to the fallback chain here, mirroring the rate-
                # limit/billing failover above. When no fallback is configured
                # (or the chain is exhausted), _try_activate_fallback returns
                # False and we fall through to the existing terminal handling
                # + provider-specific troubleshooting guidance unchanged.
                if (
                    classified.is_auth
                    and not _retry.auth_failover_attempted
                    and agent._fallback_index < len(agent._fallback_chain)
                ):
                    _retry.auth_failover_attempted = True
                    agent._buffer_status(
                        "🔐 Authentication failed and could not be refreshed — "
                        "switching to fallback provider..."
                    )
                    if agent._try_activate_fallback(reason=classified.reason):
                        active_system_prompt = _sync_failover_system_message(
                            agent, api_messages, active_system_prompt)
                        retry_count = 0
                        compression_attempts = 0
                        _retry.primary_recovery_attempted = False
                        continue

                # ── Nous Portal: record rate limit & skip retries ─────
                # When Nous returns a 429 that is a genuine account-
                # level rate limit, record the reset time to a shared
                # file so ALL sessions (cron, gateway, auxiliary) know
                # not to pile on, then skip further retries -- each
                # one burns another RPH request and deepens the hole.
                # The retry loop's top-of-iteration guard will catch
                # this on the next pass and try fallback or bail.
                #
                # IMPORTANT: Nous Portal multiplexes multiple upstream
                # providers (DeepSeek, Kimi, MiMo, Hermes).  A 429 can
                # also mean an UPSTREAM provider is out of capacity
                # for one specific model -- transient, clears in
                # seconds, nothing to do with the caller's quota.
                # Tripping the cross-session breaker on that would
                # block every Nous model for minutes.  We use
                # ``is_genuine_nous_rate_limit`` to tell the two
                # apart via the 429's own x-ratelimit-* headers and
                # the last-known-good state captured on the previous
                # successful response.
                if (
                    is_rate_limited
                    and agent.provider == "nous"
                    and classified.reason == FailoverReason.rate_limit
                    and not recovered_with_pool
                ):
                    _genuine_nous_rate_limit = False
                    try:
                        from agent.nous_rate_guard import (
                            is_genuine_nous_rate_limit,
                            record_nous_rate_limit,
                        )
                        _err_resp = getattr(api_error, "response", None)
                        _err_hdrs = (
                            getattr(_err_resp, "headers", None)
                            if _err_resp else None
                        )
                        _genuine_nous_rate_limit = is_genuine_nous_rate_limit(
                            headers=_err_hdrs,
                            last_known_state=agent._rate_limit_state,
                        )
                        if _genuine_nous_rate_limit:
                            record_nous_rate_limit(
                                headers=_err_hdrs,
                                error_context=error_context,
                            )
                        else:
                            logger.info(
                                "Nous 429 looks like upstream capacity "
                                "(no exhausted bucket in headers or "
                                "last-known state) -- not tripping "
                                "cross-session breaker."
                            )
                    except Exception:
                        pass
                    if _genuine_nous_rate_limit:
                        # Re-enter the loop exactly once so the
                        # top-of-loop Nous guard handles fallback or
                        # bails cleanly. (Setting retry_count to
                        # max_retries would make the while condition
                        # false immediately and the guard would never
                        # run -- no fallback, generic exhaustion error.)
                        retry_count = max(0, max_retries - 1)
                        continue
                    # Upstream capacity 429: fall through to normal
                    # retry logic.  A different model (or the same
                    # model a moment later) will typically succeed.

                is_payload_too_large = (
                    classified.reason == FailoverReason.payload_too_large
                )

                # Actionable hint for GitHub Models (Azure) 413 errors.
                # The free tier enforces a hard 8K token cap per request,
                # which Hermes' system prompt + tool schemas alone exceed.
                # Compression can't help — the floor is the system prompt
                # itself, not the conversation — so surface a clear "not
                # compatible" message instead of looping into three futile
                # compression attempts.
                if (
                    status_code == 413
                    and isinstance(agent.base_url, str)
                    and "models.inference.ai.azure.com" in agent.base_url
                ):
                    agent._vprint(
                        f"{agent.log_prefix}   💡 GitHub Models free tier (models.inference.ai.azure.com) caps every",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      request at ~8K tokens. Hermes' system prompt + tool schemas baseline",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      exceeds that floor, so this endpoint cannot run an agentic loop.",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      Use the `copilot` provider with a Copilot subscription token (`hermes",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      setup` → GitHub Copilot), or pick any other provider.",
                        force=True,
                    )

                if is_payload_too_large:
                    compression_attempts += 1
                    if compression_attempts > max_compression_attempts:
                        # Terminal — surface the buffered retry trace.
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logger.error("%s413 compression failed after %d attempts.", agent.log_prefix, max_compression_attempts)
                        agent._persist_session(messages, conversation_history)
                        _final_response = f"Request payload too large: max compression attempts ({max_compression_attempts}) reached."
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": _final_response,
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }
                    agent._buffer_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

                    original_len = len(messages)
                    original_tokens = estimate_messages_tokens_rough(messages)
                    _overflow_input = messages
                    # Option A (LCM issue 441): overhead-aware request size so recovery arms on the
                    # true request (msgs + tools + system), not the tool-blind message count.
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message,
                        approx_tokens=estimate_request_tokens_rough(api_messages, tools=agent.tools or None),
                        task_id=effective_task_id,
                    )
                    if messages is _overflow_input and compression_skipped_due_to_lock(agent):
                        # #69870 lock-skip: the provider proved the request
                        # does not fit, but this compression pass no-oped only
                        # because another path holds the session's compression
                        # lock. Temporary defer, not exhaustion — refund the
                        # attempt and end the turn softly so the gateway does
                        # NOT auto-reset the session (#9893/#35809).
                        compression_attempts -= 1
                        agent._persist_session(messages, conversation_history)
                        return _compression_deferred_result(
                            agent, messages, api_call_count
                        )
                    conversation_history = conversation_history_after_compression(
                        agent, messages, conversation_history
                    )

                    # Re-estimate tokens after compression.  Same-message-count
                    # compression (tool-result pruning, in-place summarization)
                    # can materially reduce request size without reducing the
                    # message array.  (#39550)
                    new_tokens = estimate_messages_tokens_rough(messages)
                    approx_tokens = new_tokens  # update for downstream logging

                    if len(messages) < original_len or (new_tokens > 0 and new_tokens < original_tokens * 0.95):
                        if len(messages) < original_len:
                            agent._buffer_status(COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=original_len, after=len(messages)))
                        else:
                            agent._buffer_status(COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE.format(before=original_tokens, after=new_tokens))
                        time.sleep(2)  # Brief pause between compression retries
                        _retry.restart_with_compressed_messages = True
                        break
                    else:
                        if agent._try_strip_image_parts_from_tool_messages(
                            api_messages,
                            remember_model=False,
                        ):
                            agent._buffer_status(
                                "📐 Compression could not reduce the request further — "
                                "removed retained vision payloads and retrying..."
                            )
                            continue

                        # Terminal — surface buffered context so the user
                        # sees what compression attempts were made.
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logger.error("%s413 payload too large. Cannot compress further.", agent.log_prefix)
                        agent._persist_session(messages, conversation_history)
                        _final_response = "Request payload too large (413). Cannot compress further."
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": _final_response,
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }

                # Check for context-length errors BEFORE generic 4xx handler.
                # The classifier detects context overflow from: explicit error
                # messages, generic 400 + large session heuristic (#1630), and
                # server disconnect + large session pattern (#2153).
                is_context_length_error = (
                    classified.reason == FailoverReason.context_overflow
                )

                if is_context_length_error:
                    compressor = agent.context_compressor
                    old_ctx = compressor.context_length

                    # ── Distinguish two very different errors ───────────
                    # 1. "Prompt too long": the INPUT exceeds the context window.
                    #    Fix: reduce context_length + compress history.
                    # 2. "max_tokens too large": input is fine, but
                    #    input_tokens + requested max_tokens > context_window.
                    #    Fix: reduce max_tokens (the OUTPUT cap) for this call.
                    #    Do NOT shrink context_length — the window is unchanged.
                    #
                    # Note: max_tokens = output token cap (one response).
                    #       context_length = total window (input + output combined).
                    available_out = parse_available_output_tokens_from_error(error_msg)
                    if available_out is not None:
                        # This is an output-cap error, not input overflow.
                        # The provider's available_tokens is the authoritative
                        # cap for the failed request, so keep it as an upper
                        # bound.  Also estimate the current API request shape
                        # (system prompt, injected context, tool schemas) because
                        # Hermes may add API-only content not present in persisted
                        # messages.  Use the smaller budget and apply a small
                        # safety margin.  Do not alter context_length.
                        request_input_estimate = estimate_request_tokens_rough(
                            api_messages, tools=agent.tools or None,
                        )
                        local_available_out = old_ctx - request_input_estimate
                        if local_available_out > 0:
                            safe_out = max(1, min(available_out, local_available_out) - 64)
                        else:
                            # The rough local estimate can overshoot the real
                            # request size.  Fall back to the provider-reported
                            # budget, which is authoritative for the failed
                            # request.
                            safe_out = max(1, available_out - 64)
                        agent._ephemeral_max_output_tokens = safe_out
                        agent._buffer_vprint(
                            f"⚠️  Output cap too large for current prompt — "
                            f"retrying with max_tokens={safe_out:,} "
                            f"(provider_available={available_out:,}, "
                            f"estimated_request_tokens={request_input_estimate:,}; "
                            f"context_length unchanged at {old_ctx:,})"
                        )
                        # Still count against compression_attempts so we don't
                        # loop forever if the error keeps recurring.
                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            agent._flush_status_buffer()
                            agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                            agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                            logger.error("%sContext compression failed after %d attempts.", agent.log_prefix, max_compression_attempts)
                            agent._persist_session(messages, conversation_history)
                            _final_response = f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached."
                            return {
                                "final_response": _final_response,
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": _final_response,
                                "partial": True,
                                "failed": True,
                                "compression_exhausted": True,
                            }
                        # Also compress the message history so the output-cap
                        # retry does not just spin on max_tokens alone.  The
                        # compressor drops the middle window, freeing enough
                        # tokens for the total to fit inside context_length.
                        # (#55546)
                        try:
                            original_len = len(messages)
                            original_tokens = estimate_messages_tokens_rough(messages)
                            _overflow_input = messages
                            messages, active_system_prompt = agent._compress_context(
                                messages, system_message,
                                approx_tokens=request_input_estimate,
                                task_id=effective_task_id,
                            )
                            if messages is _overflow_input and compression_skipped_due_to_lock(agent):
                                compression_attempts -= 1
                                agent._persist_session(messages, conversation_history)
                                return _compression_deferred_result(
                                    agent, messages, api_call_count
                                )
                            conversation_history = conversation_history_after_compression(
                                agent, messages, conversation_history
                            )
                            new_tokens = estimate_messages_tokens_rough(messages)
                            if len(messages) < original_len:
                                agent._buffer_status(COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=original_len, after=len(messages)))
                            elif new_tokens > 0 and new_tokens < original_tokens * 0.95:
                                agent._buffer_status(COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE.format(before=original_tokens, after=new_tokens))
                        except Exception:
                            # Compression must never turn an output-cap error
                            # fatal — fall through and retry on max_tokens alone.
                            logger.warning(
                                "%sOutput-cap compression hit an error; retrying on max_tokens only.",
                                agent.log_prefix,
                            )
                        _retry.restart_with_compressed_messages = True
                        break

                    # The error is output-cap-shaped (about max_tokens being
                    # too large) but the provider's wording didn't let us parse
                    # the available output budget.  Compression CANNOT help here
                    # — the input already fits; the call fails deterministically
                    # on the oversized max_tokens.  Routing it into compression
                    # re-sends the same max_tokens, gets the identical 400, and
                    # death-loops until "cannot compress further" (#55546).
                    # Fail fast with an actionable message instead of looping.
                    if is_output_cap_error(error_msg):
                        agent._flush_status_buffer()
                        agent._vprint(
                            f"{agent.log_prefix}❌ The provider rejected the request because "
                            f"max_tokens exceeds its output cap for this model.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}   💡 Lower model.max_tokens in your config.yaml to "
                            f"at or below the model's max-output limit. "
                            f"(This is an output-cap error, not a context overflow — "
                            f"compression cannot fix it.)",
                            force=True,
                        )
                        logger.error(
                            f"{agent.log_prefix}Output-cap error not routed into compression "
                            f"(max_tokens over provider cap): {error_msg[:200]}"
                        )
                        agent._persist_session(messages, conversation_history)
                        _final_response = (
                            "max_tokens exceeds the provider's output cap for this model. "
                            "Lower model.max_tokens in config.yaml."
                        )
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": _final_response,
                            "partial": True,
                            "failed": True,
                        }

                    # Error is about the INPUT being too large.  Only reduce
                    # context_length when the provider explicitly reports the
                    # real lower limit.  If the provider only says "input
                    # exceeds the context window", keep the configured window
                    # and try compression; guessing probe tiers can incorrectly
                    # turn a user-configured 1M window into 256K/128K/64K.
                    new_ctx = get_context_length_from_provider_error(error_msg, old_ctx)
                    _provider_lower = (getattr(agent, "provider", "") or "").lower()
                    _base_lower = (getattr(agent, "base_url", "") or "").rstrip("/").lower()
                    is_minimax_provider = (
                        _provider_lower in {"minimax", "minimax-cn"}
                        or _base_lower.startswith((
                            "https://api.minimax.io/anthropic",
                            "https://api.minimaxi.com/anthropic",
                        ))
                    )
                    minimax_delta_only_overflow = (
                        is_minimax_provider
                        and new_ctx is None
                        and "context window exceeds limit (" in error_msg
                    )

                    if new_ctx is not None:
                        agent._buffer_vprint(f"Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})")
                        compressor.update_model(
                            model=agent.model,
                            context_length=new_ctx,
                            base_url=agent.base_url,
                            api_key=getattr(agent, "api_key", ""),
                            provider=agent.provider,
                            api_mode=agent.api_mode,
                        )
                        # Persist an explicit provider-reported limit before
                        # compression/retry. The next request can be rate
                        # limited, omit usage, or the process can restart; none
                        # of those should discard metadata the provider already
                        # confirmed. Keep the probe flags as a best-effort
                        # post-success retry if this write cannot complete.
                        save_context_length(agent.model, agent.base_url, new_ctx)
                        # Context probing flags — only set on built-in
                        # compressor (plugin engines manage their own).  This
                        # value came from the provider, so it is safe to cache.
                        if hasattr(compressor, "_context_probed"):
                            compressor._context_probed = True
                            compressor._context_probe_persistable = True
                        agent._buffer_vprint(f"⚠️  Context length exceeded — using provider limit: {old_ctx:,} → {new_ctx:,} tokens")
                    elif minimax_delta_only_overflow:
                        agent._buffer_vprint(
                            f"Provider reported overflow amount only; "
                            f"keeping context_length at {old_ctx:,} tokens and compressing."
                        )
                    else:
                        agent._buffer_vprint(
                            f"⚠️  Context length exceeded, but provider did not report a max context length; "
                            f"keeping context_length at {old_ctx:,} tokens and compressing."
                        )

                    compression_attempts += 1
                    if compression_attempts > max_compression_attempts:
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logger.error("%sContext compression failed after %d attempts.", agent.log_prefix, max_compression_attempts)
                        agent._persist_session(messages, conversation_history)
                        _final_response = f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached."
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": _final_response,
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }
                    agent._buffer_status(COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE.format(tokens=approx_tokens, attempt=compression_attempts, cap=max_compression_attempts))

                    original_len = len(messages)
                    original_tokens = estimate_messages_tokens_rough(messages)
                    _overflow_input = messages
                    # Option A (LCM issue 441): pass the OVERHEAD-AWARE request size (msgs + tool
                    # schemas + system), not the tool-blind message count, so LCM forced-overflow
                    # recovery arms on the TRUE request that overflowed. See hermes-lcm engine
                    # _should_force_overflow_recovery. (approx_tokens stays for the status display.)
                    messages, active_system_prompt = agent._compress_context(
                        messages, system_message,
                        approx_tokens=estimate_request_tokens_rough(api_messages, tools=agent.tools or None),
                        task_id=effective_task_id,
                    )
                    if messages is _overflow_input and compression_skipped_due_to_lock(agent):
                        # #69870 lock-skip: the provider proved the request
                        # does not fit, but this compression pass no-oped only
                        # because another path holds the session's compression
                        # lock. Temporary defer, not exhaustion — refund the
                        # attempt and end the turn softly so the gateway does
                        # NOT auto-reset the session (#9893/#35809).
                        compression_attempts -= 1
                        agent._persist_session(messages, conversation_history)
                        return _compression_deferred_result(
                            agent, messages, api_call_count
                        )
                    conversation_history = conversation_history_after_compression(
                        agent, messages, conversation_history
                    )

                    # Re-estimate tokens after compression.  Same-message-count
                    # compression (tool-result pruning, in-place summarization)
                    # can materially reduce request size without reducing the
                    # message array.  (#39550)
                    new_tokens = estimate_messages_tokens_rough(messages)
                    approx_tokens = new_tokens  # update for downstream logging

                    if len(messages) < original_len or (new_tokens > 0 and new_tokens < original_tokens * 0.95) or (new_ctx and new_ctx < old_ctx):
                        if len(messages) < original_len:
                            agent._buffer_status(COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=original_len, after=len(messages)))
                        elif new_tokens > 0 and new_tokens < original_tokens * 0.95:
                            agent._buffer_status(COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE.format(before=original_tokens, after=new_tokens))
                        time.sleep(2)  # Brief pause between compression retries
                        _retry.restart_with_compressed_messages = True
                        break
                    else:
                        # Can't compress further and already at minimum tier
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
                        logger.error("%sContext length exceeded: %s tokens. Cannot compress further.", agent.log_prefix, f"{new_tokens:,}")
                        agent._persist_session(messages, conversation_history)
                        _final_response = f"Context length exceeded ({new_tokens:,} tokens). Cannot compress further."
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": _final_response,
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        }

                # Check for non-retryable client errors.  The classifier
                # already accounts for 413, 429, 529 (transient), context
                # overflow, and generic-400 heuristics.  Local validation
                # errors (ValueError, TypeError) are programming bugs.
                # Exclude UnicodeEncodeError — it's a ValueError subclass
                # but is handled separately by the surrogate sanitization
                # path above.  Exclude json.JSONDecodeError — also a
                # ValueError subclass, but it indicates a transient
                # provider/network failure (malformed response body,
                # truncated stream, routing layer corruption), not a
                # local programming bug, and should be retried (#14782).
                is_local_validation_error = (
                    isinstance(api_error, (ValueError, TypeError))
                    and not isinstance(
                        api_error, (UnicodeEncodeError, json.JSONDecodeError)
                    )
                    # ssl.SSLError (and its subclass SSLCertVerificationError)
                    # inherits from OSError *and* ValueError via Python MRO,
                    # so the isinstance(ValueError) check above would
                    # misclassify a TLS transport failure as a local
                    # programming bug and abort without retrying.  Exclude
                    # ssl.SSLError explicitly so the error classifier's
                    # retryable=True mapping takes effect instead.
                    and not isinstance(api_error, ssl.SSLError)
                    # Provider/SDK "NoneType is not iterable" failures are
                    # shape mismatches from upstream (e.g. chatgpt.com Codex
                    # backend response.completed.output=null) — not local
                    # programming bugs.  Even after #33042 made our own
                    # consumer immune, third-party shims and mocked clients
                    # can still surface this shape via TypeError.  Treat
                    # them as retryable so the error classifier's normal
                    # retry/fallback path runs instead of killing the turn
                    # as non-retryable (which left Telegram users staring
                    # at a bare "Non-retryable error" with no recovery).
                    and not (
                        isinstance(api_error, TypeError)
                        and "nonetype" in str(api_error).lower()
                        and "not iterable" in str(api_error).lower()
                    )
                )
                # ``FailoverReason.billing`` (HTTP 402) is NOT in this
                # exclusion set.  By the time we reach this block:
                #   • credential-pool rotation (line ~2031) has already
                #     fired for billing and either ``continue``d or
                #     returned (False, ...) — pool is exhausted or absent.
                #   • the eager-fallback branch above (line ~2422) also
                #     fires on billing and ``continue``s if a fallback
                #     provider is configured.
                # Falling through to here means BOTH recovery paths
                # gave up.  Treating 402 as retryable from this point
                # just burns more paid requests against a depleted
                # balance with no recovery mechanism left — see #31273
                # (real-world: ~$40 in 48h on a 24/7 gateway).  Aborting
                # mirrors how 401/403 (also ``should_fallback=True``)
                # already behave once their recovery paths have failed.
                is_client_error = (
                    is_local_validation_error
                    or (
                        not classified.retryable
                        and not classified.should_compress
                        and classified.reason not in {
                            FailoverReason.rate_limit,
                            FailoverReason.overloaded,
                            FailoverReason.context_overflow,
                            FailoverReason.payload_too_large,
                            FailoverReason.long_context_tier,
                            FailoverReason.thinking_signature,
                        }
                    )
                ) and not is_context_length_error

                if is_client_error:
                    # Copilot self-heal BEFORE fallback: a stale/degraded
                    # credential surfaces as a 400
                    # ``model_not_available_for_integrator`` /
                    # ``model_not_supported`` (not a clean 401), so the 401
                    # refresh path above never fired. Force a fresh token
                    # exchange + client rebuild and retry once on the SAME
                    # provider — a fresh 437-char API token routes to the
                    # correct integrator and the model becomes available again.
                    # Single-shot guard prevents looping on a genuinely
                    # unavailable model. Copilot-scoped so other providers'
                    # real 400s are untouched.
                    if (
                        _is_copilot_provider(agent)
                        and not _retry.copilot_stale_cred_retry_attempted
                        and _is_stale_copilot_credential_error(
                            status_code, str(getattr(api_error, "message", "") or api_error)
                        )
                    ):
                        _retry.copilot_stale_cred_retry_attempted = True
                        if agent._try_recover_stale_copilot_credential():
                            agent._buffer_vprint(
                                "🔐 Copilot credential re-exchanged after "
                                "model_not_available 400. Retrying request..."
                            )
                            retry_count = 0
                            continue
                    # Try fallback before aborting — a different provider may
                    # not have the same issue (rate limit, auth, etc.). Only
                    # announce the attempt when a fallback chain actually
                    # exists; otherwise "trying fallback..." is a lie and the
                    # session looks like it's recovering when it's about to
                    # abort silently (#35314, #17446).
                    if agent._has_pending_fallback():
                        if classified.reason == FailoverReason.content_policy_blocked:
                            agent._buffer_status("⚠️ Provider safety filter blocked this request — trying fallback...")
                        elif classified.reason == FailoverReason.ssl_cert_verification:
                            agent._buffer_status("⚠️ TLS certificate verification failed — trying fallback...")
                        else:
                            agent._buffer_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
                    if agent._try_activate_fallback():
                        active_system_prompt = _sync_failover_system_message(
                            agent, api_messages, active_system_prompt)
                        retry_count = 0
                        compression_attempts = 0
                        _retry.primary_recovery_attempted = False
                        continue
                    if api_kwargs is not None:
                        agent._dump_api_request_debug(
                            api_kwargs, reason="non_retryable_client_error", error=api_error,
                        )
                    # Terminal — flush buffered context so the user sees
                    # what was tried before the abort.
                    agent._flush_status_buffer()
                    # Summarize once: Cloudflare/proxy HTML challenge pages and
                    # other raw provider bodies must be collapsed to a short
                    # one-liner here, otherwise the full page leaks into the
                    # returned ``error`` field and downstream consumers deliver
                    # it verbatim (e.g. a cron failure notification dumped a
                    # ~60KB Cloudflare challenge page as 31 Discord messages).
                    _nonretryable_summary = agent._summarize_api_error(api_error)
                    if classified.reason == FailoverReason.content_policy_blocked:
                        agent._emit_status(
                            f"❌ Provider safety filter blocked this request: "
                            f"{_nonretryable_summary}"
                        )
                    elif classified.reason == FailoverReason.ssl_cert_verification:
                        agent._emit_status(
                            f"❌ TLS certificate verification failed: "
                            f"{_nonretryable_summary}"
                        )
                    else:
                        agent._emit_status(
                            f"❌ Non-retryable error (HTTP {status_code}): "
                            f"{_nonretryable_summary}"
                        )
                    agent._vprint(f"{agent.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
                    agent._vprint(f"{agent.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                    agent._vprint(f"{agent.log_prefix}   🌐 Endpoint: {_base}", force=True)
                    # Actionable guidance for common auth errors
                    if classified.is_auth or classified.reason == FailoverReason.billing:
                        if classified.reason == FailoverReason.billing and _print_billing_or_entitlement_guidance(
                            agent,
                            capability="model access",
                            provider=_provider,
                            base_url=str(_base),
                            model=_model,
                        ):
                            pass
                        elif _provider == "nous" and _print_nous_entitlement_guidance(
                            agent,
                            "Nous model access",
                        ):
                            pass
                        elif _provider in {"openai-codex", "xai-oauth", "nous"} and status_code == 401:
                            if _provider == "openai-codex":
                                agent._vprint(f"{agent.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                                agent._vprint(f"{agent.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                                agent._vprint(f"{agent.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                                agent._vprint(f"{agent.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
                            elif _provider == "xai-oauth":
                                agent._vprint(f"{agent.log_prefix}   💡 xAI OAuth token was rejected (HTTP 401). To fix:", force=True)
                                agent._vprint(f"{agent.log_prefix}      re-authenticate with xAI Grok OAuth (SuperGrok / Premium+) from `hermes model`.", force=True)
                            else:  # nous
                                agent._vprint(f"{agent.log_prefix}   💡 Nous Portal OAuth token was rejected (HTTP 401). Your token may be", force=True)
                                agent._vprint(f"{agent.log_prefix}      expired, revoked, or your account may be out of credits. To fix:", force=True)
                                agent._vprint(f"{agent.log_prefix}      1. Re-authenticate: hermes portal", force=True)
                                agent._vprint(f"{agent.log_prefix}      2. Check your portal account: https://portal.nousresearch.com", force=True)
                                # ``:free`` is OpenRouter slug syntax; Nous Portal will reject
                                # the model name even after a successful re-auth.
                                if isinstance(_model, str) and _model.endswith(":free"):
                                    agent._vprint(f"{agent.log_prefix}      ⚠️  Note: `{_model}` looks like an OpenRouter slug (`:free` suffix).", force=True)
                                    agent._vprint(f"{agent.log_prefix}         Nous Portal won't recognize that model name. Either switch to a", force=True)
                                    agent._vprint(f"{agent.log_prefix}         Nous catalog model, or run `/model openrouter:{_model}` to use OpenRouter.", force=True)
                        else:
                            agent._vprint(f"{agent.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
                            agent._vprint(f"{agent.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
                            agent._vprint(f"{agent.log_prefix}      • Does your account have access to {_model}?", force=True)
                            if base_url_host_matches(str(_base), "openrouter.ai"):
                                agent._vprint(f"{agent.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
                    else:
                        agent._vprint(f"{agent.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
                    # Content-policy blocks deserve their own actionable
                    # guidance — neither "fix your API key" nor "retry won't
                    # help" tells the user what to actually do. The provider
                    # has refused this specific prompt, so the recovery is
                    # either a rephrase or routing to a different model.
                    if classified.reason == FailoverReason.content_policy_blocked:
                        agent._vprint(
                            f"{agent.log_prefix}   💡 The provider's safety filter rejected this specific prompt.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      • Try rephrasing the request, narrowing the context, or splitting into smaller steps.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      • Configure a fallback provider so future blocks route automatically:",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}        hermes fallback add   (interactive picker — same as `hermes model`)",
                            force=True,
                        )
                    # TLS certificate failures are environment problems, not
                    # provider/prompt problems — tell the user exactly which
                    # knobs fix each common cause. Inspired by Claude Code
                    # v2.1.199's immediate SSL fix hints.
                    if classified.reason == FailoverReason.ssl_cert_verification:
                        agent._vprint(
                            f"{agent.log_prefix}   💡 The TLS certificate chain could not be verified. This fails the same",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      way on every retry — fix the environment, then try again:",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      • Corporate TLS-inspecting proxy? Point Python at its CA bundle:",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}        export SSL_CERT_FILE=/path/to/corp-ca.pem  (also REQUESTS_CA_BUNDLE)",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      • Missing/stale system CA store? Install/refresh it:",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}        pip install --upgrade certifi   (macOS: run 'Install Certificates.command')",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      • Self-signed local endpoint (llama.cpp, LM Studio, vLLM)? Use http://",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}        for localhost, or add the server's cert to your trust store.",
                            force=True,
                        )
                    logger.error("%sNon-retryable client error: %s", agent.log_prefix, api_error)
                    # Skip session persistence when the error is likely
                    # context-overflow related (status 400 + large session).
                    # Persisting the failed user message would make the
                    # session even larger, causing the same failure on the
                    # next attempt. (#1630)
                    if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Skipping session persistence "
                            f"for large failed session to prevent growth loop.",
                            force=True,
                        )
                    else:
                        agent._persist_session(messages, conversation_history)
                    if classified.reason == FailoverReason.content_policy_blocked:
                        _policy_response = (
                            "⚠️  The model provider's safety filter blocked this request "
                            "(not a Hermes/gateway failure).\n\n"
                            f"Provider message: {_nonretryable_summary}\n\n"
                            f"{_CONTENT_POLICY_RECOVERY_HINT}"
                        )
                        return _content_policy_blocked_result(
                            messages,
                            api_call_count,
                            final_response=_policy_response,
                            error_detail=_nonretryable_summary,
                        )
                    # Billing walls are the common non-retryable abort: enrich
                    # the result with the same structured recovery descriptor as
                    # the max-retries path so every surface (CLI, TUI, desktop)
                    # renders one consistent billing signal.
                    if classified.reason == FailoverReason.billing:
                        _ce_guidance = _billing_or_entitlement_message(
                            capability="model access",
                            provider=_provider,
                            base_url=str(_base),
                            model=_model,
                        )
                        _ce_final = f"Billing or credits exhausted: {_nonretryable_summary}"
                        if _ce_guidance:
                            _ce_final += f"\n\n{_ce_guidance}"
                        _ce_block = _billing_block_dict(_provider, _base, _model, _ce_guidance)
                        return {
                            "final_response": _ce_final,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _nonretryable_summary,
                            "failure_reason": classified.reason.value,
                            "billing_block": _ce_block,
                        }
                    return {
                        "final_response": _nonretryable_summary,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": _nonretryable_summary,
                    }

                if retry_count >= max_retries:
                    # Before falling back, try rebuilding the primary
                    # client once for transient transport errors (stale
                    # connection pool, TCP reset).  Only attempted once
                    # per API call block.
                    if not _retry.primary_recovery_attempted and agent._try_recover_primary_transport(
                        api_error, retry_count=retry_count, max_retries=max_retries,
                    ):
                        _retry.primary_recovery_attempted = True
                        retry_count = 0
                        # Primary transport recovery starts a fresh attempt
                        # cycle. Re-open fallback state so a follow-on 429 can
                        # still activate fallback_providers after stale
                        # pre-recovery fallback/credential-pool bookkeeping.
                        _retry.has_retried_429 = False
                        agent._fallback_index = 0
                        agent._fallback_activated = False
                        continue
                    # Try fallback before giving up entirely
                    if agent._has_pending_fallback():
                        agent._buffer_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
                    if agent._try_activate_fallback():
                        active_system_prompt = _sync_failover_system_message(
                            agent, api_messages, active_system_prompt)
                        retry_count = 0
                        compression_attempts = 0
                        _retry.primary_recovery_attempted = False
                        continue
                    # Terminal — flush buffered retry/fallback trace.
                    agent._flush_status_buffer()
                    _final_summary = agent._summarize_api_error(api_error)
                    _billing_guidance = ""
                    if classified.reason == FailoverReason.billing:
                        agent._emit_status(f"❌ Billing or credits exhausted — {_final_summary}")
                        _billing_guidance = _billing_or_entitlement_message(
                            capability="model access",
                            provider=_provider,
                            base_url=str(_base),
                            model=_model,
                        )
                        _print_billing_or_entitlement_guidance(
                            agent,
                            capability="model access",
                            provider=_provider,
                            base_url=str(_base),
                            model=_model,
                        )
                    elif is_rate_limited:
                        agent._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
                    else:
                        agent._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
                    agent._vprint(f"{agent.log_prefix}   💀 Final error: {_final_summary}", force=True)

                    # Detect SSE stream-drop pattern (e.g. "Network
                    # connection lost") and surface actionable guidance.
                    # This typically happens when the model generates a
                    # very large tool call (write_file with huge content)
                    # and the proxy/CDN drops the stream mid-response.
                    _is_stream_drop = (
                        not getattr(api_error, "status_code", None)
                        and any(p in error_msg for p in (
                            "connection lost", "connection reset",
                            "connection closed", "network connection",
                            "network error", "terminated",
                        ))
                    )
                    if _is_stream_drop:
                        agent._vprint(
                            f"{agent.log_prefix}   💡 The provider's stream "
                            f"connection keeps dropping. This often happens "
                            f"when the model tries to write a very large "
                            f"file in a single tool call.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      Try asking the model "
                            f"to use execute_code with Python's open() for "
                            f"large files, or to write the file in smaller "
                            f"sections.",
                            force=True,
                        )

                    # Detect thinking-timeout pattern: a known reasoning model
                    # hit a transport-layer error before the first content
                    # token arrived.  Distinct from _is_stream_drop above
                    # (which fires for large file-write stream drops) and
                    # from any classifier reason that's not a transport
                    # timeout.  Reuses the reasoning-model allowlist from
                    # agent/reasoning_timeouts.py (Fixes #52217) so the
                    # trigger is consistent with what the per-model
                    # stale-timeout floor covers.  After the classifier
                    # override at agent/error_classifier.py:720-738 (this
                    # PR), transport disconnects on reasoning models route
                    # to FailoverReason.timeout rather than
                    # context_overflow, so this branch actually fires.
                    # Detection and message text live in
                    # agent.thinking_timeout_guidance so they're
                    # unit-testable without driving the full retry loop.
                    # (Part 2 of Fixes #52310.)
                    from agent.thinking_timeout_guidance import (
                        is_thinking_timeout,
                    )
                    _is_thinking_timeout = is_thinking_timeout(
                        classified,
                        _model,
                        error_msg,
                    )
                    if _is_thinking_timeout:
                        agent._vprint(
                            f"{agent.log_prefix}   💡 The model's thinking "
                            f"phase exceeded the upstream proxy's idle "
                            f"timeout before the first content token "
                            f"arrived. This is a known issue with "
                            f"reasoning models behind cloud gateways "
                            f"(NVIDIA NIM, OpenAI, Anthropic, DeepSeek).",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      Workarounds in priority order:",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      1. Set "
                            f"`providers.{_provider}.models.{_model}.stale_timeout_seconds: 900` "
                            f"in `~/.hermes/config.yaml` to extend the per-call "
                            f"timeout. (Hermes's built-in floor is 600s for "
                            f"known reasoning models — if you still see this "
                            f"after raising, the upstream cap is even shorter.)",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      2. Lower `reasoning_budget` or set "
                            f"`reasoning_effort: medium` on this model if the provider supports it.",
                            force=True,
                        )
                        agent._vprint(
                            f"{agent.log_prefix}      3. Use a smaller / faster reasoning "
                            f"model if the task doesn't require deep thinking.",
                            force=True,
                        )

                    logger.error(
                        "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
                        agent.log_prefix, max_retries, _final_summary,
                        _provider, _model, len(api_messages), f"{approx_tokens:,}",
                    )
                    if api_kwargs is not None:
                        agent._dump_api_request_debug(
                            api_kwargs, reason="max_retries_exhausted", error=api_error,
                        )
                    agent._persist_session(messages, conversation_history)
                    _billing_block = None
                    if classified.reason == FailoverReason.billing:
                        _final_response = f"Billing or credits exhausted: {_final_summary}"
                        if _billing_guidance:
                            _final_response += f"\n\n{_billing_guidance}"
                        # Structured recovery descriptor so every surface renders
                        # the same link + label from one signal (see helper).
                        _billing_block = _billing_block_dict(_provider, _base, _model, _billing_guidance)
                    else:
                        _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
                    if _is_thinking_timeout:
                        # Thinking-timeout guidance overrides the generic
                        # stream-drop guidance — the latter is wrong for
                        # this case (it suggests splitting large file
                        # writes, which isn't what happened).  See the
                        # reasoning-model override at
                        # agent/error_classifier.py:720-738 and the
                        # detection block above for context.
                        from agent.thinking_timeout_guidance import (
                            build_thinking_timeout_guidance,
                        )
                        _final_response += build_thinking_timeout_guidance(
                            provider=_provider,
                            model=_model,
                        )
                    elif _is_stream_drop:
                        _final_response += (
                            "\n\nThe provider's stream connection keeps "
                            "dropping — this often happens when generating "
                            "very large tool call responses (e.g. write_file "
                            "with long content). Try asking me to use "
                            "execute_code with Python's open() for large "
                            "files, or to write in smaller sections."
                        )
                    return {
                        "final_response": _final_response,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": _final_summary,
                        # Surface the classified reason so callers (notably the
                        # kanban worker path in cli.py) can distinguish a
                        # transient throttle from a real failure and choose a
                        # different exit code. ``rate_limit`` / ``billing`` here
                        # mean "quota wall, not a task error".
                        "failure_reason": classified.reason.value,
                        # Present only for billing walls: structured recovery
                        # descriptor (provider, billing_url, is_nous, message).
                        "billing_block": _billing_block,
                    }

                # For rate limits, respect the Retry-After header if present
                _retry_after = None
                if is_rate_limited:
                    _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                    if _resp_headers and hasattr(_resp_headers, "get"):
                        _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                        if _ra_raw:
                            try:
                                # Cap at 10 minutes. Anthropic Tier 1 input-token
                                # buckets reset in ~171s, so a 120s cap caused us to
                                # retry before the actual reset window and re-trip the
                                # limit. 600s covers all realistic provider reset
                                # windows while still rejecting pathological values. (#26293)
                                _retry_after = min(float(_ra_raw), 600)
                            except (TypeError, ValueError):
                                pass
                wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
                _backoff_policy = None
                if (is_rate_limited or _is_zai_coding_overload) and not _retry_after:
                    wait_time, _backoff_policy = adaptive_rate_limit_backoff(
                        retry_count,
                        base_url=str(_base),
                        model=_model,
                        error=api_error,
                        default_wait=wait_time,
                    )
                if is_rate_limited or _is_zai_coding_overload:
                    _policy_note = ""
                    if _backoff_policy == "zai_coding_overload_long":
                        _policy_note = " (Z.AI Coding overload adaptive long backoff)"
                    elif _backoff_policy == "zai_coding_overload_short":
                        _policy_note = " (Z.AI Coding overload short retry)"
                    _wait_reason = "Provider overloaded" if _is_zai_coding_overload and not is_rate_limited else "Rate limited"
                    _rate_limit_status = f"⏱️ {_wait_reason}. Waiting {wait_time:.1f}s (attempt {retry_count + 1}/{max_retries}){_policy_note}..."
                    # Normal retries are buffered to avoid noisy transient chatter. Long
                    # Z.AI Coding waits are different: they can last minutes, so surface
                    # progress immediately instead of making the TUI look frozen.
                    if _backoff_policy == "zai_coding_overload_long":
                        agent._emit_status(_rate_limit_status)
                    else:
                        agent._buffer_status(_rate_limit_status)
                else:
                    agent._buffer_status(f"⏳ Retrying in {wait_time:.1f}s (attempt {retry_count}/{max_retries})...")
                logger.warning(
                    "Retrying API call in %ss (attempt %s/%s) %s policy=%s error=%s",
                    wait_time,
                    retry_count,
                    max_retries,
                    agent._client_log_context(),
                    _backoff_policy or "default",
                    api_error,
                )
                # Sleep in small increments so we can respond to interrupts quickly
                # instead of blocking the entire wait_time in one sleep() call
                sleep_end = time.time() + wait_time
                _backoff_touch_counter = 0
                while time.time() < sleep_end:
                    if agent._interrupt_requested:
                        # Same preserve-redirect rule as the retry-wait above:
                        # a steering correction must survive backoff, not die
                        # as "Operation interrupted".
                        if agent.clear_interrupt(preserve_redirect=True):
                            _retry.restart_with_redirected_messages = True
                            break
                        agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                        _interrupt_text = f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries})."
                        close_interrupted_tool_sequence(messages, _interrupt_text)
                        agent._persist_session(messages, conversation_history)
                        agent.clear_interrupt()
                        return {
                            "final_response": _interrupt_text,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    time.sleep(0.2)  # Check interrupt every 200ms
                    # Touch activity every ~30s so the gateway's inactivity
                    # monitor knows we're alive during backoff waits.
                    _backoff_touch_counter += 1
                    if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                        agent._touch_activity(
                            f"error retry backoff ({retry_count}/{max_retries}), "
                            f"{int(sleep_end - time.time())}s remaining"
                        )
                if _retry.restart_with_redirected_messages:
                    # Leave the retry loop — the check right below rebuilds this
                    # iteration from the correction instead of re-firing the
                    # stale request.
                    break
        
        if _retry.restart_with_redirected_messages:
            # The cancelled request produced no valid assistant item. Reuse the
            # same logical iteration after the outer loop appends the displayed
            # partial context and correction to ``messages``.
            api_call_count -= 1
            agent.iteration_budget.refund()
            _retry.restart_with_redirected_messages = False
            continue

        # If the API call was interrupted, skip response processing
        if interrupted:
            _turn_exit_reason = "interrupted_during_api_call"
        _run_phase(prepare_iteration, agent, s)
        _run_phase(assemble_api_request, agent, s)
        _pg = _run_phase(run_preflight_gate, agent, s)
        if _pg.action == "return":
            return _pg.result
        if _pg.action == "break":
            break
        if _pg.action == "continue":
            continue
        _run_phase(announce_api_call, agent, s)

        s.api_start_time, s.retry_count, s.max_retries = time.time(), 0, agent._api_max_retries
        s._retry, s.finish_reason, s.response, s.api_kwargs = TurnRetryState(), "stop", None, None
        s.api_request_id = agent._current_api_request_id = f"{s.turn_id}:api:{s.api_call_count}"

        early_result = _run_api_retry_loop(agent, s)
        if early_result is not None:
            return early_result

        _rs = _run_phase(apply_retry_restarts, agent, s)
        if _rs.action == "break":
            break
        if _rs.action == "continue":
            continue

        try:
            _ri = _run_phase(normalize_model_response, agent, s)
            if _ri.action == "return":
                return _ri.result
            if _ri.action == "continue":
                continue
            _v = _run_phase(
                run_tool_round if s.assistant_message.tool_calls else finish_text_response, agent, s
            )
            if _v.action == "return":
                return _v.result
            if _v.action == "break":
                break
            if _v.action == "continue":
                continue
        except Exception as e:
            if _run_phase(handle_outer_loop_error, agent, s, e=e).action == "break":
                break

    # Post-loop finalization lives in agent/turn_finalizer.finalize_turn.
    result = finalize_turn(agent, **{
        name: getattr(s, name)
        for name in inspect.signature(finalize_turn).parameters if name != "agent"
    })
    if s._compression_timeout_exhausted:
        # Reuse the gateway's context-recovery contract: transcript stays intact while
        # future input can move to a clean session (#98722).
        result.update(error=_COMPRESSION_TIMEOUT_FINAL_RESPONSE, partial=True, compression_exhausted=True)
    return result


def run_conversation(
    agent,
    user_message: Any,
    system_message: str = None,
    conversation_history: List[Dict[str, Any]] = None,
    task_id: str = None,
    stream_callback: Optional[callable] = None,
    persist_user_message: Optional[Any] = None,
    persist_user_timestamp: Optional[float] = None,
    persist_user_display_kind: Optional[str] = None,
    persist_user_display_metadata: Optional[Dict[str, Any]] = None,
    persist_user_platform_id: Optional[str] = None,
    moa_config: Optional[dict[str, Any]] = None,
    turn_author: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one turn (see ``_run_conversation_turn``) and export the current-turn boundary.

    Every envelope that leaves the loop — success, partial/error, interrupt, retry-exhausted,
    tool-limit, preflight timeout, codex runtime — passes through here, so the
    ``{turn_id, current_turn_user_idx}`` pair is stamped beside the exact ``messages`` it
    addresses, after every history rewrite including post-turn micro-compaction.
    """
    from agent.turn_context import export_current_turn_boundary

    result = _run_conversation_turn(
        agent,
        user_message,
        system_message=system_message,
        conversation_history=conversation_history,
        task_id=task_id,
        stream_callback=stream_callback,
        persist_user_message=persist_user_message,
        persist_user_timestamp=persist_user_timestamp,
        persist_user_display_kind=persist_user_display_kind,
        persist_user_display_metadata=persist_user_display_metadata,
        persist_user_platform_id=persist_user_platform_id,
        moa_config=moa_config,
        turn_author=turn_author,
    )
    return export_current_turn_boundary(agent, result, user_message)


__all__ = ["run_conversation"]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
import random  # noqa: F401,E402
import ssl  # noqa: F401,E402
import sys  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE': ('agent.conversation_compression', 'COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE'),
    'COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE': ('agent.conversation_compression', 'COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE'),
    'COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE': ('agent.conversation_compression', 'COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE'),
    'COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE': ('agent.conversation_compression', 'COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE'),
    'FailoverReason': ('agent.error_classifier', 'FailoverReason'),
    'KawaiiSpinner': ('agent.display', 'KawaiiSpinner'),
    'PARTIAL_STREAM_STUB_ID': ('hermes_constants', 'PARTIAL_STREAM_STUB_ID'),
    'PRE_API_COMPRESSION_STATUS_TEMPLATE': ('agent.conversation_compression', 'PRE_API_COMPRESSION_STATUS_TEMPLATE'),
    'adaptive_rate_limit_backoff': ('agent.retry_utils', 'adaptive_rate_limit_backoff'),
    'anchored_context_tokens': ('agent.usage_anchor', 'anchored_context_tokens'),
    'automatic_compaction_status_message': ('agent.context_engine', 'automatic_compaction_status_message'),
    'capture_usage_anchor': ('agent.usage_anchor', 'capture_usage_anchor'),
    'classify_api_error': ('agent.error_classifier', 'classify_api_error'),
    'close_interrupted_tool_sequence': ('agent.message_sanitization', 'close_interrupted_tool_sequence'),
    'coalesce_tool_call_id': ('agent.message_sanitization', 'coalesce_tool_call_id'),
    'compose_user_api_content': ('agent.turn_context', 'compose_user_api_content'),
    'compression_blocked_transiently': ('agent.conversation_compression', 'compression_blocked_transiently'),
    'compression_skipped_due_to_lock': ('agent.conversation_compression', 'compression_skipped_due_to_lock'),
    'context_compression_timed_out': ('agent.conversation_compression', 'context_compression_timed_out'),
    'conversation_history_after_compression': ('agent.conversation_compression', 'conversation_history_after_compression'),
    'env_var_enabled': ('utils', 'env_var_enabled'),
    'estimate_messages_tokens_rough': ('agent.model_metadata', 'estimate_messages_tokens_rough'),
    'estimate_request_tokens_rough': ('agent.model_metadata', 'estimate_request_tokens_rough'),
    'estimate_usage_cost': ('agent.usage_pricing', 'estimate_usage_cost'),
    'get_context_length_from_provider_error': ('agent.model_metadata', 'get_context_length_from_provider_error'),
    'has_incomplete_scratchpad': ('agent.trajectory', 'has_incomplete_scratchpad'),
    'is_output_cap_error': ('agent.model_metadata', 'is_output_cap_error'),
    'is_repetition_dominated': ('agent.repetition_guard', 'is_repetition_dominated'),
    'is_zai_coding_overload_error': ('agent.retry_utils', 'is_zai_coding_overload_error'),
    'jittered_backoff': ('agent.retry_utils', 'jittered_backoff'),
    'normalize_usage': ('agent.usage_pricing', 'normalize_usage'),
    'parse_available_output_tokens_from_error': ('agent.model_metadata', 'parse_available_output_tokens_from_error'),
    'reanchor_current_turn_user_idx': ('agent.turn_context', 'reanchor_current_turn_user_idx'),
    'save_context_length': ('agent.model_metadata', 'save_context_length'),
    'serialized_messages_bytes': ('agent.message_sanitization', 'serialized_messages_bytes'),
    'splice_provider_projection': ('agent.provider_projection', 'splice_provider_projection'),
    'zai_coding_overload_retry_ceiling': ('agent.retry_utils', 'zai_coding_overload_retry_ceiling'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
