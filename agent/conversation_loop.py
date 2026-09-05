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
from agent.prompt_caching import (
    build_prompt_cache_plan,
    effective_cache_ttl,
    strip_anthropic_cache_control,
    strip_anthropic_tool_cache_control,
)
from agent.runtime_cwd import resolve_agent_cwd
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
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool":
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
        return format_nous_portal_entitlement_message(account_info, capability=capability) or ""
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
    """Return whether routed Bot Mode prompt state must be rebuilt once."""
    try:
        from tools.bot_mode_probe import (
            bot_mode_session_state,
            stored_bot_chat_prompt_needs_upgrade,
            stored_prompt_capability_stale,
            stored_prompt_has_bot_mode_protocol,
        )
        home = None
        try:
            from agent.system_prompt import _agent_home
            home = _agent_home(agent)
        except Exception:
            pass
        routed = bool(bot_mode_session_state(agent)["session_kind"])
        if not routed and stored_prompt_has_bot_mode_protocol(stored_prompt):
            return True
        if not routed:
            return False
        return bool(
            stored_prompt_capability_stale(stored_prompt, home)
            or stored_bot_chat_prompt_needs_upgrade(stored_prompt, home)
        )
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
            # The skills index cache (LRU + disk snapshot) does not watch the skills
            # dir; a capability refresh must rebuild THROUGH it or new skills are lost.
            try:
                from agent.prompt_builder import clear_skills_system_prompt_cache
                clear_skills_system_prompt_cache(clear_snapshot=True)
            except Exception:
                pass
            agent._cached_system_prompt = agent._build_system_prompt(system_message)
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
        # Same contract for tools[]: pin the array to the order this session already
        # sent (tools freeze) instead of re-probing every check_fn on a fresh AIAgent.
        try:
            saved_tools = session_row.get("tool_names") if session_row else None
            if saved_tools:
                from tools.mcp_tool_agent import restore_agent_tool_prefix
                restore_agent_tool_prefix(agent, json.loads(saved_tools))
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

    # Plugin hook: on_session_start — fired once for a brand-new session, not on continuation.
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

    lines = prompt.splitlines()

    def line_value(label: str) -> str:
        """Last matching line wins — safe ONLY for volatile-tier fields at the END of the
        prompt (embedded project context could shadow earlier fields; see ``host_info_value``)."""
        prefix = f"{label}:"
        matches = [line[len(prefix):].strip() for line in lines if line.startswith(prefix)]
        return matches[-1] if matches else ""

    def host_info_value(label: str) -> str:
        """Read a field from the prompt's own host-info block, anchored on the FIRST ``User
        home directory:`` line so a user's ``AGENTS.md`` row cannot force a rebuild every turn."""
        prefix = f"{label}:"
        for idx, line in enumerate(lines):
            if line.startswith("User home directory:"):
                for candidate in lines[idx + 1: idx + 4]:
                    if candidate.startswith(prefix):
                        return candidate[len(prefix):].strip()
        return ""

    # Model/provider identity, then cwd drift, then runtime-surface drift (reusing a
    # desktop-built prompt on a terminal session would inject the wrong runtime hints).
    for label, attr in (("Model", "model"), ("Provider", "provider")):
        stored = line_value(label)
        current = str(getattr(agent, attr, "") or "").strip()
        if stored and current and stored != current:
            return False
    # Compare against resolve_agent_cwd() — the SAME resolver used to build the
    # prompt — so TERMINAL_CWD sessions are not falsely rejected.
    stored_cwd = host_info_value("Current working directory")
    if stored_cwd and stored_cwd != str(resolve_agent_cwd()):
        return False
    stored_platform = line_value("Platform")
    current_platform = str(getattr(agent, "platform", "") or "").strip()
    return not (stored_platform and current_platform and stored_platform != current_platform)


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
    'anchored_context_tokens': ('agent.model_metadata', 'anchored_context_tokens'),
    'automatic_compaction_status_message': ('agent.context_engine', 'automatic_compaction_status_message'),
    'capture_usage_anchor': ('agent.model_metadata', 'capture_usage_anchor'),
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
