"""Post-call response verification for the conversation turn's retry loop: stop the thinking
spinner, validate the response shape (retry / eager fallback / terminal invalid-response
result), derive ``finish_reason`` per api_mode, route content-policy refusals and
``length`` truncation, fold usage into the compressor, and mark the logical relay call
complete. Nothing here imports ``agent.conversation_loop`` at module level (cycle) —
loop-internal helpers resolve lazily so ``patch("agent.conversation_loop.X")`` keeps intercepting.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, Optional

from agent.error_classifier import FailoverReason
from agent.turn_api_call import stop_thinking_spinner
from agent.turn_failure_copy import invalid_response_failure_reason, provider_label_for, site_copy, stamp_failure
from agent.turn_truncation import handle_content_policy_refusal, recover_from_truncation
from agent.turn_usage import record_response_usage

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class ResponseCheckVerdict:
    """``action``: ``"break"`` (leave the retry loop — success, or a fallback/refusal restart
    armed on ``_retry``), ``"continue"`` (retry the API call) or ``"return"`` (``result`` is
    the turn's result dict). The other fields are the retry-loop locals rebound."""

    action: str
    thinking_spinner: Any
    messages: Any
    active_system_prompt: Any
    finish_reason: Any
    retry_count: Any
    compression_attempts: Any
    length_continue_retries: Any
    truncated_response_parts: Any
    truncated_tool_call_retries: Any
    codex_ack_continuations: Any
    _preflight_compression_blocked: Any
    _last_preflight_pressure: Any
    api_duration: Any
    result: Optional[Dict[str, Any]] = None


def _codex_finish_reason(response: Any) -> str:
    """Responses API max-output exhaustion is a normal Codex incomplete turn: route it to
    the Codex continuation path (``"incomplete"``), not the length rollback."""
    status = getattr(response, "status", None)
    if isinstance(status, str):
        status = status.strip().lower()
    incomplete_details = getattr(response, "incomplete_details", None)
    if isinstance(incomplete_details, dict):
        incomplete_reason = incomplete_details.get("reason")
    else:
        incomplete_reason = getattr(incomplete_details, "reason", None)
    if incomplete_reason is not None:
        incomplete_reason = str(incomplete_reason).strip().lower()
    if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
        return "incomplete"
    if status == "incomplete" and incomplete_reason == "content_filter":
        return "content_filter"
    return "stop"


def _derive_finish_reason(agent: Any, response: Any, messages: Any) -> str:
    if agent.api_mode == "codex_responses":
        finish_reason = _codex_finish_reason(response)
        # A function_call cut off by max_output_tokens is not a text turn to continue: the
        # Codex incomplete path would replay the partial and re-hit the same cap. Route it
        # to the length path so the same call is retried with a boosted budget (#91770).
        if finish_reason == "incomplete" and agent._get_transport().normalize_response(response).tool_calls:
            return "length"
        return finish_reason
    transport = agent._get_transport()
    if agent.api_mode == "anthropic_messages":
        return transport.response_finish_reason(response)
    normalized = transport.normalize_response(response)  # Bedrock already normalized at dispatch
    finish_reason = normalized.finish_reason
    if agent.api_mode != "bedrock_converse" and agent._should_treat_stop_as_truncated(
        finish_reason, normalized, messages
    ):
        agent._vprint(
            f"{agent.log_prefix}⚠️  Treating suspicious Ollama/GLM stop response as truncated",
            force=True, diagnostic=True,
        )
        return "length"
    return finish_reason


def check_api_response(
    agent: Any, *, response: Any, _retry: Any, thinking_spinner: Any, messages: Any,
    api_messages: Any, api_kwargs: Any, active_system_prompt: Any, conversation_history: Any,
    finish_reason: Any, retry_count: Any, max_retries: Any, compression_attempts: Any,
    max_compression_attempts: Any, length_continue_retries: Any, truncated_response_parts: Any,
    truncated_tool_call_retries: Any, codex_ack_continuations: Any, current_turn_user_idx: Any, api_call_count: Any,
    api_request_id: Any, api_start_time: Any, effective_task_id: Any, turn_id: Any,
    _preflight_compression_blocked: Any, _last_preflight_pressure: Any,
) -> ResponseCheckVerdict:
    """Verify ``response`` in the original order. The retry buffer is NOT cleared on success
    (bytes back != usable content); ``_preflight_compression_blocked``/``_last_preflight_pressure``
    reset only when the usage fold re-arms the compression budget."""
    from agent.turn_recovery import validate_response_shape

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ResponseCheckVerdict:
        return ResponseCheckVerdict(
            action=action, thinking_spinner=thinking_spinner, messages=messages,
            active_system_prompt=active_system_prompt, finish_reason=finish_reason,
            retry_count=retry_count, compression_attempts=compression_attempts,
            length_continue_retries=length_continue_retries,
            truncated_response_parts=truncated_response_parts,
            truncated_tool_call_retries=truncated_tool_call_retries,
            codex_ack_continuations=codex_ack_continuations,
            _preflight_compression_blocked=_preflight_compression_blocked,
            _last_preflight_pressure=_last_preflight_pressure, api_duration=api_duration,
            result=result,
        )

    api_duration = time.time() - api_start_time

    # Silent stop: the response box / tool messages that follow are more informative.
    thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)

    if not agent.quiet_mode:
        agent._vprint(f"{agent.log_prefix}⏱️  API call completed in {api_duration:.2f}s")

    if agent.verbose_logging:
        resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
        logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")

    response_invalid, error_details = validate_response_shape(agent, response)
    if response_invalid:
        _iv = retry_invalid_response(
            agent, response=response, error_details=error_details, _retry=_retry,
            thinking_spinner=thinking_spinner, messages=messages, api_messages=api_messages,
            api_kwargs=api_kwargs, active_system_prompt=active_system_prompt,
            conversation_history=conversation_history, retry_count=retry_count,
            max_retries=max_retries, compression_attempts=compression_attempts,
            api_call_count=api_call_count, api_request_id=api_request_id,
            api_start_time=api_start_time, api_duration=api_duration,
            effective_task_id=effective_task_id, turn_id=turn_id,
        )
        thinking_spinner = _iv.thinking_spinner
        active_system_prompt = _iv.active_system_prompt
        retry_count = _iv.retry_count
        compression_attempts = _iv.compression_attempts
        if _iv.action != "fallthrough":
            return _verdict(_iv.action, _iv.result)

    agent._turn_received_provider_response = True
    finish_reason = _derive_finish_reason(agent, response, messages)

    # HTTP-200 refusals are deterministic: one fallback try, else return the refusal.
    if finish_reason == "content_filter":
        _rv = handle_content_policy_refusal(
            agent, response, _retry, thinking_spinner=thinking_spinner, messages=messages,
            api_messages=api_messages, api_kwargs=api_kwargs,
            active_system_prompt=active_system_prompt, conversation_history=conversation_history,
            api_call_count=api_call_count, effective_task_id=effective_task_id, turn_id=turn_id,
            api_request_id=api_request_id, api_start_time=api_start_time, retry_count=retry_count,
            max_retries=max_retries,
        )
        thinking_spinner = None
        active_system_prompt = _rv.active_system_prompt
        if _rv.action == "return":
            return _verdict("return", _rv.result)
        retry_count = 0
        compression_attempts = 0
        return _verdict("break")

    if finish_reason == "length":
        _tv = recover_from_truncation(
            agent, response, finish_reason, _retry, messages=messages,
            conversation_history=conversation_history, api_kwargs=api_kwargs,
            api_call_count=api_call_count, effective_task_id=effective_task_id,
            current_turn_user_idx=current_turn_user_idx,
            length_continue_retries=length_continue_retries,
            truncated_response_parts=truncated_response_parts,
            truncated_tool_call_retries=truncated_tool_call_retries, retry_count=retry_count,
            compression_attempts=compression_attempts,
            codex_ack_continuations=codex_ack_continuations,
        )
        messages = _tv.messages
        length_continue_retries = _tv.length_continue_retries
        truncated_response_parts = _tv.truncated_response_parts
        truncated_tool_call_retries = _tv.truncated_tool_call_retries
        retry_count = _tv.retry_count
        compression_attempts = _tv.compression_attempts
        codex_ack_continuations = _tv.codex_ack_continuations
        if _tv.action in ("return", "break", "continue"):
            return _verdict(_tv.action, _tv.result)

    # Fold provider usage into compressor / anchors / session counters / state.db
    # (agent/turn_usage.py). A rearmed budget also clears the preflight-block latch.
    _usage_outcome = record_response_usage(
        agent, response, messages=messages, api_call_count=api_call_count,
        api_duration=api_duration, compression_attempts=compression_attempts,
        max_compression_attempts=max_compression_attempts,
    )
    compression_attempts = _usage_outcome.compression_attempts
    if _usage_outcome.rearmed:
        _preflight_compression_blocked = False
        _last_preflight_pressure = None

    _retry.has_retried_429 = False
    # Clearing Nous rate-limit state proves the limit reset so other sessions may resume.
    if agent.provider == "nous":
        try:
            from agent.nous_rate_guard import clear_nous_rate_limit
            from hermes_cli.anon_auth import is_anonymous_agent
            clear_nous_rate_limit(anonymous=is_anonymous_agent(agent))
        except Exception:
            pass
    from agent import relay_llm

    relay_llm.complete_logical_call(api_request_id, outcome="success")
    agent._touch_activity(f"API call #{api_call_count} completed")
    return _verdict("break")


@dataclass
class InvalidResponseVerdict:
    """``action``: ``"continue"`` (retry the API call after backoff), ``"break"`` (fallback
    armed / redirect pending) or ``"return"`` (``result``: terminal invalid-response result or
    interrupt during backoff). Rebinds ``thinking_spinner``/``active_system_prompt``/
    ``retry_count``/``compression_attempts``."""

    action: str
    thinking_spinner: Any
    active_system_prompt: Any
    retry_count: Any
    compression_attempts: Any
    result: Optional[Dict[str, Any]] = None


def retry_invalid_response(
    agent: Any, *, response: Any, error_details: Any, _retry: Any, thinking_spinner: Any,
    messages: Any, api_messages: Any, api_kwargs: Any, active_system_prompt: Any,
    conversation_history: Any, retry_count: Any, max_retries: Any, compression_attempts: Any,
    api_call_count: Any, api_request_id: Any, api_start_time: Any, api_duration: Any,
    effective_task_id: Any, turn_id: Any,
) -> InvalidResponseVerdict:
    """Malformed/empty provider response: fire the error hook, stop the spinner, eager
    fallback (empty responses often mean rate limiting), terminal result at max retries,
    else jittered backoff that preserves a pending redirect."""
    from agent.conversation_loop import _arm_fallback_restart
    from agent.retry_utils import jittered_backoff
    from agent.turn_recovery import (
        classify_codex_soft_failure, describe_invalid_response, interruptible_backoff_sleep,
    )

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> InvalidResponseVerdict:
        return InvalidResponseVerdict(
            action=action, thinking_spinner=thinking_spinner,
            active_system_prompt=active_system_prompt, retry_count=retry_count,
            compression_attempts=compression_attempts, result=result,
        )

    agent._invoke_api_request_error_hook(
        task_id=effective_task_id, turn_id=turn_id, api_request_id=api_request_id,
        api_call_count=api_call_count, api_start_time=api_start_time, api_kwargs=api_kwargs,
        error_type="InvalidAPIResponse",
        error_message=", ".join(error_details) or "Invalid API response",
        status_code=getattr(getattr(response, "error", None), "code", None),
        retry_count=retry_count, max_retries=max_retries, retryable=True, reason="invalid_response",
    )
    # Retry status is buffered and only surfaced if every retry+fallback exhausts.
    thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)

    # Codex reports quota exhaustion as HTTP 200 ``status=failed`` — the SDK never raises, so the
    # exception path's credential-pool rotation never sees it. Same-provider recovery for the
    # pool-recoverable reasons FIRST (a healthy sibling account beats burning cross-provider
    # fallback); content-policy and other failures keep the fallback/retry path (#24159).
    _soft, _soft_ctx = classify_codex_soft_failure(agent, response)
    if _soft is not None and (_soft.reason in (FailoverReason.rate_limit, FailoverReason.billing) or _soft.is_auth):
        _recovered, _retry.has_retried_429 = agent._recover_with_credential_pool(
            status_code=None, has_retried_429=_retry.has_retried_429, classified_reason=_soft.reason,
            error_context=_soft_ctx, billing_unverified=_soft.billing_unverified,
        )
        if _recovered:
            agent._buffer_diagnostic_status(f"🔄 Codex soft failure ({_soft.reason.value}) — switched to the next pool credential, retrying...")
            return _verdict("continue")
    retry_count += 1

    # Eager fallback: empty/malformed responses often mean rate limiting.
    if agent._fallback_index < len(agent._fallback_chain):
        agent._buffer_diagnostic_status("⚠️ Empty/malformed response — switching to fallback...")
    if agent._try_activate_fallback():
        active_system_prompt = _arm_fallback_restart(
            agent, api_messages, active_system_prompt, _retry)
        retry_count = 0
        compression_attempts = 0
        return _verdict("break")

    error_msg, provider_name, _failure_hint = describe_invalid_response(
        agent, response, api_duration
    )
    agent._buffer_vprint(f"⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}")
    agent._buffer_vprint(f"   🏢 Provider: {provider_name}")
    agent._buffer_vprint(f"   📝 Provider message: {agent._clean_error_message(error_msg)}")
    agent._buffer_vprint(f"   ⏱️  {_failure_hint}")

    if retry_count >= max_retries:
        if agent._has_pending_fallback():
            agent._buffer_diagnostic_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
        if agent._try_activate_fallback():
            active_system_prompt = _arm_fallback_restart(
                agent, api_messages, active_system_prompt, _retry)
            retry_count = 0
            compression_attempts = 0
            return _verdict("break")
        # Terminal — flush buffered retry trace so user sees what happened.
        agent._flush_status_buffer()
        agent._emit_diagnostic_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
        logger.error("%sInvalid API response after %d retries.", agent.log_prefix, max_retries)
        agent._persist_session(messages, conversation_history)
        # "model=<id>" is describe_invalid_response's OpenRouter fallback, not a provider name.
        _label = (
            provider_label_for(agent.provider)
            if provider_name in ("Unknown", "") or provider_name.startswith("model=")
            else provider_name
        )
        _final_response = site_copy(
            "invalid_response", label=_label, attempts=max_retries, detail=_failure_hint,
        )
        return _verdict("return", stamp_failure({
            "final_response": _final_response,
            "messages": messages,
            "completed": False,
            "api_calls": api_call_count,
            "error": f"Invalid API response after {max_retries} retries: {_failure_hint}",
            "failed": True,
        }, invalid_response_failure_reason(response), True))

    wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
    agent._buffer_vprint(f"⏳ Retrying in {wait_time:.1f}s ({_failure_hint})...")
    logger.warning("Invalid API response (retry %d/%d): %s | Provider: %s", retry_count, max_retries, ', '.join(error_details), provider_name)

    # A redirect cancels only the live request; the helper preserves the pending
    # correction (restart_with_redirected_messages) instead of clear_interrupt()-ing it.
    _interrupted = interruptible_backoff_sleep(
        agent, wait_time, _retry, messages=messages, conversation_history=conversation_history,
        api_call_count=api_call_count,
        abort_message="Interrupt detected during retry wait, aborting.",
        interrupt_text=f"Operation interrupted during retry ({_failure_hint}, attempt {retry_count}/{max_retries}).",
        activity_label=f"retry backoff ({retry_count}/{max_retries})",
    )
    if _interrupted is not None:
        return _verdict("return", _interrupted)
    if _retry.restart_with_redirected_messages:
        return _verdict("break")  # rebuild this iteration from the correction
    return _verdict("continue")
