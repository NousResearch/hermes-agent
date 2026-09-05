"""API-call exception handler for the conversation turn's retry loop: pre-/post-classification
recovery, interpreter-shutdown abandon, classified-error routing, overflow recovery, the
non-retryable client-error exit, max-retries exhaustion (primary transport recovery →
fallback → terminal result) and the interruptible backoff. Nothing here imports
``agent.conversation_loop`` at module level (cycle) — loop-internal helpers resolve lazily so
``patch("agent.conversation_loop.X")`` sites keep intercepting.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import ssl
import time
from typing import Any, Dict, Optional

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_overflow import recover_from_overflow
from agent.turn_recovery import (
    _NONRETRYABLE_LABELS, abort_turn_on_interrupt, compute_error_backoff, interruptible_backoff_sleep,
    log_api_error_attempt,
    max_retries_exhausted_result, nonretryable_client_error_result, recover_after_classification,
    recover_before_classification, route_classified_error,
)

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class ApiErrorVerdict:
    """``action``: ``"continue"`` (retry the API call), ``"break"`` (leave the retry loop:
    fallback armed / redirect pending) or ``"return"`` (``result`` is the turn's result dict);
    ``"fallthrough"`` never happens — the handler always ends in an exit. The other fields
    are the retry-loop locals the handler rebinds; ``_provider_overflow_recovery_pending`` is
    merge-only (caller sets True when set)."""

    action: str
    thinking_spinner: Any
    messages: Any
    active_system_prompt: Any
    conversation_history: Any
    approx_tokens: Any
    retry_count: Any
    max_retries: Any
    compression_attempts: Any
    _provider_overflow_recovery_pending: Any
    result: Optional[Dict[str, Any]] = None


def handle_api_error(
    agent: Any, *, api_error: Any, _retry: Any, thinking_spinner: Any, messages: Any,
    api_messages: Any, api_kwargs: Any, system_message: Any, active_system_prompt: Any,
    conversation_history: Any, approx_tokens: Any, retry_count: Any, max_retries: Any,
    compression_attempts: Any, max_compression_attempts: Any, api_call_count: Any,
    api_request_id: Any, api_start_time: Any, effective_task_id: Any, turn_id: Any,
) -> ApiErrorVerdict:
    """Recover from ``api_error`` in the original order. Every fallback activation must leave
    the retry loop with ``restart_with_rebuilt_messages`` armed (``"break"``) so the pre-API
    preflight re-runs against the fallback's context window (#84733)."""
    _provider_overflow_recovery_pending = False

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> ApiErrorVerdict:
        return ApiErrorVerdict(
            action=action, thinking_spinner=thinking_spinner, messages=messages,
            active_system_prompt=active_system_prompt, conversation_history=conversation_history,
            approx_tokens=approx_tokens, retry_count=retry_count, max_retries=max_retries,
            compression_attempts=compression_attempts,
            _provider_overflow_recovery_pending=_provider_overflow_recovery_pending, result=result,
        )

    # Stop spinner silently — retry status is buffered and only flushed when every
    # retry+fallback is exhausted.
    if thinking_spinner:
        thinking_spinner.stop("")
        thinking_spinner = None
    if agent.thinking_callback:
        agent.thinking_callback("")

    _recovered, active_system_prompt = recover_before_classification(
        agent, api_error, messages=messages, api_messages=api_messages, api_kwargs=api_kwargs,
        active_system_prompt=active_system_prompt,
    )
    if _recovered:
        return _verdict("continue")

    status_code = getattr(api_error, "status_code", None)
    error_context = agent._extract_api_error_context(api_error)

    # Process is exiting mid-flight: retries/rotation/fallbacks are futile and the
    # retry trace spams the shell. One log line.
    from tools.interpreter_shutdown import interpreter_shutting_down

    if interpreter_shutting_down(api_error):
        logger.warning(
            "%sInterpreter is shutting down — abandoning turn "
            "during API call #%d (%s)",
            agent.log_prefix, api_call_count, api_error,
        )
        _shutdown_summary = "Turn abandoned: the process was shutting down before the model call could complete."
        return _verdict("return", {
            "final_response": _shutdown_summary, "messages": messages, "api_calls": api_call_count,
            "completed": False, "failed": True, "error": _shutdown_summary,
            "failure_reason": "interpreter_shutdown", "failure_retryable": False,
        })

    _compressor = getattr(agent, "context_compressor", None)
    _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
    classified = classify_api_error(
        api_error, provider=getattr(agent, "provider", "") or "",
        model=getattr(agent, "model", "") or "", approx_tokens=approx_tokens,
        context_length=_ctx_len, num_messages=len(api_messages) if api_messages else 0,
    )
    logger.debug(
        "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
        classified.reason.value, classified.status_code,
        classified.retryable, classified.should_compress,
        classified.should_rotate_credential, classified.should_fallback,
    )
    agent._invoke_api_request_error_hook(
        task_id=effective_task_id, turn_id=turn_id, api_request_id=api_request_id,
        api_call_count=api_call_count, api_start_time=api_start_time, api_kwargs=api_kwargs,
        error_type=type(api_error).__name__, error_message=str(api_error), status_code=status_code,
        retry_count=retry_count, max_retries=max_retries, retryable=classified.retryable,
        reason=classified.reason.value,
    )

    _recovered, recovered_with_pool = recover_after_classification(
        agent, api_error, classified, _retry, status_code=status_code, error_context=error_context,
        messages=messages, api_messages=api_messages,
    )
    if _recovered:
        return _verdict("continue")

    retry_count += 1
    elapsed_time = time.time() - api_start_time
    agent._touch_activity(f"API error recovery (attempt {retry_count}/{max_retries})")

    error_type, error_msg, _provider, _base, _model = log_api_error_attempt(
        agent, api_error, retry_count=retry_count, max_retries=max_retries, status_code=status_code,
        elapsed_time=elapsed_time, api_messages=api_messages, approx_tokens=approx_tokens,
    )

    if agent._interrupt_requested:
        # Preserve a pending redirect: the user is steering, not stopping — rebuild the
        # turn from the correction instead of aborting.
        if agent.clear_interrupt(preserve_redirect=True):
            _retry.restart_with_redirected_messages = True
            return _verdict("break")
        return _verdict("return", abort_turn_on_interrupt(
            agent, messages, conversation_history, api_call_count,
            abort_message="Interrupt detected during error handling, aborting retries.",
            interrupt_text=f"Operation interrupted: handling API error ({error_type}: {agent._clean_error_message(str(api_error))}).",
        ))

    _ce = route_classified_error(
        agent, api_error, classified, _retry, error_msg=error_msg, error_context=error_context,
        recovered_with_pool=recovered_with_pool, base_url=_base, model=_model, messages=messages,
        api_messages=api_messages, system_message=system_message,
        active_system_prompt=active_system_prompt, conversation_history=conversation_history,
        retry_count=retry_count, max_retries=max_retries, compression_attempts=compression_attempts,
        max_compression_attempts=max_compression_attempts, api_call_count=api_call_count,
        effective_task_id=effective_task_id,
    )
    status_code = _ce.status_code
    messages = _ce.messages
    active_system_prompt = _ce.active_system_prompt
    conversation_history = _ce.conversation_history
    retry_count = _ce.retry_count
    max_retries = _ce.max_retries
    compression_attempts = _ce.compression_attempts
    is_rate_limited = _ce.is_rate_limited
    _wrapped_output_cap_budget = _ce.wrapped_output_cap_budget
    _is_zai_coding_overload = _ce.is_zai_coding_overload
    if _ce.provider_overflow_recovery_pending:
        _provider_overflow_recovery_pending = True
    if _ce.action != "fallthrough":
        return _verdict(_ce.action, _ce.result)

    _ov = recover_from_overflow(
        agent, api_error, classified, _retry, status_code=status_code, error_msg=error_msg,
        wrapped_output_cap_budget=_wrapped_output_cap_budget, messages=messages,
        api_messages=api_messages, system_message=system_message,
        active_system_prompt=active_system_prompt, conversation_history=conversation_history,
        approx_tokens=approx_tokens, compression_attempts=compression_attempts,
        max_compression_attempts=max_compression_attempts, api_call_count=api_call_count,
        effective_task_id=effective_task_id,
    )
    messages = _ov.messages
    active_system_prompt = _ov.active_system_prompt
    conversation_history = _ov.conversation_history
    approx_tokens = _ov.approx_tokens
    compression_attempts = _ov.compression_attempts
    is_context_length_error = _ov.is_context_length_error
    if _ov.provider_overflow_recovery_pending:
        _provider_overflow_recovery_pending = True
    if _ov.action != "fallthrough":
        return _verdict(_ov.action, _ov.result)

    _ue = settle_unrecovered_error(
        agent, api_error=api_error, classified=classified, _retry=_retry, status_code=status_code,
        error_msg=error_msg, is_context_length_error=is_context_length_error,
        is_rate_limited=is_rate_limited, _is_zai_coding_overload=_is_zai_coding_overload,
        _provider=_provider, _base=_base, _model=_model, messages=messages,
        api_messages=api_messages, api_kwargs=api_kwargs, active_system_prompt=active_system_prompt,
        conversation_history=conversation_history, approx_tokens=approx_tokens,
        retry_count=retry_count, max_retries=max_retries, compression_attempts=compression_attempts,
        api_call_count=api_call_count, effective_task_id=effective_task_id, turn_id=turn_id,
    )
    active_system_prompt = _ue.active_system_prompt
    retry_count = _ue.retry_count
    compression_attempts = _ue.compression_attempts
    return _verdict(_ue.action, _ue.result)


def _is_local_validation_error(api_error: Any) -> bool:
    """ValueError/TypeError are local bugs, except: UnicodeEncodeError (surrogate recovery
    path), json.JSONDecodeError (transient provider/network failure, must retry),
    ssl.SSLError (inherits OSError *and* ValueError — a TLS failure is not a local bug)
    and "NoneType is not iterable" TypeErrors (upstream shape mismatches, e.g. Codex
    response.completed.output=null — retryable so the fallback path runs)."""
    if not isinstance(api_error, (ValueError, TypeError)):
        return False
    if isinstance(api_error, (UnicodeEncodeError, json.JSONDecodeError, ssl.SSLError)):
        return False
    _text = str(api_error).lower()
    return not (isinstance(api_error, TypeError) and "nonetype" in _text and "not iterable" in _text)


# Non-retryable per the classifier, yet handled by the overflow/backoff paths instead.
_RETRYABLE_CLIENT_REASONS = frozenset({
    FailoverReason.rate_limit, FailoverReason.overloaded, FailoverReason.context_overflow,
    FailoverReason.payload_too_large, FailoverReason.long_context_tier, FailoverReason.thinking_signature,
})


@dataclass
class UnrecoveredErrorVerdict:
    """``action``: ``"continue"`` (retry), ``"break"`` (fallback armed / redirect pending) or
    ``"return"`` (``result`` is the terminal result dict). Rebinds ``active_system_prompt``,
    ``retry_count`` and ``compression_attempts``."""

    action: str
    active_system_prompt: Any
    retry_count: Any
    compression_attempts: Any
    result: Optional[Dict[str, Any]] = None


def settle_unrecovered_error(
    agent: Any, *, api_error: Any, classified: Any, _retry: Any, status_code: Any, error_msg: Any,
    is_context_length_error: Any, is_rate_limited: Any, _is_zai_coding_overload: Any,
    _provider: Any, _base: Any, _model: Any, messages: Any, api_messages: Any, api_kwargs: Any,
    active_system_prompt: Any, conversation_history: Any, approx_tokens: Any, retry_count: Any,
    max_retries: Any, compression_attempts: Any, api_call_count: Any, effective_task_id: Any,
    turn_id: Any,
) -> UnrecoveredErrorVerdict:
    """Decide the fate of an API error that every recovery chain declined: local validation /
    non-retryable client errors (Copilot stale-credential self-heal first, then fallback, then a
    terminal result), max-retries exhaustion (primary transport recovery -> fallback -> terminal
    result), else the interruptible error backoff. ``FailoverReason.billing`` (402) is deliberately
    treated as non-retryable (#31273)."""
    from agent.conversation_loop import (
        _arm_fallback_restart, _is_copilot_provider, _is_stale_copilot_credential_error
    )

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> UnrecoveredErrorVerdict:
        return UnrecoveredErrorVerdict(
            action=action, active_system_prompt=active_system_prompt, retry_count=retry_count,
            compression_attempts=compression_attempts, result=result,
        )

    # ``FailoverReason.billing`` (402) is deliberately NOT excluded: pool rotation and
    # eager fallback already gave up, so retrying only burns paid requests on a depleted
    # balance. Mirrors 401/403.
    is_client_error = (
        _is_local_validation_error(api_error)
        or (
            not classified.retryable
            and not classified.should_compress
            and classified.reason not in _RETRYABLE_CLIENT_REASONS
        )
    ) and not is_context_length_error

    if is_client_error:
        # Copilot self-heal BEFORE fallback: a stale credential yields a 400
        # ``model_not_available_for_integrator`` / ``model_not_supported``, not a 401.
        # Fresh token + client rebuild, one retry, SAME provider.
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
                return _verdict("continue")
        # Announce the fallback only when a chain exists, else "trying fallback..." lies
        # before a silent abort.
        if agent._has_pending_fallback():
            _label = _NONRETRYABLE_LABELS.get(classified.reason, f"Non-retryable error (HTTP {status_code})")
            agent._buffer_status(f"⚠️ {_label} — trying fallback...")
        if agent._try_activate_fallback():
            # Direct ``return _verdict("break")`` is load-bearing: the restart handler
            # re-runs the pre-API preflight against the fallback's context window.
            active_system_prompt = _arm_fallback_restart(agent, api_messages, active_system_prompt, _retry)
            retry_count = compression_attempts = 0
            return _verdict("break")
        return _verdict("return", nonretryable_client_error_result(
            agent, api_error, classified, status_code=status_code, api_kwargs=api_kwargs,
            api_messages=api_messages, messages=messages, conversation_history=conversation_history,
            api_call_count=api_call_count, approx_tokens=approx_tokens, provider=_provider,
            base_url=_base, model=_model, effective_task_id=effective_task_id, turn_id=turn_id,
        ))

    if retry_count >= max_retries:
        # Before fallback, rebuild the primary client once per API call block for
        # transient transport errors (stale pool, TCP reset).
        if not _retry.primary_recovery_attempted and agent._try_recover_primary_transport(
            api_error, retry_count=retry_count, max_retries=max_retries,
        ):
            _retry.primary_recovery_attempted = True
            retry_count = 0
            # Fresh attempt cycle: re-open fallback state so a follow-on 429 can still
            # activate fallback_providers.
            _retry.has_retried_429 = False
            agent._fallback_index = 0
            agent._fallback_activated = False
            return _verdict("continue")
        if agent._has_pending_fallback():
            agent._buffer_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
        if agent._try_activate_fallback():
            # Direct ``return _verdict("break")`` is load-bearing: the restart handler
            # re-runs the pre-API preflight against the fallback's context window.
            active_system_prompt = _arm_fallback_restart(agent, api_messages, active_system_prompt, _retry)
            retry_count = compression_attempts = 0
            return _verdict("break")
        return _verdict("return", max_retries_exhausted_result(
            agent, api_error, classified, max_retries=max_retries, is_rate_limited=is_rate_limited,
            error_msg=error_msg, api_kwargs=api_kwargs, api_messages=api_messages,
            messages=messages, conversation_history=conversation_history,
            api_call_count=api_call_count, approx_tokens=approx_tokens, provider=_provider,
            base_url=_base, model=_model, effective_task_id=effective_task_id, turn_id=turn_id,
        ))

    wait_time = compute_error_backoff(
        agent, api_error, retry_count=retry_count, max_retries=max_retries,
        is_rate_limited=is_rate_limited, is_zai_coding_overload=_is_zai_coding_overload,
        base_url=_base, model=_model,
    )
    # Same preserve-redirect rule as the invalid-response wait: a steering correction
    # must survive backoff, not die as "Operation interrupted".
    _interrupted = interruptible_backoff_sleep(
        agent, wait_time, _retry, messages=messages, conversation_history=conversation_history,
        api_call_count=api_call_count,
        abort_message="Interrupt detected during retry wait, aborting.",
        interrupt_text=f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
        activity_label=f"error retry backoff ({retry_count}/{max_retries})",
    )
    if _interrupted is not None:
        return _verdict("return", _interrupted)
    if _retry.restart_with_redirected_messages:
        # Leave the retry loop — the caller rebuilds this iteration from the correction
        # instead of re-firing the stale request.
        return _verdict("break")
    return _verdict("fallthrough")
