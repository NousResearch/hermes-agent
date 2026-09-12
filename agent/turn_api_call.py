"""The provider call for the conversation turn's retry loop: ``nous_rate_limit_guard`` (skip
the attempt while another session's Nous Portal rate limit is active), ``perform_api_call``
(streaming decision, MoA prepared-request handshake, LLM execution middleware wrapper, the
redirect ``_model_request_active`` bracket and the response-vs-redirect crossing check) and
``handle_api_interrupt`` (``InterruptedError`` mid-call). Nothing here imports
``agent.conversation_loop`` at module level (cycle).
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, Optional

from agent.message_metadata import append_message

logger = logging.getLogger("agent.conversation_loop")


def stop_thinking_spinner(agent: Any, thinking_spinner: Any) -> None:
    """Stop the spinner silently and clear the thinking callback; returns ``None`` so
    callers can rebind ``thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)``."""
    if thinking_spinner:
        thinking_spinner.stop("")
    if agent.thinking_callback:
        agent.thinking_callback("")
    return None


@dataclass
class ApiCallVerdict:
    """``action``: ``"fallthrough"`` (``response`` is ready for verification) or ``"break"``
    (a redirect crossed the response — rebuild armed on ``_retry`` or ``interrupted``)."""

    action: str
    response: Any
    response_issuer_model: Any
    thinking_spinner: Any
    interrupted: Any


def _codex_request_for_dispatch(
    agent: Any, request: Any, built_model: Any, api_messages: Any, *,
    built_with_native_compaction: bool, built_input: Any,
) -> tuple[Any, Any]:
    """Bind encrypted replay and response provenance to the final middleware model."""
    from agent.codex_responses_adapter import _wire_model_identity

    dispatched_model = _wire_model_identity(
        request.get("model") if isinstance(request, dict) else None
    )
    if dispatched_model == _wire_model_identity(built_model) or not isinstance(request, dict):
        return request, dispatched_model
    input_items = request.get("input")
    request = dict(request)
    request.pop("context_management", None)
    if (
        built_with_native_compaction
        and isinstance(api_messages, list)
        and input_items == built_input
    ):
        from agent.codex_responses_adapter import classify_responses_route

        route = classify_responses_route(agent)._asdict()
        source_messages = agent._prepare_messages_for_non_vision_model(api_messages)
        if source_messages and source_messages[0].get("role") == "system":
            source_messages = source_messages[1:]
        request["input"] = agent._get_transport().convert_messages(
            source_messages,
            replay_encrypted_reasoning=bool(
                getattr(agent, "_codex_reasoning_replay_enabled", True)
            ),
            base_url=agent.base_url,
            context_management=None,
            model=dispatched_model,
            **route,
        )
        return request, dispatched_model
    if not isinstance(input_items, list):
        return request, dispatched_model
    request["input"] = [
        item for item in input_items
        if not (
            isinstance(item, dict)
            and item.get("type") in {"reasoning", "compaction"}
        )
    ]
    return request, dispatched_model


def _should_stream(agent: Any) -> bool:
    """Streaming is preferred even without consumers (stale-stream / read-timeout health
    checks); disabled on provider signal, ACP schemes, MoA without a display consumer, or
    Mock clients in tests (SimpleNamespace, not stream iterators)."""
    if getattr(agent, "_disable_streaming", False):
        return False
    _base = str(agent.base_url or "").lower()
    if agent.provider in {"copilot-acp"} or _base.startswith(("acp://", "acp+tcp://")):
        return False
    if not agent._has_stream_consumers():
        if agent.provider == "moa":
            return False
        from unittest.mock import Mock
        if isinstance(getattr(agent, "client", None), Mock):
            return False
    return True


def perform_api_call(
    agent: Any, *, api_kwargs: Any, _original_api_kwargs: Any, _llm_middleware_trace: Any,
    _moa_prepared_request: Any, _retry: Any, thinking_spinner: Any, retry_count: Any,
    api_call_count: Any, api_request_id: Any, effective_task_id: Any, turn_id: Any,
    interrupted: Any, api_messages: Any,
) -> ApiCallVerdict:
    """Issue the request (see ``_should_stream`` for the streaming decision)."""
    response = None
    response_issuer_model = None
    built_issuer_model = None
    built_input = None
    built_with_native_compaction = False
    if agent.api_mode == "codex_responses":
        from agent.codex_responses_adapter import _wire_model_identity

        built_issuer_model = _wire_model_identity(
            _original_api_kwargs.get("model")
            if isinstance(_original_api_kwargs, dict)
            else None
        )
        response_issuer_model = _wire_model_identity(
            api_kwargs.get("model") if isinstance(api_kwargs, dict) else None
        )
        built_input = deepcopy(
            _original_api_kwargs.get("input")
            if isinstance(_original_api_kwargs, dict)
            else None
        )
        built_with_native_compaction = (
            isinstance(_original_api_kwargs, dict)
            and "context_management" in _original_api_kwargs
        )

    def _verdict(action: str) -> ApiCallVerdict:
        return ApiCallVerdict(
            action=action, response=response, response_issuer_model=response_issuer_model,
            thinking_spinner=thinking_spinner,
            interrupted=interrupted,
        )

    def _stop_spinner():
        nonlocal thinking_spinner
        thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)

    _use_streaming = _should_stream(agent)

    def _perform_api_call(next_api_kwargs):
        nonlocal response_issuer_model
        if agent.api_mode == "codex_responses":
            next_api_kwargs, response_issuer_model = _codex_request_for_dispatch(
                agent, next_api_kwargs, built_issuer_model, api_messages,
                built_with_native_compaction=built_with_native_compaction,
                built_input=built_input,
            )
            next_api_kwargs = agent._get_transport().preflight_kwargs(
                next_api_kwargs, allow_stream=False, is_github_responses=agent._is_copilot_url(),
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

    # The ``_model_request_active`` bracket is taken under the redirect lock when one exists,
    # so redirect() can't observe a half-toggled flag.
    _model_request_active = getattr(agent, "_model_request_active", None)
    _redirect_lock = getattr(agent, "_pending_redirect_lock", None)
    _bracket = nullcontext() if _redirect_lock is None else _redirect_lock
    with _bracket:
        if _model_request_active is not None:
            _model_request_active.set()
    try:
        response = run_llm_execution_middleware(
            api_kwargs, _perform_api_call, original_request=_original_api_kwargs,
            task_id=effective_task_id, turn_id=turn_id, api_request_id=api_request_id,
            session_id=agent.session_id or "", platform=agent.platform or "", model=agent.model,
            provider=agent.provider, base_url=agent.base_url, api_mode=agent.api_mode,
            api_call_count=api_call_count, middleware_trace=list(_llm_middleware_trace),
        )
    finally:
        with _bracket:
            if _model_request_active is not None:
                _model_request_active.clear()
            _redirect_crossed_response = (
                bool(agent._pending_redirect) if _redirect_lock is not None
                else agent._has_pending_redirect()
            )
    if _redirect_crossed_response:
        # Response and redirect can cross threads: discard the now-stale
        # response and rebuild from the correction rather than lose it.
        thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)
        if agent.clear_interrupt(preserve_redirect=True):
            _retry.restart_with_redirected_messages = True
        else:
            interrupted = True
        return _verdict("break")
    return _verdict("fallthrough")


@dataclass
class ApiInterruptVerdict:
    """Always ``action == "break"`` (leave the retry loop): either a redirect restart was
    armed on ``_retry`` or the turn is ``interrupted`` with ``final_response`` set."""

    action: str
    thinking_spinner: Any
    interrupted: Any
    final_response: Any


def handle_api_interrupt(
    agent: Any, *, _retry: Any, thinking_spinner: Any, messages: Any, conversation_history: Any,
    api_start_time: Any, interrupted: Any, final_response: Any,
) -> ApiInterruptVerdict:
    """``InterruptedError`` during the provider call: a pending redirect keeps its correction
    queued for the outer-loop rebuild; otherwise keep any streamed partial text so the next
    turn has a record of the half-finished reply."""
    from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

    thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)
    # redirect() cancelled only this request: keep the correction queued, clear the
    # cancellation bit, let the outer loop rebuild. Never materialize incomplete
    # signed/encrypted reasoning items.
    if agent._has_pending_redirect() and agent.clear_interrupt(preserve_redirect=True):
        _retry.restart_with_redirected_messages = True
        return ApiInterruptVerdict("break", thinking_spinner, interrupted, final_response)
    api_elapsed = time.time() - api_start_time
    agent._vprint(f"{agent.log_prefix}⚡ Interrupted during API call.", force=True)
    interrupted = True
    _partial = agent._strip_think_blocks(
        getattr(agent, "_current_streamed_assistant_text", "") or ""
    ).strip()
    if _partial:
        append_message(messages, {"role": "assistant", "content": _partial})
        final_response = _partial
    else:
        final_response = f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}{api_elapsed:.1f}s elapsed)."
    agent._persist_session(messages, conversation_history)
    return ApiInterruptVerdict("break", thinking_spinner, interrupted, final_response)


@dataclass
class NousRateGuardVerdict:
    """``action``: ``"fallthrough"`` (no active limit — make the call), ``"break"``
    (fallback armed on ``_retry``) or ``"return"`` (``result``: no fallback available)."""

    action: str
    active_system_prompt: Any
    retry_count: Any
    compression_attempts: Any
    result: Optional[Dict[str, Any]] = None


def nous_rate_limit_guard(
    agent: Any, *, _retry: Any, api_messages: Any, messages: Any, conversation_history: Any,
    active_system_prompt: Any, retry_count: Any, compression_attempts: Any, api_call_count: Any,
) -> NousRateGuardVerdict:
    """Skip the call if another session recorded a Nous Portal rate limit: every attempt (incl.
    SDK retries) counts against RPH. Never lets the guard itself break the agent loop."""
    from agent.conversation_loop import _arm_fallback_restart

    def _verdict(action: str, result: Optional[Dict[str, Any]] = None) -> NousRateGuardVerdict:
        return NousRateGuardVerdict(
            action=action, active_system_prompt=active_system_prompt, retry_count=retry_count,
            compression_attempts=compression_attempts, result=result,
        )

    if agent.provider == "nous":
        # A gateway ``x-nous-model-switch`` recorded on the previous response moves this session
        # (and the config default, when it still names the free tier's model) before the next call.
        try:
            from hermes_cli.anon_auth import apply_model_switch
            apply_model_switch(agent)
        except Exception:
            pass
        try:
            from agent.nous_rate_guard import (
                nous_rate_limit_remaining, format_remaining as _fmt_nous_remaining
            )
            _nous_remaining = nous_rate_limit_remaining()
            if _nous_remaining is not None and _nous_remaining > 0:
                from hermes_cli import anon_auth
                reset = _fmt_nous_remaining(_nous_remaining)
                if anon_auth.route_is_welcome_host(getattr(agent, "base_url", "")):
                    _nous_msg = anon_auth.FREE_TIER_RATE_LIMIT_CHAT.format(reset=reset)
                else:
                    _nous_msg = f"Nous Portal rate limit active — resets in {reset}."
                agent._buffer_vprint(f"⏳ {_nous_msg} Trying fallback...")
                agent._buffer_status(f"⏳ {_nous_msg}")
                if agent._try_activate_fallback():
                    active_system_prompt = _arm_fallback_restart(
                        agent, api_messages, active_system_prompt, _retry)
                    retry_count = 0
                    compression_attempts = 0
                    return _verdict("break")
                # No fallback — surface the buffered rate-limit context that led here.
                agent._flush_status_buffer()
                agent._persist_session(messages, conversation_history)
                return _verdict("return", {
                    "final_response": (
                        f"⏳ {_nous_msg}\n\n"
                        "No fallback provider available. Try again after the reset, or add a "
                        "fallback provider in config.yaml."
                    ),
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "failed": True,
                    "error": _nous_msg,
                })
        except Exception:
            pass  # Never let rate guard break the agent loop
    return _verdict("fallthrough")
