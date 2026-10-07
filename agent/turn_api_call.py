"""The provider call for the conversation turn's retry loop: ``perform_api_call``
(streaming decision, MoA prepared-request handshake, LLM execution middleware wrapper, the
redirect ``_model_request_active`` bracket and the response-vs-redirect crossing check) and
``handle_api_interrupt`` (``InterruptedError`` mid-call). Nothing here imports
``agent.conversation_loop`` at module level (cycle).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, Optional

from agent.error_classifier import FailoverReason
from agent.agent_runtime_helpers import _INTERRUPTED_PLACEHOLDER
from agent.message_metadata import append_message
from agent.repetition_guard import REPETITION_LOOP_INTERRUPTED, is_runaway_repetition
from agent.turn_failure_copy import site_copy, stamp_failure

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
    thinking_spinner: Any
    interrupted: Any


def _should_stream(agent: Any) -> bool:
    """Streaming is preferred even without consumers (stale-stream / read-timeout health
    checks); disabled on provider signal, ACP providers (``acp://`` scheme or an
    external-process provider profile), MoA without a display consumer, or Mock clients in
    tests (SimpleNamespace, not stream iterators)."""
    if getattr(agent, "_disable_streaming", False):
        return False
    _base = str(agent.base_url or "").lower()
    from rabbit_cli.runtime_provider_backends import _is_external_process_provider

    if _base.startswith(("acp://", "acp+tcp://")) or _is_external_process_provider(agent.provider):
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
    interrupted: Any,
) -> ApiCallVerdict:
    """Issue the request (see ``_should_stream`` for the streaming decision)."""
    response = None

    def _verdict(action: str) -> ApiCallVerdict:
        return ApiCallVerdict(
            action=action, response=response, thinking_spinner=thinking_spinner,
            interrupted=interrupted,
        )

    def _stop_spinner():
        nonlocal thinking_spinner
        thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)

    _use_streaming = _should_stream(agent)

    def _perform_api_call(next_api_kwargs):
        if agent.api_mode == "codex_responses":
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

    from rabbit_cli.middleware import run_llm_execution_middleware

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
    # A Stop during the empty-response nudge request leaves the synthetic assistant+nudge
    # pair after an executed tool result; strip it so the row appended below follows the tool
    # row (the finalizer then closes the tail with this exit's own reason).
    agent._drop_trailing_empty_response_scaffolding(messages)
    _partial = agent._strip_think_blocks(
        getattr(agent, "_current_streamed_assistant_text", "") or ""
    ).strip()
    if _partial and is_runaway_repetition(_partial):
        # The interrupted row is replayed next turn; looped bytes there re-seed the loop
        # (#112764). Same hidden shape as the redirect placeholder: nothing visible in the
        # transcript, a neutral api_content so the pre-call sanitizer does not re-heal it.
        append_message(messages, {
            "role": "assistant", "content": "", "display_kind": "hidden",
            "api_content": _INTERRUPTED_PLACEHOLDER,
        })
        final_response = REPETITION_LOOP_INTERRUPTED
    elif _partial:
        append_message(messages, {
            "role": "assistant", "content": _partial, "display_metadata": {"interrupted": True},
        })
        final_response = _partial
    else:
        final_response = f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}{api_elapsed:.1f}s elapsed)."
    agent._persist_session(messages, conversation_history)
    return ApiInterruptVerdict("break", thinking_spinner, interrupted, final_response)
