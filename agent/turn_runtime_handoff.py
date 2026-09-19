"""Whole-turn runtimes and the replay-safe hand-off between them and the generic wire.

``codex_app_server`` and ``claude_agent_sdk`` hand the entire turn to an external runtime,
bypassing the generic API loop in :func:`agent.conversation_loop.run_conversation`. Claude
may cross to the shared provider-fallback chain only when its result proves the request is
replay-safe (nothing reached the user and no tool ran); real request counts and iteration
consumption stay cumulative across the hand-off so the turn's budget and final provenance
are honest. (#25267)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)


def _sdk_result_failover_reason(result: Dict[str, Any]) -> Optional[FailoverReason]:
    """Return a validated provider failure that is safe to replay elsewhere."""
    reason_value = result.get("failover_reason")
    if not reason_value or not result.get("failed") or result.get("interrupted"):
        return None
    effects = result.get("sdk_effects")
    if isinstance(effects, dict) and any(bool(value) for value in effects.values()):
        logger.warning("claude_agent_sdk declined provider handoff after observable turn effects")
        return None
    try:
        return reason_value if isinstance(reason_value, FailoverReason) else FailoverReason(str(reason_value))
    except (TypeError, ValueError):
        logger.warning("claude_agent_sdk returned unknown failover reason %r; not crossing provider boundary",
                       reason_value)
        return None


def _sdk_handoff_is_at_untouched_user_boundary(
    agent: Any, messages: List[Dict[str, Any]], current_turn_user_idx: int,
) -> bool:
    """Whether an SDK fallback cannot replay output or a tool side effect."""
    return (
        0 <= current_turn_user_idx == len(messages) - 1
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "user"
        and not bool(getattr(agent, "_current_streamed_assistant_text", ""))
    )


def _with_runtime_attempt_provenance(agent: Any, result: Dict[str, Any], api_calls: int) -> Dict[str, Any]:
    """Attach cumulative accounting and the runtime that produced the result."""
    enriched = dict(result)
    enriched["api_calls"] = api_calls
    enriched["model"] = agent.model
    enriched["provider"] = agent.provider
    return enriched


@dataclass
class RuntimeHandoffState:
    """Cumulative accounting across the whole-turn runtime attempts of one turn."""

    api_calls: int = 0
    # The first SDK result that qualified for a provider hand-off: its error and final
    # response are what the user should see if every later attempt is exhausted too.
    first_actionable_sdk_result: Optional[Dict[str, Any]] = None

    def note(self, sdk_result: Dict[str, Any], reason: Optional[FailoverReason]) -> None:
        if reason is not None and self.first_actionable_sdk_result is None:
            self.first_actionable_sdk_result = sdk_result

    def exhaustion_result(self, current_result: Dict[str, Any]) -> Dict[str, Any]:
        if self.first_actionable_sdk_result is None or _sdk_result_failover_reason(current_result) is None:
            return current_result
        preserved = dict(current_result)
        preserved["error"] = self.first_actionable_sdk_result.get("error")
        preserved["final_response"] = self.first_actionable_sdk_result.get("final_response")
        return preserved


def _run_sdk_turn(agent: Any, *, user_message, original_user_message, messages, effective_task_id,
                  _should_review_memory) -> Dict[str, Any]:
    return agent._run_claude_agent_sdk_turn(
        user_message=user_message, original_user_message=original_user_message, messages=messages,
        effective_task_id=effective_task_id, should_review_memory=_should_review_memory,
    )


def _budget_exhausted(agent: Any, api_call_count: int) -> bool:
    return api_call_count >= agent.max_iterations or agent.iteration_budget.remaining <= 0


def _activate_fallback(agent: Any, active_system_prompt: Any, reason: Optional[FailoverReason] = None) -> Optional[Any]:
    """Switch to the next provider in the fallback chain; the refreshed prompt, or None if none is left."""
    activated = agent._try_activate_fallback(reason) if reason is not None else agent._try_activate_fallback()
    if not activated:
        return None
    from agent.conversation_loop import _sync_failover_system_message

    return _sync_failover_system_message(agent, [], active_system_prompt)


@dataclass
class WholeTurnVerdict:
    """``action``: ``"return"`` (the runtime finished the turn — ``result`` is the turn result) or
    ``"fallthrough"`` (a replay-safe provider fallback left the whole-turn lane: continue on the
    generic wire with the cumulative ``api_call_count``)."""

    action: str
    result: Any
    active_system_prompt: Any
    api_call_count: Any


def run_whole_turn_runtime(
    agent: Any, *, user_message: Any, original_user_message: Any, messages: Any, effective_task_id: Any,
    _should_review_memory: Any, active_system_prompt: Any, api_call_count: Any, _runtime_handoff: Any,
) -> WholeTurnVerdict:
    """Hand the turn to the whole-turn runtime selected by ``agent.api_mode`` (none: fallthrough)."""
    handoff: RuntimeHandoffState = _runtime_handoff

    def _verdict(action: str, result: Any = None) -> WholeTurnVerdict:
        return WholeTurnVerdict(action=action, result=result, active_system_prompt=active_system_prompt,
                                api_call_count=api_call_count)

    while agent.api_mode in {"codex_app_server", "claude_agent_sdk"}:
        if agent.api_mode == "codex_app_server":
            whole_turn_result = agent._run_codex_app_server_turn(
                user_message=user_message, original_user_message=original_user_message,
                messages=messages, effective_task_id=effective_task_id,
                should_review_memory=_should_review_memory,
            )
            return _verdict("return", _with_runtime_attempt_provenance(
                agent, whole_turn_result, handoff.api_calls + int(whole_turn_result.get("api_calls", 0) or 0)))

        sdk_result = _run_sdk_turn(
            agent, user_message=user_message, original_user_message=original_user_message,
            messages=messages, effective_task_id=effective_task_id, _should_review_memory=_should_review_memory,
        )
        sdk_calls = int(sdk_result.get("api_calls", 0) or 0)
        handoff.api_calls += sdk_calls
        for _ in range(sdk_calls):
            agent.iteration_budget.consume()
        agent._api_call_count = handoff.api_calls
        sdk_reason = _sdk_result_failover_reason(sdk_result)
        handoff.note(sdk_result, sdk_reason)
        if sdk_reason is None or _budget_exhausted(agent, handoff.api_calls):
            return _verdict("return", _with_runtime_attempt_provenance(
                agent, handoff.exhaustion_result(sdk_result), handoff.api_calls))
        refreshed_prompt = _activate_fallback(agent, active_system_prompt, sdk_reason)
        if refreshed_prompt is None:
            return _verdict("return", _with_runtime_attempt_provenance(
                agent, handoff.exhaustion_result(sdk_result), handoff.api_calls))
        active_system_prompt = refreshed_prompt
    api_call_count = handoff.api_calls
    return _verdict("fallthrough")


@dataclass
class SdkFallbackVerdict:
    """``action``: ``"continue"`` (a fallback provider took over — re-enter the loop), ``"break"``
    (the turn ends: the hand-off was blocked after visible effects and no fallback is left) or
    ``"return"`` (the SDK produced the turn result, or the iteration budget is exhausted —
    ``result`` set)."""

    action: str
    result: Any
    active_system_prompt: Any
    api_call_count: Any
    failed: Any
    final_response: Any
    _turn_exit_reason: Any


def run_sdk_fallback_iteration(
    agent: Any, *, user_message: Any, original_user_message: Any, messages: Any, effective_task_id: Any,
    _should_review_memory: Any, current_turn_user_idx: Any, active_system_prompt: Any, api_call_count: Any,
    _provider_fallback_call_refunds: Any, failed: Any, final_response: Any, _turn_exit_reason: Any,
    _iteration_budget_consumed: Any, _runtime_handoff: Any,
) -> SdkFallbackVerdict:
    """One generic-loop iteration whose provider is the Claude SDK: a fallback INTO the SDK after
    another provider failed. The SDK may only take over at an untouched user boundary, so the
    current request is never replayed after output or a tool effect reached the user."""
    handoff: RuntimeHandoffState = _runtime_handoff

    def _verdict(action: str, result: Any = None) -> SdkFallbackVerdict:
        return SdkFallbackVerdict(
            action=action, result=result, active_system_prompt=active_system_prompt,
            api_call_count=api_call_count, failed=failed, final_response=final_response,
            _turn_exit_reason=_turn_exit_reason,
        )

    def _refund_reserved_iteration() -> None:
        # begin_iteration reserved a logical request that this route did not perform. The grace
        # call consumes its flag instead of the budget, so it is re-armed rather than refunded.
        nonlocal api_call_count
        api_call_count -= 1
        if _iteration_budget_consumed:
            agent.iteration_budget.refund()
        else:
            agent._budget_grace_call = True

    if not _sdk_handoff_is_at_untouched_user_boundary(agent, messages, current_turn_user_idx):
        _refund_reserved_iteration()
        agent._buffer_status(
            "⚠️ Skipping Claude SDK fallback after turn output or tool effects to avoid replaying "
            "the current user request..."
        )
        refreshed_prompt = _activate_fallback(agent, active_system_prompt)
        if refreshed_prompt is not None:
            active_system_prompt = refreshed_prompt
            return _verdict("continue")
        failed = True
        _turn_exit_reason = "sdk_fallback_blocked_after_turn_effects"
        final_response = (
            "The current provider failed after producing output or tool effects, so the SDK "
            "fallback was not replayed."
        )
        return _verdict("break")

    sdk_result = _run_sdk_turn(
        agent, user_message=user_message, original_user_message=original_user_message,
        messages=messages, effective_task_id=effective_task_id, _should_review_memory=_should_review_memory,
    )
    sdk_calls = int(sdk_result.get("api_calls", 0) or 0)
    if sdk_calls == 0:
        _refund_reserved_iteration()
    elif sdk_calls > 1:
        # begin_iteration already accounted for one request; the SDK may have made more.
        api_call_count += sdk_calls - 1
        for _ in range(sdk_calls - 1):
            agent.iteration_budget.consume()
    agent._api_call_count = api_call_count
    sdk_reason = _sdk_result_failover_reason(sdk_result)
    handoff.note(sdk_result, sdk_reason)
    total_calls = api_call_count + _provider_fallback_call_refunds
    if sdk_reason is None or _budget_exhausted(agent, api_call_count):
        return _verdict("return", _with_runtime_attempt_provenance(
            agent, handoff.exhaustion_result(sdk_result), total_calls))
    refreshed_prompt = _activate_fallback(agent, active_system_prompt, sdk_reason)
    if refreshed_prompt is None:
        return _verdict("return", _with_runtime_attempt_provenance(
            agent, handoff.exhaustion_result(sdk_result), total_calls))
    active_system_prompt = refreshed_prompt
    return _verdict("continue")
