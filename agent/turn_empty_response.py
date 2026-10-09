"""Empty / thinking-only final-response recovery ladder for the conversation turn loop.

Runs when the model returned no visible text after ``<think>`` blocks. Ladder order is
load-bearing: partial-stream recovery → reuse prior turn content (housekeeping tools only)
→ one post-tool-call nudge → thinking-only prefill continuation (×2) → empty-response
retries (budgeted, deterministic-empty short-circuit) → fallback provider → terminal
``(empty)`` sentinel. Machinery turns (``display_kind=internal_notification``) are not
owed a reply, so their paid retry rung collapses to zero (#135816); the free repair
paths above it stay available. Nothing here imports ``agent.conversation_loop`` at
module level.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent import empty_response_guard as _empty_guard
from agent.message_metadata import append_message
from agent.turn_context_compaction import _refund_api_call
from agent.turn_recovery import interruptible_backoff_sleep

logger = logging.getLogger("agent.conversation_loop")

_INLINE_THINK_RE = re.compile(r'<think>|<thinking>|<reasoning>', re.IGNORECASE)

# The persisted user-row kinds the gateway types as self-injected machinery turns
# (gateway/response_filters.py MACHINERY_DISPLAY_KINDS). agent/ cannot import
# gateway/ (it imports agent/, and the layering is one-way), so the one kind the
# gateway produces is mirrored here; a drift is caught by the gateway marker tests.
MACHINERY_TURN_DISPLAY_KIND = "internal_notification"
MACHINERY_TURN_DISPLAY_KINDS = frozenset({MACHINERY_TURN_DISPLAY_KIND})

# Agent-object slot for the current turn's persisted user-row kind. Staged by
# ``turn_context._reset_turn_scoped_state`` (the ``display_kind`` key lives on the
# user row only, and the recovery ladder sees ``messages`` after compaction may have
# rewritten history) and cleared by the same reset on every later turn.
_TURN_DISPLAY_KIND_ATTR = "_machinery_turn_display_kind"


def current_turn_display_kind(agent: Any, explicit: Optional[str] = None) -> Optional[str]:
    """The current turn's persisted user-row kind, from the caller's explicit value when
    given (a direct ladder caller holding the staged row) else the slot
    ``turn_context`` staged at reset. Only machinery kinds survive the check: human
    turns and a caller that skipped the turn prologue both read as None."""
    value = explicit if explicit is not None else getattr(agent, _TURN_DISPLAY_KIND_ATTR, None)
    return value if value in MACHINERY_TURN_DISPLAY_KINDS else None


def _reset_turn_scoped_state(agent: Any, user_msg: Any) -> None:
    """Stage the just-built user row's ``display_kind`` onto the agent (and clear the
    slot when untyped). Called after ``_reset_per_turn_agent_state`` staged the user
    message, so the slot always describes the turn that is about to run."""
    kind = user_msg.get("display_kind") if isinstance(user_msg, dict) else None
    setattr(agent, _TURN_DISPLAY_KIND_ATTR, kind if kind in MACHINERY_TURN_DISPLAY_KINDS else None)


@dataclass
class EmptyResponseVerdict:
    """Outcome of ``recover_empty_response``.

    ``action``: ``"break"`` (turn is done — ``final_response`` is set), ``"continue"``
    (re-enter the OUTER turn loop: a nudge/prefill row was appended, a retry wait
    elapsed, or a fallback was activated and preflight must re-run), ``"return"``
    (interrupted during a retry wait — return ``result``) or ``"fallthrough"``
    (unreachable: every path exits; kept for the contract)."""

    action: str
    result: Optional[dict[str, Any]]
    final_response: Any
    turn_exit_reason: Any
    active_system_prompt: Any
    preflight_compression_blocked: bool
    api_call_count: int


def _retry_empty(
    agent: Any, response: Any, finish_reason: str, empty_candidate: bool, *, messages: Any,
    conversation_history: Any, api_call_count: int, observed_generation: bool = False,
    display_kind: Optional[str] = None,
) -> tuple:
    """Budgeted empty-response retry. Each empty attempt re-bills the full input, so the
    signature is recorded and deterministic empties stop burning paid retries (fails
    open: missing usage or any output keeps the budget). A machinery turn
    (``display_kind=internal_notification``, #82888) is not owed a reply, so its paid
    budget collapses to zero (#135816: every paid retry on those turns came back empty
    too, re-billing the full input with nobody waiting) — the attempts are still
    recorded so the deterministic-empty short-circuit keeps its evidence, and the free
    repair paths (nudge, prefill, fallback) stay available. Returns
    ``(action_or_None, interrupt_result, deterministic_empty)``."""
    from agent.retry_utils import jittered_backoff

    if empty_candidate:
        _empty_guard.record_empty_attempt(
            agent, finish_reason=finish_reason, response=response, observed_generation=observed_generation,
        )
    budget = (
        _empty_guard.empty_retry_budget(agent, response)
        if empty_candidate else _empty_guard.DEFAULT_EMPTY_RETRY_BUDGET
    )
    if empty_candidate and current_turn_display_kind(agent, display_kind):
        budget = 0
    deterministic = empty_candidate and _empty_guard.deterministic_empty(agent)
    if not (empty_candidate and agent._empty_content_retries < budget and not deterministic):
        return None, None, deterministic
    agent._empty_content_retries += 1
    n = agent._empty_content_retries
    wait_time = jittered_backoff(n, base_delay=5.0, max_delay=60.0)
    logger.warning(
        "Empty response (no content or reasoning) — retry %d/%d in %.1fs (model=%s)",
        n, budget, wait_time, agent.model,
    )
    _budget_note = (
        " — high-cost request, reduced retry budget"
        if budget < _empty_guard.DEFAULT_EMPTY_RETRY_BUDGET else ""
    )
    agent._buffer_diagnostic_status(
        f"⚠️ Empty response from model — retrying ({n}/{budget}) in {wait_time:.0f}s{_budget_note}"
    )
    _interrupted = interruptible_backoff_sleep(
        agent, wait_time, None,
        messages=messages,
        conversation_history=conversation_history,
        api_call_count=api_call_count,
        abort_message="Interrupt detected during empty-response retry wait, aborting.",
        interrupt_text=(
            f"Operation interrupted: retrying empty response from model (retry {n}/{budget})."
        ),
        activity_label=f"empty response retry backoff ({n}/{budget})",
    )
    if _interrupted is not None:
        return "return", _interrupted, deterministic
    return "continue", None, deterministic


def _terminal_empty(agent: Any, assistant_message: Any, finish_reason: str, messages: Any) -> str:
    """Retries and fallback exhausted: persist the ``(empty)`` sentinel row and return the
    delivery text. Reasoning is never copied into the delivery text here — the persisted row
    keeps the sentinel so later "continue" turns don't replay it and loop on empties."""
    _streak_cost = _empty_guard.streak_cost_usd(agent)
    if _streak_cost is not None:
        agent._buffer_diagnostic_status(
            f"ℹ️ Estimated cost of these empty attempts: ~${_streak_cost:.2f} (input tokens are billed "
            f"per attempt even when no answer is produced)"
        )
    agent._flush_status_buffer()
    reasoning_text = agent._extract_reasoning(assistant_message)
    agent._drop_trailing_empty_response_scaffolding(messages)
    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
    assistant_msg["content"] = "(empty)"
    assistant_msg["_empty_terminal_sentinel"] = True
    append_message(messages, assistant_msg)

    if not reasoning_text:
        logger.warning(
            "Empty response (no content or reasoning) after %d retries. No fallback available. "
            "model=%s provider=%s",
            agent._empty_content_retries, agent.model,
            agent.provider,
        )
        agent._emit_diagnostic_status(
            "❌ Model returned no content after all retries"
            + (" and fallback attempts." if agent._fallback_chain else
               ". No fallback providers configured.")
        )
        return "(empty)"

    logger.warning(
        "Reasoning-only response after %d retries; keeping it out of the terminal result "
        "(model=%s provider=%s)",
        agent._empty_content_retries, agent.model, agent.provider,
    )
    agent._emit_diagnostic_status(
        "⚠️ Model produced reasoning but no visible response after all retries. Returning empty."
    )
    return "(empty)"


def recover_empty_response(
    agent: Any, assistant_message: Any, response: Any, finish_reason: str, *, final_response: Any,
    messages: list[dict[str, Any]], api_messages: Any, conversation_history: Any,
    active_system_prompt: Any, api_call_count: int, turn_exit_reason: Any,
    preflight_compression_blocked: bool,
) -> EmptyResponseVerdict:
    """Recover from a final response with no visible content (see module docstring for
    the ladder). Role alternation is preserved: the post-tool nudge appends the empty
    assistant row BEFORE the user-level hint (APIs reject tool→user)."""
    from agent.conversation_loop import _EMPTY_TOOL_RESPONSE_NUDGE, _sync_failover_system_message

    _turn_exit_reason = turn_exit_reason
    _preflight_compression_blocked = preflight_compression_blocked

    def _verdict(action: str, result: Optional[dict[str, Any]] = None) -> EmptyResponseVerdict:
        return EmptyResponseVerdict(
            action=action, result=result, final_response=final_response,
            turn_exit_reason=_turn_exit_reason, active_system_prompt=active_system_prompt,
            preflight_compression_blocked=_preflight_compression_blocked,
            api_call_count=api_call_count,
        )

    # Partial stream recovery: content streamed before the connection died becomes the
    # final response instead of fallback or retries.
    _partial_streamed = getattr(agent, "_current_streamed_assistant_text", "") or ""
    if agent._has_content_after_think_block(_partial_streamed):
        _turn_exit_reason = "partial_stream_recovery"
        _recovered = agent._strip_think_blocks(_partial_streamed).strip()
        logger.info(
            "Partial stream content delivered (%d chars) — using as final response",
            len(_recovered),
        )
        agent._emit_diagnostic_status("↻ Stream interrupted — using delivered content " "as final response")
        final_response = _recovered
        # A streamed fragment isn't a confirmed preview: gateway fallback delivery
        # sends the text plus the abnormal-turn explanation.
        agent._response_was_previewed = False
        return _verdict("break")

    # Prior turn had real content + ONLY housekeeping tools: model is done, reuse it.
    # With substantive tools it was mid-task narration and the empty reply is a choke;
    # let the post-tool nudge handle it.
    fallback = getattr(agent, '_last_content_with_tools', None)
    if fallback and getattr(agent, '_last_content_tools_all_housekeeping', False):
        _turn_exit_reason = "fallback_prior_turn_content"
        logger.info("Empty follow-up after tool calls — using prior turn content as final response")
        agent._emit_diagnostic_status("↻ Empty response after tool calls — using earlier content as final answer")
        agent._last_content_with_tools = None
        agent._last_content_tools_all_housekeeping = False
        agent._empty_content_retries = 0
        # Do NOT modify the assistant message content (injected text poisoned history).
        final_response = agent._strip_think_blocks(fallback).strip()
        agent._response_was_previewed = True
        # The final IS the response streamed before the housekeeping tool: name that identity
        # so a client settles the text it already shows instead of painting it a second time.
        agent._reused_response_text = final_response
        return _verdict("break")

    # Post-tool-call empty (no prior content, or only mid-task narration): nudge once.
    _prior_was_tool = any(m.get("role") == "tool" for m in messages[-5:])
    # Ollama puts <think> in content, not reasoning_content, so _has_structured misses
    # it; detect here to route to prefill.
    _has_inline_thinking = bool(_INLINE_THINK_RE.search(final_response or ""))
    if (
        _prior_was_tool
        and not getattr(agent, "_post_tool_empty_retried", False)
        and not _has_inline_thinking  # thinking model still working — let prefill handle
    ):
        agent._post_tool_empty_retried = True
        # Clear stale narration so it doesn't resurface on a later empty response.
        agent._last_content_with_tools = None
        agent._last_content_tools_all_housekeeping = False
        logger.info("Empty response after tool calls — nudging model " "to continue processing")
        agent._buffer_diagnostic_status("⚠️ Model returned empty after tool calls — " "nudging to continue")
        # tool → assistant("(empty)") → user keeps the sequence valid.
        _nudge_msg = agent._build_assistant_message(assistant_message, finish_reason)
        _nudge_msg["content"] = "(empty)"
        _nudge_msg["_empty_recovery_synthetic"] = True
        append_message(messages, _nudge_msg)
        append_message(messages, {
            "role": "user", "content": _EMPTY_TOOL_RESPONSE_NUDGE, "_empty_recovery_synthetic": True
        })
        return _verdict("continue")

    # Thinking-only prefill: append the reasoning as-is and continue so the model sees
    # its own reasoning and writes text.
    _has_structured = bool(
        getattr(assistant_message, "reasoning", None)
        or getattr(assistant_message, "reasoning_content", None)
        or getattr(assistant_message, "reasoning_details", None)
        or _has_inline_thinking
    )
    if _has_structured and agent._thinking_prefill_retries < 2:
        agent._thinking_prefill_retries += 1
        logger.info(
            "Thinking-only response (no visible content) — prefilling to continue (%d/2)",
            agent._thinking_prefill_retries,
        )
        agent._buffer_diagnostic_status(
            f"↻ Thinking-only response — prefilling to continue ({agent._thinking_prefill_retries}/2)"
        )
        interim_msg = agent._build_assistant_message(assistant_message, "incomplete")
        interim_msg["_thinking_prefill"] = True
        append_message(messages, interim_msg)
        agent._session_messages = messages
        return _verdict("continue")

    # Empty-response retries: truly empty replies AND reasoning-only replies after
    # prefill exhaustion.
    _truly_empty = not agent._strip_think_blocks(final_response).strip()
    _empty_candidate = _truly_empty and (not _has_structured or agent._thinking_prefill_retries >= 2)
    action, interrupt_result, _deterministic_empty = _retry_empty(
        agent, response, finish_reason, _empty_candidate, messages=messages,
        conversation_history=conversation_history, api_call_count=api_call_count,
        observed_generation=_has_structured,
        display_kind=current_turn_display_kind(agent),
    )
    if action is not None:
        return _verdict(action, interrupt_result)

    if _truly_empty and _deterministic_empty:
        logger.warning(
            "Repeated empty response detected (model=%s provider=%s finish_reason=%s) — "
            "skipping remaining retries",
            agent.model, agent.provider, finish_reason,
        )
        agent._buffer_diagnostic_status(
            "⚠️ Model is repeatedly returning empty content — skipping further retries "
            "to avoid repeat charges"
        )

    # Exhausted retries — try the next provider in the chain before "(empty)".
    if _truly_empty and agent._fallback_chain:
        logger.warning(
            "Empty response after %d retries — attempting fallback (model=%s, provider=%s)",
            agent._empty_content_retries, agent.model, agent.provider,
        )
        agent._buffer_diagnostic_status("⚠️ Model returning empty responses — " "switching to fallback provider...")
        if agent._try_activate_fallback():
            active_system_prompt = _sync_failover_system_message(agent, api_messages, active_system_prompt)
            agent._empty_content_retries = 0
            agent._buffer_diagnostic_status(f"↻ Switched to fallback: {agent.model} " f"({agent.provider})")
            logger.info(
                "Fallback activated after empty responses: now using %s on %s",
                agent.model, agent.provider,
            )
            # OUTER loop: `continue` re-runs preflight against the fallback's window;
            # `break` would end the turn without calling the fallback.
            _preflight_compression_blocked = False
            # The fallback hop is a provider switch, not a model turn: refund the empty
            # call so a mid-turn fallback doesn't eat the iteration budget (#77305).
            api_call_count = _refund_api_call(agent, api_call_count)
            return _verdict("continue")

    _turn_exit_reason = "empty_response_exhausted"
    final_response = _terminal_empty(agent, assistant_message, finish_reason, messages)
    return _verdict("break")
