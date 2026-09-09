"""claude-agent-sdk runtime — the subscription-Claude agent-loop path.

The structural twin of ``agent/codex_runtime.py``'s app-server path: hands the
entire turn to Anthropic's official ``claude-agent-sdk`` (which drives the
Claude Code CLI's own agent loop under subscription OAuth by default, with
known metered lanes refused unless explicitly enabled) and projects its typed
message stream back into Hermes'
messages list so transcript persistence and recall keep working. GitHub
issue #25267.

* ``run_claude_agent_sdk_turn`` — drives one turn through a lazily-created
  ``ClaudeAgentSdkSession`` (used when ``agent.api_mode == "claude_agent_sdk"``).
  This facade orchestrates the phases in their load-bearing order; the phase
  bodies live in the ``claude_sdk_runtime_<topic>`` siblings and share one
  per-turn ``_SdkTurnState``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text
from agent.claude_sdk_runtime_continuity import _persist_turn
from agent.claude_sdk_runtime_fallback import (
    _consume_agent_interrupt,
    _reconcile_turn_outcome,
)
from agent.claude_sdk_runtime_session import (
    _refresh_turn_visibility,
    _run_sdk_attempts,
)
from agent.claude_sdk_runtime_state import _SdkTurnState
from agent.claude_sdk_runtime_usage import _account_turn

logger = logging.getLogger(__name__)


def run_claude_agent_sdk_turn(
    agent,
    *,
    user_message: Any,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """claude-agent-sdk runtime path. Hands the entire turn to the SDK's
    agent loop and projects its messages back into Hermes' list.

    Called from run_conversation() when agent.api_mode == "claude_agent_sdk".
    Returns the same dict shape as the chat_completions path.

    Continuity retire matrix (#25267):
      /new, session expiry      → NEW Hermes session row → no persisted id → fresh
      gateway restart/eviction  → same row, id persisted  → RESUME
      error/timeout retire      → id CLEARED → next turn fresh + digest
      stale/failed resume       → retire → clear → ONE fresh retry with digest

    Phase order is load-bearing: attempt effects are reset BEFORE session
    startup; terminal/stop state is reconciled BEFORE the provider-fallback
    decision; projected messages are flushed BEFORE the resume id is
    persisted. Approval context and the visibility callbacks refresh every
    turn; prompt-append construction is session-creation work
    (``claude_sdk_runtime_session``).
    """
    from agent.transports.claude_agent_sdk_session import _coerce_turn_input

    user_input = _coerce_turn_input(user_message)
    rejection = _reject_empty_turn(agent, user_input, messages)
    if rejection is not None:
        return rejection

    _refresh_approval_turn_context(agent)

    # NOTE: the user message is ALREADY appended to messages by the standard
    # run_conversation() flow before the early return reaches us. Do NOT
    # append again — that would duplicate. (Same contract as codex_runtime.)

    # An interrupt that landed before the SDK session exists (first turn, or
    # right after a retire) only set agent._interrupt_requested — honor it
    # here, mirroring the native loop's top-of-loop check, and consume the
    # flag so the NEXT turn runs normally.
    if getattr(agent, "_interrupt_requested", False):
        return _interrupted_before_start(agent, messages)

    # Stream/replay state belongs to this SDK attempt, never to the cached
    # gateway agent.  Reset before session startup too: an authoritative auth
    # failure may happen before a model call and must not inherit prior output.
    agent._current_streamed_assistant_text = ""
    agent._sdk_issued_tool_effect = False
    state = _SdkTurnState(
        user_input=user_input,
        original_user_message=original_user_message,
        messages=messages,
        messages_before_attempt=copy.deepcopy(messages),
    )

    _refresh_turn_visibility(agent, state)
    failure = _run_sdk_attempts(agent, state)
    if failure is not None:
        return failure
    _reconcile_turn_outcome(agent, state)
    _persist_turn(agent, state)
    _account_turn(agent, state)
    _maybe_spawn_background_review(
        agent, state, should_review_memory=should_review_memory
    )
    return _assemble_turn_result(agent, state)


def _reject_empty_turn(
    agent, user_input: Any, messages: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """The empty-message rejection (no session, no model call), or None to proceed."""
    if not (isinstance(user_input, str) and not user_input.strip()):
        return None
    _consume_agent_interrupt(agent)
    live_session = getattr(agent, "_claude_sdk_session", None)
    if live_session is not None:
        try:
            live_session.consume_interrupt()
        except Exception:
            logger.debug("consume_interrupt failed", exc_info=True)
    rejection = (
        "This Claude Agent SDK route can't process an empty message. "
        "Please send text or a supported image."
    )
    return {
        "final_response": rejection,
        "messages": messages,
        "api_calls": 0,
        "completed": False,
        "partial": True,
        "error": rejection,
        "interrupted": False,
        "agent_persisted": True,
    }


def _refresh_approval_turn_context(agent) -> None:
    # P1.b: refresh the approval-context snapshot EVERY turn (including
    # session-reuse turns). This runs on the agent turn thread, where the
    # session contextvars are visible; the SDK invokes the approval callback
    # from its own loop thread, where they never are — the callback reads
    # this holder instead. A cron turn writes gateway=False (honest deny, no
    # block); a later interactive turn on the SAME SDK session rewrites it
    # and un-freezes the cron-born session.
    try:
        from tools.approval_sdk_gateway import current_approval_turn_context

        agent._sdk_approval_turn_ctx = current_approval_turn_context()
    except Exception:
        logger.debug("approval turn-context refresh failed", exc_info=True)


def _interrupted_before_start(agent, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Honour an interrupt that landed before this turn's session existed."""
    _consume_agent_interrupt(agent)
    live_session = getattr(agent, "_claude_sdk_session", None)
    if live_session is not None:
        # interrupt() also set the live session's event; consume it here
        # or the NEXT legitimate message dies on the stale event with no
        # model call.
        try:
            live_session.consume_interrupt()
        except Exception:
            logger.debug("consume_interrupt failed", exc_info=True)
    return {
        "final_response": "",
        "messages": messages,
        "api_calls": 0,
        "completed": False,
        "partial": True,
        # Without this key the gateway's empty-response normalizer has no
        # branch to take (interrupted absent, api_calls 0, partial True →
        # every arm defeated) and the user's message dies in SILENCE.
        # With it, api_calls==0 + interrupted surfaces the honest
        # "interrupted before processing — send it again" path.
        "interrupted": True,
        "error": None,
        "agent_persisted": True,
    }


def _maybe_spawn_background_review(
    agent, state: _SdkTurnState, *, should_review_memory: bool
) -> None:
    turn = state.turn
    if (
        turn.final_text
        and not turn.interrupted
        and not agent.skip_background_review
        and (should_review_memory or state.should_review_skills)
    ):
        # #25267 suppressed this spawn unconditionally: the fork inherits
        # api_mode="claude_agent_sdk" and early-returns into a fresh SDK
        # session whose tool surface has no `memory` / `skill_manage`, so it
        # burned a subscription turn and wrote nothing.
        #
        # That reasoning only holds when the fork stays on THIS runtime. With
        # auxiliary.background_review naming a concrete different
        # provider/model, _resolve_review_runtime reports routed=True and the
        # review runs over there — on a normal tool surface that can write.
        # Suppressing it in that case is what left this lane unable to record
        # anything durable across sessions.
        #
        # Unrouted, #25267 still applies: skip, and let the counters above
        # keep ticking for a bounded replacement pass.
        routed = False
        try:
            from agent.background_review import _resolve_review_runtime

            routed = bool(_resolve_review_runtime(agent).get("routed"))
        except Exception:
            logger.debug("review-runtime resolve failed", exc_info=True)

        if routed:
            try:
                agent._spawn_background_review(
                    messages_snapshot=list(state.messages),
                    review_memory=should_review_memory,
                    review_skills=state.should_review_skills,
                )
            except Exception:
                logger.debug("background review spawn raised", exc_info=True)
        else:
            logger.debug(
                "claude-sdk runtime: background review skipped "
                "(memory=%s, skills=%s) — the review fork cannot write on "
                "this runtime",
                should_review_memory,
                state.should_review_skills,
            )


def _assemble_turn_result(agent, state: _SdkTurnState) -> Dict[str, Any]:
    turn = state.turn
    result = {
        "final_response": turn.final_text,
        "messages": state.messages,
        "api_calls": int(getattr(turn, "api_call_made", True)),
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "failed": bool(turn.error) and not state.effects.interrupted,
        "error": redact_sensitive_text(str(turn.error or ""), force=True) if turn.error else None,
        "interrupted": state.user_interrupted,
        "sdk_effects": state.effects.as_result_dict(),
        **(
            {"failover_reason": state.failover_reason.value}
            if state.failover_reason is not None
            else {}
        ),
        # Same persistence contract as the codex app-server path: we flushed
        # the projected rows ourselves, so the gateway must not re-write the
        # user turn (append_message has no dedup).
        "agent_persisted": True,
        "claude_sdk_session_id": turn.thread_id,
        **state.usage_result,
    }
    # Fatal startup/auth/billing refusals surface the same machine-readable
    # fields the chat_completions path sets (conversation_loop), so the -Q
    # exit contract and gateway consumers see a real failure instead of a
    # recoverable partial. getattr: test doubles build TurnResult-shaped
    # namespaces that may predate the field.
    _fatal = getattr(turn, "fatal_reason", None)
    if _fatal:
        result["failed"] = True
        result["failure_reason"] = _fatal
    return result


__all__ = ["run_claude_agent_sdk_turn"]
