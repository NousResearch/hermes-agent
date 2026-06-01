"""ACP client runtime — one turn through an ACP-compliant agent subprocess.

Extracted from AIAgent to keep the agent loop file focused.
Takes the parent AIAgent as its first argument (``agent``).
AIAgent keeps thin forwarder methods for backward compatibility.

``run_acp_client_turn`` — drives one turn through an
``ACPClientSession`` subprocess (used when api_mode == "acp_client").
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def run_acp_client_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """ACP client runtime path. Hands the entire turn to an ACP-compliant
    agent subprocess and projects its streaming events back into Hermes'
    messages list so memory/skill review keep working.

    Called from run_conversation() when agent.api_mode == "acp_client".
    Returns the same dict shape as the chat_completions path.

    Lazy session: one ACPClientSession per AIAgent instance.
    Spawned on first turn, reused across turns; kernel reclaims stdin on
    process exit — no explicit teardown hook on AIAgent.
    """
    from agent.transports.acp_client_session import ACPClientSession

    if not hasattr(agent, "_acp_session") or agent._acp_session is None:
        command = getattr(agent, "acp_command", None) or "acp-agent"
        args = getattr(agent, "acp_args", None) or []

        # on_delta: bridge streaming text deltas to Hermes' live-output path.
        # _fire_stream_delta is the same hook the chat_completions path uses
        # (see conversation_loop.py). Falls back to None in contexts that
        # don't have streaming hooked up (cron, batch).
        on_delta = getattr(agent, "_fire_stream_delta", None)

        # model: read from agent.model so the ACP server uses the same model
        # Hermes is configured for, rather than its own default (Fix 1).
        # Falls back to None when not set -- ACPClientSession skips the
        # session/set_config_option call in that case.
        model = getattr(agent, "model", None) or None

        agent._acp_session = ACPClientSession(
            command=command,
            args=list(args),
            model=model,
            on_delta=on_delta,
        )

    # NOTE: the user message is ALREADY appended to messages by the
    # standard run_conversation() flow before the early return reaches us.
    # Do NOT append again — that would duplicate.

    # cwd priority: agent.session_cwd > HERMES_ACP_SESSION_CWD env > os.getcwd()
    # HERMES_ACP_SESSION_CWD lets operators point the ACP session at a per-agent
    # sandbox directory (containing CLAUDE.md + .claude/settings.local.json) on
    # the gateway launch env without requiring a new config key in the provider
    # resolver chain.  Production Janet and janet_test run as separate processes
    # with distinct HERMES_HOME, so the env var is scoped to the right gateway.
    cwd = (
        getattr(agent, "session_cwd", None)
        or os.environ.get("HERMES_ACP_SESSION_CWD", "").strip()
        or os.getcwd()
    )

    try:
        turn = agent._acp_session.run_turn(user_input=user_message, cwd=cwd)
    except Exception as exc:
        logger.exception("ACP client turn failed")
        # Crash → unconditionally drop the session so the next turn
        # respawns from scratch instead of reusing a dead client.
        try:
            agent._acp_session.close()
        except Exception:
            pass
        agent._acp_session = None
        return {
            "final_response": (
                f"ACP client turn failed: {exc}. "
                f"Check acp_command/acp_args in your config."
            ),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
        }

    # If the turn signalled the underlying client is wedged (deadline
    # blown, subprocess exited, protocol error), retire the session so
    # the next turn respawns the agent from scratch.
    if getattr(turn, "should_retire", False):
        logger.warning(
            "ACP client session retired (turn error: %s)",
            turn.error,
        )
        try:
            agent._acp_session.close()
        except Exception:
            pass
        agent._acp_session = None

    # Splice projected messages into the conversation. The session emits
    # standard {role, content} entries, which is what curator.py / sessions
    # DB expect.
    if turn.projected_messages:
        messages.extend(turn.projected_messages)

    # Counter ticks for the agent-improvement loop.
    # _turns_since_memory and _user_turn_count are ALREADY incremented
    # in the run_conversation() pre-loop block before the early return,
    # so do NOT touch them here — that would double-count.
    # Only _iters_since_skill needs explicit increment, since the
    # chat_completions loop bumps it per tool iteration and that loop is
    # bypassed on this path.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )

    # Check the skill nudge AFTER iters were incremented — same pattern
    # as the chat_completions path.
    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider sync — skip on interrupt/error to avoid
    # feeding partial transcripts to memory.
    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    # Background review fork — same cadence + signature as the default path.
    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": 1,  # one ACP session/prompt call maps to one logical API call
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
    }


__all__ = ["run_acp_client_turn"]
