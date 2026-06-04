"""Claude Code subprocess runtime — drives turns through the ``claude`` CLI.

Extracted from :class:`AIAgent` to keep the agent loop file focused.
Mirrors ``codex_runtime.py`` for OpenAI's Codex CLI.

* ``run_claude_subprocess_turn`` — drives one turn through the ``claude``
  CLI subprocess (used when ``claude_subprocess`` is the active runtime).

The ``claude`` CLI authenticates via macOS Keychain OAuth tokens from the
user's Claude Pro/Max subscription.  All usage bills against that
subscription — no separate API key required.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def run_claude_subprocess_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Claude subprocess runtime path.  Hands the entire turn to the
    ``claude`` CLI and projects its response back into Hermes' messages
    list so memory/skill review keep working.

    Called from run_conversation() when agent.api_mode == "claude_subprocess".
    Returns the same dict shape as the chat_completions / codex_app_server paths.
    """
    from agent.transports.claude_subprocess import ClaudeSubprocessSession

    # Lazy session: one ClaudeSubprocessSession per AIAgent instance.
    # Spawned on first turn, reused across turns (resume via session_id).
    if not hasattr(agent, "_claude_session") or agent._claude_session is None:
        cwd = getattr(agent, "session_cwd", None) or os.getcwd()

        # Read model preference from config — fall back to "sonnet".
        model = "sonnet"
        config = getattr(agent, "config", None)
        if isinstance(config, dict):
            claude_cfg = config.get("claude_runtime") or {}
            if isinstance(claude_cfg, dict):
                model = claude_cfg.get("model", model)

        # Read system prompt from the agent if available
        system_prompt = None
        if hasattr(agent, "system_message") and agent.system_message:
            system_prompt = str(agent.system_message)

        agent._claude_session = ClaudeSubprocessSession(
            cwd=cwd,
            model=model,
            system_prompt=system_prompt,
        )

    # NOTE: the user message is ALREADY appended to messages by the
    # standard run_conversation() flow before the early return reaches us.
    # Do NOT append again — that would duplicate.

    try:
        turn = agent._claude_session.run_turn(user_input=user_message)
    except Exception as exc:
        logger.exception("Claude subprocess turn failed")
        # Crash → drop the session so the next turn respawns from scratch.
        try:
            agent._claude_session.close()
        except Exception:
            pass
        agent._claude_session = None
        return {
            "final_response": (
                f"Claude subprocess turn failed: {exc}. "
                f"Fall back to default runtime with `/claude-runtime auto`."
            ),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
        }

    # If the turn signalled the underlying client is wedged (timeout,
    # auth failure, etc.), retire the session so the next turn respawns.
    if turn.should_retire:
        logger.warning(
            "Claude subprocess session retired (error: %s)",
            turn.error,
        )
        try:
            agent._claude_session.close()
        except Exception:
            pass
        agent._claude_session = None

    if turn.is_error:
        return {
            "final_response": turn.error or "Claude subprocess returned an error",
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": turn.error,
        }

    # Project the response back into Hermes' message format.
    assistant_msg = {
        "role": "assistant",
        "content": turn.final_text,
    }

    # Add cost metadata to the message for tracking.
    if turn.cost_usd > 0:
        assistant_msg["_claude_cost_usd"] = turn.cost_usd
    if turn.model:
        assistant_msg["_claude_model"] = turn.model
    if turn.session_id:
        assistant_msg["_claude_session_id"] = turn.session_id

    messages.append(assistant_msg)

    # Log cost
    if turn.cost_usd > 0:
        logger.info(
            "Claude subprocess turn complete: model=%s cost=$%.4f duration=%.0fms turns=%d",
            turn.model,
            turn.cost_usd,
            turn.duration_ms,
            turn.num_turns,
        )

    # Memory/skill review — mirror the codex_runtime path.
    if should_review_memory and hasattr(agent, "_maybe_review_memory_and_skills"):
        try:
            agent._maybe_review_memory_and_skills(
                messages=messages,
                effective_task_id=effective_task_id,
                tool_iterations_since_review=turn.num_turns,
            )
        except Exception:
            logger.debug("memory/skill review after claude turn failed", exc_info=True)

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": 1,
        "completed": True,
        "partial": False,
        "error": None,
        "claude_cost_usd": turn.cost_usd,
        "claude_model": turn.model,
        "claude_session_id": turn.session_id,
    }
