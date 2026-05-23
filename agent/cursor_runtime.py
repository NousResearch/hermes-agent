"""Cursor SDK runtime — hands Hermes turns to a Cursor Agent subprocess."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def run_cursor_sdk_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Cursor SDK runtime path. Called when ``agent.api_mode == cursor_sdk_runtime``."""
    from agent.transports.cursor_sdk_session import CursorSDKSession, preflight_cursor_sdk

    progress_callback = getattr(agent, "thinking_callback", None)

    try:
        preflight_cursor_sdk(progress_callback=progress_callback)
    except (ImportError, RuntimeError) as exc:
        logger.warning("cursor SDK preflight failed: %s", exc)
        return {
            "final_response": str(exc),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
            "interrupted": False,
        }

    if not hasattr(agent, "_cursor_session") or agent._cursor_session is None:
        cwd = getattr(agent, "session_cwd", None) or os.getcwd()
        api_key = getattr(agent, "api_key", None) or os.environ.get("CURSOR_API_KEY")
        model = getattr(agent, "model", None) or "composer-2.5"
        agent._cursor_session = CursorSDKSession(
            cwd=cwd,
            api_key=api_key,
            model=model,
            progress_callback=progress_callback,
        )
    elif progress_callback is not None:
        agent._cursor_session._progress_callback = progress_callback

    # Skills are available via the hermes-tools MCP callback (skill_view /
    # skills_list). Injecting the full skills system prompt here duplicates
    # context and can stall the first Cursor turn.
    prompt = user_message or ""
    logger.info(
        "cursor SDK turn starting session=%s model=%s",
        getattr(agent, "session_id", None) or "none",
        getattr(agent, "model", None) or "composer-2.5",
    )
    try:
        turn = agent._cursor_session.run_turn(user_input=prompt)
    except Exception as exc:
        logger.exception("cursor SDK turn failed")
        try:
            agent._cursor_session.close()
        except Exception:
            pass
        agent._cursor_session = None
        return {
            "final_response": (
                f"Cursor SDK turn failed: {exc}. "
                "Check CURSOR_API_KEY and pip install cursor-sdk."
            ),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
            "interrupted": bool(getattr(agent, "_interrupt_requested", False)),
        }

    if getattr(turn, "should_retire", False):
        logger.warning("cursor SDK session retired (error: %s)", turn.error)
        try:
            agent._cursor_session.close()
        except Exception:
            pass
        agent._cursor_session = None

    if turn.projected_messages:
        messages.extend(turn.projected_messages)

    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )

    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

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

    interrupted = turn.interrupted or bool(getattr(agent, "_interrupt_requested", False))

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": 1,
        "completed": not interrupted and turn.error is None,
        "partial": interrupted or turn.error is not None,
        "error": turn.error,
        "interrupted": interrupted,
        "cursor_agent_id": turn.agent_id,
        "cursor_run_id": turn.run_id,
    }
