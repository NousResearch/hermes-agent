"""Project-local plugin template for first-turn subagent context injection."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_SUBAGENT_GOAL = "帮我生成100个字的文案，不要问为什么，什么场景，就是随机一些文案，100个字即可。"


def _extract_summary(tool_result: Any) -> str | None:
    if not isinstance(tool_result, dict):
        return None

    results = tool_result.get("results")
    if not isinstance(results, list) or not results:
        return None

    first_result = results[0]
    if not isinstance(first_result, dict):
        return None

    summary = first_result.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()

    return None


def _is_top_level_cli_session(ctx: Any, session_id: str) -> bool:
    cli = getattr(getattr(ctx, "_manager", None), "_cli_ref", None)
    active_agent = getattr(cli, "agent", None) if cli else None
    active_session_id = getattr(active_agent, "session_id", None)
    if not active_session_id:
        return True
    return session_id == active_session_id


def register(ctx):
    def inject_first_turn_copy(
        session_id: str,
        user_message: str,
        is_first_turn: bool,
        **kwargs: Any,
    ) -> dict[str, str] | None:
        if not _is_top_level_cli_session(ctx, session_id):
            logger.debug(
                "first-turn-copywriter skipped delegated child session %s",
                session_id,
            )
            return None

        try:
            raw_result = ctx.dispatch_tool(
                "delegate_task",
                {
                    "goal": _SUBAGENT_GOAL,
                    "context": f"当前用户消息：{user_message}",
                    "max_iterations": 2,
                },
            )
            tool_result = json.loads(raw_result)
        except Exception as exc:
            logger.warning(
                "first-turn-copywriter failed for session %s: %s",
                session_id,
                exc,
            )
            return None

        if isinstance(tool_result, dict) and tool_result.get("error"):
            logger.warning(
                "first-turn-copywriter delegate_task error for session %s: %s",
                session_id,
                tool_result["error"],
            )
            return None

        generated_copy = _extract_summary(tool_result)
        if not generated_copy:
            return None

        return {
            "context": (
                "以下内容请在最终回答的末尾进行追加返回，这很重要，作为我的测试标记：\n"
                f"{generated_copy}"
            )
        }

    ctx.register_hook("pre_llm_call", inject_first_turn_copy)