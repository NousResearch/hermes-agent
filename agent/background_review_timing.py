"""Review timing snapshots, inline contention handling and terminal-result disposition."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def snapshot_turn_settings(agent) -> None:
    agent._deferred_completion_banner = None

    # One config snapshot governs both streaming and finalization. In strict
    # ``before_final`` mode provider token streaming is suppressed for this turn;
    # tool-round commentary is still delivered after each non-final response is
    # classified, while the final answer leaves only in the terminal result.
    try:
        from agent.background_review import background_review_timing, load_background_review_settings

        _review_enabled, _review_task_cfg = load_background_review_settings()
        _review_timing = background_review_timing(_review_task_cfg)
    except Exception:
        logger.warning("Failed to load background review turn settings; using background mode", exc_info=True)
        _review_enabled, _review_task_cfg, _review_timing = True, {}, "background"
    if (
        not _review_enabled
        or getattr(agent, "skip_background_review", False)
        or getattr(agent, "_delegate_depth", 0) > 0
    ):
        _review_timing = "background"
    agent._background_review_turn_settings = {
        "enabled": bool(_review_enabled),
        "timing": _review_timing,
        "task_cfg": dict(_review_task_cfg),
    }


def run_inline_review(
    agent: Any, messages_snapshot: List[Dict], review_memory: bool = False,
    review_skills: bool = False, task_cfg: Optional[Dict[str, Any]] = None,
) -> dict:
    """Run inline, or report why a previous request still owns the review slot."""
    from agent.background_review import (
        cancel_background_review_for_live_turn, finish_background_review_run,
        prepare_background_review_run, spawn_background_review_thread)
    from agent.turn_finalizer import _clone_background_review_messages

    if getattr(agent, "_delegate_depth", 0) > 0:
        return {"timing": "before_final", "status": "skipped", "reason": "delegated_agent"}
    run = prepare_background_review_run(agent)
    if run is None:
        cancel_background_review_for_live_turn(
            agent, message="superseded by before-final review", tool_reason="before-final review superseded")
        run = prepare_background_review_run(agent)
    if run is None:
        logger.warning("Before-final review skipped: previous review still running (session=%s)",
                       getattr(agent, "session_id", ""))
        return {"timing": "before_final", "status": "skipped", "reason": "previous_review_still_running"}
    try:
        target, _prompt = spawn_background_review_thread(
            agent, _clone_background_review_messages(messages_snapshot),
            review_memory=review_memory, review_skills=review_skills, task_cfg=task_cfg, review_run=run)
        target()
        return {"timing": "before_final", "status": "ran"}
    finally:
        finish_background_review_run(agent, run)


def finalize_review(agent, messages, final_response, interrupted, review_memory, review_skills):
    """Share eligibility and result reporting between generic and app-server finalization."""
    settings = getattr(agent, "_background_review_turn_settings", None) or {}
    if (not final_response or interrupted or getattr(agent, "skip_background_review", False)
            or not settings.get("enabled", True) or not (review_memory or review_skills)):
        return None
    kwargs = dict(messages_snapshot=list(messages), review_memory=review_memory, review_skills=review_skills)
    if settings.get("timing", "background") == "before_final":
        try:
            disposition = agent._run_background_review_before_final(
                **kwargs, task_cfg=settings.get("task_cfg") or {})
            return disposition if isinstance(disposition, dict) else None
        except Exception:
            logger.warning("Before-final memory/skill review failed", exc_info=True)
            return {"timing": "before_final", "status": "failed", "reason": "review_failed"}
    try:
        agent._spawn_background_review(**kwargs)
    except Exception:
        logger.debug("Background memory/skill review spawn failed", exc_info=True)
    return None


def finish_terminal_boundary(agent) -> None:
    banner = getattr(agent, "_deferred_completion_banner", None)
    agent._deferred_completion_banner = None
    if banner:
        try:
            agent._safe_print(banner)
        except Exception:
            logger.debug("Deferred completion banner failed", exc_info=True)
    agent._stream_callback = None
    agent._background_review_turn_settings = None


def publish_completion_banner(agent, api_call_count) -> None:
    if agent.quiet_mode:
        return
    banner = f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)"
    settings = getattr(agent, "_background_review_turn_settings", None) or {}
    if settings.get("timing") == "before_final":
        agent._deferred_completion_banner = banner
    else:
        agent._safe_print(banner)
