"""App-server usage, memory sync and review completion for one foreground turn."""

from __future__ import annotations

from typing import Any, Dict, List


def finish_app_server_turn(agent, turn, messages: List[Dict[str, Any]], *, original_user_message: Any,
                       should_review_memory: bool) -> dict[str, Any]:
    """Post-turn bookkeeping mirroring the chat_completions loop; returns usage fields."""
    from agent.codex_runtime import (
        _record_codex_app_server_compaction, _record_codex_app_server_usage, _call_guarded)

    # run_conversation() already bumped _turns_since_memory / _user_turn_count; only _iters_since_skill is ours.
    agent._iters_since_skill = getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    _record_codex_app_server_compaction(agent, turn)
    usage_result = _record_codex_app_server_usage(agent, turn, messages=messages)
    # Skill nudge check AFTER iters were incremented (same as chat_completions).
    should_review_skills = (0 < agent._skill_nudge_interval <= agent._iters_since_skill
                            and "skill_manage" in agent.valid_tool_names)
    if should_review_skills:
        agent._iters_since_skill = 0
    # External memory sync skipped on interrupt/error (no partial transcripts).
    if not turn.interrupted and turn.error is None:
        _call_guarded(getattr(agent, "_sync_external_memory_for_turn", None), "external memory sync raised", kwargs=dict(
            original_user_message=original_user_message, final_response=turn.final_text, interrupted=False, messages=messages,
        ))
    from agent.background_review_timing import finalize_review
    review_disposition = finalize_review(
        agent, messages, turn.final_text, turn.interrupted, should_review_memory, should_review_skills)
    if review_disposition is not None:
        usage_result["background_review"] = review_disposition
    return usage_result


