"""Canonical slash-command name to ``GatewayRunner`` handler method name.

Pure binding data: command metadata stays in ``hermes_cli.commands``. Values are
names rather than function objects so dispatch still binds ``self`` at runtime
without importing leaf modules here.
"""

from __future__ import annotations

GATEWAY_SLASH_HANDLERS: dict[str, str] = {
    "approvals": "_handle_approvals_command",
    "branch": "_handle_branch_command",
    "bundles": "_handle_bundles_command",
    "codex-runtime": "_handle_codex_runtime_command",
    "compress": "_handle_compress_command",
    "debug": "_handle_debug_command",
    "diff": "_handle_diff_command",
    "fast": "_handle_fast_command",
    "goal": "_handle_goal_command",
    "insights": "_handle_insights_command",
    "loop": "_handle_loop_command",
    "memory": "_handle_memory_command",
    "model": "_handle_model_command",
    "personality": "_handle_personality_command",
    "platform": "_handle_platform_command",
    "reasoning": "_handle_reasoning_command",
    "refine": "_handle_refine_command",
    "reload-mcp": "_handle_reload_mcp_command",
    "reload-skills": "_handle_reload_skills_command",
    "resume": "_handle_resume_command",
    "retry": "_handle_retry_command",
    "review": "_handle_review_command",
    "rollback": "_handle_rollback_command",
    "save": "_handle_save_command",
    "sessions": "_handle_sessions_command",
    "sethome": "_handle_set_home_command",
    "skills": "_handle_skills_command",
    "stop": "_handle_stop_command",
    "title": "_handle_title_command",
    "topic": "_handle_topic_command",
    "topup": "_handle_topup_command",
    "usage": "_handle_usage_command",
    "voice": "_handle_voice_command",
    "whoami": "_handle_whoami_command",
}
