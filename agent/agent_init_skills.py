"""Skill-review settings read once at agent init (``skills.*`` in config.yaml)."""
from contextlib import suppress


def _apply_skill_review_config(agent, agent_cfg) -> None:
    """``skills.creation_nudge_interval`` (skill reminders every N tool iterations) plus the
    slow-turn rule ``turn_finalizer._slow_turn_review_flags`` applies: a turn with many tool
    rounds, or a tool that failed and then worked, is reviewed regardless of the nudge
    counters, so short chats learn too."""
    agent._skill_nudge_interval = 10
    agent._review_slow_turns, agent._slow_turn_tool_rounds = True, 4
    skills_cfg = agent_cfg.get("skills", {}) if isinstance(agent_cfg, dict) else {}
    if not isinstance(skills_cfg, dict):
        return
    with suppress(TypeError, ValueError):
        agent._skill_nudge_interval = int(skills_cfg.get("creation_nudge_interval", 10))
    agent._review_slow_turns = bool(skills_cfg.get("review_slow_turns", True))
    with suppress(TypeError, ValueError):
        agent._slow_turn_tool_rounds = int(skills_cfg.get("slow_turn_tool_rounds", 4))
