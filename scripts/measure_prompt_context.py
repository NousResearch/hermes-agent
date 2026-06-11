"""Measure high-level Hermes system-prompt context sections.

This is intentionally lightweight and approximate: character counts plus an
OpenAI-style token estimate (chars / 4). It avoids model/provider calls and is
safe to run in local checkouts.
"""

from __future__ import annotations

from agent.prompt_builder import (
    HERMES_AGENT_HELP_GUIDANCE,
    KANBAN_GUIDANCE,
    MEMORY_GUIDANCE,
    OPENAI_MODEL_EXECUTION_GUIDANCE,
    SKILLS_GUIDANCE,
    TASK_COMPLETION_GUIDANCE,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    build_environment_hints,
    build_skills_system_prompt,
)


def estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def main() -> None:
    sections = {
        "skills_index": build_skills_system_prompt(),
        "environment_hints": build_environment_hints(),
        "hermes_help_guidance": HERMES_AGENT_HELP_GUIDANCE,
        "memory_guidance": MEMORY_GUIDANCE,
        "skills_guidance": SKILLS_GUIDANCE,
        "kanban_guidance": KANBAN_GUIDANCE,
        "task_completion_guidance": TASK_COMPLETION_GUIDANCE,
        "tool_use_enforcement": TOOL_USE_ENFORCEMENT_GUIDANCE,
        "execution_discipline": OPENAI_MODEL_EXECUTION_GUIDANCE,
    }
    for name, text in sections.items():
        print(f"{name}\tchars={len(text)}\test_tokens={estimate_tokens(text)}")
    total_chars = sum(len(text) for text in sections.values())
    print(f"measured_total\tchars={total_chars}\test_tokens={estimate_tokens(''.join(sections.values()))}")


if __name__ == "__main__":
    main()
