"""Skill-governance bundled plugin.

Registers read-only tools that turn Vladimir's approved skill-to-plugin roadmap
into machine-readable summaries and implementation plans. The plugin performs no
external API calls and has no live side effects.
"""

from __future__ import annotations

from .schemas import (
    SKILLS_FIND_PLUGIN_CANDIDATES_SCHEMA,
    SKILLS_PLUGIN_ROADMAP_SUMMARY_SCHEMA,
    SKILLS_TO_PLUGIN_PLAN_SCHEMA,
)
from .tools import (
    skills_find_plugin_candidates,
    skills_plugin_roadmap_summary,
    skills_to_plugin_plan,
)

_TOOLS = (
    (
        "skills_plugin_roadmap_summary",
        SKILLS_PLUGIN_ROADMAP_SUMMARY_SCHEMA,
        skills_plugin_roadmap_summary,
        "🧭",
    ),
    (
        "skills_find_plugin_candidates",
        SKILLS_FIND_PLUGIN_CANDIDATES_SCHEMA,
        skills_find_plugin_candidates,
        "🔎",
    ),
    ("skills_to_plugin_plan", SKILLS_TO_PLUGIN_PLAN_SCHEMA, skills_to_plugin_plan, "🧩"),
)


def register(ctx) -> None:
    """Register read-only skill-governance tools with Hermes."""

    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="skills",
            schema=schema,
            handler=handler,
            emoji=emoji,
        )
