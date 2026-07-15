"""Study OS plugin.

Registers a small Obsidian-backed study toolset. The plugin is bundled and
auto-loaded, but the tools only enter a model schema when the ``study`` toolset
is enabled for the active platform/profile.
"""

from __future__ import annotations

from pathlib import Path

from agent.skill_utils import parse_frontmatter
from plugins.study_os.context import active_learning_context
from plugins.study_os.domain_packs import domain_pack_registry
from plugins.study_os.learning import (
    STUDY_ACTIVITY_SCHEMA,
    STUDY_COACH_SCHEMA,
    handle_study_activity,
    handle_study_coach,
)


_TOOLS = (
    ("study_activity", STUDY_ACTIVITY_SCHEMA, handle_study_activity, "study"),
    ("study_coach", STUDY_COACH_SCHEMA, handle_study_coach, "study"),
)


def register(ctx) -> None:
    """Register Study OS tools and the opt-in Study OS skill."""
    for name, schema, handler, toolset in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=handler,
            emoji="📚",
        )

    ctx.register_hook("pre_llm_call", active_learning_context)

    skills_root = Path(__file__).resolve().parent / "skills"
    for name, description in (
        ("study-os", "Route StudyOS learning workflows."),
        ("study-plan", "Plan StudyOS projects, interventions, and schedules."),
        ("study-organize", "Organize problems into StudyOS notes."),
        ("study-review", "Run StudyOS spaced repetition reviews."),
        ("study-teach", "Teach through StudyOS learning records."),
        ("study-lesson", "Create visual StudyOS lesson artifacts."),
        ("study-assessment", "Analyze StudyOS exams and mistakes."),
        ("study-grill", "Bridge grilling sessions into StudyOS decisions."),
    ):
        ctx.register_skill(name, skills_root / name / "SKILL.md", description)

    for pack in domain_pack_registry().values():
        if pack.prompt_skill is None:
            continue
        skill_path = skills_root / pack.prompt_skill / "SKILL.md"
        frontmatter, _body = parse_frontmatter(skill_path.read_text(encoding="utf-8"))
        skill_name = str(frontmatter.get("name") or "").strip()
        description = str(frontmatter.get("description") or "").strip()
        if skill_name != pack.prompt_skill or not description:
            raise ValueError(
                f"DomainPack {pack.id} prompt skill metadata must declare "
                f"name={pack.prompt_skill!r} and a description"
            )
        ctx.register_skill(skill_name, skill_path, description)
