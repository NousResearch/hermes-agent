"""Study OS plugin.

Registers a small Obsidian-backed study toolset. The plugin is bundled and
auto-loaded, but the tools only enter a model schema when the ``study`` toolset
is enabled for the active platform/profile.
"""

from __future__ import annotations

from pathlib import Path

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

    skills_root = Path(__file__).resolve().parent / "skills"
    for name, description in (
        ("study-os", "Route StudyOS learning workflows."),
        ("study-plan", "Plan StudyOS projects and schedules."),
        ("study-organize", "Organize problems into StudyOS notes."),
        ("study-review", "Run StudyOS spaced repetition reviews."),
        ("study-teach", "Teach through StudyOS learning records."),
        ("study-lesson", "Create visual StudyOS lesson artifacts."),
        ("study-assessment", "Analyze StudyOS exams and mistakes."),
        ("study-kaoyan", "Guide 考研 learning with StudyOS."),
        ("study-engineering", "Guide engineering and skill learning with StudyOS."),
        ("study-grill", "Bridge grilling sessions into StudyOS decisions."),
    ):
        ctx.register_skill(name, skills_root / name / "SKILL.md", description)
