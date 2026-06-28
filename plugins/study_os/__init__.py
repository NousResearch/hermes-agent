"""Study OS plugin.

Registers a small Obsidian-backed study toolset. The plugin is bundled and
auto-loaded, but the tools only enter a model schema when the ``study`` toolset
is enabled for the active platform/profile.
"""

from __future__ import annotations

from pathlib import Path

from plugins.study_os.tools import (
    STUDY_CREATE_REVIEW_TASK_SCHEMA,
    STUDY_EXPORT_ANKI_CANDIDATES_SCHEMA,
    STUDY_EXTRACT_CONCEPTS_SCHEMA,
    STUDY_GENERATE_WEEKLY_REPORT_SCHEMA,
    STUDY_LIST_NOTES_SCHEMA,
    STUDY_LOG_ERROR_SCHEMA,
    STUDY_READ_NOTE_SCHEMA,
    handle_study_create_review_task,
    handle_study_export_anki_candidates,
    handle_study_extract_concepts,
    handle_study_generate_weekly_report,
    handle_study_list_notes,
    handle_study_log_error,
    handle_study_read_note,
)


_TOOLS = (
    ("study_list_notes", STUDY_LIST_NOTES_SCHEMA, handle_study_list_notes, "study"),
    ("study_read_note", STUDY_READ_NOTE_SCHEMA, handle_study_read_note, "study"),
    ("study_extract_concepts", STUDY_EXTRACT_CONCEPTS_SCHEMA, handle_study_extract_concepts, "study"),
    ("study_log_error", STUDY_LOG_ERROR_SCHEMA, handle_study_log_error, "study"),
    ("study_create_review_task", STUDY_CREATE_REVIEW_TASK_SCHEMA, handle_study_create_review_task, "study"),
    ("study_generate_weekly_report", STUDY_GENERATE_WEEKLY_REPORT_SCHEMA, handle_study_generate_weekly_report, "study"),
    ("study_export_anki_candidates", STUDY_EXPORT_ANKI_CANDIDATES_SCHEMA, handle_study_export_anki_candidates, "study"),
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

    skill_md = Path(__file__).resolve().parent / "skills" / "study-os" / "SKILL.md"
    ctx.register_skill(
        "study-os",
        skill_md,
        "Use Study OS tools to review Obsidian learning notes, log mistakes, plan reviews, and export Anki candidates.",
    )
