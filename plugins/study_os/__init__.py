"""Study OS plugin.

Registers a small Obsidian-backed study toolset. The plugin is bundled and
auto-loaded, but the tools only enter a model schema when the ``study`` toolset
is enabled for the active platform/profile.
"""

from __future__ import annotations

from pathlib import Path

from plugins.study_os.tools import (
    STUDY_CONCEPT_GRAPH_SCHEMA,
    STUDY_CREATE_CURRICULUM_SCHEMA,
    STUDY_CREATE_REVIEW_TASK_SCHEMA,
    STUDY_DECISION_SCHEMA,
    STUDY_DUE_REVIEWS_SCHEMA,
    STUDY_EXPORT_ANKI_CANDIDATES_SCHEMA,
    STUDY_EXTRACT_CONCEPTS_SCHEMA,
    STUDY_GENERATE_WEEKLY_REPORT_SCHEMA,
    STUDY_IMPORT_PLAN_SCHEMA,
    STUDY_LEARNING_QUEUE_SCHEMA,
    STUDY_LEARNING_RECORD_SCHEMA,
    STUDY_LESSON_SCHEMA,
    STUDY_LIST_CURRICULA_SCHEMA,
    STUDY_LIST_NOTES_SCHEMA,
    STUDY_LOG_ERROR_SCHEMA,
    STUDY_LOG_SESSION_SCHEMA,
    STUDY_PLAN_PROGRESS_SCHEMA,
    STUDY_PROJECT_SCHEMA,
    STUDY_PROMPT_CONTEXT_SCHEMA,
    STUDY_READ_NOTE_SCHEMA,
    STUDY_RECORD_REVIEW_SCHEMA,
    STUDY_REVIEW_STATS_SCHEMA,
    STUDY_SCHEDULE_SCHEMA,
    STUDY_SYNC_MEMORY_SCHEMA,
    STUDY_UPDATE_CONCEPT_STATE_SCHEMA,
    handle_study_concept_graph,
    handle_study_create_curriculum,
    handle_study_create_review_task,
    handle_study_decision,
    handle_study_due_reviews,
    handle_study_export_anki_candidates,
    handle_study_extract_concepts,
    handle_study_generate_weekly_report,
    handle_study_import_plan,
    handle_study_learning_queue,
    handle_study_learning_record,
    handle_study_lesson,
    handle_study_list_curricula,
    handle_study_list_notes,
    handle_study_log_error,
    handle_study_log_session,
    handle_study_plan_progress,
    handle_study_project,
    handle_study_prompt_context,
    handle_study_read_note,
    handle_study_record_review,
    handle_study_review_stats,
    handle_study_schedule,
    handle_study_sync_memory,
    handle_study_update_concept_state,
)


_TOOLS = (
    ("study_list_notes", STUDY_LIST_NOTES_SCHEMA, handle_study_list_notes, "study"),
    ("study_read_note", STUDY_READ_NOTE_SCHEMA, handle_study_read_note, "study"),
    ("study_extract_concepts", STUDY_EXTRACT_CONCEPTS_SCHEMA, handle_study_extract_concepts, "study"),
    ("study_log_error", STUDY_LOG_ERROR_SCHEMA, handle_study_log_error, "study"),
    ("study_create_review_task", STUDY_CREATE_REVIEW_TASK_SCHEMA, handle_study_create_review_task, "study"),
    ("study_decision", STUDY_DECISION_SCHEMA, handle_study_decision, "study"),
    ("study_generate_weekly_report", STUDY_GENERATE_WEEKLY_REPORT_SCHEMA, handle_study_generate_weekly_report, "study"),
    ("study_export_anki_candidates", STUDY_EXPORT_ANKI_CANDIDATES_SCHEMA, handle_study_export_anki_candidates, "study"),
    ("study_due_reviews", STUDY_DUE_REVIEWS_SCHEMA, handle_study_due_reviews, "study"),
    ("study_record_review", STUDY_RECORD_REVIEW_SCHEMA, handle_study_record_review, "study"),
    ("study_sync_memory", STUDY_SYNC_MEMORY_SCHEMA, handle_study_sync_memory, "study"),
    ("study_concept_graph", STUDY_CONCEPT_GRAPH_SCHEMA, handle_study_concept_graph, "study"),
    ("study_review_stats", STUDY_REVIEW_STATS_SCHEMA, handle_study_review_stats, "study"),
    ("study_learning_queue", STUDY_LEARNING_QUEUE_SCHEMA, handle_study_learning_queue, "study"),
    ("study_learning_record", STUDY_LEARNING_RECORD_SCHEMA, handle_study_learning_record, "study"),
    ("study_lesson", STUDY_LESSON_SCHEMA, handle_study_lesson, "study"),
    ("study_log_session", STUDY_LOG_SESSION_SCHEMA, handle_study_log_session, "study"),
    ("study_update_concept_state", STUDY_UPDATE_CONCEPT_STATE_SCHEMA, handle_study_update_concept_state, "study"),
    ("study_import_plan", STUDY_IMPORT_PLAN_SCHEMA, handle_study_import_plan, "study"),
    ("study_plan_progress", STUDY_PLAN_PROGRESS_SCHEMA, handle_study_plan_progress, "study"),
    ("study_create_curriculum", STUDY_CREATE_CURRICULUM_SCHEMA, handle_study_create_curriculum, "study"),
    ("study_list_curricula", STUDY_LIST_CURRICULA_SCHEMA, handle_study_list_curricula, "study"),
    ("study_project", STUDY_PROJECT_SCHEMA, handle_study_project, "study"),
    ("study_schedule", STUDY_SCHEDULE_SCHEMA, handle_study_schedule, "study"),
    ("study_prompt_context", STUDY_PROMPT_CONTEXT_SCHEMA, handle_study_prompt_context, "study"),
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
