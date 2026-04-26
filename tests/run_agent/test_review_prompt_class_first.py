"""Behavior tests for the class-first skill review prompts.

The skill review / combined review prompts steer the background review agent
toward generalizing existing skills rather than accumulating near-duplicates.
These tests assert the behavioral *instructions* are present — they do NOT
snapshot the full prompt text (change-detector).
"""

from run_agent import AIAgent


def test_skill_review_prompt_instructs_survey_first():
    """Prompt must tell the reviewer to list existing skills before deciding."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "skills_list" in prompt, "must instruct the reviewer to call skills_list"
    assert "skill_view" in prompt, "must instruct the reviewer to skill_view candidates"
    assert "SURVEY" in prompt, "must name the survey step explicitly"


def test_skill_review_prompt_is_class_first():
    """Prompt must steer toward the CLASS of task, not the specific task."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "CLASS" in prompt, "must tell the reviewer to think about the task class"
    assert "class level" in prompt, "must anchor naming at the class level"


def test_skill_review_prompt_prefers_updating_existing():
    """Prompt must prefer generalizing an existing skill over creating a new one."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "PREFER GENERALIZING" in prompt or "PREFER UPDATING" in prompt, (
        "must state the update-over-create preference"
    )
    assert "ONLY CREATE A NEW SKILL" in prompt, (
        "must gate new-skill creation behind a last-resort clause"
    )


def test_skill_review_prompt_flags_overlap_for_followup():
    """Prompt must ask the reviewer to note overlapping skills for future review."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "overlap" in prompt.lower(), "must mention the overlap-flagging protocol"


def test_skill_review_prompt_preserves_opt_out_clause():
    """The 'Nothing to save.' escape clause must remain."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "Nothing to save." in prompt


def test_combined_review_prompt_keeps_memory_section():
    """Combined prompt must still cover memory review."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "**Memory**" in prompt
    assert "memory tool" in prompt


def test_combined_review_prompt_skills_section_is_class_first():
    """The **Skills** half of the combined prompt must follow the same protocol."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "**Skills**" in prompt
    assert "SURVEY" in prompt
    assert "CLASS" in prompt
    assert "skills_list" in prompt
    assert "ONLY CREATE A NEW SKILL" in prompt


def test_combined_review_prompt_preserves_opt_out_clause():
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "Nothing to save." in prompt


def test_memory_review_prompt_unchanged_in_structure():
    """Memory-only review prompt stays focused on user facts — not touched by this change."""
    prompt = AIAgent._MEMORY_REVIEW_PROMPT
    # Guardrails: the memory-only prompt must NOT mention skills/surveys.
    assert "skills_list" not in prompt
    assert "SURVEY" not in prompt
    assert "memory tool" in prompt
