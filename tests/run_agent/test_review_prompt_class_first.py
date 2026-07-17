"""Behavior tests for background-review information routing.

The review prompts must be selective: most turns should not mutate durable
state. When a durable procedural lesson exists, skill updates use this order:
  1. Patching currently-loaded skills first,
  2. Patching existing umbrellas next,
  3. Adding references/ files under an existing umbrella,
  4. Creating a new class-level umbrella only when nothing else fits.

Global user preferences belong in USER memory. Task-class procedures belong in
skills. Project state and completed work belong outside always-injected memory.

These tests assert behavioral *instructions* are present — they do NOT
snapshot the full prompt text (change-detector).
"""

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# _SKILL_REVIEW_PROMPT
# ---------------------------------------------------------------------------

def test_skill_review_prompt_is_selective_by_default():
    """Routine successful sessions must not be treated as learning events."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    lower = prompt.lower()
    assert "most sessions produce" not in lower
    assert "missed learning opportunity" not in lower
    assert "not whether you update" not in lower
    assert "nothing to save" in lower
    assert any(k in lower for k in ("selective", "durable signal", "default outcome"))


def test_skill_review_prompt_routes_preferences_by_scope():
    """Global user preferences and task-class procedures have different homes."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    lower = prompt.lower()
    assert "global" in lower and "user" in lower and "memory" in lower
    assert "task-class" in lower or "class of task" in lower
    assert "not just memory signals" not in lower


def test_skill_review_prompt_prefers_loaded_skills_first():
    """Currently-loaded skills must be the first patch target."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "LOADED" in prompt or "loaded" in prompt, (
        "must mention currently-loaded skills"
    )
    # Must name the mechanisms for detecting loaded skills
    assert "skill_view" in prompt and "/skill" in prompt, (
        "must name skill_view and /skill-name as loaded-skill signals"
    )


def test_skill_review_prompt_has_four_step_preference_order():
    """The 4-step patch/support-file/create ladder must be present."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "PATCH" in prompt
    assert "references/" in prompt or "REFERENCE" in prompt
    assert "CREATE" in prompt
    assert "UMBRELLA" in prompt or "umbrella" in prompt


def test_skill_review_prompt_names_three_support_file_kinds():
    """Support-file step must name references/, templates/, and scripts/."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "references/" in prompt, "must name references/ as a support-file kind"
    assert "templates/" in prompt, "must name templates/ as a support-file kind"
    assert "scripts/" in prompt, "must name scripts/ as a support-file kind"
    # Purpose hints for each kind
    assert "knowledge" in prompt.lower() or "research" in prompt.lower() or "API docs" in prompt, (
        "must mention knowledge-bank / research / API-docs role of references/"
    )
    assert "copied" in prompt.lower() or "starter" in prompt.lower() or "reproduce" in prompt.lower(), (
        "must mention that templates/ are starter files to copy/modify"
    )
    assert "re-runnable" in prompt.lower() or "verification" in prompt.lower() or "probe" in prompt.lower(), (
        "must mention that scripts/ are re-runnable actions"
    )


def test_skill_review_prompt_has_name_veto_for_create():
    """Creating a new skill must be gated behind class-level naming."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "class level" in prompt.lower() or "CLASS-LEVEL" in prompt
    assert "MUST NOT" in prompt or "must not" in prompt, (
        "must have a name-veto clause blocking session-artifact names"
    )


def test_skill_review_prompt_does_not_duplicate_preferences_into_skills():
    """A preference alone must not be copied into both USER and SKILL.md."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    lower = prompt.lower()
    assert "do not duplicate" in lower or "do not copy" in lower
    assert "preference" in lower and "skill" in lower


def test_skill_review_prompt_flags_overlap_and_defers_to_curator():
    """Reviewer should not consolidate live; flag overlap for the curator."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "overlap" in prompt.lower()
    assert "curator" in prompt.lower(), "must defer consolidation to the curator"


def test_skill_review_prompt_still_has_opt_out_clause():
    """'Nothing to save.' must remain as a real-but-not-default option."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "Nothing to save." in prompt


# ---------------------------------------------------------------------------
# _COMBINED_REVIEW_PROMPT
# ---------------------------------------------------------------------------

def test_combined_review_prompt_has_memory_section():
    """Memory half must still cover user facts and preferences."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "**Memory**" in prompt
    assert "memory tool" in prompt


def test_combined_review_prompt_is_selective_by_default():
    """The combined review must not manufacture writes from routine turns."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    lower = prompt.lower()
    assert "**Skills**" in prompt
    assert "most sessions produce" not in lower
    assert "missed learning opportunity" not in lower
    assert "nothing to save" in lower


def test_combined_review_prompt_routes_preferences_by_scope():
    """Combined review separates global USER preferences from procedures."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    lower = prompt.lower()
    assert "global" in lower and "user" in lower and "memory" in lower
    assert "task-class" in lower or "class of task" in lower
    assert "not just memory signals" not in lower


def test_combined_review_prompt_prefers_loaded_skills_first():
    """Combined prompt must also prefer loaded skills first."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "LOADED" in prompt or "loaded" in prompt
    assert "skill_view" in prompt and "/skill" in prompt


def test_combined_review_prompt_has_four_step_skill_ladder():
    """Combined prompt must keep the patch/support-file/create ladder on the Skills half."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "PATCH" in prompt
    assert "references/" in prompt or "REFERENCE" in prompt
    assert "CREATE" in prompt
    assert "CLASS-LEVEL" in prompt or "class-level" in prompt or "class level" in prompt.lower()


def test_combined_review_prompt_names_three_support_file_kinds():
    """Combined prompt must also name all three support-file kinds."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "references/" in prompt
    assert "templates/" in prompt
    assert "scripts/" in prompt


def test_combined_review_prompt_preserves_opt_out_clause():
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "Nothing to save." in prompt


# ---------------------------------------------------------------------------
# Anti-pattern guidance — see issue #6051. The reviewer was learning transient
# environment failures (e.g. "browser tools do not work" from a fresh-install
# Playwright miss) as durable skill rules, then citing them against itself for
# weeks after the environment was fixed. Both review prompts must explicitly
# tell the reviewer not to capture environment-dependent or negative-framing
# content as skills.
# ---------------------------------------------------------------------------


def _assert_anti_pattern_guidance(prompt: str, label: str) -> None:
    """Both review prompts must carry the same anti-pattern section."""
    lower = prompt.lower()
    assert "do not capture" in lower, (
        f"{label}: must have an explicit 'Do NOT capture' section"
    )
    # Environment-dependent failures (the #6051 root cause)
    assert any(k in lower for k in ("missing binar", "command not found", "uninstalled", "fresh-install")), (
        f"{label}: must call out environment/setup failures as not-skill-worthy"
    )
    # Negative-framing avoidance
    assert any(k in lower for k in ("negative claim", "do not work", "is broken")), (
        f"{label}: must call out negative-claim phrasings as the failure mode"
    )
    # Positive reframing — "capture the fix, not the failure"
    assert "capture the fix" in lower or "capture the fix " in lower, (
        f"{label}: must redirect tool-failure capture toward the fix, not the constraint"
    )
    # One-off task narratives (#12812 family)
    assert "one-off" in lower, (
        f"{label}: must call out one-off task narratives as not-skill-worthy"
    )


def test_skill_review_prompt_has_anti_pattern_guidance():
    """_SKILL_REVIEW_PROMPT must tell the reviewer NOT to capture transient env failures (#6051)."""
    _assert_anti_pattern_guidance(AIAgent._SKILL_REVIEW_PROMPT, "_SKILL_REVIEW_PROMPT")


def test_combined_review_prompt_has_anti_pattern_guidance():
    """_COMBINED_REVIEW_PROMPT must carry the same guidance — same failure mode applies."""
    _assert_anti_pattern_guidance(AIAgent._COMBINED_REVIEW_PROMPT, "_COMBINED_REVIEW_PROMPT")


# ---------------------------------------------------------------------------
# _MEMORY_REVIEW_PROMPT — bounded always-loaded memory only
# ---------------------------------------------------------------------------

def test_memory_review_prompt_still_focused_on_user_facts():
    """Memory-only review stays focused on stable user/environment facts."""
    prompt = AIAgent._MEMORY_REVIEW_PROMPT
    # The memory-only prompt should NOT drift into skill territory
    assert "skills_list" not in prompt
    assert "SURVEY" not in prompt
    assert "memory tool" in prompt


def test_memory_review_prompt_rejects_operational_history():
    """Completed work and mutable project state must not enter every prompt."""
    lower = AIAgent._MEMORY_REVIEW_PROMPT.lower()
    assert "completed work" in lower
    assert "project state" in lower
    assert "do not save" in lower or "never save" in lower
    assert "session" in lower and "project" in lower


def test_combined_review_prompt_has_explicit_information_router():
    """The combined reviewer must classify information before choosing a tool."""
    lower = AIAgent._COMBINED_REVIEW_PROMPT.lower()
    for destination in ("user", "memory", "skill", "project state", "session history"):
        assert destination in lower
    assert "classify" in lower or "route" in lower
