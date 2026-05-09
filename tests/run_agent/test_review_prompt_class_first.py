"""Behavior tests for the skill review / combined review prompts.

The review prompts steer the background review agent toward actively updating
the skill library after most sessions, with a strong bias toward:
  1. Patching currently-loaded skills first,
  2. Patching existing umbrellas next,
  3. Adding references/ files under an existing umbrella,
  4. Creating a new class-level umbrella only when nothing else fits.

User-preference corrections (style, format, verbosity, legibility) are
first-class skill signals, not just memory signals.

These tests assert behavioral *instructions* are present — they do NOT
snapshot the full prompt text (change-detector).
"""

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# _SKILL_REVIEW_PROMPT
# ---------------------------------------------------------------------------

def test_skill_review_prompt_biases_toward_active_updates():
    """Prompt must frame updating as the default stance, not something rare."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "ACTIVE" in prompt or "active" in prompt.lower(), (
        "must tell the reviewer to be active"
    )
    # "missed learning opportunity" or equivalent framing for not acting
    assert "missed" in prompt.lower() or "opportunity" in prompt.lower(), (
        "must frame inaction as a miss, not a neutral outcome"
    )


def test_skill_review_prompt_treats_user_corrections_as_skill_signal():
    """Style/format/verbosity complaints must be FIRST-CLASS skill signals, not just memory."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    lower = prompt.lower()
    # Must mention style/format/verbosity-family corrections
    assert any(k in lower for k in ("style", "format", "verbos", "legib", "tone")), (
        "must name style/format/verbosity/legibility as signals"
    )
    # Must frame these as first-class skill signals (not memory-only)
    assert "FIRST-CLASS" in prompt or "first-class" in prompt, (
        "must explicitly label user-preference corrections as first-class skill signals"
    )
    # Must mention the correction-type phrases to tune the model's ear
    assert "stop doing" in lower or "don't" in lower or "hate" in lower or "frustrat" in lower, (
        "must give concrete phrasing examples so the model recognizes corrections"
    )


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


def test_skill_review_prompt_embeds_user_preferences_in_skills():
    """Must explicitly say user-preference lessons belong in SKILL.md, not only memory."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    lower = prompt.lower()
    assert "preference" in lower, "must mention user preferences"
    assert "memory" in lower and "skill" in lower, (
        "must contrast memory vs skill responsibilities"
    )


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


def test_combined_review_prompt_skills_biased_toward_active_updates():
    """Skills half must carry the active-update bias."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "**Skills**" in prompt
    assert "ACTIVE" in prompt or "active" in prompt.lower()
    assert "missed" in prompt.lower() or "opportunity" in prompt.lower()


def test_combined_review_prompt_treats_user_corrections_as_skill_signal():
    """Combined prompt must carry the same user-preference-is-skill-signal rule."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    lower = prompt.lower()
    assert any(k in lower for k in ("style", "format", "verbos", "legib", "tone"))
    assert "FIRST-CLASS" in prompt or "first-class" in prompt


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
# _MEMORY_REVIEW_PROMPT — unchanged, still memory-focused
# ---------------------------------------------------------------------------

def test_memory_review_prompt_still_focused_on_user_facts():
    """Memory-only review prompt stays focused on user facts — not touched by this change."""
    prompt = AIAgent._MEMORY_REVIEW_PROMPT
    # The memory-only prompt should NOT drift into skill territory
    assert "skills_list" not in prompt
    assert "SURVEY" not in prompt
    assert "memory tool" in prompt
