"""Behavior tests for safe background skill-review prompts.

The review prompts must preserve reusable skill maintenance while preventing
background review from persisting user-specific, session-specific, private, or
untrusted content into the global skill library.
"""

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Shared prompt safety expectations
# ---------------------------------------------------------------------------


def _assert_safe_skill_review_stance(prompt: str, label: str) -> None:
    lower = prompt.lower()
    assert "global skill" in lower or "global skills" in lower, (
        f"{label}: must frame skills as globally shared/reusable context"
    )
    assert "conservative" in lower, (
        f"{label}: background review must be conservative, not write-biased"
    )
    assert "nothing to save" in lower, (
        f"{label}: must preserve a no-op path when no safe update exists"
    )
    assert "most sessions produce" not in lower, (
        f"{label}: must not bias the reviewer toward writes in most sessions"
    )
    assert "missed learning opportunity" not in lower, (
        f"{label}: must not frame safe inaction as a failure"
    )


def _assert_private_content_exclusions(prompt: str, label: str) -> None:
    lower = prompt.lower()
    for phrase in (
        "user-specific",
        "session-specific",
        "private",
        "tenant",
        "credentials",
        "tokens",
        "secrets",
        "personal data",
    ):
        assert phrase in lower, f"{label}: must exclude {phrase} content from skills"
    assert "error transcripts" in lower and "do not" in lower, (
        f"{label}: must not persist raw/session error transcripts into global skills"
    )


def _assert_untrusted_source_guidance(prompt: str, label: str) -> None:
    lower = prompt.lower()
    assert "untrusted" in lower, f"{label}: must address untrusted content"
    assert "independently validated" in lower, (
        f"{label}: must require source/content validation before skill writes"
    )
    assert "stripped" in lower and "instructions" in lower, (
        f"{label}: must strip embedded instructions from persisted skill text"
    )


def _assert_user_preferences_stay_out_of_global_skills(prompt: str, label: str) -> None:
    lower = prompt.lower()
    assert any(k in lower for k in ("style", "format", "verbos", "legib", "tone")), (
        f"{label}: must mention style/format/verbosity-family preferences"
    )
    assert "memory" in lower and "skill" in lower, (
        f"{label}: must distinguish memory/user profile from skills"
    )
    assert "shared skill library must not" in lower or "not in the shared skill library" in lower, (
        f"{label}: user-specific preferences must not be written to global skills"
    )
    assert "first-class skill" not in lower, (
        f"{label}: must not label user preferences as first-class skill signals"
    )


def _assert_support_file_safety(prompt: str, label: str) -> None:
    lower = prompt.lower()
    assert "references/" in prompt, f"{label}: must still document references/ support files"
    assert "templates/" in prompt, f"{label}: must still document templates/ support files"
    assert "scripts/" in prompt, f"{label}: must still document scripts/ support files"
    assert "sanitized" in lower and "source-checked" in lower, (
        f"{label}: support files must require sanitization and source checks"
    )
    assert "deterministic" in lower and "non-destructive" in lower, (
        f"{label}: generated scripts/templates must be constrained to safe reusable content"
    )
    assert "never include private transcripts" in lower, (
        f"{label}: support files must forbid private transcripts"
    )


def _assert_skill_update_shape(prompt: str, label: str) -> None:
    lower = prompt.lower()
    assert "loaded" in lower and "skill_view" in prompt and "/skill" in prompt, (
        f"{label}: may still prefer loaded skills after safety checks"
    )
    assert "patch" in lower, f"{label}: must preserve patch guidance"
    assert "create" in lower, f"{label}: must preserve create guidance"
    assert "class-level" in lower or "class level" in lower, (
        f"{label}: skill names must remain class-level"
    )
    assert "must not" in lower and "session artifact" in lower, (
        f"{label}: must veto session-artifact skill names"
    )


# ---------------------------------------------------------------------------
# _SKILL_REVIEW_PROMPT
# ---------------------------------------------------------------------------


def test_skill_review_prompt_uses_conservative_global_write_stance():
    _assert_safe_skill_review_stance(AIAgent._SKILL_REVIEW_PROMPT, "_SKILL_REVIEW_PROMPT")


def test_skill_review_prompt_excludes_private_and_session_content():
    _assert_private_content_exclusions(AIAgent._SKILL_REVIEW_PROMPT, "_SKILL_REVIEW_PROMPT")


def test_skill_review_prompt_handles_untrusted_sources_safely():
    _assert_untrusted_source_guidance(AIAgent._SKILL_REVIEW_PROMPT, "_SKILL_REVIEW_PROMPT")


def test_skill_review_prompt_keeps_user_preferences_out_of_global_skills():
    _assert_user_preferences_stay_out_of_global_skills(AIAgent._SKILL_REVIEW_PROMPT, "_SKILL_REVIEW_PROMPT")


def test_skill_review_prompt_support_files_are_sanitized_and_reusable():
    _assert_support_file_safety(AIAgent._SKILL_REVIEW_PROMPT, "_SKILL_REVIEW_PROMPT")


def test_skill_review_prompt_preserves_safe_skill_update_shape():
    _assert_skill_update_shape(AIAgent._SKILL_REVIEW_PROMPT, "_SKILL_REVIEW_PROMPT")


def test_skill_review_prompt_flags_overlap_and_defers_to_curator():
    """Reviewer should not consolidate live; flag overlap for the curator."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    assert "overlap" in prompt.lower()
    assert "curator" in prompt.lower(), "must defer consolidation to the curator"


# ---------------------------------------------------------------------------
# _COMBINED_REVIEW_PROMPT
# ---------------------------------------------------------------------------


def test_combined_review_prompt_has_memory_section():
    """Memory half must still cover user facts and preferences."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "**Memory**" in prompt
    assert "memory tool" in prompt


def test_combined_review_prompt_uses_conservative_global_write_stance():
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "**Skills**" in prompt
    _assert_safe_skill_review_stance(prompt, "_COMBINED_REVIEW_PROMPT")


def test_combined_review_prompt_excludes_private_and_session_content():
    _assert_private_content_exclusions(AIAgent._COMBINED_REVIEW_PROMPT, "_COMBINED_REVIEW_PROMPT")


def test_combined_review_prompt_handles_untrusted_sources_safely():
    _assert_untrusted_source_guidance(AIAgent._COMBINED_REVIEW_PROMPT, "_COMBINED_REVIEW_PROMPT")


def test_combined_review_prompt_keeps_user_preferences_out_of_global_skills():
    _assert_user_preferences_stay_out_of_global_skills(AIAgent._COMBINED_REVIEW_PROMPT, "_COMBINED_REVIEW_PROMPT")


def test_combined_review_prompt_support_files_are_sanitized_and_reusable():
    _assert_support_file_safety(AIAgent._COMBINED_REVIEW_PROMPT, "_COMBINED_REVIEW_PROMPT")


def test_combined_review_prompt_preserves_safe_skill_update_shape():
    _assert_skill_update_shape(AIAgent._COMBINED_REVIEW_PROMPT, "_COMBINED_REVIEW_PROMPT")


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
# _MEMORY_REVIEW_PROMPT — unchanged, still memory-focused
# ---------------------------------------------------------------------------


def test_memory_review_prompt_still_focused_on_user_facts():
    """Memory-only review prompt stays focused on user facts — not touched by this change."""
    prompt = AIAgent._MEMORY_REVIEW_PROMPT
    # The memory-only prompt should NOT drift into skill territory
    assert "skills_list" not in prompt
    assert "SURVEY" not in prompt
    assert "memory tool" in prompt
