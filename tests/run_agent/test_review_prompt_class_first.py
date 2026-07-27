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
















# ---------------------------------------------------------------------------
# _COMBINED_REVIEW_PROMPT
# ---------------------------------------------------------------------------

def test_combined_review_prompt_has_memory_section():
    """Memory half must still cover user facts and preferences."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "**Memory**" in prompt
    assert "memory tool" in prompt














def _assert_unresolved_failure_guidance(prompt: str, label: str) -> None:
    """Unresolved task attempts must not become persistent skill guidance."""
    lower = prompt.lower()
    assert "unresolved failures" in lower, f"{label}: must identify unresolved failures"
    assert "working method" in lower, f"{label}: must require a working method"
    assert "told the user to check manually" in lower, (
        f"{label}: must recognize an explicitly unresolved session"
    )
    assert "never the dead ends" in lower, f"{label}: must exclude failed attempts"
    assert "independently confident" in lower, (
        f"{label}: must limit exceptions to verified alternatives"
    )


def test_skill_review_prompt_rejects_unresolved_failures():
    _assert_unresolved_failure_guidance(AIAgent._SKILL_REVIEW_PROMPT, "_SKILL_REVIEW_PROMPT")


def test_combined_review_prompt_rejects_unresolved_failures():
    _assert_unresolved_failure_guidance(AIAgent._COMBINED_REVIEW_PROMPT, "_COMBINED_REVIEW_PROMPT")


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






# ---------------------------------------------------------------------------
# _MEMORY_REVIEW_PROMPT — unchanged, still memory-focused
# ---------------------------------------------------------------------------

