"""Tests for agent.verification_shadow — shadow-mode verification detector.

These tests cover the 16 adversarially-validated cases plus edge cases for
the attempt cap, nudge injection, and evidence logging.
"""

import pytest

from agent.verification_shadow import (
    detect_verifiable_claims,
    has_evidence,
    shadow_check,
    MAX_SHADOW_ATTEMPTS,
)


# ── detect_verifiable_claims: should NOT flag ──────────────────────────


class TestNoFalsePositives:
    """Cases that should produce zero claims."""

    def test_conversational_filler(self):
        text = (
            "I can see what you mean. That makes sense. "
            "The agent works by calling tools in sequence."
        )
        assert detect_verifiable_claims(text) == []

    def test_general_prose(self):
        text = (
            "The code works by iterating over a list. "
            "This works differently on Windows."
        )
        assert detect_verifiable_claims(text) == []

    def test_explanatory_context(self):
        text = (
            "It works by checking the DOM. "
            "It is working that way on all browsers."
        )
        assert detect_verifiable_claims(text) == []

    def test_comma_continuation(self):
        text = "It works, and here is the output from the test."
        assert detect_verifiable_claims(text) == []

    def test_plain_description(self):
        text = "The function takes two arguments and returns a list."
        assert detect_verifiable_claims(text) == []

    def test_short_non_claim(self):
        text = "OK, done."
        assert detect_verifiable_claims(text) == []


# ── detect_verifiable_claims: SHOULD flag ──────────────────────────────


class TestDetectsRealClaims:
    """Cases that should produce at least one claim."""

    def test_verification_claims(self):
        text = "I verified that the button turns red. Tests pass and it renders correctly."
        claims = detect_verifiable_claims(text)
        assert len(claims) > 0

    def test_task_observation(self):
        text = "I can see the output is correct and the result matches expectations."
        claims = detect_verifiable_claims(text)
        assert len(claims) > 0

    def test_fix_claim(self):
        text = "The fix works now and the change is correct."
        claims = detect_verifiable_claims(text)
        assert len(claims) > 0

    def test_standalone_claim_with_exclamation(self):
        text = "Great, it works now!"
        claims = detect_verifiable_claims(text)
        assert len(claims) > 0

    def test_period_terminated(self):
        text = "Done. It works."
        claims = detect_verifiable_claims(text)
        assert len(claims) > 0

    def test_fix_works_period(self):
        text = "The fix works."
        claims = detect_verifiable_claims(text)
        assert len(claims) > 0

    def test_solution_is_fixed(self):
        text = "The solution is fixed."
        claims = detect_verifiable_claims(text)
        assert len(claims) > 0

    def test_now_working(self):
        text = "Now it is working."
        claims = detect_verifiable_claims(text)
        assert len(claims) > 0

    def test_succeeds_already(self):
        text = "It succeeds already."
        claims = detect_verifiable_claims(text)
        assert len(claims) > 0

    def test_passes_again(self):
        text = "It passes again."
        claims = detect_verifiable_claims(text)
        assert len(claims) > 0

    def test_comma_uppercase_continuation(self):
        """Comma + uppercase letter IS terminal (not a lowercase continuation)."""
        text = "It works, Here is the output."
        claims = detect_verifiable_claims(text)
        assert len(claims) > 0

    def test_comma_standalone(self):
        """Comma at end of string is terminal."""
        text = "It works,"
        claims = detect_verifiable_claims(text)
        assert len(claims) > 0


# ── has_evidence ───────────────────────────────────────────────────────


class TestHasEvidence:
    """Evidence detection at response level."""

    def test_exit_code_is_evidence(self):
        assert has_evidence("exit_code: 0") is True

    def test_return_code_is_evidence(self):
        assert has_evidence("return code: 0") is True

    def test_traceback_is_evidence(self):
        assert has_evidence("Traceback (most recent call last):") is True

    def test_console_log_is_evidence(self):
        assert has_evidence("console.log output: hello") is True

    def test_screenshot_path_is_evidence(self):
        assert has_evidence("screenshot_path: /tmp/screen.png") is True

    def test_plain_text_no_evidence(self):
        assert has_evidence("I think it works, I didn't test it.") is False

    def test_claim_without_evidence(self):
        assert has_evidence("The fix works and tests pass.") is False


# ── shadow_check ───────────────────────────────────────────────────────


class TestShadowCheck:
    """Integration: shadow_check with different modes and states.

    shadow_check takes keyword-only args: response_text, shadow_mode, threshold.
    It returns None (no claims), a dict with mode='shadow' (log-only),
    or a dict with mode='gate' (includes a nudge string).
    """

    def test_shadow_mode_returns_dict_not_nudge(self):
        """Shadow mode returns a dict with flagged claims, not a nudge string."""
        result = shadow_check(
            response_text="It works.",
            shadow_mode="shadow",
            threshold=0.0,
        )
        assert result is not None
        assert result["mode"] == "shadow"
        assert "flagged_claims" in result
        assert "nudge" not in result

    def test_gate_mode_returns_nudge(self):
        """Gate mode returns a dict with a nudge string."""
        result = shadow_check(
            response_text="It works.",
            shadow_mode="gate",
            threshold=0.0,
        )
        assert result is not None
        assert result["mode"] == "gate"
        assert "nudge" in result
        assert isinstance(result["nudge"], str)
        assert "verification" in result["nudge"].lower() or "evidence" in result["nudge"].lower()

    def test_off_mode_returns_none(self):
        result = shadow_check(
            response_text="It works.",
            shadow_mode="off",
        )
        assert result is None

    def test_no_claims_returns_none(self):
        """No verifiable claims means no check needed."""
        result = shadow_check(
            response_text="The function takes two arguments.",
            shadow_mode="gate",
        )
        assert result is None

    def test_evidence_present_returns_none(self):
        """Claims with evidence don't trigger a nudge."""
        result = shadow_check(
            response_text="It works. exit_code: 0",
            shadow_mode="gate",
        )
        assert result is None

    def test_constant_exists(self):
        """MAX_SHADOW_ATTEMPTS is exported and is a small int."""
        assert isinstance(MAX_SHADOW_ATTEMPTS, int)
        assert 1 <= MAX_SHADOW_ATTEMPTS <= 5

    def test_threshold_filters_low_confidence(self):
        """Claims below the threshold are not flagged."""
        result = shadow_check(
            response_text="It works.",
            shadow_mode="gate",
            threshold=0.99,  # Very high threshold — nothing should pass
        )
        assert result is None

    def test_low_threshold_catches_more(self):
        """Lower threshold catches more claims."""
        result = shadow_check(
            response_text="It works.",
            shadow_mode="shadow",
            threshold=0.0,
        )
        assert result is not None
        assert result["count"] > 0
