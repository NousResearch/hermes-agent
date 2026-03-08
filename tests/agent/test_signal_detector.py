"""Tests for agent.signal_detector — implicit signal detection & nudges."""

import pytest

from agent.signal_detector import (
    SIGNALS,
    NUDGES,
    detect_signals,
    build_nudge_prefix,
)


# ---------------------------------------------------------------------------
# detect_signals() — individual signal categories
# ---------------------------------------------------------------------------


class TestFrustration:
    def test_explicit_frustration(self):
        result = detect_signals("ugh this is so stupid, nothing works")
        assert result["frustration"] is True

    def test_still_not_working(self):
        result = detect_signals("it still won't compile after all my changes")
        assert result["frustration"] is True

    def test_keeping_failing(self):
        result = detect_signals("the tests keep failing for hours")
        assert result["frustration"] is True


class TestConfusion:
    def test_dont_understand(self):
        result = detect_signals("I don't understand what this function does")
        assert result["confusion"] is True

    def test_makes_no_sense(self):
        result = detect_signals("this error makes no sense to me")
        assert result["confusion"] is True

    def test_stuck(self):
        result = detect_signals("I'm completely stuck on this problem")
        assert result["confusion"] is True


class TestUrgency:
    def test_asap(self):
        result = detect_signals("I need this fixed asap")
        assert result["urgency"] is True

    def test_deadline(self):
        result = detect_signals("the deadline is due tomorrow")
        assert result["urgency"] is True

    def test_immediately(self):
        result = detect_signals("please do this immediately")
        assert result["urgency"] is True


class TestFatigue:
    def test_exhausted(self):
        result = detect_signals("I'm exhausted, been debugging all day")
        assert result["fatigue"] is True

    def test_brain_fried(self):
        result = detect_signals("my brain is fried, can't think straight")
        assert result["fatigue"] is True

    def test_burned_out(self):
        result = detect_signals("I'm completely burned out on this")
        assert result["fatigue"] is True


class TestLearning:
    def test_new_to(self):
        result = detect_signals("I'm new to Python, can you help?")
        assert result["learning"] is True

    def test_explain(self):
        result = detect_signals("can you explain how async works?")
        assert result["learning"] is True

    def test_eli5(self):
        result = detect_signals("eli5 what is a closure?")
        assert result["learning"] is True


class TestExploration:
    def test_what_if(self):
        result = detect_signals("what if we used a different algorithm?")
        assert result["exploration"] is True

    def test_brainstorm(self):
        result = detect_signals("let's brainstorm some ideas for the API")
        assert result["exploration"] is True


class TestCelebration:
    def test_it_works(self):
        result = detect_signals("it works! the tests are all passing!")
        assert result["celebration"] is True

    def test_figured_it_out(self):
        result = detect_signals("I figured it out, the bug was in the parser")
        assert result["celebration"] is True


class TestAnxiety:
    def test_worried(self):
        result = detect_signals("I'm worried this change might break production")
        assert result["anxiety"] is True

    def test_what_if_goes_wrong(self):
        result = detect_signals("what if the migration goes wrong?")
        assert result["anxiety"] is True


class TestOverwhelm:
    def test_too_much(self):
        result = detect_signals("there's too much to do, where do I start?")
        assert result["overwhelm"] is True

    def test_information_overload(self):
        result = detect_signals("information overload, I can't process this")
        assert result["overwhelm"] is True


class TestDeepWork:
    def test_focus(self):
        result = detect_signals("I need to focus, just give me the code")
        assert result["deep_work"] is True

    def test_dont_interrupt(self):
        result = detect_signals("don't interrupt, I'm in flow state")
        assert result["deep_work"] is True


# ---------------------------------------------------------------------------
# detect_signals() — edge cases & cross-cutting
# ---------------------------------------------------------------------------


class TestNeutralMessages:
    """Neutral/casual messages should not trigger any signals."""

    def test_simple_greeting(self):
        result = detect_signals("hello, how are you?")
        assert not any(result.values())

    def test_normal_request(self):
        result = detect_signals("please create a function that sorts a list")
        assert not any(result.values())

    def test_technical_discussion(self):
        result = detect_signals("the API returns JSON with a status field")
        assert not any(result.values())


class TestMultipleSignals:
    """A single message can trigger multiple signals simultaneously."""

    def test_frustration_and_fatigue(self):
        result = detect_signals(
            "this stupid thing still won't compile, I've been at this for hours"
        )
        assert result["frustration"] is True
        assert result["fatigue"] is True

    def test_confusion_and_learning(self):
        result = detect_signals(
            "I'm new to React and I don't understand hooks at all"
        )
        assert result["learning"] is True
        assert result["confusion"] is True

    def test_urgency_and_anxiety(self):
        result = detect_signals(
            "deadline is due tomorrow and I'm worried it might break"
        )
        assert result["urgency"] is True
        assert result["anxiety"] is True


class TestEmptyAndEdgeCases:
    def test_empty_string(self):
        result = detect_signals("")
        assert not any(result.values())

    def test_case_insensitive(self):
        result = detect_signals("UGH THIS IS SO STUPID")
        assert result["frustration"] is True

    def test_all_signal_names_have_nudges(self):
        """Every signal category must have a corresponding nudge string."""
        for name in SIGNALS:
            assert name in NUDGES, f"Signal '{name}' missing from NUDGES"

    def test_returns_all_signal_keys(self):
        """detect_signals always returns a key for every signal category."""
        result = detect_signals("hello")
        assert set(result.keys()) == set(SIGNALS.keys())


# ---------------------------------------------------------------------------
# build_nudge_prefix()
# ---------------------------------------------------------------------------


class TestBuildNudgePrefix:
    def test_no_signals_returns_empty(self):
        prefix = build_nudge_prefix("hello, how are you?")
        assert prefix == ""

    def test_single_signal_format(self):
        prefix = build_nudge_prefix("ugh this is so stupid")
        assert prefix.startswith("[Context: ")
        assert prefix.endswith("]\n\n")
        assert "frustrated" in prefix.lower() or "frustrat" in prefix.lower()

    def test_multiple_signals_combined(self):
        prefix = build_nudge_prefix(
            "I'm exhausted and nothing works, tried everything"
        )
        assert prefix.startswith("[Context: ")
        assert prefix.endswith("]\n\n")
        # Should contain both frustration and fatigue nudges
        assert "frustrat" in prefix.lower()
        assert "tired" in prefix.lower() or "fatigue" in prefix.lower()

    def test_empty_message(self):
        assert build_nudge_prefix("") == ""

    def test_prefix_does_not_modify_message(self):
        """build_nudge_prefix returns only the prefix, not the message."""
        msg = "ugh this is broken"
        prefix = build_nudge_prefix(msg)
        assert msg not in prefix
