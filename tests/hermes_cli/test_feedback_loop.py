"""
Tests for feedback_loop module (P2-10).

Covers: message classification, signal recording, quality scoring,
reporting, implicit skip detection, and eval label integration.
"""

import os
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from hermes_cli.feedback_loop import (
    SignalType,
    SignalStrength,
    FeedbackSignal,
    FeedbackReport,
    FeedbackLoop,
    classify_message,
    _extract_correction,
)


# ---------------------------------------------------------------------------
# Message classification
# ---------------------------------------------------------------------------


class TestClassifyApprove:
    def test_explicit_approval(self):
        signal = classify_message("looks good, approved")
        assert signal.signal_type == SignalType.APPROVE
        assert signal.strength == SignalStrength.EXPLICIT

    def test_lgtm(self):
        signal = classify_message("lgtm, ship it")
        assert signal.signal_type == SignalType.APPROVE

    def test_emoji_thumbs_up(self):
        signal = classify_message("👍")
        assert signal.signal_type == SignalType.APPROVE

    def test_merge_command(self):
        signal = classify_message("go ahead and merge")
        assert signal.signal_type == SignalType.APPROVE

    def test_simple_yes(self):
        signal = classify_message("yes")
        assert signal.signal_type == SignalType.APPROVE


class TestClassifyReject:
    def test_explicit_rejection(self):
        signal = classify_message("no, this is wrong")
        assert signal.signal_type == SignalType.REJECT
        assert signal.strength == SignalStrength.EXPLICIT

    def test_redo_command(self):
        signal = classify_message("redo this entire thing")
        assert signal.signal_type == SignalType.REJECT

    def test_revert(self):
        signal = classify_message("revert the last change")
        assert signal.signal_type == SignalType.REJECT

    def test_thumbs_down(self):
        signal = classify_message("👎")
        assert signal.signal_type == SignalType.REJECT

    def test_stop_command(self):
        signal = classify_message("don't do that, revert it")
        assert signal.signal_type == SignalType.REJECT


class TestClassifyCorrect:
    def test_not_x_but_y(self):
        signal = classify_message(
            "that's not the right path, but /home/kensei/.hermes is"
        )
        assert signal.signal_type == SignalType.CORRECT
        assert signal.correction_target != ""
        assert signal.correction_replacement != ""

    def test_should_be(self):
        signal = classify_message("the port should be 8080")
        assert signal.signal_type == SignalType.CORRECT

    def test_change_x_to_y(self):
        signal = classify_message("change the model to claude-sonnet-4")
        assert signal.signal_type == SignalType.CORRECT
        assert "model" in signal.correction_target.lower()
        assert "claude-sonnet-4" in signal.correction_replacement.lower()

    def test_actually(self):
        signal = classify_message("actually the server runs on port 3000")
        assert signal.signal_type == SignalType.CORRECT

    def test_correct_takes_priority_over_reject(self):
        # "no, that's not X but Y" — should be CORRECT, not REJECT
        signal = classify_message("no, that's not postgres but sqlite")
        assert signal.signal_type == SignalType.CORRECT


class TestClassifyClarify:
    def test_what_about(self):
        signal = classify_message("what about the database migration?")
        assert signal.signal_type == SignalType.CLARIFY

    def test_explain(self):
        signal = classify_message("can you explain why?")
        assert signal.signal_type == SignalType.CLARIFY

    def test_question_mark(self):
        signal = classify_message("why was this changed?")
        assert signal.signal_type == SignalType.CLARIFY

    def test_not_clear(self):
        signal = classify_message("I don't understand, this is confusing")
        assert signal.signal_type == SignalType.CLARIFY


class TestClassifyUnknown:
    def test_neutral_message(self):
        signal = classify_message("let me know when it's done")
        assert signal.signal_type == SignalType.UNKNOWN

    def test_empty_message(self):
        signal = classify_message("")
        assert signal.signal_type == SignalType.UNKNOWN


# ---------------------------------------------------------------------------
# Correction extraction
# ---------------------------------------------------------------------------


class TestExtractCorrection:
    def test_not_x_but_y(self):
        target, replacement = _extract_correction("not port 3000 but 8080")
        assert "3000" in target
        assert "8080" in replacement

    def test_should_be(self):
        target, replacement = _extract_correction("the config path should be /etc/hermes")
        assert "config" in target.lower()
        assert "/etc/hermes" in replacement

    def test_change_x_to_y(self):
        target, replacement = _extract_correction("change the timeout to 600")
        assert "timeout" in target.lower()
        assert "600" in replacement

    def test_no_match(self):
        target, replacement = _extract_correction("random message without patterns")
        assert target == ""
        assert replacement == ""


# ---------------------------------------------------------------------------
# FeedbackLoop persistence
# ---------------------------------------------------------------------------


class TestFeedbackLoopPersistence:
    def test_record_and_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "feedback.db")
            loop = FeedbackLoop(store_path=db_path)

            signal = classify_message("approved, looks great")
            signal.linked_task_id = "t_abc123"
            loop.record(signal)

            report = loop.report(days=365)
            assert report.total_signals == 1
            assert report.approval_rate == 1.0

    def test_record_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "feedback.db")
            loop = FeedbackLoop(store_path=db_path)

            signal = classify_message("change the port to 8080")
            signal.linked_task_id = "t_xyz"
            loop.record(signal)

            report = loop.report(days=365)
            assert report.correction_rate == 1.0

    def test_rejection_affects_quality_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "feedback.db")
            loop = FeedbackLoop(store_path=db_path)

            # Record an approval
            loop.record(classify_message("approved"))
            assert loop.agent_quality_score(days=365) > 0.5

            # Record a rejection — score should drop
            loop.record(classify_message("no, this is wrong"))
            score = loop.agent_quality_score(days=365)
            assert score <= 0.5  # rejections outweigh approvals


# ---------------------------------------------------------------------------
# Agent quality scoring
# ---------------------------------------------------------------------------


class TestAgentQualityScore:
    def test_neutral_with_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = FeedbackLoop(store_path=os.path.join(tmp, "fb.db"))
            assert loop.agent_quality_score(days=30) == 0.5

    def test_all_approvals_high_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = FeedbackLoop(store_path=os.path.join(tmp, "fb.db"))
            for _ in range(5):
                loop.record(classify_message("approved, looks great"))
            score = loop.agent_quality_score(days=30)
            assert score >= 0.8

    def test_mixed_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = FeedbackLoop(store_path=os.path.join(tmp, "fb.db"))
            loop.record(classify_message("approved"))
            loop.record(classify_message("approved"))
            loop.record(classify_message("no, wrong"))
            score = loop.agent_quality_score(days=30)
            # (2 - 1) / 3 = 0.33, + 0.5 = 0.83
            assert 0.5 < score < 1.0


# ---------------------------------------------------------------------------
# Feedback report
# ---------------------------------------------------------------------------


class TestFeedbackReport:
    def test_empty_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = FeedbackLoop(store_path=os.path.join(tmp, "fb.db"))
            report = loop.report(days=7)
            assert report.total_signals == 0
            assert report.trend == "stable"

    def test_task_grouping(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = FeedbackLoop(store_path=os.path.join(tmp, "fb.db"))
            s1 = classify_message("approved")
            s1.linked_task_id = "t_1"
            s2 = classify_message("no, wrong")
            s2.linked_task_id = "t_2"
            loop.record(s1)
            loop.record(s2)

            report = loop.report(days=365)
            assert "t_1" in report.by_task
            assert "t_2" in report.by_task

    def test_trend_calculation(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = FeedbackLoop(store_path=os.path.join(tmp, "fb.db"))
            # Two approvals recorded — should be stable (not enough variance)
            loop.record(classify_message("approved"))
            loop.record(classify_message("approved"))
            report = loop.report(days=365)
            assert report.trend in ("stable", "improving")


# ---------------------------------------------------------------------------
# Implicit skip detection
# ---------------------------------------------------------------------------


class TestImplicitSkip:
    def test_detects_skip_after_silence(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = FeedbackLoop(store_path=os.path.join(tmp, "fb.db"))
            # Decision made 5 hours ago, no response — should trigger
            decision_time = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
            result = loop.detect_implicit_skips(
                task_id="t_test",
                decision_timestamp=decision_time,
                silence_hours=4,
            )
            assert result is True

    def test_no_skip_within_grace_period(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = FeedbackLoop(store_path=os.path.join(tmp, "fb.db"))
            # Decision made 1 hour ago — still within grace period
            decision_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            result = loop.detect_implicit_skips(
                task_id="t_test",
                decision_timestamp=decision_time,
                silence_hours=4,
            )
            assert result is False

    def test_no_skip_with_existing_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = FeedbackLoop(store_path=os.path.join(tmp, "fb.db"))
            decision_time = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()

            # Pre-record an approval after the decision
            signal = classify_message("approved")
            signal.linked_task_id = "t_test"
            signal.timestamp = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
            loop.record(signal)

            result = loop.detect_implicit_skips(
                task_id="t_test",
                decision_timestamp=decision_time,
                silence_hours=4,
            )
            assert result is False  # got feedback, no skip


# ---------------------------------------------------------------------------
# Eval label integration
# ---------------------------------------------------------------------------


class TestEvalLabelIntegration:
    def test_signals_as_eval_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = FeedbackLoop(store_path=os.path.join(tmp, "fb.db"))

            s1 = classify_message("approved")
            s1.linked_task_id = "t_eval"
            s2 = classify_message("wrong, redo")
            s2.linked_task_id = "t_eval"
            loop.record(s1)
            loop.record(s2)

            labels = loop.signals_as_eval_labels(task_id="t_eval", days=365)
            assert len(labels) == 2
            verdicts = [l["verdict"] for l in labels]
            assert "pass" in verdicts
            assert "fail" in verdicts

    def test_empty_labels_for_unknown_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = FeedbackLoop(store_path=os.path.join(tmp, "fb.db"))
            labels = loop.signals_as_eval_labels(task_id="nonexistent")
            assert labels == []
