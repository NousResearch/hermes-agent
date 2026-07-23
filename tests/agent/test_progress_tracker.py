"""Tests for agent.progress_tracker — cross-turn progress detection."""

from agent.progress_tracker import ProgressTracker


class TestProgressTracker:
    def test_no_warning_within_threshold(self):
        tracker = ProgressTracker(warn_after=5, halt_after=10)
        for _ in range(4):
            tracker.record_iteration()
            assert tracker.check() is None

    def test_warning_at_threshold(self):
        tracker = ProgressTracker(warn_after=5, halt_after=10)
        for _ in range(5):
            tracker.record_iteration()
        result = tracker.check()
        assert result is not None
        assert "not converging" in result

    def test_halt_at_threshold(self):
        tracker = ProgressTracker(warn_after=5, halt_after=10)
        for _ in range(10):
            tracker.record_iteration()
        result = tracker.check()
        assert result is not None
        assert "MUST provide a final response" in result

    def test_text_response_resets_counter(self):
        tracker = ProgressTracker(warn_after=5, halt_after=10)
        for _ in range(8):
            tracker.record_iteration()
        assert tracker.check() is not None  # warning
        tracker.record_text_response()
        assert tracker.check() is None  # reset

    def test_file_mutation_resets_counter(self):
        tracker = ProgressTracker(warn_after=5, halt_after=10)
        for _ in range(8):
            tracker.record_iteration()
        tracker.record_file_mutation("write_file")
        assert tracker.check() is None

    def test_non_mutation_tool_does_not_reset(self):
        tracker = ProgressTracker(warn_after=5, halt_after=10)
        for _ in range(8):
            tracker.record_iteration()
        tracker.record_file_mutation("read_file")  # not a mutation tool
        assert tracker.check() is not None  # still warned

    def test_disabled_tracker(self):
        tracker = ProgressTracker(enabled=False)
        for _ in range(50):
            tracker.record_iteration()
        assert tracker.check() is None

    def test_reset(self):
        tracker = ProgressTracker(warn_after=3, halt_after=5)
        for _ in range(4):
            tracker.record_iteration()
        assert tracker.check() is not None
        tracker.reset()
        assert tracker.check() is None

    def test_terminal_resets_counter(self):
        tracker = ProgressTracker(warn_after=5, halt_after=10)
        for _ in range(8):
            tracker.record_iteration()
        tracker.record_file_mutation("terminal")
        assert tracker.check() is None

    def test_patch_resets_counter(self):
        tracker = ProgressTracker(warn_after=5, halt_after=10)
        for _ in range(8):
            tracker.record_iteration()
        tracker.record_file_mutation("patch")
        assert tracker.check() is None

    def test_sequential_warnings_escalate(self):
        tracker = ProgressTracker(warn_after=3, halt_after=6)
        for _ in range(3):
            tracker.record_iteration()
        warn = tracker.check()
        assert warn is not None and "not converging" in warn
        for _ in range(3):
            tracker.record_iteration()
        halt = tracker.check()
        assert halt is not None and "MUST provide a final response" in halt

    def test_custom_thresholds(self):
        tracker = ProgressTracker(warn_after=2, halt_after=4)
        tracker.record_iteration()
        assert tracker.check() is None
        tracker.record_iteration()
        assert tracker.check() is not None
