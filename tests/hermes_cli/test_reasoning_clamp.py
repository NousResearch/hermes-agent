"""Tests for the shared reasoning-clamp logic (hermes_cli/reasoning_clamp.py).

These cover the pure threshold/footer/counter primitives that both the
streaming and non-streaming reasoning paths share, so a regression in either
path shows up here first.
"""

from hermes_cli.reasoning_clamp import (
    REASONING_CLAMP_LINES,
    StreamingReasoningClamp,
    clamp_lines,
    clamp_notice,
)


def test_threshold_default_is_ten():
    # The non-streaming recap historically clamped at 10; keep parity.
    assert REASONING_CLAMP_LINES == 10


def test_clamp_notice_is_ascii_and_mentions_count():
    notice = clamp_notice(7)
    assert "7 more lines" in notice
    assert "/reasoning full" in notice
    assert notice.isascii()


class TestClampLines:
    def test_short_block_unchanged(self):
        lines = ["a", "b", "c"]
        visible, hidden = clamp_lines(lines, show_full=False, threshold=10)
        assert visible == lines
        assert hidden == 0

    def test_exactly_threshold_not_clamped(self):
        lines = [str(i) for i in range(10)]
        visible, hidden = clamp_lines(lines, show_full=False, threshold=10)
        assert visible == lines
        assert hidden == 0

    def test_over_threshold_clamped(self):
        lines = [str(i) for i in range(25)]
        visible, hidden = clamp_lines(lines, show_full=False, threshold=10)
        assert visible == lines[:10]
        assert hidden == 15

    def test_show_full_disables_clamp(self):
        lines = [str(i) for i in range(50)]
        visible, hidden = clamp_lines(lines, show_full=True, threshold=10)
        assert visible == lines
        assert hidden == 0


class TestStreamingReasoningClamp:
    def test_shows_up_to_threshold_then_hides(self):
        clamp = StreamingReasoningClamp(show_full=False, threshold=3)
        results = [clamp.should_show() for _ in range(5)]
        assert results == [True, True, True, False, False]
        assert clamp.hidden == 2
        assert clamp.emitted == 5

    def test_show_full_never_hides(self):
        clamp = StreamingReasoningClamp(show_full=True, threshold=3)
        assert all(clamp.should_show() for _ in range(20))
        assert clamp.hidden == 0

    def test_under_threshold_no_hidden(self):
        clamp = StreamingReasoningClamp(show_full=False, threshold=10)
        for _ in range(4):
            clamp.should_show()
        assert clamp.hidden == 0

    def test_hidden_count_matches_clamp_lines(self):
        # The streaming counter and the batch splitter must agree.
        lines = [str(i) for i in range(30)]
        clamp = StreamingReasoningClamp(show_full=False, threshold=10)
        for _ in lines:
            clamp.should_show()
        _, batch_hidden = clamp_lines(lines, show_full=False, threshold=10)
        assert clamp.hidden == batch_hidden
