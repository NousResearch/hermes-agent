"""Tests for the user correction detector."""

from __future__ import annotations

import pytest

from agent.signal_sources.user_correction_detector import (
    DEFAULT_CORRECTION_PATTERNS,
    detect,
    reset_patterns,
)


@pytest.fixture(autouse=True)
def _restore_default_patterns():
    """Make sure each test starts with the default patterns, even if
    a previous test called ``reset_patterns``.
    """
    from agent.signal_sources import user_correction_detector as mod

    mod.reset_to_defaults()
    yield
    mod.reset_to_defaults()


class TestDetectEnglish:
    @pytest.mark.parametrize(
        "msg",
        [
            "that's wrong",
            "That is wrong.",
            "you're wrong",
            "You are wrong",
            "Wrong.",
            "incorrect",
            "redo",
            "redo it",
            "try again",
            "do it again",
            "not quite",
            "not right",
            "fix this",
            "fix it",
            "this is wrong",
            "this isn't right",
            "wrong answer",
        ],
    )
    def test_positive(self, msg: str):
        assert detect([msg]) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "thanks!",
            "great work",
            "looks good",
            "this is helpful",
            "please continue",
            "good answer",
            "all correct",
        ],
    )
    def test_negative(self, msg: str):
        assert detect([msg]) is False


class TestDetectChinese:
    @pytest.mark.parametrize(
        "msg",
        [
            "不对",
            "不对的",
            "错了",
            "重新做",
            "再试一次",
            "改一下",
            "改成蓝色",
            "不是这个",
            "应该是红色",
        ],
    )
    def test_positive(self, msg: str):
        assert detect([msg]) is True


class TestDetectMulti:
    def test_empty(self):
        assert detect([]) is False
        assert detect([""]) is False
        assert detect(["", ""]) is False

    def test_first_match_wins(self):
        assert detect(["thanks!", "that's wrong", "thanks again"]) is True

    def test_no_match_anywhere(self):
        assert detect(["nice work", "thanks", "see you"]) is False


class TestDetectOtherLanguages:
    @pytest.mark.parametrize(
        "msg",
        [
            "está mal",
            "incorrecto",
            "hazlo de nuevo",
            "c'est faux",
            "refais",
            "falsch",
            "nochmal",
        ],
    )
    def test_positive(self, msg: str):
        assert detect([msg]) is True


class TestResetPatterns:
    def test_empty_patterns_disables_detection(self):
        reset_patterns([])
        assert detect(["that's wrong", "不对", "redo"]) is False

    def test_custom_patterns(self):
        reset_patterns(["nope"])
        assert detect(["nope"]) is True
        assert detect(["that's wrong"]) is False
