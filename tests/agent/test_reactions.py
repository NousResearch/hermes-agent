"""Behavior tests for the token-free reaction detector."""

import pytest

from agent.reactions import VIBE, detect_reaction


@pytest.mark.parametrize(
    "text",
    [
        "good bot",
        "Good Bot!",
        "ily",
        "ilysm",
        "i love you",
        "love you",
        "love u",
        "luv ya",
        "thanks",
        "thank you",
        "thx",
        "ty",
        "tysm",
        "you're the best <3",
        "here you go <33",
        "❤️",
        "🥰 amazing",
        "sending 💖",
        "great job, thank you so much!",
    ],
)
def test_affection_fires_vibe(text):
    assert detect_reaction(text) == VIBE


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "run the tests",
        "this is great",  # positive sentiment, NOT affection — must not fire
        "awesome work on the refactor",
        "it's broken </3",  # broken heart is not affection
        "the ferry departs at 3",  # 'ty' must be word-bounded, not match 'ferTY'-like
        "commit the changes",
    ],
)
def test_neutral_or_negative_does_not_fire(text):
    assert detect_reaction(text) is None


def test_case_insensitive_invariant():
    # Casing must never change the classification.
    for text in ("ILY", "iLy", "ily"):
        assert detect_reaction(text) == detect_reaction("ily")
