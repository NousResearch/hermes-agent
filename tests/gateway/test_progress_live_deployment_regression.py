"""Deployment regression for plain-English Discord progress questions."""

import pytest


def test_live_tree_recognizes_plain_english_burndown_progress_question():
    """Removing the progress runtime or its natural-language admission must fail."""
    try:
        from gateway.progress_queries import is_progress_query
    except ModuleNotFoundError:
        pytest.fail("plain-English progress routing is absent from the live tree")

    assert is_progress_query(
        "How’s the burndown patches going how much more do we have to do till we can send a PR"
    )
