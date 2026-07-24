from __future__ import annotations

from unittest.mock import patch

from agent.turn_finalizer import _should_spawn_background_review


def test_background_review_is_suppressed_for_active_goal():
    with patch("agent.turn_finalizer._goal_active_for_session", return_value=True):
        assert not _should_spawn_background_review(
            final_response="progress checkpoint",
            interrupted=False,
            should_review_memory=True,
            should_review_skills=False,
            session_id="session-with-goal",
        )


def test_background_review_runs_without_active_goal():
    with patch("agent.turn_finalizer._goal_active_for_session", return_value=False):
        assert _should_spawn_background_review(
            final_response="done for now",
            interrupted=False,
            should_review_memory=True,
            should_review_skills=False,
            session_id="session-without-goal",
        )


def test_background_review_still_respects_existing_guards():
    with patch("agent.turn_finalizer._goal_active_for_session", return_value=False):
        assert not _should_spawn_background_review(
            final_response="",
            interrupted=False,
            should_review_memory=True,
            should_review_skills=False,
            session_id="session-without-goal",
        )
        assert not _should_spawn_background_review(
            final_response="done for now",
            interrupted=True,
            should_review_memory=True,
            should_review_skills=False,
            session_id="session-without-goal",
        )
        assert not _should_spawn_background_review(
            final_response="done for now",
            interrupted=False,
            should_review_memory=False,
            should_review_skills=False,
            session_id="session-without-goal",
        )
