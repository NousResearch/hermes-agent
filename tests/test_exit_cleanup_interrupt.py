"""Tests for KeyboardInterrupt handling in exit cleanup paths.

``except Exception`` does not catch ``KeyboardInterrupt`` (which inherits
from ``BaseException``).  A second Ctrl+C during exit cleanup must not
abort remaining cleanup steps.  These tests exercise the actual production
code paths — not a copy of the try/except pattern.
"""

import atexit
import weakref
from unittest.mock import MagicMock, patch, call

import pytest


class TestCronJobCleanup:
    """cron/scheduler.py — end_session + close in the finally block."""

    def test_keyboard_interrupt_in_end_session_does_not_skip_close(self):
        """If end_session raises KeyboardInterrupt, close() must still run."""
        mock_db = MagicMock()
        mock_db.end_session.side_effect = KeyboardInterrupt

        from cron import scheduler

        job = {
            "id": "test-job-1",
            "name": "test cleanup",
            "prompt": "hello",
            "schedule": "0 9 * * *",
            "model": "test/model",
        }

        with patch("hermes_state.SessionDB", return_value=mock_db), \
             patch.object(scheduler, "_build_job_prompt", return_value="hello"), \
             patch.object(scheduler, "_resolve_origin", return_value=None), \
             patch.object(scheduler, "_resolve_delivery_target", return_value=None), \
             patch("dotenv.load_dotenv", return_value=None), \
             patch("run_agent.AIAgent") as MockAgent:
            # Make the agent raise immediately so we hit the finally block
            MockAgent.return_value.run_conversation.side_effect = RuntimeError("boom")
            scheduler.run_job(job)

        mock_db.end_session.assert_called_once()
        mock_db.close.assert_called_once()

    def test_keyboard_interrupt_in_close_does_not_propagate(self):
        """If close() raises KeyboardInterrupt, it must not escape run_job."""
        mock_db = MagicMock()
        mock_db.close.side_effect = KeyboardInterrupt

        from cron import scheduler

        job = {
            "id": "test-job-2",
            "name": "test close interrupt",
            "prompt": "hello",
            "schedule": "0 9 * * *",
            "model": "test/model",
        }

        with patch("hermes_state.SessionDB", return_value=mock_db), \
             patch.object(scheduler, "_build_job_prompt", return_value="hello"), \
             patch.object(scheduler, "_resolve_origin", return_value=None), \
             patch.object(scheduler, "_resolve_delivery_target", return_value=None), \
             patch("dotenv.load_dotenv", return_value=None), \
             patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value.run_conversation.side_effect = RuntimeError("boom")
            # Must not raise
            scheduler.run_job(job)

        mock_db.end_session.assert_called_once()
        mock_db.close.assert_called_once()
