"""Regression tests verifying create_execution failure does not wedge _running_job_ids."""

from unittest.mock import MagicMock, patch
import pytest

from cron.scheduler import tick, get_running_job_ids, _running_job_ids, _running_lock


class TestCronCreateExecutionWedge:
    """Verify _running_job_ids is cleared if create_execution raises an exception in _tick_impl."""

    def test_create_execution_failure_cleans_running_job_ids(self):
        job = {"id": "test_wedge_job_1", "name": "Test Wedge Job"}

        with patch("cron.scheduler.open", MagicMock()), \
             patch("cron.scheduler.get_due_jobs", return_value=[job]), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.create_execution", side_effect=RuntimeError("DB Lock Error")):
            tick(verbose=False)

        assert "test_wedge_job_1" not in get_running_job_ids()

        with _running_lock:
            assert "test_wedge_job_1" not in _running_job_ids

    def test_subsequent_tick_succeeds_after_create_execution_error(self):
        job = {"id": "test_wedge_job_2", "name": "Test Wedge Job 2"}

        # First tick attempt fails at create_execution
        with patch("cron.scheduler.open", MagicMock()), \
             patch("cron.scheduler.get_due_jobs", return_value=[job]), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.create_execution", side_effect=RuntimeError("DB Lock Error")):
            tick(verbose=False)

        assert "test_wedge_job_2" not in get_running_job_ids()

        # Second tick attempt succeeds at create_execution
        mock_exec = {"id": "exec_123"}
        mock_pool = MagicMock()

        with patch("cron.scheduler.open", MagicMock()), \
             patch("cron.scheduler.get_due_jobs", return_value=[job]), \
             patch("cron.scheduler.advance_next_run"), \
             patch("cron.scheduler.create_execution", return_value=mock_exec), \
             patch("cron.scheduler._get_parallel_pool", return_value=mock_pool):
            tick(verbose=False)

        mock_pool.submit.assert_called_once()

        # Clean up running state after mock submit
        with _running_lock:
            _running_job_ids.discard("test_wedge_job_2")
