#!/usr/bin/env python3
"""
Tests for the queue worker daemon.

Validates that the worker can poll for jobs, execute them, and handle
lease coordination properly.
"""

import pytest
import tempfile
import os
import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path so we can import from the hermes-agent modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.worker_daemon import QueueWorker, DEFAULT_WORKER_ID


class TestQueueWorker:
    """Test suite for the QueueWorker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.worker = QueueWorker(
            worker_id="test-worker",
            poll_interval=1,
            max_concurrent=2
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        self.worker.shutdown()
    
    def test_worker_initialization(self):
        """Test that worker initializes correctly."""
        assert self.worker.worker_id == "test-worker"
        assert self.worker.poll_interval == 1
        assert self.worker.max_concurrent == 2
        assert self.worker.executor is not None
    
    @patch('cron.worker_daemon.get_due_jobs')
    def test_poll_with_no_jobs(self, mock_get_due_jobs):
        """Test polling when no jobs are due."""
        mock_get_due_jobs.return_value = []
        
        jobs_started = self.worker.poll_and_execute()
        
        assert jobs_started == 0
        mock_get_due_jobs.assert_called_once()
    
    @patch('cron.worker_daemon.get_due_jobs')
    @patch('cron.worker_daemon.mark_job_started')
    @patch('cron.worker_daemon.run_job')
    @patch('cron.worker_daemon.save_job_output')
    @patch('cron.worker_daemon._deliver_result')
    @patch('cron.worker_daemon.mark_job_run')
    def test_successful_job_execution(
        self,
        mock_mark_job_run,
        mock_deliver_result,
        mock_save_job_output,
        mock_run_job,
        mock_mark_job_started,
        mock_get_due_jobs
    ):
        """Test successful execution of a single job."""
        # Setup mocks
        test_job = {"id": "test-job-1", "name": "Test Job"}
        mock_get_due_jobs.return_value = [test_job]
        mock_run_job.return_value = (True, "output content", "Job completed", None)
        mock_save_job_output.return_value = "/tmp/output.md"
        mock_deliver_result.return_value = None
        
        # Execute
        jobs_started = self.worker.poll_and_execute()
        
        # Wait briefly for async execution
        time.sleep(0.1)
        
        # Verify
        assert jobs_started == 1
        mock_get_due_jobs.assert_called_once()
        mock_mark_job_started.assert_called_once_with("test-job-1")
        mock_run_job.assert_called_once_with(test_job)
        mock_save_job_output.assert_called_once_with("test-job-1", "output content")
        mock_deliver_result.assert_called_once()
        mock_mark_job_run.assert_called_once_with(
            "test-job-1", success=True, error=None, delivery_error=None
        )
    
    @patch('cron.worker_daemon.get_due_jobs')
    @patch('cron.worker_daemon.mark_job_started')
    @patch('cron.worker_daemon.run_job')
    @patch('cron.worker_daemon.save_job_output')
    @patch('cron.worker_daemon.mark_job_run')
    def test_failed_job_execution(
        self,
        mock_mark_job_run,
        mock_save_job_output,
        mock_run_job,
        mock_mark_job_started,
        mock_get_due_jobs
    ):
        """Test handling of a failed job."""
        # Setup mocks
        test_job = {"id": "test-job-2", "name": "Failed Job"}
        mock_get_due_jobs.return_value = [test_job]
        mock_run_job.return_value = (False, "error output", "", "Something went wrong")
        mock_save_job_output.return_value = "/tmp/error_output.md"
        
        # Execute
        jobs_started = self.worker.poll_and_execute()
        
        # Wait briefly for async execution
        time.sleep(0.1)
        
        # Verify
        assert jobs_started == 1
        mock_mark_job_run.assert_called_once_with(
            "test-job-2", success=False, error="Something went wrong", delivery_error=None
        )
    
    @patch('cron.worker_daemon.get_due_jobs')
    @patch('cron.worker_daemon.mark_job_started')
    @patch('cron.worker_daemon.run_job')
    @patch('cron.worker_daemon.save_job_output')
    @patch('cron.worker_daemon._deliver_result')
    @patch('cron.worker_daemon.mark_job_run')
    def test_silent_job_no_delivery(
        self,
        mock_mark_job_run,
        mock_deliver_result,
        mock_save_job_output,
        mock_run_job,
        mock_mark_job_started,
        mock_get_due_jobs
    ):
        """Test that SILENT jobs don't trigger delivery."""
        # Setup mocks
        test_job = {"id": "test-job-3", "name": "Silent Job"}
        mock_get_due_jobs.return_value = [test_job]
        mock_run_job.return_value = (True, "output content", "[SILENT]", None)
        mock_save_job_output.return_value = "/tmp/silent_output.md"
        
        # Execute
        jobs_started = self.worker.poll_and_execute()
        
        # Wait briefly for async execution
        time.sleep(0.1)
        
        # Verify no delivery was attempted
        assert jobs_started == 1
        mock_deliver_result.assert_not_called()
        mock_mark_job_run.assert_called_once_with(
            "test-job-3", success=True, error=None, delivery_error=None
        )
    
    @patch('cron.worker_daemon.get_due_jobs')
    def test_duplicate_job_prevention(self, mock_get_due_jobs):
        """Test that the same job isn't started twice concurrently."""
        # Setup
        test_job = {"id": "test-job-4", "name": "Duplicate Test"}
        mock_get_due_jobs.return_value = [test_job, test_job]  # Same job twice
        
        with patch('cron.worker_daemon.mark_job_started'), \
             patch('cron.worker_daemon.run_job') as mock_run_job:
            
            # Make run_job slow so we can test concurrency
            def slow_run_job(job):
                time.sleep(0.2)
                return (True, "output", "result", None)
            mock_run_job.side_effect = slow_run_job
            
            # Execute - should only start one job even though two are "due"
            jobs_started = self.worker.poll_and_execute()
            
            # Second poll while first is still running
            jobs_started_2 = self.worker.poll_and_execute()
            
            # Wait for jobs to complete
            time.sleep(0.5)
            
            # Verify only one job was actually started
            assert jobs_started == 1
            assert jobs_started_2 == 0
    
    def test_shutdown(self):
        """Test graceful shutdown."""
        # Worker should shutdown without hanging
        self.worker.shutdown()
        # Executor should be shut down
        assert self.worker.executor._shutdown
    
    @patch('cron.worker_daemon.get_due_jobs')
    def test_poll_exception_handling(self, mock_get_due_jobs):
        """Test that polling exceptions are handled gracefully."""
        mock_get_due_jobs.side_effect = Exception("Database error")
        
        # Should not raise, should return 0
        jobs_started = self.worker.poll_and_execute()
        assert jobs_started == 0


class TestWorkerDaemonCLI:
    """Test the CLI interface for the worker daemon."""
    
    def test_default_worker_id_generation(self):
        """Test that default worker IDs are generated correctly."""
        worker_id = DEFAULT_WORKER_ID
        assert worker_id.startswith("worker-")
        assert len(worker_id) > len("worker-")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])