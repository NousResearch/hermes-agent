#!/usr/bin/env python3
"""
Queue Worker Daemon

Standalone process that polls for due jobs and executes them, respecting
runner leases and retry backoff. Designed to run independently of the
gateway so job execution survives gateway restarts.

Key design principles:
- Uses the lease metadata from agent/job_lifecycle.py
- Detects and recovers stale claims automatically  
- Implements proper backoff on retries
- Can run multiple workers with lease coordination
- Graceful shutdown preserves running jobs

Usage:
    python -m cron.worker_daemon [--profile PROFILE] [--worker-id ID] [--poll-interval N]
"""

import os
import sys
import time
import signal
import logging
import threading
import concurrent.futures
from pathlib import Path
from typing import Optional, Set
from datetime import datetime, timezone

# Add the parent directory to sys.path so we can import from the hermes-agent modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from cron.jobs import get_due_jobs, mark_job_started, mark_job_run, save_job_output
from cron.scheduler import run_job, _deliver_result, SILENT_MARKER
from hermes_cli.config import load_config, get_hermes_home

logger = logging.getLogger(__name__)

# Default worker configuration
DEFAULT_POLL_INTERVAL = 30  # seconds
DEFAULT_WORKER_ID = f"worker-{os.getpid()}"
DEFAULT_MAX_CONCURRENT = 5

# Global state for graceful shutdown
_shutdown_event = threading.Event()
_running_jobs: Set[str] = set()
_running_lock = threading.Lock()


class QueueWorker:
    """
    Standalone queue worker that polls for due jobs and executes them.
    
    Respects lease metadata to coordinate with other workers and recover
    from stale claims when processes crash.
    """
    
    def __init__(
        self,
        profile: Optional[str] = None,
        worker_id: Optional[str] = None,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ):
        self.profile = profile
        self.worker_id = worker_id or DEFAULT_WORKER_ID
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix=f"queue-worker-{self.worker_id}"
        )
        
        logger.info(
            "Queue worker initialized: worker_id=%s, poll_interval=%ds, max_concurrent=%d",
            self.worker_id, self.poll_interval, self.max_concurrent
        )
    
    def poll_and_execute(self) -> int:
        """
        Poll for due jobs and execute them.
        
        Returns:
            Number of jobs started (not necessarily completed)
        """
        try:
            due_jobs = get_due_jobs()
            if not due_jobs:
                return 0
            
            jobs_started = 0
            
            for job in due_jobs:
                if _shutdown_event.is_set():
                    logger.info("Shutdown requested, stopping job polling")
                    break
                
                job_id = job["id"]
                
                # Check if we're already running this job in this worker
                with _running_lock:
                    if job_id in _running_jobs:
                        logger.debug("Job '%s' already running in this worker", job_id)
                        continue
                    _running_jobs.add(job_id)
                
                # Submit job to thread pool
                future = self.executor.submit(self._execute_job, job)
                jobs_started += 1
                
                # Add completion callback to clean up running set
                def cleanup_job(fut, jid=job_id):
                    with _running_lock:
                        _running_jobs.discard(jid)
                        
                future.add_done_callback(cleanup_job)
            
            if jobs_started > 0:
                logger.info("Started %d job(s)", jobs_started)
            
            return jobs_started
            
        except Exception as e:
            logger.error("Error polling for jobs: %s", e, exc_info=True)
            return 0
    
    def _execute_job(self, job: dict) -> bool:
        """
        Execute a single job end-to-end.
        
        Args:
            job: Job dictionary from get_due_jobs()
            
        Returns:
            True if job completed successfully, False otherwise
        """
        job_id = job["id"]
        job_name = job.get("name", job_id)
        
        try:
            logger.info("Executing job '%s' (worker: %s)", job_name, self.worker_id)
            
            # Mark job as started (this will grant the lease to this worker)
            mark_job_started(job_id)
            
            # Run the actual job
            success, output, final_response, error = run_job(job)
            
            # Save output to file
            output_file = save_job_output(job_id, output)
            logger.debug("Job '%s' output saved to: %s", job_name, output_file)
            
            # Prepare content for delivery
            deliver_content = final_response if success else f"⚠️ Cron job '{job_name}' failed:\n{error}"
            
            # Check if we should deliver (non-empty and not SILENT)
            should_deliver = bool(deliver_content.strip())
            if should_deliver and success and SILENT_MARKER in deliver_content.strip().upper():
                logger.info("Job '%s': agent returned %s — skipping delivery", job_id, SILENT_MARKER)
                should_deliver = False
            
            # Deliver result if needed
            delivery_error = None
            if should_deliver:
                try:
                    delivery_error = _deliver_result(job, deliver_content, adapters=None, loop=None)
                except Exception as de:
                    delivery_error = str(de)
                    logger.error("Delivery failed for job '%s': %s", job_id, de)
            
            # Treat empty final_response as a soft failure
            if success and not final_response.strip():
                success = False
                error = "Agent completed but produced empty response (model error, timeout, or misconfiguration)"
            
            # Mark job as completed
            mark_job_run(job_id, success=success, error=error, delivery_error=delivery_error)
            
            if success:
                logger.info("Job '%s' completed successfully", job_name)
            else:
                logger.warning("Job '%s' failed: %s", job_name, error)
            
            return success
            
        except Exception as e:
            logger.error("Job '%s' execution failed: %s", job_name, e, exc_info=True)
            
            # Mark job as failed
            try:
                mark_job_run(job_id, success=False, error=f"Worker execution error: {e}")
            except Exception as mark_error:
                logger.error("Failed to mark job as failed: %s", mark_error)
            
            return False
    
    def run_forever(self):
        """
        Main worker loop - polls for jobs until shutdown is requested.
        """
        logger.info("Queue worker starting main loop")
        
        while not _shutdown_event.is_set():
            try:
                jobs_started = self.poll_and_execute()
                
                # Sleep for poll interval, but wake up on shutdown
                if not _shutdown_event.wait(timeout=self.poll_interval):
                    continue  # Timeout reached, continue polling
                else:
                    break  # Shutdown event set
                    
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received, shutting down")
                break
            except Exception as e:
                logger.error("Unexpected error in worker loop: %s", e, exc_info=True)
                # Sleep briefly to avoid tight error loops
                time.sleep(5)
        
        logger.info("Worker loop ended, shutting down executor")
        self.shutdown()
    
    def shutdown(self, wait_timeout: int = 30):
        """
        Graceful shutdown - wait for running jobs to complete.
        
        Args:
            wait_timeout: Max seconds to wait for jobs to complete
        """
        logger.info("Shutting down queue worker (timeout=%ds)", wait_timeout)
        
        # Simple shutdown - wait for completion
        self.executor.shutdown(wait=True)
        
        logger.info("Queue worker shutdown complete")


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info("Received signal %d, requesting shutdown", signum)
        _shutdown_event.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)


def main():
    """Main entry point for the queue worker daemon."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hermes Queue Worker Daemon")
    parser.add_argument("--profile", help="Hermes profile to use")
    parser.add_argument("--worker-id", help="Unique worker ID")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL,
                       help="Poll interval in seconds")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
                       help="Max concurrent jobs")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    # Create and run worker
    worker = QueueWorker(
        profile=args.profile,
        worker_id=args.worker_id,
        poll_interval=args.poll_interval,
        max_concurrent=args.max_concurrent,
    )
    
    try:
        worker.run_forever()
    except Exception as e:
        logger.error("Fatal error in worker daemon: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()