"""
Cron job scheduler - executes due jobs.

Provides tick() which checks for due jobs and runs them. The gateway
calls this every 60 seconds from a background thread.

Uses a file-based lock (~/.hermes/cron/.tick.lock) so only one tick
runs at a time if multiple processes overlap.
"""

import fcntl
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cron.jobs import get_due_jobs, mark_job_run, save_job_output

# File-based lock prevents concurrent ticks from gateway + daemon + systemd timer
_LOCK_DIR = Path.home() / ".hermes" / "cron"
_LOCK_FILE = _LOCK_DIR / ".tick.lock"


def run_job(job: dict) -> tuple[bool, str, Optional[str]]:
    """
    Execute a single cron job.
    
    Returns:
        Tuple of (success, output, error_message)
    """
    from run_agent import AIAgent
    
    job_id = job["id"]
    job_name = job["name"]
    prompt = job["prompt"]
    
    logger.info("Running job '%s' (ID: %s)", job_name, job_id)
    logger.info("Prompt: %s", prompt[:100])
    
    try:
        # Create agent with default settings
        # Jobs run in isolated sessions (no prior context)
        agent = AIAgent(
            model=os.getenv("HERMES_MODEL", "anthropic/claude-opus-4.6"),
            quiet_mode=True,
            session_id=f"cron_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Run the conversation
        result = agent.run_conversation(prompt)
        
        # Extract final response
        final_response = result.get("final_response", "")
        if not final_response:
            final_response = "(No response generated)"
        
        # Build output document
        output = f"""# Cron Job: {job_name}

**Job ID:** {job_id}
**Run Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}

## Prompt

{prompt}

## Response

{final_response}
"""
        
        logger.info("Job '%s' completed successfully", job_name)
        return True, output, None
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error("Job '%s' failed: %s", job_name, error_msg)
        
        # Build error output
        output = f"""# Cron Job: {job_name} (FAILED)

**Job ID:** {job_id}
**Run Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}

## Prompt

{prompt}

## Error

```
{error_msg}

{traceback.format_exc()}
```
"""
        return False, output, error_msg


def tick(verbose: bool = True) -> int:
    """
    Check and run all due jobs.
    
    Uses a file lock so only one tick runs at a time, even if the gateway's
    in-process ticker and a standalone daemon or manual tick overlap.
    
    Args:
        verbose: Whether to print status messages
    
    Returns:
        Number of jobs executed (0 if another tick is already running)
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)

    try:
        lock_fd = open(_LOCK_FILE, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError):
        # Another tick is already running — skip silently
        logger.debug("Tick skipped — another instance holds the lock")
        return 0

    try:
        due_jobs = get_due_jobs()

        if verbose and not due_jobs:
            logger.info("%s - No jobs due", datetime.now().strftime('%H:%M:%S'))
            return 0

        if verbose:
            logger.info("%s - %s job(s) due", datetime.now().strftime('%H:%M:%S'), len(due_jobs))

        executed = 0
        for job in due_jobs:
            try:
                success, output, error = run_job(job)

                output_file = save_job_output(job["id"], output)
                if verbose:
                    logger.info("Output saved to: %s", output_file)

                mark_job_run(job["id"], success, error)
                executed += 1

            except Exception as e:
                logger.error("Error processing job %s: %s", job['id'], e)
                mark_job_run(job["id"], False, str(e))

        return executed
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


if __name__ == "__main__":
    tick(verbose=True)
