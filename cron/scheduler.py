"""
Cron job scheduler - executes due jobs.

This module provides:
- tick(): Run all due jobs once (for system cron integration)
- run_daemon(): Run continuously, checking every 60 seconds
"""

import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cron.jobs import get_due_jobs, mark_job_run, save_job_output


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
    
    This is designed to be called by system cron every minute:
        */1 * * * * cd ~/hermes-agent && python -c "from cron import tick; tick()"
    
    Args:
        verbose: Whether to print status messages
    
    Returns:
        Number of jobs executed
    """
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
            
            # Save output to file
            output_file = save_job_output(job["id"], output)
            if verbose:
                logger.info("Output saved to: %s", output_file)
            
            # Mark job as run (handles repeat counting, next_run computation)
            mark_job_run(job["id"], success, error)
            executed += 1
            
        except Exception as e:
            logger.error("Error processing job %s: %s", job['id'], e)
            mark_job_run(job["id"], False, str(e))
    
    return executed


def run_daemon(check_interval: int = 60, verbose: bool = True):
    """
    Run the cron daemon continuously.
    
    Checks for due jobs every `check_interval` seconds.
    
    Args:
        check_interval: Seconds between checks (default: 60)
        verbose: Whether to print status messages
    """
    logger.info("Starting daemon (checking every %ss)", check_interval)
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            try:
                tick(verbose=verbose)
            except Exception as e:
                logger.error("Tick error: %s", e)
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        logger.info("Daemon stopped")


if __name__ == "__main__":
    # Allow running directly: python cron/scheduler.py [daemon|tick]
    import argparse
    
    parser = argparse.ArgumentParser(description="Hermes Cron Scheduler")
    parser.add_argument("mode", choices=["daemon", "tick"], default="tick", nargs="?",
                        help="Mode: 'tick' to run once, 'daemon' to run continuously")
    parser.add_argument("--interval", type=int, default=60,
                        help="Check interval in seconds for daemon mode")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress status messages")
    
    args = parser.parse_args()
    
    if args.mode == "daemon":
        run_daemon(check_interval=args.interval, verbose=not args.quiet)
    else:
        tick(verbose=not args.quiet)
