"""
Cron job scheduling system for Hermes Agent.

This module provides scheduled task execution, allowing the agent to:
- Run automated tasks on schedules (cron expressions, intervals, one-shot)
- Self-schedule reminders and follow-up tasks
- Execute tasks in isolated sessions (no prior context)

Usage:
    # Run due jobs (for system cron integration)
    python -c "from cron import tick; tick()"
    
    # Or via CLI
    python cli.py --cron-daemon
"""

from cron.jobs import (
    create_job,
    get_job,
    list_jobs,
    remove_job,
    update_job,
    JOBS_FILE,
)
from cron.scheduler import tick, run_daemon

__all__ = [
    "create_job",
    "get_job", 
    "list_jobs",
    "remove_job",
    "update_job",
    "tick",
    "run_daemon",
    "JOBS_FILE",
]
