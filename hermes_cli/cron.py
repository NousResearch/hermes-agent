"""
Cron subcommand for hermes CLI.

Handles: hermes cron [list|daemon|tick]
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ANSI colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"

def color(text: str, *codes) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + Colors.RESET


def cron_list(show_all: bool = False):
    """List all scheduled jobs."""
    from cron.jobs import list_jobs
    
    jobs = list_jobs(include_disabled=show_all)
    
    if not jobs:
        print(color("No scheduled jobs.", Colors.DIM))
        print(color("Create one with: hermes cron add <schedule> <prompt>", Colors.DIM))
        return
    
    print()
    print(color("┌─────────────────────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                         Scheduled Jobs                                  │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────────────────────┘", Colors.CYAN))
    print()
    
    for job in jobs:
        job_id = job.get("id", "?")[:8]
        name = job.get("name", "(unnamed)")
        schedule = job.get("schedule_display", job.get("schedule", {}).get("value", "?"))
        enabled = job.get("enabled", True)
        next_run = job.get("next_run_at", "?")
        
        # Repeat info
        repeat_info = job.get("repeat", {})
        repeat_times = repeat_info.get("times")
        repeat_completed = repeat_info.get("completed", 0)
        
        if repeat_times:
            repeat_str = f"{repeat_completed}/{repeat_times}"
        else:
            repeat_str = "∞"
        
        # Delivery targets
        deliver = job.get("deliver", ["local"])
        if isinstance(deliver, str):
            deliver = [deliver]
        deliver_str = ", ".join(deliver)
        
        # Status indicator
        if not enabled:
            status = color("[disabled]", Colors.RED)
        else:
            status = color("[active]", Colors.GREEN)
        
        print(f"  {color(job_id, Colors.YELLOW)} {status}")
        print(f"    Name:      {name}")
        print(f"    Schedule:  {schedule}")
        print(f"    Repeat:    {repeat_str}")
        print(f"    Next run:  {next_run}")
        print(f"    Deliver:   {deliver_str}")
        print()


def cron_daemon(interval: int = 60):
    """Run the cron daemon."""
    from cron.scheduler import start_daemon
    
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│              🦋 Hermes Cron Daemon                      │", Colors.CYAN))
    print(color("├─────────────────────────────────────────────────────────┤", Colors.CYAN))
    print(color("│  Press Ctrl+C to stop                                   │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    print()
    
    try:
        start_daemon(interval=interval)
    except KeyboardInterrupt:
        print()
        print(color("Cron daemon stopped.", Colors.YELLOW))


def cron_tick():
    """Run due jobs once (for system cron integration)."""
    from cron.scheduler import tick
    
    print(f"[{datetime.now().isoformat()}] Running cron tick...")
    tick()


def cron_command(args):
    """Handle cron subcommands."""
    subcmd = getattr(args, 'cron_command', None)
    
    if subcmd is None or subcmd == "list":
        show_all = getattr(args, 'all', False)
        cron_list(show_all)
    
    elif subcmd == "daemon":
        interval = getattr(args, 'interval', 60)
        cron_daemon(interval)
    
    elif subcmd == "tick":
        cron_tick()
    
    else:
        print(f"Unknown cron command: {subcmd}")
        print("Usage: hermes cron [list|daemon|tick]")
        sys.exit(1)
