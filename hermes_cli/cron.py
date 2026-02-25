"""
Cron subcommand for hermes CLI.

Handles: hermes cron [list|status|tick]

Cronjobs are executed automatically by the gateway daemon (hermes gateway).
Install the gateway as a service for background execution:
    hermes gateway install
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli.colors import Colors, color


def cron_list(show_all: bool = False):
    """List all scheduled jobs."""
    from cron.jobs import list_jobs

    jobs = list_jobs(include_disabled=show_all)

    if not jobs:
        print(color("No scheduled jobs.", Colors.DIM))
        print(color("Create one with the /cron add command in chat, or via Telegram.", Colors.DIM))
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

        repeat_info = job.get("repeat", {})
        repeat_times = repeat_info.get("times")
        repeat_completed = repeat_info.get("completed", 0)

        if repeat_times:
            repeat_str = f"{repeat_completed}/{repeat_times}"
        else:
            repeat_str = "∞"

        deliver = job.get("deliver", ["local"])
        if isinstance(deliver, str):
            deliver = [deliver]
        deliver_str = ", ".join(deliver)

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

    # Warn if gateway isn't running
    from hermes_cli.gateway import find_gateway_pids

    if not find_gateway_pids():
        print(color("  ⚠  Gateway is not running — jobs won't fire automatically.", Colors.YELLOW))
        print(color("     Start it with: hermes gateway install", Colors.DIM))
        print()


def cron_tick():
    """Run due jobs once and exit."""
    from cron.scheduler import tick

    tick(verbose=True)


def cron_status():
    """Show cron execution status."""
    from cron.jobs import list_jobs
    from hermes_cli.gateway import find_gateway_pids

    print()

    pids = find_gateway_pids()
    if pids:
        print(color("✓ Gateway is running — cron jobs will fire automatically", Colors.GREEN))
        print(f"  PID: {', '.join(map(str, pids))}")
    else:
        print(color("✗ Gateway is not running — cron jobs will NOT fire", Colors.RED))
        print()
        print("  To enable automatic execution:")
        print("    hermes gateway install    # Install as system service (recommended)")
        print("    hermes gateway            # Or run in foreground")

    print()

    jobs = list_jobs(include_disabled=False)
    if jobs:
        next_runs = [j.get("next_run_at") for j in jobs if j.get("next_run_at")]
        print(f"  {len(jobs)} active job(s)")
        if next_runs:
            print(f"  Next run: {min(next_runs)}")
    else:
        print("  No active jobs")

    print()


def cron_command(args):
    """Handle cron subcommands."""
    subcmd = getattr(args, "cron_command", None)

    if subcmd is None or subcmd == "list":
        show_all = getattr(args, "all", False)
        cron_list(show_all)

    elif subcmd == "tick":
        cron_tick()

    elif subcmd == "status":
        cron_status()

    else:
        print(f"Unknown cron command: {subcmd}")
        print("Usage: hermes cron [list|status|tick]")
        sys.exit(1)
