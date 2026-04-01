"""
Record subcommand for hermes CLI.

Handles standalone recording management commands: list, show, start, stop,
run, schedule, and delete.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli.colors import Colors, color


def record_list():
    """List all saved recordings."""
    from recording.store import list_recordings

    recordings = list_recordings()
    if not recordings:
        print(color("No saved recordings.", Colors.DIM))
        print(color("Start one with: hermes record start <name>", Colors.DIM))
        return

    print()
    print(
        color(
            "┌─────────────────────────────────────────────────────────────────────────┐",
            Colors.CYAN,
        )
    )
    print(
        color(
            "│                          Recordings                                    │",
            Colors.CYAN,
        )
    )
    print(
        color(
            "└─────────────────────────────────────────────────────────────────────────┘",
            Colors.CYAN,
        )
    )
    print()

    for rec in recordings:
        name = rec.get("name", "?")
        desc = rec.get("description", "")
        steps = rec.get("step_count", 0)
        created = rec.get("created_at", "?")

        print(f"  {color(name, Colors.YELLOW)}")
        if desc:
            print(f"    Description: {desc}")
        print(f"    Steps:       {steps}")
        print(f"    Created:     {created}")
        print()


def record_show(name):
    """Show full details of a recording including steps."""
    from recording.store import get_recording

    rec = get_recording(name)
    if not rec:
        print(color(f"Recording not found: {name}", Colors.RED))
        return 1

    print()
    print(f"  {color('Name:', Colors.CYAN)} {rec.get('name', '?')}")
    print(f"  {color('Description:', Colors.CYAN)} {rec.get('description', '')}")
    print(f"  {color('Created:', Colors.CYAN)} {rec.get('created_at', '?')}")
    print()

    steps = rec.get("steps", [])
    if not steps:
        print(color("  No steps recorded.", Colors.DIM))
        return 0

    print(f"  {color('Steps:', Colors.CYAN)} ({len(steps)} total)")
    print()
    for i, step in enumerate(steps):
        tool = step.get("tool", "?")
        expected = step.get("expected_status", "?")
        args = step.get("arguments", {})
        args_preview = json.dumps(args, ensure_ascii=False)
        if len(args_preview) > 120:
            args_preview = args_preview[:120] + "..."

        status_color = Colors.GREEN if expected == "success" else Colors.YELLOW
        print(f"  {i + 1}. {color(tool, Colors.YELLOW)} [{color(expected, status_color)}]")
        print(f"     {args_preview}")
        print()

    return 0


def record_start(name, description=""):
    """Start recording tool calls in the current session."""
    from recording.capture import RecordingSession

    try:
        session = RecordingSession(name, description)
        session.start()
    except (ValueError, RuntimeError) as e:
        print(color(f"Cannot start recording: {e}", Colors.RED))
        return 1

    print(color(f"Recording started: {name}", Colors.GREEN))
    print(color("Tool calls in this session will be captured.", Colors.DIM))
    print(color("Stop with: hermes record stop", Colors.DIM))
    return 0


def record_stop():
    """Stop the active recording session."""
    from recording.capture import get_active_session

    session = get_active_session()
    if session is None:
        print(color("No active recording session.", Colors.YELLOW))
        return 1

    summary = session.stop()
    print(color(f"Recording stopped: {summary['name']}", Colors.GREEN))
    print(f"  Steps captured: {summary['step_count']}")
    return 0


def record_run(name):
    """Replay a recording."""
    from recording.store import get_recording
    from recording.replay import replay_recording

    rec = get_recording(name)
    if not rec:
        print(color(f"Recording not found: {name}", Colors.RED))
        return 1

    steps = rec.get("steps", [])
    if not steps:
        print(color(f"Recording '{name}' has no steps.", Colors.YELLOW))
        return 0

    print(f"Replaying '{name}' ({len(steps)} steps)...")
    print()

    def on_step(i, step, result):
        tool = step.get("tool", "?")
        status = "ok" if not result.startswith("Error") else "error"
        print(f"  Step {i + 1}/{len(steps)}: {tool} [{status}]")

    def on_deviation(i, step, result):
        tool = step.get("tool", "?")
        expected = step.get("expected_status", "?")
        print(color(f"  Deviation at step {i + 1} ({tool}): expected {expected}", Colors.YELLOW))
        return True  # Continue by default

    summary = replay_recording(rec, on_step=on_step, on_deviation=on_deviation)

    print()
    if summary["success"]:
        print(color(f"Replay completed: {summary['steps_completed']}/{summary['steps_total']} steps", Colors.GREEN))
    else:
        print(color(f"Replay failed: {summary['error']}", Colors.RED))
    return 0 if summary["success"] else 1


def record_schedule(name, schedule, name_override=None):
    """Schedule a recording as a cron job."""
    from recording.store import get_recording
    from cron.jobs import create_job

    rec = get_recording(name)
    if not rec:
        print(color(f"Recording not found: {name}", Colors.RED))
        return 1

    job_name = name_override or f"recording:{rec['name']}"
    prompt = f"[REPLAY_RECORDING:{name}] Execute the recorded action sequence '{name}'."

    try:
        job = create_job(
            prompt=prompt,
            schedule=schedule,
            name=job_name,
        )
    except Exception as e:
        print(color(f"Failed to schedule: {e}", Colors.RED))
        return 1

    print(color(f"Scheduled recording '{name}' as cron job", Colors.GREEN))
    print(f"  Job ID:    {job.get('id', '?')}")
    print(f"  Name:      {job_name}")
    print(f"  Schedule:  {schedule}")
    print(f"  Next run:  {job.get('next_run_at', '?')}")
    return 0


def record_delete(name):
    """Delete a recording."""
    from recording.store import delete_recording

    if delete_recording(name):
        print(color(f"Deleted recording: {name}", Colors.GREEN))
        return 0
    else:
        print(color(f"Recording not found: {name}", Colors.RED))
        return 1


def record_command(args):
    """Handle record subcommands."""
    subcmd = getattr(args, "record_command", None)

    if subcmd is None or subcmd == "list":
        record_list()
        return 0

    if subcmd == "show":
        return record_show(args.name)

    if subcmd == "start":
        return record_start(args.name, getattr(args, "description", ""))

    if subcmd == "stop":
        return record_stop()

    if subcmd == "run":
        return record_run(args.name)

    if subcmd == "schedule":
        return record_schedule(
            args.name,
            args.schedule,
            name_override=getattr(args, "name_override", None),
        )

    if subcmd in {"delete", "rm"}:
        return record_delete(args.name)

    print(f"Unknown record command: {subcmd}")
    print("Usage: hermes record [list|show|start|stop|run|schedule|delete]")
    sys.exit(1)
