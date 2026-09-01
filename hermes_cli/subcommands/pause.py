"""``hermes pause`` / ``hermes resume`` — the Kanban dispatch pause.

``pause`` writes the ESTOP sentinel at ``$HERMES_HOME/ESTOP``; new Kanban worker spawns halt on
their next check, while chat turns and cron dispatch keep running (in-flight work is never
killed). ``resume`` removes it and dispatch resumes on the next tick — no restart. Ported from
gastownhall/gastown estop.go (MIT).
"""

from __future__ import annotations

import argparse


def cmd_pause(args: argparse.Namespace) -> int:
    """Engage the Kanban dispatch pause."""
    from agent.estop import engage, get_state, is_engaged

    reason = getattr(args, "reason", None)
    already = is_engaged()
    path = engage(reason=reason)
    state = get_state() or {}
    verb = "Still paused" if already else "Hermes paused"
    detail = f" — reason: {state['reason']}" if state.get("reason") else ""
    print(f"⏸️  {verb}{detail}")
    print(f"    sentinel: {path}")
    print(
        "    New Kanban worker spawns are on hold; chat and cron keep running.\n"
        "    In-flight work keeps running. Run `hermes resume` to lift the pause.")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Disengage the Kanban dispatch pause."""
    from agent.estop import disengage, sentinel_path

    if disengage():
        print("▶️  Hermes resumed — dispatch picks up on the next tick.")
    else:
        print(f"Hermes is not paused (no sentinel at {sentinel_path()}).")
    return 0


def build_pause_parser(subparsers) -> None:
    """Attach the ``pause`` and ``resume`` subcommands to ``subparsers``."""
    pause_parser = subparsers.add_parser(
        "pause", help="Pause Kanban dispatch (chat and cron keep running)",
        description="Engage the Kanban dispatch pause. Halts NEW Kanban worker "
            "spawns only — chat turns and cron dispatch are unaffected — "
            "until `hermes resume`. In-flight work is never killed.")
    pause_parser.add_argument(
        "--reason", default=None, help="Optional reason stored in the sentinel and shown to users")
    pause_parser.set_defaults(func=cmd_pause)

    resume_parser = subparsers.add_parser(
        "resume", help="Lift the emergency stop set by `hermes pause`",
        description="Remove the ESTOP sentinel; dispatch resumes on the next tick.")
    resume_parser.set_defaults(func=cmd_resume)
