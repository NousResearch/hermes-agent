"""
Hermes CLI: `hermes dashboard` command
========================================
Renders the agent observability dashboard.
"""

from __future__ import annotations

import argparse
import os
import sys


def show_dashboard(args: argparse.Namespace):
    """Entry point for the dashboard command."""
    # Add agent dir to path to find agent.dashboard
    agent_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    sys.path.insert(0, os.path.abspath(agent_dir))

    try:
        from agent.dashboard import render_snapshot, render_live
        from hermes_state import SessionDB
    except ImportError as exc:
        print(f"Error: could not import dashboard components: {exc}", file=sys.stderr)
        print("Please ensure Hermes Agent is installed correctly.", file=sys.stderr)
        sys.exit(1)

    session_id = args.session or os.environ.get("HERMES_SESSION_ID")
    if not session_id:
        # Get the most recent session
        try:
            db = SessionDB()
            sessions = db.get_recent_sessions(limit=1)
            if sessions:
                session_id = sessions[0]["id"]
            else:
                print("No active or recent sessions found.", file=sys.stderr)
                sys.exit(1)
        except Exception as exc:
            print(f"Could not automatically find the last session: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.live:
        try:
            render_live(session_id, refresh_seconds=args.refresh)
        except KeyboardInterrupt:
            print("\nDashboard closed.")
            sys.exit(0)
    else:
        print(render_snapshot(session_id))


def build_dashboard_parser(subparsers: argparse._SubParsersAction):
    """Add 'dashboard' command to the CLI arg parser."""
    p = subparsers.add_parser(
        "dashboard",
        help="Show agent observability dashboard",
        description="Displays a live or snapshot view of the agent's performance, health, and activity.",
    )
    p.add_argument(
        "--session",
        type=str,
        help="Session ID to show. Defaults to the current or most recent session.",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Enable live-updating dashboard view (refreshes every 2s).",
    )
    p.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="Refresh interval in seconds for --live mode.",
    )
    p.set_defaults(func=show_dashboard)
