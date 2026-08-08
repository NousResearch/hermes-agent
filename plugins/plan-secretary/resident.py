"""Per-session resident entry for plan-secretary, driven by Hermes cron.

Each tick: incremental capture scan (window: last N minutes; the cursor
advances forever) + session-scoped reminder pass. Pure Python, stdlib only.
Windows-safe (no bash dependency).

Usage::

    python -m plugins.plan_secretary.resident <session_id> [--since-minutes 10] [--max-messages 120]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from hermes_constants import get_hermes_home

from . import core


def run_watcher(session_id: str, since_minutes: int, max_messages: int) -> int:
    cmd = [
        sys.executable,
        "-m", "plugins.plan_secretary.watcher",
        "--source", "auto",
        "--since-minutes", str(since_minutes),
        "--current-session-only",
        "--session-id", session_id,
        "--max-messages", str(max_messages),
        "--cursor", str(core.state_dir() / f"watcher_cursor_{session_id}.json"),
    ]
    return subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[2]), text=True).returncode


def run_notifier(session_id: str) -> int:
    messages = core.notify(
        session_id=session_id,
        state_path=core.state_dir() / f"notification_state_{session_id}.json",
        due_repeat_minutes=10,
    )
    if messages:
        print("\n\n".join(messages))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-session plan-secretary resident (cron-driven).")
    parser.add_argument("session_id")
    parser.add_argument("--since-minutes", type=int, default=10)
    parser.add_argument("--max-messages", type=int, default=120)
    args = parser.parse_args(argv)
    watcher_rc = run_watcher(args.session_id, args.since_minutes, args.max_messages)
    notifier_rc = run_notifier(args.session_id)
    return watcher_rc or notifier_rc


if __name__ == "__main__":
    raise SystemExit(main())
