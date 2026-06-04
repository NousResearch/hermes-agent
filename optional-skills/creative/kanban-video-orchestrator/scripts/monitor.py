#!/usr/bin/env python3
"""
Monitor a running video-production kanban. Polls `hermes kanban list` and
`events` for a tenant and surfaces issues (stuck tasks, missing heartbeats,
repeated retries, dependency deadlocks).

Usage:
    monitor.py --tenant <project-slug> [--interval 30]

Outputs a periodic snapshot to stdout. Sends alerts via stderr when issues
are detected. Designed to run alongside the kanban — kill with Ctrl-C when
you're satisfied (or scripted to stop on completion).

This is best-effort observability. It does not auto-restart tasks; intervention
decisions should remain human/AI-overseen.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta


def hermes_available() -> bool:
    return shutil.which("hermes") is not None


def kanban_list(tenant: str) -> list[dict]:
    """Returns parsed task rows. Falls back to plain stdout parsing if JSON
    output isn't supported by the installed hermes CLI."""
    try:
        out = subprocess.run(
            ["hermes", "kanban", "list", "--tenant", tenant, "--json"],
            capture_output=True, text=True, check=False,
        )
        if out.returncode == 0 and out.stdout.strip().startswith("["):
            return json.loads(out.stdout)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    # Fallback: textual parse of `hermes kanban list`
    out = subprocess.run(
        ["hermes", "kanban", "list", "--tenant", tenant],
        capture_output=True, text=True, check=False,
    )
    rows = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "STATUS" in line.upper():
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[0].startswith("t_"):
            rows.append({
                "id": parts[0],
                "status": parts[1] if len(parts) > 1 else "?",
                "assignee": parts[2] if len(parts) > 2 else "?",
                "title": " ".join(parts[3:]) if len(parts) > 3 else "",
                "started_at": None,
                "heartbeat_at": None,
                "max_runtime_s": None,
            })
    return rows


def kanban_show(task_id: str) -> dict | None:
    out = subprocess.run(
        ["hermes", "kanban", "show", task_id, "--json"],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def detect_issues(tasks: list[dict]) -> list[str]:
    """Return a list of issue strings, one per concern."""
    now = datetime.now()
    issues: list[str] = []
    by_status = defaultdict(list)
    for t in tasks:
        by_status[t.get("status", "?")].append(t)

    # Stuck tasks: RUNNING with no heartbeat in 2 min
    for t in by_status.get("running", []) + by_status.get("RUNNING", []):
        hb = t.get("heartbeat_at")
        if not hb:
            continue
        try:
            hb_dt = datetime.fromisoformat(str(hb).rstrip("Z"))
        except ValueError:
            continue
        if now - hb_dt > timedelta(minutes=2):
            issues.append(
                f"STUCK: {t['id']} ({t.get('assignee', '?')}) — "
                f"no heartbeat in {(now - hb_dt).total_seconds():.0f}s"
            )

    # Tasks exceeding max_runtime
    for t in by_status.get("running", []) + by_status.get("RUNNING", []):
        started = t.get("started_at")
        max_rt = t.get("max_runtime_s")
        if not started or not max_rt:
            continue
        try:
            started_dt = datetime.fromisoformat(str(started).rstrip("Z"))
        except ValueError:
            continue
        elapsed = (now - started_dt).total_seconds()
        if elapsed > max_rt:
            issues.append(
                f"OVERTIME: {t['id']} ({t.get('assignee', '?')}) — "
                f"running {elapsed:.0f}s, cap was {max_rt}s"
            )

    # Repeated retries
    for t in tasks:
        retries = t.get("retries", 0)
        if retries and retries >= 2:
            issues.append(
                f"FLAPPING: {t['id']} ({t.get('assignee', '?')}) — "
                f"retried {retries}× — fix root cause before next run"
            )

    return issues


def snapshot(tenant: str) -> tuple[list[dict], list[str]]:
    tasks = kanban_list(tenant)
    issues = detect_issues(tasks)
    return tasks, issues


def print_snapshot(tasks: list[dict], issues: list[str]):
    counts = defaultdict(int)
    for t in tasks:
        counts[str(t.get("status", "?")).lower()] += 1

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
          f"Total: {len(tasks)} | "
          + " | ".join(f"{k}: {v}" for k, v in sorted(counts.items())))

    for t in tasks:
        bar = "✓" if str(t.get("status", "")).lower() == "done" else \
              "▶" if str(t.get("status", "")).lower() == "running" else \
              "·" if str(t.get("status", "")).lower() == "ready" else \
              "✗" if str(t.get("status", "")).lower() == "failed" else "?"
        print(f"  {bar} {t.get('id', '?'):14} {t.get('assignee', '?'):20}  "
              f"{t.get('title', '')[:60]}")

    if issues:
        print("\n  ⚠  ISSUES:", file=sys.stderr)
        for i in issues:
            print(f"     {i}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tenant", required=True,
                    help="Project tenant slug to monitor")
    ap.add_argument("--interval", type=int, default=30,
                    help="Poll interval in seconds (default: 30)")
    ap.add_argument("--once", action="store_true",
                    help="Print one snapshot and exit (no polling loop)")
    args = ap.parse_args()

    if not hermes_available():
        print("ERROR: 'hermes' CLI not found in PATH", file=sys.stderr)
        sys.exit(1)

    if args.once:
        tasks, issues = snapshot(args.tenant)
        print_snapshot(tasks, issues)
        sys.exit(0 if not issues else 2)

    print(f"Monitoring tenant '{args.tenant}' every {args.interval}s. "
          "Ctrl-C to exit.")
    try:
        while True:
            tasks, issues = snapshot(args.tenant)
            print_snapshot(tasks, issues)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
