#!/usr/bin/env python3
"""
Skill Broker Post-Completion Hook

Called after a task completes. Checks if the task had any active borrows
and revokes them automatically.

Called by: kanban completion trigger (cron job or direct invocation)
Usage:
  python3 skill-broker-revoke-hook.py <task_id> [task_result]

Where task_result is: completed | failed | blocked | cancelled
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

LEDGER_PATH = Path("/home/kensei/.hermes/governance/skill-broker-ledger.jsonl")
SCRIPT_PATH = Path("/home/kensei/.hermes/scripts/skill-broker-ledger.py")


def read_ledger():
    if not LEDGER_PATH.exists():
        return []
    entries = []
    with open(LEDGER_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def find_active_borrows(task_id):
    """Find any borrow events for this task that haven't been revoked yet."""
    entries = read_ledger()
    active = []
    for entry in entries:
        if (entry.get("task_id") == task_id
                and entry.get("success") is True
                and entry.get("revoked_at") is None):
            active.append(entry)
    return active


def revoke_borrow(event_id, task_result):
    """Run the ledger revoke command."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "revoke", event_id, task_result],
        capture_output=True, text=True, timeout=30
    )
    return result.returncode == 0, result.stdout.strip()


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: hook.py <task_id> [task_result]", "exit": 1}))
        return 1

    task_id = sys.argv[1]
    task_result = sys.argv[2] if len(sys.argv) > 2 else "completed"

    # Find active borrows for this task
    active = find_active_borrows(task_id)

    if not active:
        print(json.dumps({
            "task_id": task_id,
            "action": "noop",
            "reason": "no active borrows found for this task",
        }))
        return 0

    results = []
    for entry in active:
        event_id = entry["event_id"]
        profile = entry.get("worker_profile", "?")
        skill = entry.get("skill_borrowed", "?")
        ok, output = revoke_borrow(event_id, task_result)
        results.append({
            "event_id": event_id,
            "profile": profile,
            "skill": skill,
            "revoked": ok,
            "output": output,
        })

    print(json.dumps({
        "task_id": task_id,
        "task_result": task_result,
        "action": "revoked",
        "borrows_revoked": len(results),
        "details": results,
    }))

    return 0


if __name__ == "__main__":
    sys.exit(main())
