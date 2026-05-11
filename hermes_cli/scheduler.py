"""Hermes Scheduler — Milestone H.

Polls ~/.hermes/schedule.yaml and dispatches due tasks by invoking
the Hermes CLI as a subprocess. Designed to run as a background daemon
or be called in one-shot mode from ops.py.

No new dependencies — uses only Python stdlib (subprocess, yaml via
PyYAML which is already installed in the Hermes venv).

Usage:
    # One-shot: check and dispatch any due tasks, then exit
    python3 -m hermes_cli.scheduler --once

    # Daemon mode: poll every 5 minutes indefinitely
    python3 -m hermes_cli.scheduler

    # Via ops CLI
    python3 -m hermes_cli.ops schedule --once
    python3 -m hermes_cli.ops schedule --status

Schedule file: ~/.hermes/schedule.yaml
Reports output: ~/.hermes/reports/
Scheduler log: ~/.hermes/logs/scheduler.jsonl
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
_SCHEDULE_FILE = _HERMES_HOME / "schedule.yaml"
_REPORTS_DIR = _HERMES_HOME / "reports"
_SCHEDULER_LOG = _HERMES_HOME / "logs" / "scheduler.jsonl"

# ---------------------------------------------------------------------------
# Default schedule template (written on first run if schedule.yaml is absent)
# ---------------------------------------------------------------------------

_DEFAULT_SCHEDULE = """\
# Hermes Scheduled Tasks
# ─────────────────────
# Each task has:
#   id            — unique slug, used to track last_run
#   name          — human-readable label (shown in dashboard)
#   prompt        — the message sent to Hermes when the task fires
#   interval_hours — how often to run (24 = daily, 168 = weekly)
#   enabled       — set to false to pause without deleting the task
#   last_run      — ISO timestamp, updated automatically by the scheduler

tasks:
  - id: daily-digest
    name: Daily Activity Digest
    prompt: |
      Generate a brief digest of today's Hermes activity.
      Read ~/.hermes/logs/structured.jsonl for events in the last 24 hours.
      Count: tasks run, tools called, errors encountered, slow tools (> 3s).
      Write a concise markdown summary to ~/.hermes/reports/digest-{date}.md
      where {date} is today's date in YYYY-MM-DD format.
    interval_hours: 24
    enabled: false
    last_run: null

  - id: weekly-perf
    name: Weekly Performance Report
    prompt: |
      Generate a weekly performance report for Hermes.
      Use the hermes_cli.ops module to get: slow tool calls, model call stats,
      task completion rates. Compare tool durations and identify bottlenecks.
      Save the report to ~/.hermes/reports/perf-{date}.md
      where {date} is today's date in YYYY-MM-DD format.
    interval_hours: 168
    enabled: false
    last_run: null
"""

# ---------------------------------------------------------------------------
# YAML loading — uses PyYAML (already in Hermes venv) with stdlib fallback
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Any:
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Minimal fallback: just return empty (scheduler won't run without yaml)
        raise RuntimeError(
            "PyYAML not found. Install with: pip install pyyaml\n"
            "Or activate the Hermes venv: source ~/.hermes/hermes-agent/venv/bin/activate"
        )


def _save_yaml(path: Path, data: Any) -> None:
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Schedule file management
# ---------------------------------------------------------------------------

def load_schedule() -> Dict[str, Any]:
    """Load ~/.hermes/schedule.yaml, creating it from template if absent."""
    _HERMES_HOME.mkdir(parents=True, exist_ok=True)
    if not _SCHEDULE_FILE.exists():
        _SCHEDULE_FILE.write_text(_DEFAULT_SCHEDULE, encoding="utf-8")
        print(f"Created schedule file at {_SCHEDULE_FILE}")
    return _load_yaml(_SCHEDULE_FILE)


def get_tasks(schedule: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return all task dicts from the schedule."""
    return schedule.get("tasks") or []


def get_due_tasks(schedule: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return tasks that are enabled and whose interval has elapsed since last_run."""
    now = datetime.now(timezone.utc)
    due = []
    for task in get_tasks(schedule):
        if not task.get("enabled", False):
            continue
        interval_h = task.get("interval_hours", 24)
        last_run_str = task.get("last_run")
        if last_run_str is None:
            due.append(task)
            continue
        try:
            last_run = datetime.fromisoformat(str(last_run_str).replace("Z", "+00:00"))
            elapsed_h = (now - last_run).total_seconds() / 3600
            if elapsed_h >= interval_h:
                due.append(task)
        except Exception:
            due.append(task)  # bad timestamp → run it
    return due


def update_last_run(task_id: str) -> None:
    """Write the current UTC timestamp into schedule.yaml for the given task id."""
    try:
        schedule = _load_yaml(_SCHEDULE_FILE)
        tasks = schedule.get("tasks") or []
        for task in tasks:
            if task.get("id") == task_id:
                task["last_run"] = datetime.now(timezone.utc).isoformat()
                break
        _save_yaml(_SCHEDULE_FILE, schedule)
    except Exception as e:
        _log_event("update_last_run_error", task_id=task_id, error=str(e))


# ---------------------------------------------------------------------------
# Structured log
# ---------------------------------------------------------------------------

def _log_event(event: str, **fields: Any) -> None:
    """Append a JSON line to ~/.hermes/logs/scheduler.jsonl. Non-fatal."""
    try:
        _SCHEDULER_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {"event": event, "ts": datetime.now(timezone.utc).isoformat(), **fields}
        with open(_SCHEDULER_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Task dispatch
# ---------------------------------------------------------------------------

def _resolve_hermes_cmd() -> List[str]:
    """Find the hermes command: prefer the venv binary, fall back to PATH."""
    venv_hermes = _HERMES_HOME / "hermes-agent" / "venv" / "bin" / "hermes"
    if venv_hermes.exists():
        return [str(venv_hermes)]
    # Check if 'hermes' is on PATH
    import shutil
    h = shutil.which("hermes")
    if h:
        return [h]
    # Last resort: python -m cli
    python = _HERMES_HOME / "hermes-agent" / "venv" / "bin" / "python3"
    cli = _HERMES_HOME / "hermes-agent" / "cli.py"
    if python.exists() and cli.exists():
        return [str(python), str(cli)]
    return ["hermes"]  # will fail loudly if not found


def dispatch_task(task: Dict[str, Any], dry_run: bool = False) -> bool:
    """
    Invoke Hermes with the task's prompt as a non-interactive one-shot run.
    Returns True on success. Updates last_run on success.

    Dry run mode prints the command without executing it.
    """
    task_id = task.get("id", "unknown")
    name = task.get("name", task_id)
    prompt = task.get("prompt", "").strip()

    # Substitute {date} placeholder in prompt
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = prompt.replace("{date}", today)

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = _resolve_hermes_cmd() + [prompt]

    if dry_run:
        print(f"[dry-run] Would dispatch: {name}")
        print(f"  Command: {' '.join(cmd[:2])} <prompt>")
        print(f"  Prompt: {prompt[:120]}...")
        return True

    print(f"[scheduler] Dispatching: {name} ({task_id})")
    _log_event("task_dispatch", task_id=task_id, name=name)

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            timeout=600,  # 10 minute max per scheduled task
            capture_output=True,
            text=True,
            env={**os.environ, "HERMES_SCHEDULED": "1"},  # lets Hermes know it's scheduled
        )
        duration_s = round(time.time() - start, 1)
        success = result.returncode == 0
        _log_event(
            "task_complete",
            task_id=task_id,
            name=name,
            success=success,
            returncode=result.returncode,
            duration_s=duration_s,
            stdout_tail=result.stdout.strip()[-500:] if result.stdout else "",
            stderr_tail=result.stderr.strip()[-200:] if result.stderr else "",
        )
        if success:
            update_last_run(task_id)
            print(f"[scheduler] ✓ {name} completed in {duration_s}s")
        else:
            print(f"[scheduler] ✗ {name} failed (rc={result.returncode})")
        return success
    except subprocess.TimeoutExpired:
        _log_event("task_timeout", task_id=task_id, name=name)
        print(f"[scheduler] ✗ {name} timed out after 600s")
        return False
    except Exception as e:
        _log_event("task_error", task_id=task_id, name=name, error=str(e))
        print(f"[scheduler] ✗ {name} error: {e}")
        return False


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_once(dry_run: bool = False) -> Dict[str, int]:
    """Check for due tasks and dispatch them. Returns counts."""
    schedule = load_schedule()
    due = get_due_tasks(schedule)
    if not due:
        print("[scheduler] No tasks due.")
        return {"due": 0, "dispatched": 0, "failed": 0}

    print(f"[scheduler] {len(due)} task(s) due.")
    dispatched = failed = 0
    for task in due:
        ok = dispatch_task(task, dry_run=dry_run)
        if ok:
            dispatched += 1
        else:
            failed += 1

    return {"due": len(due), "dispatched": dispatched, "failed": failed}


def run_daemon(poll_seconds: int = 300) -> None:
    """Poll for due tasks every poll_seconds indefinitely. Ctrl+C to stop."""
    print(f"[scheduler] Daemon started. Polling every {poll_seconds}s. Ctrl+C to stop.")
    _log_event("daemon_start", poll_seconds=poll_seconds)
    try:
        while True:
            try:
                run_once()
            except Exception as e:
                _log_event("daemon_poll_error", error=str(e))
                print(f"[scheduler] Poll error (continuing): {e}")
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        _log_event("daemon_stop", reason="keyboard_interrupt")
        print("\n[scheduler] Daemon stopped.")


def status() -> List[Dict[str, Any]]:
    """Return a status summary of all tasks (due/pending/disabled)."""
    schedule = load_schedule()
    now = datetime.now(timezone.utc)
    result = []
    for task in get_tasks(schedule):
        enabled = task.get("enabled", False)
        interval_h = task.get("interval_hours", 24)
        last_run_str = task.get("last_run")
        next_run_str = None
        hours_until = None
        state = "disabled"

        if enabled:
            if last_run_str is None:
                state = "due"
                next_run_str = "now"
            else:
                try:
                    last_run = datetime.fromisoformat(str(last_run_str).replace("Z", "+00:00"))
                    elapsed_h = (now - last_run).total_seconds() / 3600
                    remaining_h = interval_h - elapsed_h
                    if remaining_h <= 0:
                        state = "due"
                        next_run_str = "now"
                    else:
                        state = "pending"
                        hours_until = round(remaining_h, 1)
                        next_run_str = f"in {hours_until}h"
                except Exception:
                    state = "due"
                    next_run_str = "now (bad timestamp)"

        result.append({
            "id": task.get("id"),
            "name": task.get("name"),
            "interval_hours": interval_h,
            "enabled": enabled,
            "state": state,
            "last_run": last_run_str,
            "next_run": next_run_str,
            "hours_until": hours_until,
        })
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Hermes task scheduler")
    p.add_argument("--once", action="store_true", help="Check and dispatch due tasks, then exit")
    p.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    p.add_argument("--status", action="store_true", help="Show status of all scheduled tasks")
    p.add_argument("--poll", type=int, default=300, help="Daemon poll interval in seconds (default 300)")
    args = p.parse_args()

    if args.status:
        rows = status()
        if not rows:
            print("No tasks defined in schedule.yaml")
            return
        print(f"{'ID':<20} {'Name':<30} {'State':<10} {'Every':<10} {'Next run'}")
        print("-" * 80)
        for r in rows:
            print(f"{r['id']:<20} {r['name']:<30} {r['state']:<10} {r['interval_hours']}h{'':<5} {r['next_run'] or '—'}")
        return

    if args.once or args.dry_run:
        counts = run_once(dry_run=args.dry_run)
        print(f"Done — due: {counts['due']}, dispatched: {counts['dispatched']}, failed: {counts['failed']}")
        return

    run_daemon(poll_seconds=args.poll)


if __name__ == "__main__":
    main()
