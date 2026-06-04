#!/usr/bin/env python3
"""
kanban_async_monitor.py — Async Kanban Chain Monitor

Watches a kanban task chain (T1→T2→T3) without blocking the orchestrator.
Uses `hermes kanban list --json` and `hermes kanban show --json` for all
state reads.

Design principle (Solution E architecture):
    Kanban   = state store (persistent, crash-safe)
    Monitor  = this script (detects blocks, dispatches stuck tasks)
    Agent    = decision maker (reviews notifications, manually unblocks)

Usage:
    # Start monitoring a 3-task chain
    python3 kanban_async_monitor.py --chain t_xxx,t_yyy,t_zzz

    # Custom poll interval and timeout
    python3 kanban_async_monitor.py --chain t_xxx,t_yyy --poll 20 --timeout 90

    # Read pending notifications
    python3 kanban_async_monitor.py --notifications

    # Clear notifications after reading
    python3 kanban_async_monitor.py --notifications --clear

    # Run as background daemon
    nohup python3 kanban_async_monitor.py --chain t_xxx,t_yyy,t_zzz &

Safety:
    NEVER auto-unblocks tasks. Instead, notifies the user for any block.
    Uses (task_id, event_id) pairs for idempotent block handling.
    File-locked JSON notifications prevent concurrent write corruption.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────

HERMES_BIN = "hermes"
POLL_INTERVAL = 30          # seconds
DEFAULT_TIMEOUT = 60         # minutes
MAX_CONSECUTIVE_ERRORS = 10  # stop polling after this many failures

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
NOTIFY_FILE = HERMES_HOME / "kanban" / "async_notifications.json"
STATE_DIR = HERMES_HOME / "kanban" / "worker_states"

_shutdown = False


def _on_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print(f"\n[{_ts()}] signal {signum}, saving state and exiting...")


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


# ── CLI Helpers ────────────────────────────────────────────────────

def _run(*args, timeout=15):
    """Run `hermes kanban ...`. Returns (stdout, stderr, returncode)."""
    cmd = [HERMES_BIN, "kanban"] + list(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", f"timeout after {timeout}s", -1
    except FileNotFoundError:
        return "", "hermes CLI not found", -1


def _run_json(*args, timeout=15):
    """Run `hermes kanban ... --json`. Returns parsed JSON or None."""
    cmd = [HERMES_BIN, "kanban"] + list(args) + ["--json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None


def get_chain_statuses(chain_ids):
    """Return {task_id: status_str} from `hermes kanban list --json`."""
    data = _run_json("list")
    if data is None:
        return None
    return {t["id"]: t["status"] for t in data if t["id"] in chain_ids}


def get_task_detail(task_id):
    """
    Return detailed task info from `hermes kanban show --json`.

    Returns dict with: status, block_reason, summary, latest_block_event_id,
    or None on failure.
    """
    data = _run_json("show", task_id)
    if data is None:
        return None

    task = data.get("task", {})
    events = data.get("events", [])
    block_reason = None
    latest_block_event_id = None

    for evt in reversed(events):
        if evt.get("kind") == "blocked":
            payload = evt.get("payload", {})
            block_reason = (
                payload.get("reason", "") if isinstance(payload, dict)
                else str(payload)
            )
            latest_block_event_id = evt.get("id")
            break

    return {
        "status": task.get("status", "unknown"),
        "block_reason": block_reason,
        "summary": data.get("latest_summary", ""),
        "latest_block_event_id": latest_block_event_id,
        "assignee": task.get("assignee", ""),
    }


# ── Notification (file-locked JSON) ─────────────────────────────────

def notify(message, chain_id):
    """Append notification with file locking + push to connected platforms."""
    NOTIFY_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── File notification (backup) ──
    notifications = []
    try:
        with open(NOTIFY_FILE, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            raw = f.read()
            if raw.strip():
                try:
                    notifications = json.loads(raw)
                except json.JSONDecodeError:
                    pass
            notifications.append({
                "time": datetime.now().strftime("%m-%d %H:%M:%S"),
                "chain": chain_id,
                "msg": message,
            })
            if len(notifications) > 100:
                notifications = notifications[-100:]
            f.seek(0)
            f.truncate()
            json.dump(notifications, f, indent=2, ensure_ascii=False)
            fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        pass

    # ── Push to messaging platforms (real-time) ──
    try:
        env = dict(os.environ)
        for v in ("HERMES_CRON_SESSION", "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
                   "HERMES_CRON_AUTO_DELIVER_PLATFORM"):
            env.pop(v, None)
        subprocess.run(
            [HERMES_BIN, "send", message, "-t", "feishu"],
            capture_output=True, text=True, timeout=15, env=env,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # Best-effort; file backup already persisted


def read_notifications(clear=False):
    """Read and optionally clear pending notifications."""
    if not NOTIFY_FILE.exists():
        print("No pending notifications.")
        return []

    try:
        with open(NOTIFY_FILE, "r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            raw = f.read()
            notifications = json.loads(raw) if raw.strip() else []
            if clear:
                f.seek(0)
                f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)
    except (OSError, json.JSONDecodeError):
        print("No pending notifications (parse error).")
        return []

    if not notifications:
        print("No pending notifications.")
        return []

    print(f"\n{'='*60}")
    print(f"Pending notifications ({len(notifications)}):")
    print(f"{'='*60}")
    for n in notifications:
        print(f"  [{n['time']}] {n['chain']}: {n['msg']}")
    if clear:
        print("\n(cleared)")
    else:
        print("\nUse --clear to remove these.")
    return notifications


# ── State Persistence ───────────────────────────────────────────────

def load_state(chain_id):
    """Load or initialize monitoring state for a chain."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = STATE_DIR / f"{chain_id}.json"
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "chain_id": chain_id,
        "started_at": datetime.now().isoformat(),
        "handled_block_events": [],
        "last_statuses": {},
        "last_poll": None,
        "consecutive_errors": 0,
    }


def save_state(state):
    """Atomically persist monitoring state (tmp+rename)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = STATE_DIR / f"{state['chain_id']}.json"
    state["last_poll"] = datetime.now().isoformat()
    tmp = state_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(state_file)


def cleanup_state(chain_id):
    """Remove state file after chain completes."""
    (STATE_DIR / f"{chain_id}.json").unlink(missing_ok=True)


# ── Core Monitoring Logic ───────────────────────────────────────────

def handle_blocked(task_id, state):
    """
    Handle a blocked task — NEVER auto-unblock.

    Notifies the user for ALL block reasons. Uses (task_id, event_id)
    pairs for idempotent handling (same task can block multiple times).
    """
    detail = get_task_detail(task_id)
    if detail is None:
        print(f"  [{_ts()}] WARN  cannot read detail for {task_id}")
        return

    event_id = detail.get("latest_block_event_id")
    reason = detail.get("block_reason", "") or ""
    block_key = f"{task_id}:{event_id}" if event_id else task_id

    if block_key in state["handled_block_events"]:
        return  # Already notified for this block

    is_review = reason.lower().startswith("review-required")
    prefix = "[review]" if is_review else "[blocked]"
    msg = f"{prefix} {task_id}: {reason[:100] or '(no reason)'}"
    print(f"  [{_ts()}] NOTIFY {msg}")
    notify(msg, state["chain_id"])

    state["handled_block_events"].append(block_key)
    save_state(state)


def dispatch_stuck_ready(task_id):
    """Nudge the kanban dispatcher if a task appears stuck ready."""
    stdout, _, rc = _run("dispatch", "--max", "2")
    if rc == 0:
        print(f"  [{_ts()}] dispatch (triggered by stuck {task_id})")


def _ts():
    """Short timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def monitor_chain(chain_ids, poll_interval=30, max_wait_minutes=60):
    """
    Main monitoring loop.

    Returns True if all tasks completed, False on timeout or fatal error.
    """
    global _shutdown
    chain_id = chain_ids[0]
    state = load_state(chain_id)
    state.setdefault("started_at", datetime.now().isoformat())
    save_state(state)

    start_time = datetime.now()
    deadline = start_time + timedelta(minutes=max_wait_minutes)

    print(f"[{_ts()}] watching chain: {' -> '.join(chain_ids)}")
    print(f"[{_ts()}]   poll: {poll_interval}s  timeout: {max_wait_minutes}m  "
          f"pid: {os.getpid()}")

    cycle = 0
    last_statuses = state.get("last_statuses", {})

    while datetime.now() < deadline and not _shutdown:
        cycle += 1
        statuses = get_chain_statuses(chain_ids)

        if statuses is None:
            state["consecutive_errors"] += 1
            print(f"[{_ts()}] ERR kanban list failed "
                  f"(#{state['consecutive_errors']}/{MAX_CONSECUTIVE_ERRORS})")
            if state["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS:
                notify(f"monitor failed: {MAX_CONSECUTIVE_ERRORS} consecutive errors",
                       state["chain_id"])
                save_state(state)
                return False
            time.sleep(poll_interval)
            continue

        state["consecutive_errors"] = 0

        # Log status changes
        changed = any(
            statuses.get(tid) != last_statuses.get(tid)
            for tid in chain_ids
        )
        if changed:
            print(f"[{_ts()}] --- cycle {cycle} ---")
            for tid in chain_ids:
                ns = statuses.get(tid, "not_found")
                os_ = last_statuses.get(tid, "start")
                if ns != os_:
                    print(f"  {tid}: {os_} -> {ns}")

        all_done = True
        for tid in chain_ids:
            s = statuses.get(tid, "not_found")

            if s == "blocked":
                all_done = False
                handle_blocked(tid, state)
            elif s == "ready":
                all_done = False
                if last_statuses.get(tid) == "ready":
                    dispatch_stuck_ready(tid)
            elif s in ("running", "todo"):
                all_done = False
            elif s == "not_found":
                all_done = False
                print(f"  [{_ts()}] WARN  {tid} not found in board")

        last_statuses = statuses.copy()
        state["last_statuses"] = last_statuses
        save_state(state)

        if all_done:
            elapsed = (datetime.now() - start_time).total_seconds()
            msg = (f"Chain complete: {len(chain_ids)} tasks, "
                   f"{elapsed/60:.1f} min")
            print(f"\n[{_ts()}] DONE {msg}")
            for tid in chain_ids:
                detail = get_task_detail(tid)
                if detail:
                    summary = detail.get("summary", "")[:300] or "(no summary)"
                    print(f"\n  [{tid}]\n  {summary}")
            notify(msg, state["chain_id"])
            cleanup_state(chain_id)
            return True

        if not changed:
            parts = []
            for tid in chain_ids:
                s = statuses.get(tid, "?")
                e = {"done": "*", "running": "~", "ready": ">",
                     "blocked": "!", "todo": "."}.get(s, "?")
                parts.append(f"{e}{tid[-6:]}")
            print(f"[{_ts()}] {' '.join(parts)}")

        time.sleep(poll_interval)

    if _shutdown:
        print(f"[{_ts()}] shutdown requested, state saved")
        save_state(state)
        return False

    msg = f"Timeout ({max_wait_minutes}min) at: {last_statuses}"
    print(f"\n[{_ts()}] TIMEOUT {msg}")
    notify(msg, state["chain_id"])
    return False


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="kanban_async_monitor — watch kanban chains without blocking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kanban_async_monitor.py --chain t_xxx,t_yyy,t_zzz
  kanban_async_monitor.py --chain t_xxx,t_yyy --poll 20 --timeout 90
  kanban_async_monitor.py --notifications
  kanban_async_monitor.py --notifications --clear
        """,
    )
    parser.add_argument("--chain", help="comma-separated task IDs (e.g. t_xxx,t_yyy)")
    parser.add_argument("--poll", type=int, default=POLL_INTERVAL,
                        help=f"poll interval in seconds (default: {POLL_INTERVAL})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"max wait in minutes (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--notifications", action="store_true",
                        help="read pending notifications")
    parser.add_argument("--clear", action="store_true",
                        help="clear notifications (with --notifications)")
    args = parser.parse_args()

    if args.notifications:
        read_notifications(clear=args.clear)
        return

    if not args.chain:
        parser.error("--chain is required (e.g. --chain t_xxx,t_yyy,t_zzz)")

    chain_ids = [t.strip() for t in args.chain.split(",")]
    if not chain_ids:
        parser.error("--chain must contain at least one task ID")

    for tid in chain_ids:
        if not (tid.startswith("t_") and len(tid) >= 10):
            print(f"WARNING: '{tid}' does not look like a valid task ID")

    success = monitor_chain(chain_ids, args.poll, args.timeout)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
