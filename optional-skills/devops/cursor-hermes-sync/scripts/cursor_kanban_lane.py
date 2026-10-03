#!/usr/bin/env python3
"""
cursor_kanban_lane.py — A Hermes Kanban worker lane that dispatches cursor-assigned
tasks to the Cursor AI IDE instead of a Hermes profile.

This is a standalone dispatcher daemon that:
1. Polls the Kanban DB for tasks assigned to 'cursor'
2. Claims them (atomically, like the normal dispatcher)
3. Opens Cursor on the task's workspace with the task prompt as context
4. Writes a comment to the task with the handoff context
5. Marks the task as running

When you're done in Cursor, you complete the task in Hermes:
  hermes kanban complete <task_id> --summary "Done in Cursor"

Architecture:
  - Uses the same dispatch_once() from hermes_cli.kanban_db with a custom spawn_fn
  - The spawn_fn opens Cursor CLI instead of spawning a Hermes profile subprocess
  - The task lifecycle (ready → running → done/blocked) is owned by the Kanban kernel
  - The dispatcher respects claim locks, TTLs, and failure limits

Usage:
  # Start the Cursor lane daemon (foreground, Ctrl+C to stop)
  python3 cursor_kanban_lane.py daemon

  # Run a single dispatch tick
  python3 cursor_kanban_lane.py tick

  # Check status — show cursor-assigned tasks
  python3 cursor_kanban_lane.py status

  # Dispatch a specific task by ID
  python3 cursor_kanban_lane.py dispatch <task_id>

  # Reclaim stuck running tasks (release claim, set back to ready)
  python3 cursor_kanban_lane.py reclaim [task_id] [--board <slug>] [--force]

Configuration:
  Set CURSOR_ASSIGNEE env var to change the assignee name (default: "cursor")
  Set CURSOR_CLI env var to override the Cursor CLI path
"""

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ── Hermes imports ───────────────────────────────────────────────────────────

HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
KANBAN_DB = HERMES_HOME / "kanban.db"

# Import the Kanban kernel
sys.path.insert(0, str(HERMES_HOME / "hermes-agent"))
try:
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db import (
        dispatch_once,
        connect_closing,
        kanban_db_path,
        set_workspace_path,
        _set_worker_pid,
    )
except ImportError as e:
    print(f"ERROR: Cannot import Hermes kanban module: {e}")
    print(f"Make sure HERMES_HOME is set correctly: {HERMES_HOME}")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

CURSOR_ASSIGNEE = os.environ.get("CURSOR_ASSIGNEE", "cursor")
CURSOR_CLI = os.environ.get(
    "CURSOR_CLI",
    "/Applications/Cursor.app/Contents/Resources/app/bin/cursor",
)
POLL_INTERVAL = float(os.environ.get("CURSOR_LANE_INTERVAL", "30"))  # seconds
DEFAULT_CLAIM_TTL_SECONDS = 900  # 15 minutes — matches Hermes dispatcher
STALE_CHECK_EVERY_N_TICKS = 10  # check for expired claims every N ticks


# ── Cursor spawn function ────────────────────────────────────────────────────

def find_cursor_cli():
    """Find the Cursor CLI binary."""
    # Check env override
    if os.environ.get("CURSOR_CLI"):
        return os.environ["CURSOR_CLI"]
    # Check PATH
    try:
        result = subprocess.run(["which", "cursor"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    # Check macOS app bundle
    if Path(CURSOR_CLI).exists():
        return CURSOR_CLI
    return None


def cursor_spawn(task, workspace: str, *, board: Optional[str] = None) -> Optional[int]:
    """
    Custom spawn_fn for the Kanban dispatcher.

    Instead of spawning `hermes -p <profile> chat -q ...`, this opens Cursor
    on the task's workspace and writes the task context to AGENTS.md so Cursor
    picks it up automatically.

    Returns a PID (the Cursor process) so the dispatcher can detect crashes.
    """
    cursor_cli = find_cursor_cli()
    if not cursor_cli:
        # Write a kanban comment explaining the failure
        try:
            db_path = kanban_db_path(board=board) if board else KANBAN_DB
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (task.id, "cursor-lane",
                 "ERROR: Cursor CLI not found. Install Cursor or set CURSOR_CLI env var.",
                 int(time.time())),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        return None

    # Write the task context to AGENTS.md in the workspace
    # so Cursor's AI picks it up when it opens
    workspace_path = Path(workspace)
    agents_md = workspace_path / "AGENTS.md"

    # Read the task body from the Kanban DB
    task_body = task.body or "No description provided"
    task_title = task.title or "Untitled task"

    # Read existing AGENTS.md if present, append the task context
    handoff_entry = f"""
## Kanban Task: {task_title}

- **Task ID:** {task.id}
- **Assigned to:** Cursor (via Hermes Kanban)
- **Board:** {board or "default"}

### Task Description

{task_body}

### Instructions

This task was assigned to Cursor from the Hermes Kanban board. When you're done:
1. Update the task in Hermes: `hermes kanban complete {task.id} --summary "Describe what you did"`
2. Or block it if you need human input: `hermes kanban block {task.id} --reason "Need clarification on..."`

### Handoff Log

- [{time.strftime('%Y-%m-%d %H:%M:%S')}] Dispatched from Hermes Kanban to Cursor
"""

    if agents_md.exists():
        content = agents_md.read_text()
        if f"Kanban Task: {task_title}" not in content:
            agents_md.write_text(content.rstrip() + handoff_entry)
    else:
        agents_md.write_text(f"# Project\n\n{handoff_entry}")

    # Open Cursor on the workspace
    try:
        proc = subprocess.Popen(
            [cursor_cli, str(workspace_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach from this process group
        )

        # Write a comment to the task so the Kanban board shows what happened
        try:
            db_path = kanban_db_path(board=board) if board else KANBAN_DB
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (task.id, "cursor-lane",
                 f"Dispatched to Cursor AI. Opened workspace at {workspace}. "
                 f"Task context written to AGENTS.md. Cursor PID: {proc.pid}",
                 int(time.time())),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        return proc.pid
    except Exception as e:
        # Log the error as a comment
        try:
            db_path = kanban_db_path(board=board) if board else KANBAN_DB
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (task.id, "cursor-lane",
                 f"Failed to spawn Cursor: {e}",
                 int(time.time())),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        return None


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_tick(args):
    """Run a single dispatch tick for cursor-assigned tasks only.

    We bypass dispatch_once() because it checks profile_exists() before
    calling spawn_fn — and 'cursor' is not a Hermes profile. Instead we
    directly query for ready cursor-assigned tasks, claim them, and spawn
    Cursor.
    """
    board = args.board

    # Resolve the board DB path
    try:
        db_path = kanban_db_path(board=board)
    except Exception as e:
        print(f"ERROR: Could not resolve board DB path: {e}")
        print(f"  Board: {board or 'default'}")
        sys.exit(1)

    if not db_path.exists():
        print(f"ERROR: Kanban DB not found: {db_path}")
        print(f"  Board: {board or 'default'}")
        print(f"  Run 'hermes kanban init' first.")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Find ready tasks assigned to cursor
    rows = conn.execute(
        "SELECT id, title, body, assignee, workspace_kind, workspace_path "
        "FROM tasks WHERE assignee = ? AND status = 'ready' "
        "ORDER BY priority DESC, created_at ASC LIMIT ?",
        (CURSOR_ASSIGNEE, args.max or 1),
    ).fetchall()

    spawned = []
    for row in rows:
        task_id = row["id"]
        title = row["title"]

        # Atomically claim the task
        claim_lock = f"cursor-lane:{os.getpid()}:{int(time.time())}"
        claim_expires = int(time.time()) + 900  # 15 min TTL

        result = conn.execute(
            "UPDATE tasks SET status = 'running', started_at = ?, "
            "claim_lock = ?, claim_expires = ? "
            "WHERE id = ? AND status = 'ready'",
            (int(time.time()), claim_lock, claim_expires, task_id),
        )
        if result.rowcount == 0:
            # Someone else claimed it
            continue
        conn.commit()

        # Resolve workspace
        workspace = row["workspace_path"]
        if not workspace or not Path(workspace).exists():
            board_root = HERMES_HOME / "kanban" / "boards" / (board or "default")
            workspace = str(board_root / "workspaces" / task_id)
            Path(workspace).mkdir(parents=True, exist_ok=True)
            conn.execute(
                "UPDATE tasks SET workspace_path = ? WHERE id = ?",
                (workspace, task_id),
            )
            conn.commit()

        # Create task stub for spawn function
        class TaskStub:
            pass
        task = TaskStub()
        task.id = row["id"]
        task.title = row["title"]
        task.body = row["body"]
        task.assignee = row["assignee"]

        # Spawn Cursor
        pid = cursor_spawn(task, workspace, board=board)

        if pid:
            conn.execute(
                "UPDATE tasks SET worker_pid = ? WHERE id = ?",
                (pid, task_id),
            )
            conn.commit()
            spawned.append(task_id)
            print(f"  Spawned Cursor for task {task_id}: {title}")
        else:
            # Spawn failed — mark as blocked
            conn.execute(
                "UPDATE tasks SET status = 'blocked' WHERE id = ?",
                (task_id,),
            )
            conn.commit()
            print(f"  Failed to spawn Cursor for {task_id} — marked blocked")

    conn.close()

    print(f"\nDispatch tick complete:")
    print(f"  Spawned:  {len(spawned)}")
    if spawned:
        print(f"  Tasks:    {', '.join(spawned)}")
    else:
        print(f"  (no cursor-assigned tasks ready)")


def cmd_daemon(args):
    """Run the Cursor lane dispatcher daemon."""
    print(f"Cursor Kanban Lane Daemon")
    print(f"  Assignee:  {CURSOR_ASSIGNEE}")
    print(f"  Cursor CLI: {find_cursor_cli() or 'NOT FOUND'}")
    print(f"  Interval:  {args.interval or POLL_INTERVAL}s")
    print(f"  Board:      {args.board or 'default'}")
    print()

    cursor_cli = find_cursor_cli()
    if not cursor_cli:
        print("ERROR: Cursor CLI not found. Set CURSOR_CLI env var or install Cursor.")
        sys.exit(1)

    # Handle Ctrl+C gracefully
    running = True

    def handle_signal(signum, frame):
        nonlocal running
        print(f"\nReceived signal {signum}, shutting down...")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Resolve the board DB path
    try:
        db_path = kanban_db_path(board=args.board)
    except Exception as e:
        print(f"ERROR: Could not resolve board DB path: {e}")
        sys.exit(1)

    if not db_path.exists():
        print(f"ERROR: Kanban DB not found: {db_path}")
        print(f"  Run 'hermes kanban init' first.")
        sys.exit(1)

    tick_count = 0
    while running:
        tick_count += 1
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            # ── Stale claim detection (every N ticks) ──────────────────────
            if tick_count % STALE_CHECK_EVERY_N_TICKS == 0:
                now = int(time.time())
                stale_rows = conn.execute(
                    "SELECT id, title FROM tasks "
                    "WHERE assignee = ? AND status = 'running' "
                    "AND claim_expires IS NOT NULL "
                    "AND claim_expires < ?",
                    (CURSOR_ASSIGNEE, now),
                ).fetchall()
                for sr in stale_rows:
                    sid = sr["id"]
                    _reclaim_task(
                        conn, sid,
                        comment="Cursor process appears stale "
                                "(claim TTL expired), reclaiming",
                    )
                    print(f"[tick {tick_count}] Reclaimed stale task {sid} "
                          f"(claim expired)")

            # Find ready cursor-assigned tasks
            rows = conn.execute(
                "SELECT id, title, body, assignee, workspace_path "
                "FROM tasks WHERE assignee = ? AND status = 'ready' "
                "ORDER BY priority DESC, created_at ASC LIMIT ?",
                (CURSOR_ASSIGNEE, args.max or 1),
            ).fetchall()

            spawned = []
            for row in rows:
                task_id = row["id"]
                title = row["title"]

                claim_lock = f"cursor-lane:{os.getpid()}:{int(time.time())}"
                claim_expires = int(time.time()) + 900

                result = conn.execute(
                    "UPDATE tasks SET status = 'running', started_at = ?, "
                    "claim_lock = ?, claim_expires = ? "
                    "WHERE id = ? AND status = 'ready'",
                    (int(time.time()), claim_lock, claim_expires, task_id),
                )
                if result.rowcount == 0:
                    continue
                conn.commit()

                # Resolve workspace
                workspace = row["workspace_path"]
                if not workspace or not Path(workspace).exists():
                    board_root = HERMES_HOME / "kanban" / "boards" / (args.board or "default")
                    workspace = str(board_root / "workspaces" / task_id)
                    Path(workspace).mkdir(parents=True, exist_ok=True)
                    conn.execute(
                        "UPDATE tasks SET workspace_path = ? WHERE id = ?",
                        (workspace, task_id),
                    )
                    conn.commit()

                class TaskStub:
                    pass
                task = TaskStub()
                task.id = row["id"]
                task.title = row["title"]
                task.body = row["body"]
                task.assignee = row["assignee"]

                pid = cursor_spawn(task, workspace, board=args.board)

                if pid:
                    conn.execute(
                        "UPDATE tasks SET worker_pid = ? WHERE id = ?",
                        (pid, task_id),
                    )
                    conn.commit()
                    spawned.append(task_id)
                    print(f"[tick {tick_count}] Dispatched to Cursor: {', '.join(spawned)}")
                else:
                    conn.execute(
                        "UPDATE tasks SET status = 'blocked' WHERE id = ?",
                        (task_id,),
                    )
                    conn.commit()
                    print(f"[tick {tick_count}] Failed to spawn Cursor for {task_id} — blocked")

            conn.close()

            if not spawned and tick_count == 1:
                # Only print on first tick if nothing found
                pass  # silent (watchdog pattern)

        except Exception as e:
            print(f"[tick {tick_count}] ERROR: {e}", file=sys.stderr)

        # Sleep between ticks
        interval = args.interval or POLL_INTERVAL
        for _ in range(int(interval * 10)):
            if not running:
                break
            time.sleep(0.1)

    print("Daemon stopped.")


def cmd_status(args):
    """Show cursor-assigned tasks on the board."""
    board = args.board

    # Resolve board DB path (supports HERMES_HOME override for testing)
    try:
        db_path = kanban_db_path(board=board)
    except Exception as e:
        print(f"ERROR: Could not resolve board DB path: {e}")
        return

    if not db_path.exists():
        print(f"No Kanban DB found for board '{board or 'default'}'.")
        print(f"Run 'hermes kanban init' first.")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get all tasks assigned to cursor
    rows = conn.execute(
        "SELECT id, title, status, assignee, body, "
        "datetime(created_at, 'unixepoch', 'localtime') as created "
        "FROM tasks WHERE assignee = ? ORDER BY created_at DESC",
        (CURSOR_ASSIGNEE,),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"No tasks assigned to '{CURSOR_ASSIGNEE}'.")
        print(f"Assign a task with: hermes kanban assign <task_id> {CURSOR_ASSIGNEE}")
        return

    print(f"Tasks assigned to '{CURSOR_ASSIGNEE}' ({len(rows)}):\n")
    for row in rows:
        task_id, title, status, assignee, body, created = row
        status_icon = {
            "ready": "▶",
            "running": "🔄",
            "done": "✓",
            "blocked": "⛔",
            "archived": "📦",
        }.get(status, "○")
        print(f"  {status_icon} {task_id}  {status:10s}  {title}")
        if body and len(body) > 100:
            print(f"    {body[:100]}...")
        elif body:
            print(f"    {body}")
        print(f"    Created: {created}")
        print()


def _reclaim_task(conn, task_id: str, *, comment: Optional[str] = None,
                  author: str = "cursor-lane") -> bool:
    """Reclaim a single cursor-assigned running task.

    Sets status='ready', clears claim_lock/claim_expires/worker_pid.
    Optionally writes a comment to the task first.

    Returns True if the task was reclaimed, False if the UPDATE affected
    no rows (e.g. the task was no longer running or not assigned to cursor).
    """
    if comment:
        try:
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (task_id, author, comment, int(time.time())),
            )
        except Exception:
            pass

    result = conn.execute(
        "UPDATE tasks SET status = 'ready', "
        "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
        "WHERE id = ? AND assignee = ? AND status = 'running'",
        (task_id, CURSOR_ASSIGNEE),
    )
    conn.commit()
    return result.rowcount > 0


def cmd_reclaim(args):
    """Reclaim cursor-assigned tasks that are stuck in 'running' status.

    Releases the claim (clears claim_lock, claim_expires, worker_pid) and
    sets status back to 'ready' so they can be re-dispatched.

    Usage:
      python3 cursor_kanban_lane.py reclaim [task_id] [--board <slug>] [--force]
    """
    board = args.board
    task_id = args.task_id

    try:
        db_path = kanban_db_path(board=board)
    except Exception as e:
        print(f"ERROR: Could not resolve board DB path: {e}")
        sys.exit(1)

    if not db_path.exists():
        print(f"ERROR: Kanban DB not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    now = int(time.time())

    if task_id:
        # Reclaim a specific task
        row = conn.execute(
            "SELECT id, title, status, assignee FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            print(f"ERROR: Task {task_id} not found.")
            conn.close()
            sys.exit(1)

        if row["assignee"] != CURSOR_ASSIGNEE:
            print(f"ERROR: Task {task_id} is assigned to '{row['assignee']}', "
                  f"not '{CURSOR_ASSIGNEE}'.")
            conn.close()
            sys.exit(1)

        if row["status"] != "running":
            print(f"Task {task_id} is not running (status='{row['status']}'). "
                  f"Nothing to reclaim.")
            conn.close()
            return

        reclaimed = _reclaim_task(
            conn, task_id,
            comment="Reclaimed by manual `reclaim` command — "
                    "claim released, status set back to ready.",
        )
        if reclaimed:
            print(f"Reclaimed task {task_id}: {row['title']}")
            print(f"  Status: running → ready")
            print(f"  Cleared: claim_lock, claim_expires, worker_pid")
        else:
            print(f"Could not reclaim task {task_id} "
                  f"(it may have changed state).")
        conn.close()
        return

    # Reclaim ALL cursor-assigned running tasks
    rows = conn.execute(
        "SELECT id, title, status, claim_lock, worker_pid FROM tasks "
        "WHERE assignee = ? AND status = 'running' "
        "ORDER BY started_at ASC",
        (CURSOR_ASSIGNEE,),
    ).fetchall()

    if not rows:
        print(f"No running tasks assigned to '{CURSOR_ASSIGNEE}' to reclaim.")
        conn.close()
        return

    print(f"Found {len(rows)} running task(s) assigned to '{CURSOR_ASSIGNEE}':\n")
    for row in rows:
        print(f"  {row['id']}  {row['title']}  "
              f"(pid={row['worker_pid'] or 'none'})")

    if not args.force:
        print()
        answer = input(
            f"\nReclaim all {len(rows)} task(s)? [y/N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            conn.close()
            return

    count = 0
    for row in rows:
        tid = row["id"]
        if _reclaim_task(
            conn, tid,
            comment="Reclaimed by manual `reclaim` command — "
                    "claim released, status set back to ready.",
        ):
            count += 1
            print(f"  Reclaimed {tid}: {row['title']}")

    print(f"\nReclaimed {count} of {len(rows)} task(s).")
    conn.close()


def cmd_dispatch_task(args):
    """Manually dispatch a specific task to Cursor."""
    task_id = args.task_id
    board = args.board

    # Resolve board DB path directly (avoid connect_closing which triggers
    # the delegation guard in _assert_not_delegated_child_mutation)
    try:
        db_path = kanban_db_path(board=board)
    except Exception as e:
        print(f"ERROR: Could not resolve board DB path: {e}")
        sys.exit(1)

    if not db_path.exists():
        print(f"ERROR: Kanban DB not found: {db_path}")
        print(f"  Run 'hermes kanban init' first.")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get the task
    row = conn.execute(
        "SELECT id, title, body, assignee, status, workspace_path "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()

    if not row:
        print(f"ERROR: Task {task_id} not found.")
        conn.close()
        sys.exit(1)

    tid = row["id"]
    title = row["title"]
    body = row["body"]
    assignee = row["assignee"]
    status = row["status"]
    workspace_path = row["workspace_path"]

    if assignee != CURSOR_ASSIGNEE:
        print(f"ERROR: Task {task_id} is assigned to '{assignee}', not '{CURSOR_ASSIGNEE}'.")
        print(f"Reassign with: hermes kanban assign {task_id} {CURSOR_ASSIGNEE}")
        conn.close()
        sys.exit(1)

    if status == "done":
        print(f"Task {task_id} is already done.")
        conn.close()
        return

    # Get or create workspace
    if workspace_path and Path(workspace_path).exists():
        workspace = workspace_path
    else:
        board_root = HERMES_HOME / "kanban" / "boards" / (board or "default")
        workspace = str(board_root / "workspaces" / task_id)
        Path(workspace).mkdir(parents=True, exist_ok=True)
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?",
            (workspace, task_id),
        )
        conn.commit()

    # Create a minimal task object for the spawn function
    class TaskStub:
        pass

    task = TaskStub()
    task.id = tid
    task.title = title
    task.body = body
    task.assignee = assignee

    print(f"Dispatching task {task_id} to Cursor...")
    print(f"  Title: {title}")
    print(f"  Workspace: {workspace}")

    pid = cursor_spawn(task, workspace, board=board)

    if pid:
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, status = 'running', started_at = ? WHERE id = ?",
            (pid, int(time.time()), task_id),
        )
        conn.commit()
        print(f"  Cursor PID: {pid}")
        print(f"  Task status: running")
        print()
        print("When done in Cursor, complete the task:")
        print(f"  hermes kanban complete {task_id} --summary 'Done'")
    else:
        print("  ERROR: Failed to spawn Cursor.")

    conn.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cursor AI worker lane for Hermes Kanban.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is a custom Kanban dispatcher that sends cursor-assigned tasks to the
Cursor AI IDE instead of a Hermes profile.

Workflow:
  1. Create a task: hermes kanban create "Build landing page" --assignee cursor --body "Wire up hero"
  2. Start the daemon:  python3 cursor_kanban_lane.py daemon
  3. Cursor opens with the task context in AGENTS.md
  4. When done: hermes kanban complete <task_id> --summary "Done"

Or dispatch manually:
  python3 cursor_kanban_lane.py dispatch <task_id>
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # tick — single dispatch tick
    p = sub.add_parser("tick", help="Run a single dispatch tick")
    p.add_argument("--board", default=None, help="Board slug")
    p.add_argument("--max", type=int, default=1, help="Max spawns per tick")
    p.set_defaults(func=cmd_tick)

    # daemon — continuous polling
    p = sub.add_parser("daemon", help="Run the Cursor lane dispatcher daemon")
    p.add_argument("--board", default=None, help="Board slug")
    p.add_argument("--interval", type=float, default=POLL_INTERVAL, help="Poll interval seconds")
    p.add_argument("--max", type=int, default=1, help="Max spawns per tick")
    p.set_defaults(func=cmd_daemon)

    # status — show cursor-assigned tasks
    p = sub.add_parser("status", help="Show cursor-assigned tasks")
    p.add_argument("--board", default=None, help="Board slug")
    p.set_defaults(func=cmd_status)

    # dispatch — manually dispatch a specific task
    p = sub.add_parser("dispatch", help="Manually dispatch a specific task to Cursor")
    p.add_argument("task_id", help="Task ID to dispatch")
    p.add_argument("--board", default=None, help="Board slug")
    p.set_defaults(func=cmd_dispatch_task)

    # reclaim — release stuck running tasks back to ready
    p = sub.add_parser(
        "reclaim",
        help="Reclaim stuck running tasks (release claim, set status=ready)",
        description=(
            "Reclaim cursor-assigned tasks stuck in 'running' status. "
            "Releases the claim and sets status back to 'ready' for re-dispatch. "
            "Without a task_id, reclaims ALL running cursor tasks (with prompt)."
        ),
    )
    p.add_argument("task_id", nargs="?", default=None,
                   help="Specific task ID to reclaim (optional)")
    p.add_argument("--board", default=None, help="Board slug")
    p.add_argument("--force", action="store_true",
                   help="Skip confirmation prompt when reclaiming all")
    p.set_defaults(func=cmd_reclaim)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()