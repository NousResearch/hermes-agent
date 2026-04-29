"""
JM (即梦) Task Queue — minimal viable module for Hermes Agent.

Handles:
- /jm add --worker <id> --prompt <text>   → create task
- /jm add --prompt <text>                 → create task (default worker)
- /jm list                                → list pending/running tasks
- /jm status                              → queue summary

Persistence: SQLite at ~/.hermes/jm_tasks.db (WAL mode, survives restart).
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WORKER = "jm-a"

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

_DB_PATH: Path = get_hermes_home() / "jm_tasks.db"

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS jm_tasks (
    task_id     TEXT PRIMARY KEY,
    worker_id   TEXT NOT NULL DEFAULT 'jm-a',
    prompt      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
"""


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection (WAL mode)."""
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    return conn


# ---------------------------------------------------------------------------
# Task ID generation: jm_YYYYMMDD_NNN
# ---------------------------------------------------------------------------

def _generate_task_id() -> str:
    """Generate a task ID like jm_20260418_001 (daily sequential)."""
    today = datetime.now().strftime("%Y%m%d")
    prefix = f"jm_{today}_"
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT task_id FROM jm_tasks WHERE task_id LIKE ? ORDER BY task_id DESC LIMIT 1",
            (f"{prefix}%",),
        ).fetchone()
        if row:
            last_seq = int(row["task_id"].split("_")[-1])
            seq = last_seq + 1
        else:
            seq = 1
        return f"{prefix}{seq:03d}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

def _parse_add_args(raw_args: str) -> tuple[str, str]:
    """Parse ``--worker <id> --prompt <text>`` from raw argument string.

    Returns (worker_id, prompt).  Raises ValueError on bad input.
    """
    worker_id = DEFAULT_WORKER
    prompt = ""

    # Extract --worker value (if present)
    worker_match = re.search(r"--worker\s+(\S+)", raw_args)
    if worker_match:
        worker_id = worker_match.group(1)
        # Remove --worker part from the string
        raw_args = raw_args[:worker_match.start()] + raw_args[worker_match.end():]

    # Extract --prompt value (everything after --prompt)
    prompt_match = re.search(r"--prompt\s+(.*)", raw_args, re.DOTALL)
    if prompt_match:
        prompt = prompt_match.group(1).strip()
    else:
        # If no --prompt flag, treat remaining text as prompt
        remaining = raw_args.strip()
        if remaining:
            prompt = remaining

    if not prompt:
        raise ValueError("缺少 prompt 参数。用法: /jm add --prompt <文本>")

    return worker_id, prompt


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def add_task(worker_id: str, prompt: str) -> dict:
    """Insert a new task and return its metadata dict."""
    task_id = _generate_task_id()
    now = time.time()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO jm_tasks (task_id, worker_id, prompt, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'pending', ?, ?)",
            (task_id, worker_id, prompt, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    logger.info("JM task created: %s worker=%s prompt=%.60s", task_id, worker_id, prompt)
    return {
        "task_id": task_id,
        "worker_id": worker_id,
        "prompt": prompt,
        "status": "pending",
    }


def list_tasks(limit: int = 50) -> list[dict]:
    """Return tasks ordered by creation time (oldest first)."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT task_id, worker_id, prompt, status, created_at "
            "FROM jm_tasks ORDER BY created_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def queue_summary() -> dict:
    """Return counts by status."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM jm_tasks GROUP BY status"
        ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Command handler (called from GatewayRunner)
# ---------------------------------------------------------------------------

async def handle_jm_command(args_text: str) -> str:
    """Process ``/jm <subcommand> [args]`` and return a reply string.

    Supported subcommands:
        add   — create a new task
        list  — show recent tasks
        status — queue summary
    """
    args_text = args_text.strip()

    # Determine subcommand
    if not args_text:
        return _help_text()

    parts = args_text.split(None, 1)
    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "add":
        return _handle_add(rest)
    elif sub == "list":
        return _handle_list()
    elif sub == "status":
        return _handle_status()
    elif sub == "help":
        return _help_text()
    else:
        return f"未知子命令 `{sub}`。\n\n{_help_text()}"


# ---------------------------------------------------------------------------
# Sub-handlers
# ---------------------------------------------------------------------------

def _handle_add(raw_args: str) -> str:
    try:
        worker_id, prompt = _parse_add_args(raw_args)
    except ValueError as e:
        return f"❌ 参数错误: {e}"

    task = add_task(worker_id, prompt)

    prompt_preview = task["prompt"][:80] + ("…" if len(task["prompt"]) > 80 else "")
    return (
        f"✅ 任务已入队\n"
        f"• Task ID: {task['task_id']}\n"
        f"• Worker: {task['worker_id']}\n"
        f"• Prompt: {prompt_preview}\n"
        f"• Status: {task['status']}"
    )


def _handle_list() -> str:
    tasks = list_tasks(limit=50)
    if not tasks:
        return "当前 JM 队列为空"

    lines = ["当前 JM 队列：\n"]
    for i, t in enumerate(tasks, 1):
        prompt_short = t["prompt"][:30] + ("..." if len(t["prompt"]) > 30 else "")
        lines.append(
            f"[{i}] {t['task_id']} | {t['worker_id']} | {t['status']} | {prompt_short}"
        )
    return "\n".join(lines)


def _handle_status() -> str:
    summary = queue_summary()
    if not summary:
        return "📭 队列为空。"

    total = sum(summary.values())
    parts = [f"📊 队列状态 (共 {total} 个任务)\n"]
    icon_map = {"pending": "⏳", "running": "🔄", "done": "✅", "failed": "❌"}
    for status, count in sorted(summary.items()):
        icon = icon_map.get(status, "❓")
        parts.append(f"{icon} {status}: {count}")
    return "\n".join(parts)


def _help_text() -> str:
    return (
        "即梦任务队列 /jm 命令\n\n"
        "• /jm add --prompt <文本> — 添加任务（默认 worker jm-a）\n"
        "• /jm add --worker jm-a --prompt <文本> — 添加任务到指定 worker\n"
        "• /jm list — 查看最近任务\n"
        "• /jm status — 队列统计\n"
        "• /jm help — 显示此帮助"
    )
