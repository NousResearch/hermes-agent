"""Shared Kanban effort estimation and asynchronous persistence."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from hermes_cli import kanban_db
from hermes_cli import kanban_db_connect as kbc

log = logging.getLogger(__name__)

_ESTIMATE_SYSTEM_PROMPT = (
    "You estimate how much work an autonomous coding agent will spend on a "
    "kanban task. Given the task title and description, respond with STRICT "
    "JSON only (no prose, no code fence):\n"
    '{"est_tokens": <integer total tokens across the whole run>, '
    '"complexity": "S"|"M"|"L", '
    '"rationale": "<one short sentence>"}\n'
    "Base the token figure on a realistic multi-turn agent run (reading files, "
    "tool calls, edits, retries) — not a single reply. S≈small/localized, M≈multi-file, L≈broad or ambiguous."
)

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kanban-estimate")


def _cap(value: Optional[str], limit: int) -> str:
    value = (value or "").strip()
    return value if len(value) <= limit else value[:limit] + "…"


def estimate_text(title: str, body: Optional[str], *, task_id: Optional[str] = None) -> dict:
    if not (title or "").strip():
        return {"ok": False, "reason": "a title is required to estimate"}
    try:
        from agent.auxiliary_client import call_llm
    except Exception:
        return {"ok": False, "reason": "auxiliary client unavailable"}
    user_msg = f"Title: {_cap(title, 400)}\n\nDescription:\n{_cap(body, 4000) or '(none)'}"
    from agent.portal_tags import get_affinity_scope, reset_affinity_scope, set_affinity_scope
    affinity_token = None if get_affinity_scope() else set_affinity_scope(f"kanban:{task_id or 'estimate'}")
    try:
        resp = call_llm(
            task="kanban_estimator",
            messages=[{"role": "system", "content": _ESTIMATE_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            temperature=0.0, max_tokens=300, timeout=60)
    except Exception as exc:
        return {"ok": False, "reason": f"LLM error: {type(exc).__name__}"}
    finally:
        if affinity_token is not None:
            reset_affinity_scope(affinity_token)
    try:
        raw = (resp.choices[0].message.content or "").strip()
        model = getattr(resp, "model", None)
    except Exception:
        raw, model = "", None
    try:
        match = None if raw.lstrip().startswith("{") else re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0) if match else raw)
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        return {"ok": False, "reason": "could not parse an estimate from the model"}
    try:
        est_tokens = int(parsed.get("est_tokens") or 0)
    except (TypeError, ValueError):
        est_tokens = 0
    complexity = str(parsed.get("complexity") or "").strip().upper()
    return {
        "ok": True,
        "est_tokens": est_tokens,
        "complexity": complexity if complexity in {"S", "M", "L"} else None,
        "rationale": str(parsed.get("rationale") or "").strip() or None,
        "model": model,
    }


def estimate_and_persist(task_id: str, *, board: Optional[str] = None) -> dict:
    task = None
    for attempt in range(5):
        with kbc.connect_closing(board=board) as conn:
            task = kanban_db.get_task(conn, task_id)
            if task is not None:
                break
        if attempt == 4:
            return {"ok": False, "reason": "task not found"}
        time.sleep(0.1)
    if task is None:
        return {"ok": False, "reason": "task not found"}
    with kbc.connect_closing(board=board) as conn:
        enabled = bool(kanban_db.read_board_metadata(board).get("auto_estimate_enabled", False))
    if not enabled or task.status not in kanban_db.AUTO_ESTIMATE_STATUSES:
        return {"ok": False, "reason": "task is no longer eligible for automatic estimation"}
    result = estimate_text(task.title, task.body, task_id=task_id)
    with kbc.connect_closing(board=board) as conn:
        current = kanban_db.get_task(conn, task_id)
        enabled = bool(kanban_db.read_board_metadata(board).get("auto_estimate_enabled", False))
        if current is None or not enabled or current.status not in kanban_db.AUTO_ESTIMATE_STATUSES:
            return {"ok": False, "reason": "task changed while estimating"}
        with kanban_db.write_txn(conn):
            conn.execute("UPDATE tasks SET estimate = ? WHERE id = ?", (json.dumps(result), task_id))
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'estimate', ?, ?)",
                (task_id, json.dumps(result), int(time.time())),
            )
    return result


def schedule_task_estimate(task_id: str, *, board: Optional[str] = None) -> None:
    """Queue estimation after creation; no caller waits on the auxiliary model."""
    def run() -> None:
        try:
            estimate_and_persist(task_id, board=board)
        except Exception:
            log.exception("kanban estimate failed for %s", task_id)
    ctx = contextvars.copy_context()
    _EXECUTOR.submit(ctx.run, run)


def schedule_board_estimates(board: str) -> int:
    """Queue missing estimates for tasks through the running column.

    The caller remains non-blocking; the return value is the number queued.
    """
    with kbc.connect_closing(board=board) as conn:
        rows = conn.execute(
            "SELECT id FROM tasks "
            "WHERE status IN ('triage', 'todo', 'scheduled', 'ready', 'running') "
            "AND COALESCE(json_extract(estimate, '$.ok'), 0) != 1 "
            "ORDER BY created_at ASC"
        ).fetchall()
    for row in rows:
        schedule_task_estimate(row["id"], board=board)
    return len(rows)
