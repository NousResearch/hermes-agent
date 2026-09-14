"""Opt-in auxiliary remediation for blocked Kanban tasks without a creator wake route."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_specify import _call_aux, _extract_json_blob

logger = logging.getLogger(__name__)

_ATTEMPT_EVENT = "blocked_remediation_started"
_AUTHOR = "blocked-resolver"

_SYSTEM_PROMPT = """You are the fallback resolver for a blocked Hermes Kanban task.
The task has no resumable creator-agent route. Decide whether the task can safely be retried
with additional context, or must remain blocked for a human.

Output exactly one JSON object:
{
  "action": "retry" | "escalate",
  "context": "concise context or instruction for the worker/human",
  "rationale": "one sentence"
}

Rules:
- Choose retry only when the supplied blocker is transient or the context you provide materially
  resolves a missing-context blocker.
- Never invent credentials, approvals, completed dependencies, facts, or user decisions.
- Never treat silence as consent. Human approval and capability blockers must stay blocked.
- Do not create tasks, reassign work, or claim that an external action occurred.
- Keep context under 1200 characters. Output JSON only.
"""

_USER_TEMPLATE = """Task id: {task_id}
Title: {title}
Assignee: {assignee}
Block kind: {block_kind}
Block reason: {reason}
Latest run: {run_summary}
Remediation attempt: {attempt}/{max_attempts}
Task body:
{body}
"""


@dataclass(frozen=True)
class ResolveOutcome:
    task_id: Optional[str]
    attempted: bool
    action: str = ""
    reason: str = ""


def _safe_text(value: Any, limit: int) -> str:
    from agent.redact import redact_sensitive_text

    text = redact_sensitive_text("" if value is None else str(value), force=True, redact_url_credentials=True)
    return text[:limit]


def _latest_block_payload(conn: Any, task_id: str) -> dict:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? "
        "AND kind IN ('blocked', 'block_loop_detected') ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None or not row["payload"]:
        return {}
    try:
        value = json.loads(row["payload"])
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _latest_run_summary(conn: Any, task_id: str) -> str:
    row = conn.execute(
        "SELECT summary, outcome FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None:
        return "(none)"
    summary = _safe_text(row["summary"], 800) or "(no summary)"
    return f"{row['outcome'] or 'unknown'}: {summary}"


def _has_creator_wake(conn: Any, task_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM kanban_notify_subs WHERE task_id = ? "
        "AND delivery_mode IN ('wake', 'notify+wake') LIMIT 1",
        (task_id,),
    ).fetchone() is not None


def _claim_candidate(conn: Any, max_attempts: int) -> tuple[Optional[kb.Task], int]:
    """Atomically reserve one eligible blocked task for an auxiliary call."""
    with kb.write_txn(conn):
        rows = conn.execute(
            "SELECT id FROM tasks WHERE status = 'blocked' "
            "AND (block_kind IS NULL OR block_kind NOT IN ('needs_input', 'capability')) "
            "ORDER BY created_at ASC, id ASC"
        ).fetchall()
        for row in rows:
            task_id = row["id"]
            if _has_creator_wake(conn, task_id):
                continue
            attempts = int(conn.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
                (task_id, _ATTEMPT_EVENT),
            ).fetchone()[0])
            if attempts >= max_attempts:
                continue
            attempt = attempts + 1
            kb._append_event(
                conn, task_id, _ATTEMPT_EVENT,
                {"attempt": attempt, "max_attempts": max_attempts},
            )
            return kb.get_task(conn, task_id), attempt
    return None, 0


def _parse_decision(raw: str) -> tuple[str, str, str]:
    parsed = _extract_json_blob(raw)
    if parsed is None:
        return "", "", "LLM returned malformed JSON"
    action = str(parsed.get("action") or "").strip().lower()
    if action not in {"retry", "escalate"}:
        return "", "", "LLM returned an unsupported action"
    context = _safe_text(parsed.get("context"), 1200).strip()
    rationale = _safe_text(parsed.get("rationale"), 500).strip()
    if not context:
        return "", "", "LLM response omitted remediation context"
    note = f"Automated blocker remediation: {context}"
    if rationale:
        note += f"\nRationale: {rationale}"
    return action, note, ""


def resolve_one(conn: Any, *, max_attempts: int = 1, timeout: int = 120) -> ResolveOutcome:
    """Resolve at most one blocked task; expected failures leave it blocked."""
    max_attempts = max(1, int(max_attempts))
    task, attempt = _claim_candidate(conn, max_attempts)
    if task is None:
        return ResolveOutcome(None, False, reason="no eligible blocked task")

    payload = _latest_block_payload(conn, task.id)
    raw, reason = _call_aux(
        "blocked resolver", task.id, aux_task="kanban_block_resolver", system=_SYSTEM_PROMPT,
        user=_USER_TEMPLATE.format(
            task_id=task.id,
            title=_safe_text(task.title, 400),
            assignee=_safe_text(task.assignee, 100),
            block_kind=task.block_kind or "unspecified",
            reason=_safe_text(payload.get("reason"), 1200) or "(no reason recorded)",
            run_summary=_latest_run_summary(conn, task.id),
            attempt=attempt,
            max_attempts=max_attempts,
            body=_safe_text(task.body, 4000) or "(no body)",
        ),
        max_tokens=1200, timeout=timeout, log=logger,
    )
    if raw is None:
        return ResolveOutcome(task.id, True, reason=reason)

    action, note, reason = _parse_decision(raw)
    if not action:
        return ResolveOutcome(task.id, True, reason=reason)

    current = kb.get_task(conn, task.id)
    if current is None or current.status != "blocked":
        return ResolveOutcome(task.id, True, action=action, reason="task changed while resolver ran")
    kb.add_comment(conn, task.id, _AUTHOR, note)
    if action == "retry":
        # needs_input/capability were excluded at claim time; recheck the live row
        # so a concurrent edit cannot turn this into an approval bypass.
        current = kb.get_task(conn, task.id)
        if current and current.status == "blocked" and current.block_kind not in {"needs_input", "capability"}:
            if kb.unblock_task(conn, task.id):
                return ResolveOutcome(task.id, True, action="retry", reason="task unblocked")
        return ResolveOutcome(task.id, True, action="escalate", reason="human gate preserved")
    return ResolveOutcome(task.id, True, action="escalate", reason="left blocked for human")
