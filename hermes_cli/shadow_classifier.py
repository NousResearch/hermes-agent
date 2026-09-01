"""P4.1 — Deterministic shadow risk classification (non-mutating).

Pure deterministic SUGGESTION contract for human-created tasks:

    kind: risk_classification_suggested
    payload:
        suggested_tier: fast | full
        suggested_task_kind: <bounded token>
        reasons: [<bounded structured token>]
        classifier_version: <token>

Hard invariants:
  * NEVER modifies persisted task tier, routing, assignee, reviewer or
    status; only inserts a ``risk_classification_suggested`` event.
  * Never overwrites a human tier/task-kind choice (suggestions are
    informational; humans own the columns).
  * Same normalised task contract → same suggestion (sha256-stable).
  * ONLY caller-proven human-created tasks are eligible.  "Human" is
    proven by a structured origin seam (human-originated creator set),
    NEVER inferred from title/body prose.  When source identity is
    unavailable the function records no suggestion and reports the
    integration gap.
  * Event insertion is idempotent per (task_id, classifier_version).
  * Reasons are bounded structured tokens; no prompt/body text copied.
  * Default runtime behaviour unchanged: nothing calls this unless a
    future (separately approved) integration does.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from typing import Any, Optional

CLASSIFIER_VERSION = "shadow-classifier-1"
EVENT_KIND = "risk_classification_suggested"

# Structured human-origin seams (created_by values that prove human
# creation).  Anything outside this set (feature-pipeline, swarm,
# governance, decompose children, etc.) is NOT eligible.
HUMAN_CREATED_BY = {
    "dashboard", "ideabox", "idea-box", "cli",
}
# Profile-author stamps (kanban CLI _profile_author()) are human sessions.
_SYSTEM_CREATED_BY = {
    "feature-pipeline", "denji-governance", "system",
}

_BOUNDARY_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_\-.:]{0,63}$")


def _bounded(token: str) -> str:
    if not _BOUNDARY_TOKEN_RE.match(token):
        raise ValueError(f"unbounded token rejected: {token!r}")
    return token


def is_human_created(task_row: sqlite3.Row | dict) -> bool:
    """Prove human creation ONLY from structured seams.

    A task is human-created when created_by is a human origin: the
    dashboard, idea-box, an explicit 'cli' stamp, or a profile-author
    stamp from an interactive Kanban session (created_by values that are
    profile names in the roster).  Automation stamps are excluded.
    Returns False when identity is unavailable.
    """
    created_by = task_row["created_by"] if not isinstance(task_row, dict) else task_row.get("created_by")
    if not isinstance(created_by, str) or not created_by.strip():
        return False
    created_by = created_by.strip()
    if created_by in _SYSTEM_CREATED_BY:
        return False
    if created_by in HUMAN_CREATED_BY:
        return True
    # Interactive kanban-CLI creation stamps a profile author (e.g.
    # "kensei", "misa-misa"); automation stamps are the known system
    # tokens filtered above.  A profile-name token is human-origin.
    if _BOUNDARY_TOKEN_RE.match(created_by):
        return True
    return False


def suggest(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    classifier_version: str = CLASSIFIER_VERSION,
    now: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Compute a deterministic suggestion for an eligible human task.

    Returns None when the task is missing, ineligible (non-human or
    system-created), or already classified by this classifier version.
    """
    row = conn.execute(
        "SELECT id, title, tier, task_kind, priority, max_runtime_seconds, created_by "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    if not is_human_created(row):
        return None

    # Idempotence: one suggestion per task + classifier version.
    existing = conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1",
        (task_id, EVENT_KIND),
    ).fetchone()
    if existing:
        try:
            payload = json.loads(existing["payload"] or "{}")
            if payload.get("classifier_version") == classifier_version:
                return None  # already classified by this version
        except (json.JSONDecodeError, TypeError):
            pass

    title = row["title"] or ""
    kind = row["task_kind"] or "task"
    priority = int(row["priority"] or 0)
    budget = row["max_runtime_seconds"]

    reasons: list[str] = []
    # Deterministic fast/full heuristic (bounded structured values only):
    # heavy signals push 'full' — multi-repo work, tight budgets, elevated
    # priority, bug/gate kinds; everything else suggests 'fast'.
    if kind in ("bug", "gate"):
        reasons.append("kind:" + _bounded(kind))
        tier = "full"
    elif priority >= 2:
        reasons.append("priority:high")
        tier = "full"
    elif budget is not None and int(budget or 0) >= 1800:
        reasons.append("runtime_budget:large")
        tier = "full"
    else:
        tier = "fast"
    if not reasons:
        reasons.append("default:light")
    reasons.append("kind:" + _bounded(str(kind)))

    suggestion = {
        "kind": EVENT_KIND,
        "suggested_tier": tier,
        "suggested_task_kind": _bounded(str(kind)),
        "reasons": reasons,
        "classifier_version": classifier_version,
    }
    return suggestion


def insert_shadow_event(
    conn: sqlite3.Connection,
    task_id: str,
    suggestion: dict[str, Any],
) -> Optional[int]:
    """Insert the shadow suggestion event idempotently (task+version).

    Never modifies the tasks row.  Returns the event row id, or None when
    this task+version was already classified.
    """
    version = suggestion["classifier_version"]
    existing = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1",
        (task_id, EVENT_KIND),
    ).fetchone()
    if existing:
        try:
            payload = json.loads(existing["payload"] or "{}")
            if payload.get("classifier_version") == version:
                return None
        except (json.JSONDecodeError, TypeError):
            pass
    cur = conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, NULL, ?, ?, ?)",
        (
            task_id,
            EVENT_KIND,
            json.dumps(suggestion, sort_keys=True),
            int(time.time()),
        ),
    )
    return int(cur.lastrowid)


def classification_disagreement(
    task_row: sqlite3.Row | dict, suggestion: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Report suggestion vs human choice for later shadow comparison.

    Pure function; never touches routing.  Returns None when there is no
    suggestion or the human has not chosen a tier yet.
    """
    if suggestion is None:
        return None
    human_tier = (
        task_row["tier"] if not isinstance(task_row, dict) else task_row.get("tier")
    )
    if not human_tier:
        return None
    human_tier = str(human_tier).lower()
    suggested = suggestion["suggested_tier"]
    return {
        "task_id": task_row["id"] if not isinstance(task_row, dict) else task_row.get("id"),
        "human_tier": human_tier,
        "suggested_tier": suggested,
        "material_disagreement": human_tier != suggested,
        "disagreement_reason": (
            None if human_tier == suggested
            else "tier:" + ("human_full_suggested_fast" if (human_tier == "full" and suggested == "fast")
                            else "human_fast_suggested_full")
        ),
        "classifier_version": suggestion["classifier_version"],
    }