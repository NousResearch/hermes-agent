"""P4.1 — Deterministic shadow risk classification (non-mutating).

Corrected per audit correction handoff (C5):

  * Human provenance FAILS CLOSED: only trusted human creation seams
    (dashboard, idea-box, cli) and REGISTRY-VERIFIED interactive profile
    authors are eligible.  Unknown/missing/malformed/automation stamps
    produce no suggestion — never inferred from token shape.
  * Per-version idempotence searches ALL prior suggestion events, not
    only the latest.
  * No duplicate kind:* reason tokens.

Pure deterministic SUGGESTION contract for human-created tasks:

    kind: risk_classification_suggested
    payload:
        suggested_tier: fast | full
        suggested_task_kind: <bounded token>
        reasons: [<bounded structured token>]
        classifier_version: <token>

Hard invariants (unchanged):
  * NEVER modifies persisted task tier, routing, assignee, reviewer or
    status; only inserts a ``risk_classification_suggested`` event.
  * Never overwrites a human tier/task-kind choice.
  * Same normalised task contract → same suggestion.
  * Event insertion is idempotent per (task_id, classifier_version).
  * Reasons are bounded structured tokens; no prompt/body text copied.
  * Default runtime behaviour unchanged.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

CLASSIFIER_VERSION = "shadow-classifier-3"
EVENT_KIND = "risk_classification_suggested"

# R2-1: atomic exactly-once invariant for suggestion events.
_SUGGESTION_UNIQUENESS_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_shadow_task_version "
    "ON task_events (task_id, kind, "
    "CAST(json_extract(payload, '$.classifier_version') AS TEXT)) "
    "WHERE kind = 'risk_classification_suggested'"
)


def ensure_suggestion_uniqueness(conn: sqlite3.Connection) -> bool:
    """R2-1: install the atomic uniqueness invariant with transaction
    preservation (execute(), never executescript) and fail-closed duplicate
    detection.  Returns True when the invariant is installed; False when
    legacy duplicates block it (never silently deduped).
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='ux_shadow_task_version'"
    ).fetchone()
    if row is not None:
        return True
    dupes = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT task_id, json_extract(payload, '$.classifier_version') AS v"
        "  FROM task_events WHERE kind = 'risk_classification_suggested'"
        "  GROUP BY task_id, v HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    if dupes:
        return False  # fail closed; reconciliation plan required (R2-3 pattern)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_shadow_task_version "
        "ON task_events (task_id, kind, "
        "CAST(json_extract(payload, '$.classifier_version') AS TEXT)) "
        "WHERE kind = 'risk_classification_suggested'"
    )
    return True


# R2-7: trusted creation seams that THEMSELVES prove human creation.
TRUSTED_HUMAN_CREATED_BY = {"dashboard", "ideabox", "idea-box", "cli"}

# Automation/derived-origin stamps that must never classify as human.
_AUTOMATION_STAMPS = {
    "feature-pipeline", "denji-governance", "system",
    "swarm", "cron", "webhook", "job", "automation", "scheduler",
    "worker", "pipeline", "governance", "decompose", "triage-router",
    "orchestrator", "market-scanner", "skill-broker", "skill-research",
}

_BOUNDARY_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_\-.:]{0,63}$")

# R2-7: structured interactive-human marker contract.  A profile-author
# stamp is eligible ONLY when a caller-proven interactive marker exists
# (an event/field distinguishing interactive user requests from
# agent/cron/webhook creation).  Default seam name; tests may stub.
INTERACTIVE_MARKER_EVENT_KIND = "task.created_interactively"


def _task_has_interactive_marker(conn: sqlite3.Connection, task_id) -> bool:
    """True only when a structured interactive-creation marker exists for
    the task (caller-proven; no prose inference)."""
    row = conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? LIMIT 1",
        (task_id, INTERACTIVE_MARKER_EVENT_KIND),
    ).fetchone()
    return row is not None


def _bounded(token: str) -> str:
    if not _BOUNDARY_TOKEN_RE.match(token):
        raise ValueError(f"unbounded token rejected: {token!r}")
    return token


def _registry_profile_names(hermes_home: Optional[Path] = None) -> Optional[set[str]]:
    """R2-7: registry-verified profile names via the FULL core validator.

    Uses ``profile_registry.load_registry()`` so an invalid registry
    (bad enums, unknown parents, cycles, duplicates) yields None —
    malformed data can never authorise eligibility.  Returns None when
    the registry is absent/unreadable/invalid (fail closed).
    """
    home = hermes_home or Path(
        __import__("os").environ.get("HERMES_HOME", "") or
        Path.home() / ".hermes"
    )
    path = home / "governance" / "profile-registry.yaml"
    if not path.exists():
        return None
    try:
        from hermes_cli.profile_registry import load_registry
        reg = load_registry(path)
    except Exception:
        return None
    return {p["name"] for p in reg["profiles"]}


def is_human_created(
    task_row: "sqlite3.Row | dict",
    conn: Optional[sqlite3.Connection] = None,
    hermes_home: Optional[Path] = None,
) -> bool:
    """Prove human creation ONLY from structured seams (fail closed).

    R2-7: eligible origins are
      * trusted creation seams that themselves prove human creation
        (dashboard / idea-box / cli), or
      * a registry-verified profile author WITH a caller-proven structured
        interactive-creation marker (``task.created_interactively``) —
        registry membership alone is NOT human evidence.

    Malformed registries authorise nobody.  Token shape is never evidence.
    """
    created_by = (
        task_row["created_by"] if not isinstance(task_row, dict)
        else task_row.get("created_by")
    )
    if not isinstance(created_by, str) or not created_by.strip():
        return False
    created_by = created_by.strip()
    if not _BOUNDARY_TOKEN_RE.match(created_by):
        return False
    if created_by in _AUTOMATION_STAMPS:
        return False
    if created_by in TRUSTED_HUMAN_CREATED_BY:
        return True
    # Profile authors: registry-verified AND structured interactive marker.
    names = _registry_profile_names(hermes_home)
    if names is None or created_by not in names:
        return False
    if conn is None:
        return False  # cannot verify the marker without the connection
    return _task_has_interactive_marker(conn, task_row["id"] if not isinstance(task_row, dict) else task_row.get("id"))


def suggest(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    classifier_version: str = CLASSIFIER_VERSION,
    now: Optional[int] = None,
    hermes_home: Optional[Path] = None,
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
    # Normalise to dict so plain (non-Row-factory) connections work too.
    if isinstance(row, dict):
        task = row
    else:
        try:
            task = {k: row[k] for k in row.keys()}
        except (AttributeError, IndexError):
            task = dict(zip(
                ("id", "title", "tier", "task_kind", "priority",
                 "max_runtime_seconds", "created_by"), row))
    if not is_human_created(task, conn=conn, hermes_home=hermes_home):
        return None

    # C5: idempotence — search ALL prior events for this classifier
    # version, not only the latest.
    prior = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
        (task_id, EVENT_KIND),
    ).fetchall()
    for p in prior:
        try:
            payload = json.loads(p[0] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if payload.get("classifier_version") == classifier_version:
            return None  # already classified by this version

    kind = task["task_kind"] or "task"
    priority = int(task["priority"] or 0)
    budget = task["max_runtime_seconds"]

    reasons: list[str] = []
    if kind in ("bug", "gate"):
        reasons.append("kind:" + _bounded(str(kind)))
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

    R2-1: atomic via the uniqueness invariant — concurrent writers race on
    INSERT, the loser's IntegrityError resolves as idempotent replay.
    Never modifies the tasks row.  Returns the event row id, or None when
    this task+version was already classified or the invariant is blocked
    by legacy duplicates (fail closed; see ensure_suggestion_uniqueness).
    """
    version = suggestion["classifier_version"]
    prior = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
        (task_id, EVENT_KIND),
    ).fetchall()
    for p in prior:
        try:
            payload = json.loads(p[0] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if payload.get("classifier_version") == version:
            return None
    if not ensure_suggestion_uniqueness(conn):
        return None  # legacy duplicates: fail closed, migration plan required
    try:
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
    except sqlite3.IntegrityError:
        # R2-1: concurrent writer won the uniqueness race — idempotent replay.
        return None


def classification_disagreement(
    task_row: "sqlite3.Row | dict", suggestion: Optional[dict[str, Any]],
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