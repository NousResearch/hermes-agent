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

CLASSIFIER_VERSION = "shadow-classifier-2"
EVENT_KIND = "risk_classification_suggested"

# C5: trusted human creation seams.  Anything outside this set (or not
# registry-verified as an interactive profile author) is NOT eligible.
TRUSTED_HUMAN_CREATED_BY = {"dashboard", "ideabox", "idea-box", "cli"}

# Automation/derived-origin stamps that must never classify as human,
# even if they later appear in a registry by mistake.
_AUTOMATION_STAMPS = {
    "feature-pipeline", "feature-pipeline", "denji-governance", "system",
    "swarm", "cron", "webhook", "job", "automation", "scheduler",
    "worker", "pipeline", "governance", "decompose", "triage-router",
    "orchestrator", "market-scanner", "skill-broker", "skill-research",
}

_BOUNDARY_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_\-.:]{0,63}$")


def _bounded(token: str) -> str:
    if not _BOUNDARY_TOKEN_RE.match(token):
        raise ValueError(f"unbounded token rejected: {token!r}")
    return token


def _registry_profile_names(hermes_home: Optional[Path] = None) -> Optional[set[str]]:
    """Registry-verified profile names, or None when the registry is
    absent/unreadable (provenance then cannot be verified → fail closed).
    """
    home = hermes_home or Path(
        __import__("os").environ.get("HERMES_HOME", "") or
        Path.home() / ".hermes"
    )
    path = home / "governance" / "profile-registry.yaml"
    if not path.exists():
        return None
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if raw.get("schema_version") != 1 or not isinstance(raw.get("profiles"), list):
        return None
    return {
        e["name"] for e in raw["profiles"]
        if isinstance(e, dict) and isinstance(e.get("name"), str)
    }


def is_human_created(
    task_row: "sqlite3.Row | dict",
    hermes_home: Optional[Path] = None,
) -> bool:
    """Prove human creation ONLY from structured seams (fail closed).

    Eligible origins:
      * trusted creation seams: dashboard / idea-box / cli;
      * an interactive profile-author stamp that is VERIFIED present in
        the deployed profile registry.

    Everything else — unknown tokens, automation stamps, missing or
    malformed identity — is NOT human.  Token shape is never evidence.
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
    if created_by in _AUTOMATION_STAMPS or created_by in _AUTOMATION_STAMPS:
        return False
    if created_by in TRUSTED_HUMAN_CREATED_BY:
        return True
    # Profile-author stamps: only registry-verified names qualify.
    names = _registry_profile_names(hermes_home)
    if names is None:
        return False  # no registry → cannot verify → fail closed
    return created_by in names


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
    if not is_human_created(row, hermes_home=hermes_home):
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

    kind = row["task_kind"] or "task"
    priority = int(row["priority"] or 0)
    budget = row["max_runtime_seconds"]

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

    Never modifies the tasks row.  Returns the event row id, or None when
    this task+version was already classified (searches ALL prior events).
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