"""Event-driven lead shared-memory injection (R5 / F-04).

Locked decision 2 (2026-08-08): per-task completion for leads is
EVENT-DRIVEN, wired at the kanban dispatcher; the nightly batch
(scripts/shared_memory_inject.py) covers workers. This module is the
lead seam: when a kanban task completes, read the team's shared surface,
distil the highest-salience points, and inject them into the LEAD's scope
as observations tagged source.event=shared_injection.

Contract (shared with the nightly script):
- Same store contract (bundle caller supplies it).
- Same provenance tag source.event=shared_injection.
- Idempotent per (team, task_id) — a repeated completion event never
  duplicates an injection.
- Scope-safe: unknown teams are a no-op; injection only ever writes into
  the lead's own scope for a configured team.
- Best-effort: never raises into the kanban watcher.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Team registry: shared profile_id -> {lead agent, participant agents}.
# Mirrors the nightly script's TEAMS map (must stay IDENTICAL — see
# governance/team-definitions-draft-20260808.md). Approved by Sahil
# 2026-08-08; all 9 former Tier-3 profiles activated and teemed.
TEAMS: dict[str, dict[str, Any]] = {
    "team-content": {
        "lead": "ceecee",
        "participants": ["ceecee-writer", "ceecee-social", "ceecee-brand",
                         "ceecee-reviewer", "content-strategist", "ceecee-seo"],
    },
    "team-research": {
        "lead": "remii",
        "participants": ["remii-digest", "remii-gitradar", "remii-market",
                         "market-scanner", "remii-deep"],
    },
    "team-build": {
        "lead": "octacon",
        "participants": ["octacon-backend", "octacon-infra", "octacon-testrunner",
                         "moss", "octacon-mobile", "octacon-frontend",
                         "octacon-architect", "octacon-techwriter"],
    },
    "team-ops": {
        "lead": "wesker",
        "participants": ["wesker-ops", "wesker-scanner", "wesker-backup"],
    },
    "team-qa": {
        "lead": "quan",
        "participants": ["quan-arch", "quan-code", "quan-perf", "quan-security",
                         "quan-ux", "quan-e2e"],
    },
    "team-design": {
        "lead": "dezzy",
        "participants": ["dezzy-brand", "dezzy-component-lib",
                         "dezzy-design-system", "dezzy-ux-prototype",
                         "dezzy-ux-architect", "dezzy-image-prompt"],
    },
    "team-knowledge": {
        "lead": "light",
        "participants": ["light-indexer", "light-wiki", "light-archivist"],
    },
    "team-governance": {
        "lead": "denji",
        "participants": ["denji-ledger", "denji-reviewer", "denji-skill",
                         "skill-broker", "skill-research", "denji-monitor"],
    },
    "team-admin": {
        "lead": "gojo",
        "participants": ["gojo-admin", "gojo-calendar", "gojo-mailbox"],
    },
    "team-orchestration": {
        "lead": "orchestrator",
        "participants": ["triage-router"],
    },
}

QUERY_TERMS = ["decision", "status", "conclusion", "learned", "blocker", "owner"]
MAX_INJECT_PER_TASK = 5


def _already_injected(bundle: Any, scope: Any, task_id: str) -> bool:
    """True when the lead scope already holds a shared_injection for task_id."""
    for obs in bundle.repository.list_observations(scope):
        src = dict(getattr(obs, "source", {}) or {})
        if src.get("event") == "shared_injection" and src.get("task_id") == task_id:
            return True
    return False


def inject_on_task_completion(
    *,
    bundle: Any,
    task_id: str,
    title: str,
    board: str,
    team: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Inject the lead's distilled shared-surface points on task completion.

    Idempotent: a second call for the same (team, task_id) is a no-op.
    Unknown team: no-op. Never raises.
    """
    from severian.domain.models import Scope

    team_cfg = TEAMS.get(team)
    if not team_cfg:
        logger.debug("shared injection: unknown team %r — no-op", team)
        return {"injected": 0, "skipped": "unknown_team"}

    lead = team_cfg["lead"]
    lead_scope = Scope(
        tenant_id="kensei",
        profile_id=team,
        collection_id="team",
        agent_id=lead,
        session_id="shared",
    )

    if _already_injected(bundle, lead_scope, task_id):
        logger.info("shared injection: task %s already injected for lead %s", task_id, lead)
        return {"injected": 0, "skipped": "already_injected"}

    # Distil the highest-salience points from the team's shared surface.
    team_scope = Scope(
        tenant_id="kensei",
        profile_id=team,
        collection_id="team",
        agent_id="team-lead",
        session_id="shared",
    )
    points: list[str] = []
    for term in QUERY_TERMS:
        hits = bundle.service.search(
            team_scope, query=term, limit=3, derive=False,
            salience_weight=0.5, salience_halflife_h=720.0,
        )
        for hit in hits:
            if hit.content not in points:
                points.append(hit.content)
        if len(points) >= MAX_INJECT_PER_TASK:
            break

    if not points:
        # Nothing to distil yet — still record the completion fact so the
        # event is visible in the ledger and future runs can extend it.
        points = [f"[task:{task_id}] {title}"]

    stamp = (now or datetime.now(timezone.utc)).isoformat()
    source: Mapping[str, Any] = {
        "provider": "severian",
        "event": "shared_injection",
        "task_id": task_id,
        "board": board,
        "team": team,
        "injected_at": stamp,
        "note": "distilled from team shared surface on task completion (R5)",
    }
    try:
        for point in points:
            bundle.service.ingest_observation(
                lead_scope,
                content=point,
                source={**source, "point_index": len(points) > 1},
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("shared injection failed for task %s: %s", task_id, exc)
        return {"injected": 0, "skipped": "error", "error": str(exc)}

    logger.info(
        "shared injection: %d point(s) -> lead %s for task %s",
        len(points), lead, task_id,
    )
    return {"injected": len(points)}
