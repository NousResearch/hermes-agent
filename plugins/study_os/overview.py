"""Read-only StudyOS projection for the learner's daily control surface."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from plugins.study_os.interventions import InterventionOrchestrator, parse_as_of
from plugins.study_os.learning import (
    _all_attempts,
    _assistance_level,
    _diagnosis,
    _evaluator_kind,
    _independently_verified,
)
from plugins.study_os.reviews import StudyReviewReadModel
from plugins.study_os.schemas import validate_plan_proposal
from plugins.study_os.workspace import StudyWorkspace


def _project_clock(project: dict[str, Any], value: Any = None) -> datetime:
    resolved = parse_as_of(value)
    try:
        return resolved.astimezone(ZoneInfo(str(project["timezone"])))
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"project timezone is not a valid IANA timezone: {project['timezone']}"
        ) from exc


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        resolved = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        return None
    return resolved


def _today_events(
    workspace: StudyWorkspace,
    project: dict[str, Any],
    clock: datetime,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    catalog = workspace.discover_schedules(str(project["project_id"]))
    for artifact in catalog.schedules:
        schedule = artifact.schedule
        for event in schedule.get("events", []):
            start = _parse_datetime(event.get("start"))
            if (
                start is None
                or start.astimezone(clock.tzinfo).date() != clock.date()
            ):
                continue
            events.append(
                {
                    **event,
                    "schedule_id": schedule["schedule_id"],
                    "schedule_title": schedule["title"],
                }
            )
    events.sort(key=lambda item: (str(item.get("start")), str(item.get("id"))))
    return events


def _due_reviews(vault: Path, clock: datetime, limit: int) -> dict[str, Any]:
    projection = StudyReviewReadModel(vault).due(
        as_of=clock.date(),
        limit=limit,
    )
    return {
        # Review notes predate Learning Projects and currently have no reliable
        # project ownership field, so this projection states its Vault scope
        # instead of pretending the active project filtered them.
        "scope": "vault",
        "count": projection["count"],
        "subjects": projection["subjects"],
        "items": projection["due"],
    }


def _attempts_today(
    attempts: list[dict[str, Any]],
    clock: datetime,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for attempt in attempts:
        occurred = _parse_datetime(attempt.get("occurred_at"))
        if occurred is not None and occurred.astimezone(clock.tzinfo).date() == clock.date():
            result.append(attempt)
    return result


def _evidence_projection(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    diagnosis = _diagnosis(attempts)
    occurred = [
        value
        for item in attempts
        if (value := _parse_datetime(item.get("occurred_at"))) is not None
    ]
    return {
        "attempt_count": len(attempts),
        "independently_verified_count": sum(
            1 for attempt in attempts if _independently_verified(attempt)
        ),
        "latest_evidence_at": (
            max(occurred).isoformat(timespec="seconds") if occurred else None
        ),
        "dimensions": diagnosis["evidence_dimensions"],
        "evaluator_provenance": dict(
            sorted(Counter(_evaluator_kind(item) for item in attempts).items())
        ),
        "assistance_provenance": dict(
            sorted(Counter(_assistance_level(item) for item in attempts).items())
        ),
    }


def _pending_plan_proposals(
    workspace: StudyWorkspace,
    project_id: str,
) -> list[dict[str, Any]]:
    root = workspace.projects_root / project_id / "plan-proposals"
    if not root.exists():
        return []
    proposals: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            ok, validated = validate_plan_proposal(raw)
            if ok and isinstance(validated, dict) and validated.get("status") == "proposed":
                proposals.append(validated)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    proposals.sort(key=lambda item: (str(item.get("created_at")), str(item.get("proposal_id"))))
    return proposals


def build_study_overview(
    workspace: StudyWorkspace,
    *,
    project_id: str | None = None,
    as_of: str | None = None,
    review_limit: int = 10,
    intervention_limit: int = 5,
) -> dict[str, Any]:
    """Build one bounded, side-effect-free projection for today's StudyOS UI."""

    project = workspace.project(project_id)
    clock = _project_clock(project, as_of)
    attempts = _all_attempts(workspace.vault, project["project_id"])
    today_attempts = _attempts_today(attempts, clock)
    orchestration = InterventionOrchestrator(
        project=project,
        diagnosis_builder=_diagnosis,
    ).build(
        attempts=attempts,
        as_of=clock,
        max_items=max(1, min(intervention_limit, 20)),
    )
    return {
        "configured": True,
        "vault_path": str(workspace.vault),
        "active_project_id": workspace.active_project_id(),
        "project": project,
        "as_of": clock.isoformat(timespec="seconds"),
        "today": clock.date().isoformat(),
        "today_events": _today_events(workspace, project, clock),
        "due_reviews": _due_reviews(
            workspace.vault,
            clock,
            max(1, min(review_limit, 100)),
        ),
        "completed_today": sum(
            1 for attempt in today_attempts if attempt.get("activity_kind") == "review"
        ),
        "activity_today": len(today_attempts),
        "evidence": _evidence_projection(attempts),
        "intervention_queue": orchestration["queue"],
        "pending_plan_proposals": _pending_plan_proposals(
            workspace,
            project["project_id"],
        ),
    }
