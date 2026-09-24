"""Designated task-event publication; never reads task prose or transcripts."""
from __future__ import annotations

import json

from gateway.work_presentation import (
    TrustedWorkAudience, audience_dict, audience_from_dict,
    validate_presentation, validate_steps,
)


def _service():
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    service = getattr(manager, "_work_presentation_registration", None)
    return service if service is not None and service.active else None


def latest_task_audience(conn, task_id: str) -> TrustedWorkAudience | None:
    row = conn.execute(
        "SELECT run_id, payload FROM task_events WHERE task_id=? "
        "AND json_extract(payload, '$.publication.version')=1 ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"] if hasattr(row, "keys") else row[1])
        publication = payload["publication"]
        row_run = row["run_id"] if hasattr(row, "keys") else row[0]
        if publication.get("version") != 1 or publication.get("run_id") != row_run:
            return None
        return audience_from_dict(publication["audience"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def project_task_publication(conn, task_id: str, current_revision: int) -> dict | None:
    row = conn.execute(
        "SELECT id, run_id, payload, created_at FROM task_events WHERE task_id=? "
        "AND json_extract(payload, '$.publication.version')=1 ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"])
        publication = payload["publication"]
        if publication.get("version") != 1 or publication.get("run_id") != row["run_id"]:
            raise ValueError
        return {
            "audience": audience_from_dict(publication["audience"]),
            "presentation": validate_presentation(publication["presentation"]),
            "steps": validate_steps(publication.get("steps", [])),
            "published_at": row["created_at"],
            "publication_stale": row["id"] != current_revision,
        }
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise ValueError("invalid task publication") from None
def _build(audience, presentation, steps, run_id):
    service = _service()
    if service is None or not service.accepts_audience(audience):
        raise ValueError("work publication is unavailable for this route")
    return {
        "version": 1, "audience": audience_dict(audience),
        "presentation": validate_presentation(presentation),
        "steps": validate_steps(steps), "run_id": run_id,
    }


def prepare_create(conn, *, presentation, steps=(), creator_task_id=None, creator_run_id=None):
    if presentation is None:
        if steps:
            raise ValueError("steps require a presentation")
        return None
    if creator_task_id:
        row = conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (creator_task_id,)).fetchone()
        current = row[0] if row is not None else None
        if creator_run_id is None or current != int(creator_run_id):
            raise ValueError("task publication requires the creator's current run")
        audience = latest_task_audience(conn, creator_task_id)
    else:
        from gateway.session_context import current_work_audience
        audience = current_work_audience()
    if audience is None:
        raise ValueError("task publication requires a trusted Telegram audience")
    return _build(audience, presentation, steps, None)


def prepare_update(conn, task_id, *, expected_run_id, presentation, steps=()):
    if presentation is None:
        if steps:
            raise ValueError("steps require a presentation")
        return None
    if expected_run_id is None:
        raise ValueError("task progress publication requires an exact worker run")
    row = conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (task_id,)).fetchone()
    if row is None or row[0] != int(expected_run_id):
        raise ValueError("task progress publication requires the current worker run")
    audience = latest_task_audience(conn, task_id)
    if audience is None:
        raise ValueError("task has no trusted publication audience")
    return _build(audience, presentation, steps, int(expected_run_id))
