"""Metadata-only Kanban export surface for external source integrations.

This module intentionally uses an explicit column allowlist and emits only
coordination metadata. It must not read or serialize task bodies, comments,
results, run output, logs, workspace paths, or other raw worker/client data.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from hermes_cli import kanban_db as kb

SCHEMA_VERSION = "kanban-metadata-export-v1"
SOURCE_SYSTEM = "Hermes Kanban"
DEFAULT_LIFECYCLE_EVENT_TYPE = "snapshot"
SAFE_SUMMARY_SENTINEL = "safe_summary"

# Keep this SQL as the only database source for the exporter. It is deliberately
# narrower than kb.list_tasks(), which hydrates full Task objects including body,
# result, workspace_path, and other fields forbidden in metadata export output.
_METADATA_TASK_SQL = """
SELECT
    id,
    status,
    assignee,
    tenant,
    priority,
    created_by,
    created_at,
    started_at,
    completed_at,
    workflow_template_id,
    current_step_key
FROM tasks
WHERE 1=1
"""


def _json_safe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def metadata_task_record(
    source: Mapping[str, Any],
    *,
    lifecycle_event_type: str = DEFAULT_LIFECYCLE_EVENT_TYPE,
) -> dict[str, Any]:
    """Map one allowlisted Kanban metadata row to the export contract.

    ``taskSummary`` is only carried when the caller has explicitly marked the
    field with ``wendyFieldSafety == "safe_summary"``. The live Kanban DB query
    does not select any summary/title/body field, so production exports default
    this to ``None``.
    """

    wendy_field_safety = source.get("wendyFieldSafety")
    task_summary = None
    if wendy_field_safety == SAFE_SUMMARY_SENTINEL:
        raw_summary = source.get("taskSummary")
        task_summary = str(raw_summary) if raw_summary is not None else None

    return {
        "schemaVersion": SCHEMA_VERSION,
        "sourceSystem": SOURCE_SYSTEM,
        "lifecycleEventType": lifecycle_event_type or DEFAULT_LIFECYCLE_EVENT_TYPE,
        "sourceTaskId": source["id"],
        "taskStatus": source["status"],
        "assignee": source.get("assignee"),
        "tenant": source.get("tenant"),
        "priority": int(source.get("priority") or 0),
        "createdBy": source.get("created_by"),
        "createdAt": _json_safe_int(source.get("created_at")),
        "startedAt": _json_safe_int(source.get("started_at")),
        "completedAt": _json_safe_int(source.get("completed_at")),
        "workflowTemplateId": source.get("workflow_template_id"),
        "currentStepKey": source.get("current_step_key"),
        "taskSummary": task_summary,
        "wendyFieldSafety": wendy_field_safety,
    }


def metadata_export_payload(
    tasks: list[dict[str, Any]],
    *,
    lifecycle_event_type: str = DEFAULT_LIFECYCLE_EVENT_TYPE,
    exported_at: int | None = None,
) -> dict[str, Any]:
    """Build the metadata-only exporter envelope."""

    return {
        "schemaVersion": SCHEMA_VERSION,
        "sourceSystem": SOURCE_SYSTEM,
        "lifecycleEventType": lifecycle_event_type or DEFAULT_LIFECYCLE_EVENT_TYPE,
        "exportedAt": int(time.time()) if exported_at is None else int(exported_at),
        "tasks": tasks,
    }


def export_metadata(
    *,
    assignee: str | None = None,
    status: str | None = None,
    tenant: str | None = None,
    include_archived: bool = False,
    lifecycle_event_type: str = DEFAULT_LIFECYCLE_EVENT_TYPE,
) -> dict[str, Any]:
    """Read and return a metadata-only Kanban export payload.

    The query is read-only and uses an explicit allowlist of safe columns. Do
    not replace it with kb.list_tasks() or any show/runs/context/log source.
    """

    query = _METADATA_TASK_SQL
    params: list[Any] = []
    if assignee is not None:
        query += " AND assignee = ?"
        params.append(kb._canonical_assignee(assignee))
    if status is not None:
        if status not in kb.VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(kb.VALID_STATUSES)}")
        query += " AND status = ?"
        params.append(status)
    if tenant is not None:
        query += " AND tenant = ?"
        params.append(tenant)
    if not include_archived and status != "archived":
        query += " AND status != 'archived'"
    query += " ORDER BY priority DESC, created_at ASC, id ASC"

    with kb.connect_closing() as conn:
        rows = conn.execute(query, params).fetchall()

    tasks = [metadata_task_record(dict(row), lifecycle_event_type=lifecycle_event_type) for row in rows]
    return metadata_export_payload(tasks, lifecycle_event_type=lifecycle_event_type)
