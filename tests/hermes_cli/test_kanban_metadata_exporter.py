"""Tests for the metadata-only Kanban exporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_metadata_exporter import metadata_export_payload, metadata_task_record


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_export_metadata_json_uses_v1_schema_and_safe_allowlist(kanban_home, tmp_path):
    workspace_path = tmp_path / "private-client-workspace"
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="private client title must not leak",
            body="secret body must not leak",
            assignee="alice",
            created_by="operator",
            workspace_kind="dir",
            workspace_path=str(workspace_path),
            tenant="tenant-a",
            priority=7,
        )
        kb.add_comment(conn, task_id, "reviewer", "comment must not leak")
        kb.complete_task(
            conn,
            task_id,
            result="result must not leak",
            summary="run summary must not leak",
            metadata={"raw": "worker output must not leak"},
        )

    payload = json.loads(kc.run_slash("export-metadata --json"))

    assert payload["schemaVersion"] == "kanban-metadata-export-v1"
    assert payload["sourceSystem"] == "Hermes Kanban"
    assert payload["lifecycleEventType"] == "snapshot"
    assert len(payload["tasks"]) == 1

    row = payload["tasks"][0]
    assert row == {
        "schemaVersion": "kanban-metadata-export-v1",
        "sourceSystem": "Hermes Kanban",
        "lifecycleEventType": "snapshot",
        "sourceTaskId": task_id,
        "taskStatus": "done",
        "assignee": "alice",
        "tenant": "tenant-a",
        "priority": 7,
        "createdBy": "operator",
        "createdAt": row["createdAt"],
        "startedAt": None,
        "completedAt": row["completedAt"],
        "workflowTemplateId": None,
        "currentStepKey": None,
        "taskSummary": None,
        "wendyFieldSafety": None,
    }

    serialized = json.dumps(payload, ensure_ascii=False)
    forbidden_fragments = [
        "private client title must not leak",
        "secret body must not leak",
        "comment must not leak",
        "result must not leak",
        "run summary must not leak",
        "worker output must not leak",
        str(workspace_path),
        "workspace_path",
        "body",
        "result",
        "comments",
        "runs",
        "events",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in serialized


def test_task_summary_only_exports_when_wendy_field_safety_is_safe_summary():
    unsafe = metadata_task_record(
        {
            "id": "t_unsafe",
            "status": "ready",
            "assignee": None,
            "tenant": None,
            "priority": 0,
            "created_by": None,
            "created_at": 1,
            "started_at": None,
            "completed_at": None,
            "workflow_template_id": None,
            "current_step_key": None,
            "wendyFieldSafety": "unsafe",
            "taskSummary": "must not leak",
        }
    )
    safe = metadata_task_record(
        {
            "id": "t_safe",
            "status": "ready",
            "assignee": None,
            "tenant": None,
            "priority": 0,
            "created_by": None,
            "created_at": 1,
            "started_at": None,
            "completed_at": None,
            "workflow_template_id": None,
            "current_step_key": None,
            "wendyFieldSafety": "safe_summary",
            "taskSummary": "safe summary",
        }
    )

    assert unsafe["taskSummary"] is None
    assert safe["taskSummary"] == "safe summary"


def test_metadata_export_payload_defaults_lifecycle_to_snapshot():
    payload = metadata_export_payload([])

    assert payload["schemaVersion"] == "kanban-metadata-export-v1"
    assert payload["sourceSystem"] == "Hermes Kanban"
    assert payload["lifecycleEventType"] == "snapshot"
    assert payload["tasks"] == []
