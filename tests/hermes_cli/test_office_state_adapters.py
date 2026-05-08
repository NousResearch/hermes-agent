"""Tests for AI Office read-only source adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.office_adapters import (
    collect_cron_office_state,
    collect_kanban_office_state,
    collect_session_office_state,
)
from hermes_cli.office_state import build_office_state
from hermes_state import SessionDB


@pytest.fixture
def isolated_kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _source(payload: dict, source_id: str) -> dict:
    return next(source for source in payload["data_sources"] if source["id"] == source_id)


def test_kanban_adapter_reports_missing_without_creating_a_db(isolated_kanban_home):
    db_path = isolated_kanban_home / "kanban.db"
    assert not db_path.exists()

    result = collect_kanban_office_state()

    assert result.source.status == "missing"
    assert result.rooms == []
    assert result.work_items == []
    assert result.events == []
    assert not db_path.exists(), "read-only adapter must not initialize Kanban storage"


def test_kanban_adapter_projects_safe_room_task_and_event_fields(isolated_kanban_home):
    kb.init_db()
    secret_title = "Deploy sk-office-redaction-sentinel from /home/alice/.hermes/.env"
    with kb.connect() as conn:
        parent = kb.create_task(
            conn,
            title="Parent task",
            body="parent body must not leak",
            assignee="planner",
            priority=2,
        )
        kb.complete_task(conn, parent, result="parent result must not leak")
        child = kb.create_task(
            conn,
            title=secret_title,
            body="body sk-body-redaction-sentinel must not leak",
            assignee="worker",
            created_by="telegram",
            priority=5,
            parents=[parent],
            workspace_path="/home/alice/private/repo",
        )
        assert kb.block_task(conn, child, reason="blocked reason should not leak as body/result")

    result = collect_kanban_office_state()
    payload = result.to_payload()
    serialized = json.dumps(payload, ensure_ascii=False)

    assert result.source.status == "ok"
    assert result.source.item_count == 2
    assert len(result.rooms) == 1
    assert result.rooms[0]["kind"] == "kanban_board"
    assert result.rooms[0]["counts"]["blocked"] == 1

    child_item = next(item for item in result.work_items if item["source_id"] == child)
    assert child_item["kind"] == "kanban_task"
    assert child_item["status"] == "blocked"
    assert child_item["priority"] == 5
    assert child_item["assignee"] == "worker"
    assert child_item["dependency_counts"] == {"parents": 1, "children": 0}
    assert child_item["provenance"] == {"status": "unknown", "missing_reason": "kanban_task_has_no_source_columns"}

    assert "sk-office-redaction-sentinel" not in child_item["title"]
    assert "/home/alice" not in child_item["title"]
    assert result.redactions.redacted_field_count >= 1

    assert result.events
    assert all("payload" not in event for event in result.events)
    assert all(event["kind"] for event in result.events)

    forbidden = [
        "body sk-bodySECRET",
        "parent body must not leak",
        "parent result must not leak",
        "blocked reason should not leak",
        "workspace_path",
        "latest_summary",
        "result",
        "/home/alice/private/repo",
    ]
    for needle in forbidden:
        assert needle not in serialized


def test_build_office_state_merges_kanban_status_and_summary(isolated_kanban_home):
    kb.init_db()
    with kb.connect() as conn:
        ready = kb.create_task(conn, title="Ready work", assignee="builder")
        blocked = kb.create_task(conn, title="Blocked work", assignee="builder")
        kb.block_task(conn, blocked, reason="needs decision")
        assert ready

    state = build_office_state(include_kanban=True)
    payload = state.to_dict()

    kanban_source = _source(payload, "kanban")
    assert kanban_source["status"] == "ok"
    assert kanban_source["item_count"] == 2
    assert payload["summary"]["active_work_count"] == 1
    assert payload["summary"]["needs_attention_count"] == 1
    assert payload["summary"]["warning_count"] == 0
    assert len(payload["rooms"]) == 1
    assert len(payload["work_items"]) == 2


def test_kanban_adapter_converts_board_read_failure_to_source_error(isolated_kanban_home):
    # A directory at the legacy DB path makes sqlite.connect fail without creating
    # or mutating the source, which exercises the adapter failure path.
    (isolated_kanban_home / "kanban.db").mkdir()

    state = build_office_state(include_kanban=True)
    payload = state.to_dict()

    kanban_source = _source(payload, "kanban")
    assert kanban_source["status"] == "error"
    assert kanban_source["item_count"] == 0
    assert "kanban.db" not in kanban_source.get("error_summary", "")
    assert payload["work_items"] == []


def test_cron_adapter_reports_missing_without_creating_storage(isolated_kanban_home):
    jobs_file = isolated_kanban_home / "cron" / "jobs.json"
    assert not jobs_file.exists()

    result = collect_cron_office_state()

    assert result.source.status == "missing"
    assert result.automations == []
    assert not jobs_file.exists(), "read-only adapter must not initialize cron storage"


def test_cron_adapter_projects_safe_automation_fields(isolated_kanban_home):
    jobs_file = isolated_kanban_home / "cron" / "jobs.json"
    output_dir = isolated_kanban_home / "cron" / "output" / "job_secret"
    output_dir.mkdir(parents=True)
    (output_dir / "2026-05-08_01-00-00.md").write_text("raw output must not leak", encoding="utf-8")
    jobs_file.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "job_secret",
                        "name": "Daily sk-cron-redaction-sentinel",
                        "prompt": "prompt must not leak",
                        "script": "/home/alice/.hermes/scripts/private.py",
                        "context_from": ["upstream_secret"],
                        "schedule": {"kind": "cron", "display": "0 8 * * *", "expr": "0 8 * * *"},
                        "schedule_display": "0 8 * * *",
                        "enabled": True,
                        "state": "scheduled",
                        "deliver": "telegram:-1003775710032:11",
                        "last_run_at": "2026-05-08T09:00:00+09:00",
                        "next_run_at": "2026-05-09T08:00:00+09:00",
                        "last_status": "error",
                        "last_error": "Script timed out: /home/alice/.hermes/scripts/private.py sk-errorSECRET123",
                        "last_delivery_error": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = collect_cron_office_state()
    serialized = json.dumps(result.to_payload(), ensure_ascii=False)

    assert result.source.status == "ok"
    assert result.source.item_count == 1
    automation = result.automations[0]
    assert automation["kind"] == "cron_job"
    assert automation["source_id"] == "job_secret"
    assert automation["enabled"] is True
    assert automation["state"] == "scheduled"
    assert automation["last_status"] == "error"
    assert automation["schedule"] == {"kind": "cron", "display": "0 8 * * *"}
    assert automation["output_artifact_count"] == 1
    assert automation["delivery_targets"] == [
        {"kind": "explicit", "platform": "telegram", "has_chat": True, "has_thread": True}
    ]
    assert automation["name"] == "Cron job job_secr"
    assert automation["last_error_summary"] == "last_error_recorded"

    forbidden = ["prompt must not leak", "script", "context_from", "private.py", "raw output must not leak", "-1003775710032"]
    for needle in forbidden:
        assert needle not in serialized


def test_session_adapter_reports_missing_without_creating_state_db(isolated_kanban_home):
    state_db = isolated_kanban_home / "state.db"
    assert not state_db.exists()

    result = collect_session_office_state()

    assert result.source.status == "missing"
    assert result.agents == []
    assert not state_db.exists(), "read-only adapter must not initialize session storage"


def test_session_adapter_projects_metadata_without_transcripts(isolated_kanban_home):
    db = SessionDB(isolated_kanban_home / "state.db")
    db.create_session(
        "sess_secret_full_id",
        "telegram",
        user_id="123456",
        model="gpt-test",
    )
    db.set_session_title("sess_secret_full_id", "Title sk-session-redaction-sentinel")
    db.append_message("sess_secret_full_id", "user", "raw prompt sk-messageSECRET must not leak")
    db.append_message(
        "sess_secret_full_id",
        "assistant",
        "assistant content must not leak",
        tool_calls='[{"args":"tool secret must not leak"}]',
        reasoning="reasoning must not leak",
    )
    db.end_session("sess_secret_full_id", "completed")
    db.close()

    result = collect_session_office_state()
    serialized = json.dumps(result.to_payload(), ensure_ascii=False)

    assert result.source.status == "ok"
    assert result.source.item_count == 1
    agent = result.agents[0]
    assert agent["kind"] == "session_actor"
    assert agent["source_platform"] == "telegram"
    assert agent["session_id_prefix"] == "sess_sec"
    assert agent["title"] is None
    assert agent["title_policy"] == "hidden_by_default"
    assert agent["message_count"] == 2
    assert agent["model"] == "gpt-test"
    assert agent["status"] == "ended"

    for needle in [
        "raw prompt",
        "assistant content",
        "tool secret",
        "reasoning must not leak",
        "sk-sessionSECRET",
        "123456",
        "sess_secret_full_id",
    ]:
        assert needle not in serialized
