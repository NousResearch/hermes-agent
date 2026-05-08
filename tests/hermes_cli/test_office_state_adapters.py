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
    collect_topic_registry_office_state,
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
    sensitive_title = "Deploy " + "sk-" + "office-redaction-sentinel" + " from /home/alice/.hermes/.env"
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
            title=sensitive_title,
            body="body " + "sk-" + "body-redaction-sentinel" + " must not leak",
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

    assert "sk-" + "office-redaction-sentinel" not in child_item["title"]
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
                        "name": "Daily " + "sk-" + "cron-redaction-sentinel",
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
                        "last_error": "Script timed out: /home/alice/.hermes/scripts/private.py " + "*" * 3,
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
        {
            "kind": "explicit",
            "platform": "telegram",
            "has_chat": True,
            "has_thread": True,
            "topic_ref": "topic:telegram:hidden:ref-bfe1f52327",
            "chat_id_display": "hidden",
            "thread_id_display": "hidden",
            "display_name": "Telegram topic (derived)",
            "confidence": "derived",
        }
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


def test_topic_registry_adapter_reports_missing_without_creating_storage(isolated_kanban_home):
    registry_file = isolated_kanban_home / "office" / "topics.json"
    assert not registry_file.exists()

    result = collect_topic_registry_office_state()

    assert result.source.status == "missing"
    assert result.topics == []
    assert result.rooms == []
    assert not registry_file.exists(), "read-only adapter must not initialize topic registry storage"


def test_topic_registry_adapter_projects_safe_topic_records(isolated_kanban_home):
    registry_file = isolated_kanban_home / "office" / "topics.json"
    registry_file.parent.mkdir(parents=True)
    registry_file.write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "id": "topic:telegram:manual:11",
                        "platform": "telegram",
                        "chat_id_raw": "-1003775710032",
                        "chat_id_display": "Hermes Hub",
                        "thread_id_raw": 11,
                        "thread_id_display": "11",
                        "display_name": "Automation " + "sk-" + "topic-redaction-sentinel",
                        "purpose": "automation",
                        "source": "manual_alias",
                        "confidence": "manual",
                        "last_observed_at": "2026-05-08T10:00:00+00:00",
                        "notes": "raw note must not leak",
                    },
                    {
                        "id": "telegram:-1003775710032:raw",
                        "platform": "telegram",
                        "chat_id_raw": "-1003775710032",
                        "thread_id": "777",
                        "display_name": "Raw-only topic",
                        "purpose": "automation",
                        "source": "manual_alias",
                        "confidence": "manual",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = collect_topic_registry_office_state()
    serialized = json.dumps(result.to_payload(), ensure_ascii=False)

    assert result.source.status == "ok"
    assert result.source.item_count == 2
    assert len(result.topics) == 2
    assert len(result.rooms) == 2
    topic = result.topics[0]
    assert topic["platform"] == "telegram"
    assert topic["display_name"] != "Automation sk-topic-redaction-sentinel"
    assert topic["purpose"] == "automation"
    assert topic["source"] == "manual_alias"
    assert topic["confidence"] == "manual"
    assert topic["thread_id_display"] == "hidden"
    room = result.rooms[0]
    assert room["kind"] == "telegram_topic"
    assert room["topic_id"] == topic["id"]
    raw_only_topic = result.topics[1]
    assert raw_only_topic["id"] == "topic:telegram:hidden:ref-a978f916ec"
    assert raw_only_topic["chat_id_display"] == "hidden"
    assert raw_only_topic["thread_id_display"] == "hidden"

    for needle in ["-1003775710032", "chat_id_raw", "thread_id_raw", "raw note must not leak", "sk-topic-redaction-sentinel", "777", "topic:telegram:manual:11"]:
        assert needle not in serialized


def test_cron_adapter_projects_delivery_topics_and_provenance_without_chat_ids(isolated_kanban_home):
    jobs_file = isolated_kanban_home / "cron" / "jobs.json"
    jobs_file.parent.mkdir(parents=True)
    jobs_file.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "job_topic",
                        "schedule": {"kind": "cron", "display": "0 7 * * *"},
                        "enabled": True,
                        "state": "scheduled",
                        "deliver": "telegram:-1003775710032:11",
                        "next_run_at": "2026-05-09T07:00:00+09:00",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = collect_cron_office_state()
    serialized = json.dumps(result.to_payload(), ensure_ascii=False)

    assert result.source.status == "ok"
    assert result.topics == [
        {
            "id": "topic:telegram:hidden:ref-bfe1f52327",
            "platform": "telegram",
            "display_name": "Telegram topic (derived)",
            "purpose": "unknown",
            "source": "cron_delivery",
            "confidence": "derived",
            "chat_id_display": "hidden",
            "thread_id_display": "hidden",
        }
    ]
    assert result.provenance == [
        {
            "id": "prov:cron:job_topic:delivered_to:telegram:hidden:ref-bfe1f52327",
            "subject_kind": "cron_job",
            "subject_id": "cron:job_topic",
            "relation": "delivered_to",
            "source": "cron_delivery",
            "target_ref": "topic:telegram:hidden:ref-bfe1f52327",
            "confidence": "derived",
        }
    ]
    assert result.automations[0]["delivery_targets"][0]["topic_ref"] == "topic:telegram:hidden:ref-bfe1f52327"
    assert "-1003775710032" not in serialized


def test_build_office_state_merges_topic_registry_and_cron_provenance(isolated_kanban_home):
    registry_file = isolated_kanban_home / "office" / "topics.json"
    registry_file.parent.mkdir(parents=True)
    registry_file.write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "id": "topic:telegram:manual:40",
                        "platform": "telegram",
                        "chat_id_display": "Hermes Hub",
                        "thread_id_display": "40",
                        "display_name": "Operations",
                        "purpose": "operations",
                        "source": "manual_alias",
                        "confidence": "manual",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    jobs_file = isolated_kanban_home / "cron" / "jobs.json"
    jobs_file.parent.mkdir(parents=True)
    jobs_file.write_text(
        json.dumps({"jobs": [{"id": "job_topic", "state": "scheduled", "deliver": "telegram:-1003775710032:11"}]}),
        encoding="utf-8",
    )

    state = build_office_state(include_kanban=False, include_sessions=False)
    payload = state.to_dict()

    assert _source(payload, "topics")["status"] == "ok"
    assert _source(payload, "provenance")["status"] == "ok"
    assert len(payload["topics"]) == 2
    assert len(payload["provenance"]) == 1
    assert "Operations" in {topic["display_name"] for topic in payload["topics"]}


def test_build_office_state_marks_derived_topics_partial_when_registry_missing(isolated_kanban_home):
    jobs_file = isolated_kanban_home / "cron" / "jobs.json"
    jobs_file.parent.mkdir(parents=True)
    jobs_file.write_text(
        json.dumps({"jobs": [{"id": "job_topic", "state": "scheduled", "deliver": "telegram:-1003775710032:11"}]}),
        encoding="utf-8",
    )

    state = build_office_state(include_kanban=False, include_sessions=False)
    payload = state.to_dict()

    assert _source(payload, "topics")["status"] == "partial"
    assert _source(payload, "topics")["item_count"] == 1
    assert _source(payload, "topics")["warning_count"] == 1
    assert payload["topics"][0]["source"] == "cron_delivery"
    assert _source(payload, "provenance")["status"] == "ok"


def test_session_adapter_projects_metadata_without_transcripts(isolated_kanban_home):
    db = SessionDB(isolated_kanban_home / "state.db")
    db.create_session(
        "sess_sensitive_full_id",
        "telegram",
        user_id="123456",
        model="gpt-test",
    )
    db.set_session_title("sess_sensitive_full_id", "Title " + "sk-" + "session-redaction-sentinel")
    db.append_message("sess_sensitive_full_id", "user", "raw prompt " + "*" * 3 + " must not leak")
    db.append_message(
        "sess_sensitive_full_id",
        "assistant",
        "assistant content must not leak",
        tool_calls='[{"args":"tool sensitive marker must not leak"}]',
        reasoning="reasoning must not leak",
    )
    db.end_session("sess_sensitive_full_id", "completed")
    db.close()

    result = collect_session_office_state()
    serialized = json.dumps(result.to_payload(), ensure_ascii=False)

    assert result.source.status == "ok"
    assert result.source.item_count == 1
    agent = result.agents[0]
    assert agent["kind"] == "session_actor"
    assert agent["source_platform"] == "telegram"
    assert agent["session_id_prefix"] == "sess_sen"
    assert agent["title"] is None
    assert agent["title_policy"] == "hidden_by_default"
    assert agent["message_count"] == 2
    assert agent["model"] == "gpt-test"
    assert agent["status"] == "ended"

    for needle in [
        "raw prompt",
        "assistant content",
        "tool sensitive",
        "reasoning must not leak",
        "sk-" + "session-redaction-sentinel",
        "123456",
        "sess_sensitive_full_id",
    ]:
        assert needle not in serialized
