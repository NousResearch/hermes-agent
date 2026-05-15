"""Tests for the focused task registry substrate."""

import copy
import json

import pytest

from agent.pending_turn_queue import PendingTurnItem, KIND_TEXT, KIND_MEDIA, BOUNDARY_HARD
from agent.task_registry import (
    FRONTDESK_BLOCKED_USER_INPUT,
    FRONTDESK_REVIEW_PASSED,
    FRONTDESK_RUNNING_WORKER,
    FRONTDESK_WORKER_DONE_PENDING_REVIEW,
    REVIEW_ACTION_ASK_USER,
    REVIEW_ACTION_PRESENT,
    REVIEW_BLOCKED,
    REVIEW_PASSED,
    STAGE_DONE,
    STAGE_NOT_STARTED,
    STAGE_QUEUED,
    STAGE_RUNNING,
    STATUS_BLOCKED,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_PROPOSED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    WORKER_CLAUDE_CODE,
    ReviewResultArtifact,
    TaskOrigin,
    TaskRegistry,
)


class UncopyableRaw:
    def __deepcopy__(self, memo):  # pragma: no cover - should never be called
        raise AssertionError("raw should not be deep-copied")


def test_create_list_and_session_filtering():
    reg = TaskRegistry()
    origin = TaskOrigin(platform="telegram", chat_id="chat-1", user_id="woo", session_key="s1")

    t1 = reg.create_task("draft briefing", origin=origin)
    t2 = reg.create_task("review code", session_key="s2", status=STATUS_QUEUED)

    assert len(reg) == 2
    assert t1.task_id in reg
    assert t1.session_key == "s1"
    assert t1.origin.platform == "telegram"
    assert [t.task_id for t in reg.list_tasks()] == [t1.task_id, t2.task_id]
    assert reg.list_tasks(session_key="s1") == [t1]
    assert reg.list_tasks(session_key="missing") == []


def test_status_updates_worker_assignment_notes_and_cancel():
    reg = TaskRegistry()
    task = reg.create_task("heavy implementation", session_key="s1")

    reg.update_status(task.task_id, STATUS_RUNNING, note="started")
    reg.assign_worker(task.task_id, "worker-1", worker_kind=WORKER_CLAUDE_CODE)
    reg.add_note(task.task_id, "needs review")

    assert task.status == STATUS_RUNNING
    assert task.active_worker_id == "worker-1"
    assert task.worker_kind == WORKER_CLAUDE_CODE
    assert task.notes == ["started", "needs review"]

    reg.clear_worker(task.task_id)
    assert task.active_worker_id is None
    assert task.worker_kind is None

    reg.cancel_task(task.task_id, reason="user reclaimed")
    assert task.status == STATUS_CANCELLED
    assert task.notes[-1] == "cancelled: user reclaimed"


def test_invalid_status_rejected_on_create_update_and_load():
    reg = TaskRegistry()
    with pytest.raises(ValueError, match="unknown task status"):
        reg.create_task("bad", status="mystery")

    task = reg.create_task("ok")
    with pytest.raises(ValueError, match="unknown task status"):
        reg.update_status(task.task_id, "mystery")

    payload = task.to_dict()
    payload["status"] = "mystery"
    with pytest.raises(ValueError, match="unknown task status"):
        TaskRegistry.from_dict({"version": 1, "tasks": [payload]})


def test_followups_attach_in_order_and_legacy_payloads_gain_session_key():
    reg = TaskRegistry()
    task = reg.create_task("collect fragmented input", session_key="s1")

    first = PendingTurnItem(text="first", session_key="s1")
    reg.attach_followup(task.task_id, first)
    second = reg.attach_followup(task.task_id, "second")
    media = reg.attach_followup(task.task_id, ("caption", ["/tmp/a.png"]))

    assert task.pending_followups[0] is first
    assert [item.text for item in task.pending_followups] == ["first", "second", "caption"]
    assert second.session_key == "s1"
    assert media.kind == KIND_MEDIA
    assert media.boundary == BOUNDARY_HARD
    assert media.media_refs == ["/tmp/a.png"]


def test_append_followup_records_plain_text_provenance_and_task_hint():
    reg = TaskRegistry()
    task = reg.create_task("collect late guidance", session_key="s1")

    item = reg.append_followup(
        task.task_id,
        "이 조건도 넣어줘",
        source="gateway",
        session_key="s1",
    )

    assert task.pending_followups == [item]
    assert item.text == "이 조건도 넣어줘"
    assert item.source == "gateway"
    assert item.session_key == "s1"
    assert item.task_hint == task.task_id


def test_serialization_roundtrip_excludes_raw_without_touching_it():
    reg = TaskRegistry()
    task = reg.create_task("serialize safely", origin={"platform": "cli", "session_key": "s1"})
    reg.attach_followup(
        task.task_id,
        PendingTurnItem(text="rawful", session_key="s1", raw=UncopyableRaw()),
    )
    reg.attach_artifact(task.task_id, {"path": "/tmp/report.md", "kind": "markdown"})
    reg.add_note(task.task_id, "review before delivery")

    data = reg.to_dict()
    assert "raw" not in data["tasks"][0]["pending_followups"][0]
    json.dumps(data)  # JSON-safe

    restored = TaskRegistry.from_dict(copy.deepcopy(data))
    restored_task = restored.get_task(task.task_id)
    assert restored_task is not None
    assert restored_task.origin.platform == "cli"
    assert restored_task.pending_followups[0].text == "rawful"
    assert restored_task.pending_followups[0].raw is None
    assert restored_task.artifacts == [{"path": "/tmp/report.md", "kind": "markdown"}]
    assert restored_task.notes == ["review before delivery"]


def test_active_filter_excludes_terminal_statuses():
    reg = TaskRegistry()
    active = reg.create_task("active", status=STATUS_PROPOSED)
    queued = reg.create_task("queued", status=STATUS_QUEUED)
    done = reg.create_task("done", status=STATUS_DONE)
    error = reg.create_task("error", status=STATUS_ERROR)
    cancelled = reg.create_task("cancelled", status=STATUS_CANCELLED)

    assert [t.task_id for t in reg.list_tasks(active_only=True)] == [
        active.task_id,
        queued.task_id,
    ]
    assert not done.is_active
    assert not error.is_active
    assert not cancelled.is_active


def test_artifacts_are_detached_json_safe_copies():
    reg = TaskRegistry()
    task = reg.create_task("artifact work")
    artifact = {"path": "/tmp/out.pdf", "kind": "pdf", "meta": {"page": 1}}

    stored = reg.attach_artifact(task.task_id, artifact)
    artifact["path"] = "/tmp/mutated.pdf"
    artifact["meta"]["page"] = 99

    assert stored == {"path": "/tmp/out.pdf", "kind": "pdf", "meta": {"page": 1}}
    serialized = task.to_dict()
    serialized["artifacts"][0]["meta"]["page"] = 2
    assert task.artifacts[0]["meta"]["page"] == 1

    with pytest.raises(TypeError, match="artifact must be a dict"):
        reg.attach_artifact(task.task_id, ["not", "dict"])

    with pytest.raises(TypeError, match="artifact must be JSON-serializable"):
        reg.attach_artifact(task.task_id, {"bad": object()})

    with pytest.raises(TypeError, match="artifact must be JSON-serializable"):
        reg.attach_artifact(task.task_id, {"metric": float("nan")})

    with pytest.raises(TypeError, match="artifact must be JSON-serializable"):
        reg.attach_artifact(task.task_id, {"metric": float("inf")})

    json.dumps(reg.to_dict(), allow_nan=False)


def test_worker_result_attaches_json_safe_snapshot_to_task():
    reg = TaskRegistry()
    task = reg.create_task("implementation", session_key="s1")

    result = reg.attach_worker_result(
        task.task_id,
        {
            "worker_id": "worker-1",
            "task_id": task.task_id,
            "status": "succeeded",
            "summary": "changed agent/task_registry.py",
            "artifacts": [{"path": "agent/task_registry.py", "kind": "source"}],
            "tests": [{"command": "pytest tests/agent/test_task_registry.py", "status": "passed"}],
            "extra_raw": object(),
        },
    )

    assert result["worker_id"] == "worker-1"
    assert result["task_id"] == task.task_id
    assert result["status"] == "succeeded"
    assert result["review_status"] == "pending_review"
    assert "extra_raw" not in result
    assert task.to_dict()["result"] == result
    json.dumps(task.to_dict(), allow_nan=False)


def test_worker_result_strips_or_rejects_raw_non_json_metadata():
    reg = TaskRegistry()
    task = reg.create_task("implementation", session_key="s1")

    result = reg.attach_worker_result(
        task.task_id,
        {
            "worker_id": "worker-1",
            "task_id": task.task_id,
            "status": "failed",
            "summary": "failed before completion",
            "error": RuntimeError("boom"),
            "metadata": {"proc": object()},
        },
    )

    assert result == {
        "worker_id": "worker-1",
        "task_id": task.task_id,
        "status": "failed",
        "summary": "failed before completion",
        "error": "boom",
        "review_status": "pending_review",
    }
    json.dumps(reg.to_dict(), allow_nan=False)


def test_frontdesk_metadata_and_review_artifact_roundtrip():
    reg = TaskRegistry()
    task = reg.create_task(
        "background implementation",
        session_key="s1",
        status=STATUS_QUEUED,
        frontdesk_state=FRONTDESK_RUNNING_WORKER,
        worker_stage=STAGE_RUNNING,
        reviewer_stage=STAGE_NOT_STARTED,
    )

    reg.update_frontdesk_metadata(
        task.task_id,
        worker_process_id=12345,
        worker_session_id="worker-1",
        last_message_path="/tmp/workers/task-1/last-message.txt",
        summary_artifact_path="/tmp/workers/task-1/summary.md",
    )
    reg.attach_worker_result(
        task.task_id,
        {"worker_id": "worker-1", "status": "succeeded", "summary": "done"},
    )
    reg.update_frontdesk_metadata(
        task.task_id,
        frontdesk_state=FRONTDESK_WORKER_DONE_PENDING_REVIEW,
        worker_stage=STAGE_DONE,
        reviewer_stage=STAGE_QUEUED,
    )
    review = reg.attach_review_result(
        task.task_id,
        ReviewResultArtifact(
            review_status=REVIEW_PASSED,
            reviewer_model="gpt-5.5",
            summary="safe to present",
            tests_run=["pytest tests/agent/test_task_registry.py"],
            changed_files=["agent/task_registry.py"],
            risks=["durability still future work"],
            recommended_next_action=REVIEW_ACTION_PRESENT,
            artifact_paths=["/tmp/workers/task-1/review.json"],
        ),
    )

    assert review["review_status"] == REVIEW_PASSED
    assert task.frontdesk_state == FRONTDESK_REVIEW_PASSED
    assert task.review_verdict == REVIEW_PASSED
    assert task.reviewer_stage == STAGE_DONE
    assert task.review_artifact_path == "/tmp/workers/task-1/review.json"

    restored = TaskRegistry.from_dict(copy.deepcopy(reg.to_dict()))
    restored_task = restored.get_task(task.task_id)
    assert restored_task is not None
    assert restored_task.worker_process_id == "12345"
    assert restored_task.worker_session_id == "worker-1"
    assert restored_task.last_message_path.endswith("last-message.txt")
    assert restored_task.summary_artifact_path.endswith("summary.md")
    assert restored_task.review_result == review
    assert restored_task.frontdesk_state == FRONTDESK_REVIEW_PASSED
    json.dumps(restored.to_dict(), allow_nan=False)


def test_blocking_review_sets_user_input_state():
    reg = TaskRegistry()
    task = reg.create_task("needs clarification", status=STATUS_RUNNING)

    review = reg.attach_review_result(
        task.task_id,
        {
            "review_status": REVIEW_BLOCKED,
            "reviewer_model": "gpt-5.5",
            "summary": "need user to choose deployment target",
            "tests_run": [],
            "changed_files": [],
            "risks": ["cannot choose target safely"],
            "recommended_next_action": REVIEW_ACTION_ASK_USER,
            "artifact_paths": [],
        },
    )

    assert review["review_status"] == REVIEW_BLOCKED
    assert task.status == STATUS_BLOCKED
    assert task.frontdesk_state == FRONTDESK_BLOCKED_USER_INPUT
    assert task.awaiting_user_input is True
    assert task.blocked_reason == "need user to choose deployment target"


def test_json_persistence_roundtrip_and_missing_file(tmp_path):
    path = tmp_path / "registry.json"
    missing = TaskRegistry.load(path)
    assert missing.path == str(path)
    assert missing.list_tasks() == []

    task = missing.create_task("persist me", session_key="s1")
    missing.attach_followup(task.task_id, PendingTurnItem(kind=KIND_TEXT, text="late note"))
    written = missing.save()

    assert written == str(path)
    loaded = TaskRegistry.load(path)
    assert loaded.path == str(path)
    assert [t.user_goal for t in loaded.list_tasks()] == ["persist me"]
    assert loaded.get_task(task.task_id).pending_followups[0].text == "late note"

    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["version"] == TaskRegistry.SCHEMA_VERSION
