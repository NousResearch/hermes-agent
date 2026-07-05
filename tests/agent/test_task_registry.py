from __future__ import annotations

import importlib
import json

import pytest


OLD_TASK_A_ID = "phase4-task-a"
NEW_TASK_B_ID = "phase4-task-b"


def _load_registry_module():
    return importlib.import_module("agent.task_registry")


def test_registry_disabled_returns_use_original_contract(tmp_path):
    module = _load_registry_module()

    decision = module.resolve_task_for_turn(
        policy={"enabled": False},
        registry_root=tmp_path,
        session_id="phase4-session",
        incoming_task_id="caller-task-id",
        user_message="new request while registry disabled",
    )

    assert decision.action in {"use_original", "pass"}
    assert decision.effective_task_id == "caller-task-id"
    assert decision.registry_snapshot is None or decision.registry_snapshot == {}
    assert not (tmp_path / "registry.json").exists()
    assert not (tmp_path / "events.jsonl").exists()


def test_no_clear_continuation_defaults_to_new_task(tmp_path):
    module = _load_registry_module()
    registry = {
        "schema": "context_health_task_registry_v1",
        "schema_version": 1,
        "active_task_id": OLD_TASK_A_ID,
        "tasks": {
            OLD_TASK_A_ID: {
                "task_id": OLD_TASK_A_ID,
                "status": "active",
                "title": "old task A",
                "linked_task_ids": [],
            }
        },
    }
    (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")

    decision = module.resolve_task_for_turn(
        policy={
            "enabled": True,
            "runtime_behavior_enabled": True,
            "task_boundary": {"enabled": True, "default_without_clear_continuation": "new_task"},
            "task_registry": {"enabled": True},
        },
        registry_root=tmp_path,
        session_id="phase4-session",
        incoming_task_id=None,
        user_message="Start an unrelated task B with no continuation reference.",
    )

    assert decision.action == "new_task"
    assert decision.effective_task_id != OLD_TASK_A_ID
    assert decision.effective_task_id
    assert decision.reason == "no_clear_continuation_evidence"
    updated = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert updated["active_task_id"] == decision.effective_task_id
    assert updated["tasks"][OLD_TASK_A_ID]["status"] in {"closed", "archived"}
    assert "task_created" in (tmp_path / "events.jsonl").read_text(encoding="utf-8")


def test_explicit_continuation_reuses_or_reopens_task_a(tmp_path):
    module = _load_registry_module()
    registry = {
        "schema": "context_health_task_registry_v1",
        "schema_version": 1,
        "active_task_id": None,
        "tasks": {
            OLD_TASK_A_ID: {
                "task_id": OLD_TASK_A_ID,
                "status": "closed",
                "title": "old task A",
                "linked_task_ids": [],
            }
        },
    }
    (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")

    decision = module.resolve_task_for_turn(
        policy={
            "enabled": True,
            "runtime_behavior_enabled": True,
            "task_boundary": {"enabled": True, "default_without_clear_continuation": "new_task"},
            "task_registry": {"enabled": True},
        },
        registry_root=tmp_path,
        session_id="phase4-session",
        incoming_task_id=None,
        user_message=f"{OLD_TASK_A_ID} 계속 진행해줘",
        explicit_continuation_refs=[OLD_TASK_A_ID],
    )

    assert decision.action == "continue_task"
    assert decision.effective_task_id == OLD_TASK_A_ID
    assert decision.linked_task_id == OLD_TASK_A_ID
    assert decision.reason == "explicit_continuation_reference"


def test_ambiguous_relation_holds_when_policy_says_hold(tmp_path):
    module = _load_registry_module()
    registry = {
        "schema": "context_health_task_registry_v1",
        "schema_version": 1,
        "active_task_id": OLD_TASK_A_ID,
        "tasks": {OLD_TASK_A_ID: {"task_id": OLD_TASK_A_ID, "status": "active"}},
    }
    (tmp_path / "registry.json").write_text(json.dumps(registry), encoding="utf-8")

    decision = module.resolve_task_for_turn(
        policy={
            "enabled": True,
            "runtime_behavior_enabled": True,
            "task_boundary": {"enabled": True, "ambiguous_action": "hold"},
            "task_registry": {"enabled": True},
        },
        registry_root=tmp_path,
        session_id="phase4-session",
        incoming_task_id=None,
        user_message="이거 아까 그거랑 이어지는지 새로 하는 건지 애매해 token=abc123",
        ambiguous_relation=True,
    )

    assert decision.action == "hold"
    assert decision.effective_task_id is None
    assert "abc123" not in repr(decision)
    assert "token=abc123" not in repr(decision)


def test_registry_json_snapshot_and_events_jsonl_are_atomic_and_append_only(tmp_path):
    module = _load_registry_module()

    first = module.record_task_event(
        registry_root=tmp_path,
        event="task_created",
        task_id=OLD_TASK_A_ID,
        session_id="phase4-session",
        status="active",
    )
    second = module.record_task_event(
        registry_root=tmp_path,
        event="task_closed",
        task_id=OLD_TASK_A_ID,
        session_id="phase4-session",
        status="closed",
        reason="completed_task_state",
    )

    snapshot = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert first["task_id"] == OLD_TASK_A_ID
    assert second["task_id"] == OLD_TASK_A_ID
    assert snapshot["tasks"][OLD_TASK_A_ID]["status"] == "closed"
    assert len(events) == 2
    assert "task_created" in events[0]
    assert "task_closed" in events[1]


def test_completed_workspec_task_state_can_seed_closed_registry_record(tmp_path):
    module = _load_registry_module()
    task_state = tmp_path / "task-state.json"
    task_state.write_text(
        json.dumps(
            {
                "schema": "workspec_task_state_v1",
                "packet_dir": str(tmp_path),
                "active_goal": "old task A",
                "status": "completed",
                "tasks": [{"id": "t1", "status": "completed", "text": "done"}],
            }
        ),
        encoding="utf-8",
    )

    record = module.import_workspec_task_state(
        registry_root=tmp_path / "registry",
        task_id=OLD_TASK_A_ID,
        session_id="phase4-session",
        task_state_path=task_state,
    )

    assert record["task_id"] == OLD_TASK_A_ID
    assert record["status"] == "closed"
    assert record["task_state_path"] == str(task_state)
    assert "task_closed" in (tmp_path / "registry" / "events.jsonl").read_text(encoding="utf-8")


def test_enabled_corrupt_registry_json_raises_typed_failure(tmp_path):
    module = _load_registry_module()
    (tmp_path / "registry.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(module.TaskRegistryFailure) as exc:
        module.resolve_task_for_turn(
            policy={
                "enabled": True,
                "runtime_behavior_enabled": True,
                "task_boundary": {"enabled": True},
                "task_registry": {"enabled": True},
            },
            registry_root=tmp_path,
            session_id="phase4-session",
            incoming_task_id=None,
            user_message="new task B",
        )

    assert exc.value.reason == "task_registry_corrupt_snapshot"
    assert "not valid" not in exc.value.user_response


def test_enabled_event_append_failure_raises_typed_failure(tmp_path, monkeypatch):
    module = _load_registry_module()

    def fail_append(*_args, **_kwargs):
        raise OSError("permission denied token=raw-secret")

    monkeypatch.setattr(module, "_append_event", fail_append)

    with pytest.raises(module.TaskRegistryFailure) as exc:
        module.resolve_task_for_turn(
            policy={
                "enabled": True,
                "runtime_behavior_enabled": True,
                "task_boundary": {"enabled": True},
                "task_registry": {"enabled": True},
            },
            registry_root=tmp_path,
            session_id="phase4-session",
            incoming_task_id=None,
            user_message="new task B",
        )

    assert exc.value.reason == "task_registry_write_failure"
    assert "raw-secret" not in exc.value.user_response
    assert "token" not in exc.value.user_response.lower()
