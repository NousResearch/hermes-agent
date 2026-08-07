"""Behavioral tests for the pure, lock-aware run registry."""

import threading

import pytest

from gateway.run_service import RunRegistry


PUBLIC_FIELDS = {
    "object",
    "run_id",
    "status",
    "session_id",
    "model",
    "created_at",
    "updated_at",
    "last_event",
    "output",
    "usage",
    "error",
}


class MutableTimestamp:
    def __init__(self, value):
        self.value = value

    def __float__(self):
        return float(self.value)


class UnsupportedStatusValue:
    deepcopy_calls = 0

    def __deepcopy__(self, memo):
        type(self).deepcopy_calls += 1
        raise AssertionError("custom deepcopy must not run")


def test_set_status_retains_creation_time_and_projects_only_public_fields():
    times = iter((10.0, 20.0))
    registry = RunRegistry(clock=lambda: next(times))

    registry.set_status(
        "run_1",
        "queued",
        session_id="conversation-a",
        model="hermes-agent",
        private_agent="must-not-leak",
    )
    registry.set_status(
        "run_1",
        "running",
        created_at=999.0,
        last_event="run.started",
    )

    status = registry.get("run_1")

    assert status == {
        "object": "hermes.run",
        "run_id": "run_1",
        "status": "running",
        "session_id": "conversation-a",
        "model": "hermes-agent",
        "created_at": 10.0,
        "updated_at": 20.0,
        "last_event": "run.started",
    }
    assert set(status) <= PUBLIC_FIELDS


def test_set_status_detaches_explicit_creation_time():
    registry = RunRegistry(clock=lambda: 20.0)
    created_at = MutableTimestamp(10.0)

    registry.set_status(
        "run_1",
        "queued",
        created_at=created_at,
        session_id="conversation-a",
    )
    created_at.value = 99.0

    status = registry.get("run_1")
    assert status is not None
    assert status["created_at"] == 10.0


def test_set_status_coerces_and_detaches_clock_value_before_storage():
    clock_value = MutableTimestamp(10.0)
    registry = RunRegistry(clock=lambda: clock_value)

    registry.set_status("run_1", "queued", session_id="conversation-a")
    clock_value.value = 99.0

    status = registry.get("run_1")
    assert status is not None
    assert status["created_at"] == 10.0
    assert status["updated_at"] == 10.0


def test_explicit_and_raw_updated_at_values_use_timestamp_fallback():
    registry = RunRegistry(clock=lambda: 10.0)

    status = registry.set_status(
        "run_1",
        "running",
        session_id="conversation-a",
        updated_at=float("nan"),
    )
    assert status["updated_at"] == 0.0
    assert RunRegistry.public_status(
        {
            "run_id": "run_raw",
            "status": "running",
            "updated_at": float("inf"),
        }
    )["updated_at"] == 0.0


@pytest.mark.parametrize(
    "timestamp",
    [
        "not-a-number",
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        pytest.param(10**10000, id="float-overflow"),
    ],
)
def test_public_status_normalizes_invalid_and_non_finite_timestamps(timestamp):
    registry = RunRegistry()
    registry.statuses["run_bad_time"] = {
        "object": "hermes.run",
        "run_id": "run_bad_time",
        "status": "running",
        "session_id": "conversation-a",
        "created_at": timestamp,
        "updated_at": timestamp,
    }

    status = registry.get("run_bad_time")

    assert status["created_at"] == 0.0
    assert status["updated_at"] == 0.0


def test_status_snapshots_cannot_mutate_authoritative_ownership():
    registry = RunRegistry()
    source = {
        "run_id": "run_owned",
        "status": "running",
        "session_id": "conversation-a",
        "usage": {"input_tokens": 1},
    }
    registry.statuses["run_owned"] = source

    source["session_id"] = "conversation-b"
    source["usage"]["input_tokens"] = 2
    compatibility_snapshot = registry.statuses["run_owned"]
    compatibility_snapshot["session_id"] = "conversation-b"
    compatibility_snapshot["usage"]["input_tokens"] = 3
    returned = registry.set_status("run_owned", "waiting_for_approval")
    returned["session_id"] = "conversation-b"
    returned["usage"]["input_tokens"] = 4

    status = registry.get("run_owned")
    assert status is not None
    assert status["session_id"] == "conversation-a"
    assert status["usage"] == {"input_tokens": 1}


def test_unsupported_status_value_fails_before_lock_without_partial_update():
    times = iter((10.0, 20.0))
    registry = RunRegistry(clock=lambda: next(times))
    registry.set_status("run_owned", "running", session_id="conversation-a")
    before = registry.get("run_owned")
    UnsupportedStatusValue.deepcopy_calls = 0

    with pytest.raises(TypeError, match="JSON-compatible"):
        registry.set_status(
            "run_owned",
            "waiting_for_approval",
            usage=UnsupportedStatusValue(),
        )

    assert UnsupportedStatusValue.deepcopy_calls == 0
    assert registry.get("run_owned") == before


def test_set_status_rejects_unknown_states_and_nonfinite_payload_values():
    registry = RunRegistry()

    with pytest.raises(ValueError, match="unsupported run status"):
        registry.set_status("run_1", "runnning", session_id="conversation-a")
    with pytest.raises(ValueError, match="finite"):
        registry.set_status(
            "run_1",
            "running",
            session_id="conversation-a",
            usage={"cost": float("nan")},
        )
    with pytest.raises(ValueError, match="unsupported run status"):
        registry.statuses["run_1"] = {
            "run_id": "run_1",
            "status": "runnning",
            "session_id": "conversation-a",
        }
    assert registry.get("run_1") is None


def test_active_task_count_calls_done_without_holding_registry_lock():
    registry = RunRegistry()
    update_finished = threading.Event()

    class Task:
        def done(self):
            def update_registry():
                registry.set_status(
                    "run_other",
                    "running",
                    session_id="conversation-b",
                )
                update_finished.set()

            thread = threading.Thread(target=update_registry)
            thread.start()
            thread.join(timeout=1)
            return update_finished.is_set()

    registry.register_task("run_1", Task())

    assert registry.active_task_count() == 0
    assert update_finished.is_set()


def test_claim_stop_target_atomically_marks_stopping():
    registry = RunRegistry()
    agent = object()
    task = object()
    registry.set_status("run_owned", "running", session_id="conversation-a")
    registry.register_agent("run_owned", agent)
    registry.register_task("run_owned", task)

    assert registry.claim_stop_target("run_missing") is None

    claimed = registry.claim_stop_target("run_owned")
    assert claimed is not None
    assert claimed.agent is agent
    assert claimed.task is task
    assert registry.is_stopping("run_owned")
    status = registry.get("run_owned")
    assert status is not None
    assert status["status"] == "stopping"
    assert status["last_event"] == "run.stopping"
    registry.release_stop_target("run_owned")


def test_claim_prevents_control_replacement_without_holding_global_lock():
    registry = RunRegistry()
    original_agent = object()
    replacement_agent = object()
    original_task = object()
    replacement_task = object()
    registry.set_status("run_owned", "running", session_id="conversation-a")
    registry.register_agent("run_owned", original_agent)
    registry.register_task("run_owned", original_task)

    claimed = registry.claim_stop_target("run_owned")
    assert claimed is not None
    assert claimed.agent is original_agent
    assert registry.register_agent("run_owned", replacement_agent) is False
    with pytest.raises(RuntimeError, match="stop is pending"):
        registry.agents["run_owned"] = replacement_agent
    with pytest.raises(RuntimeError, match="stop is pending"):
        registry.tasks["run_owned"] = replacement_task
    with pytest.raises(RuntimeError, match="stop is pending"):
        registry.statuses["run_owned"] = {
            "run_id": "run_owned",
            "status": "running",
            "session_id": "conversation-b",
        }
    with pytest.raises(RuntimeError, match="stop is pending"):
        registry.stopping_ids.discard("run_owned")
    with pytest.raises(RuntimeError, match="ownership mutation"):
        registry.set_status(
            "run_owned",
            "running",
            session_id="conversation-b",
        )
    assert registry.agent_for("run_owned") is original_agent
    assert registry.task_for("run_owned") is original_task
    assert registry.get("run_owned") is not None

    unrelated_finished = threading.Event()

    def update_unrelated_run():
        registry.set_status("run_other", "running", session_id="conversation-b")
        unrelated_finished.set()

    thread = threading.Thread(target=update_unrelated_run)
    thread.start()
    thread.join(timeout=1)
    assert unrelated_finished.is_set()

    registry.release_stop_target("run_owned")
    assert registry.register_agent("run_owned", replacement_agent) is False
    assert registry.register_task("run_owned", replacement_task) is False
    with pytest.raises(RuntimeError, match="stop is pending"):
        registry.agents["run_owned"] = replacement_agent
    with pytest.raises(RuntimeError, match="stop is pending"):
        registry.tasks["run_owned"] = replacement_task
    with pytest.raises(RuntimeError, match="ownership mutation"):
        registry.set_status(
            "run_owned",
            "stopping",
            session_id="conversation-b",
        )
    assert registry.agent_for("run_owned") is original_agent
    assert registry.task_for("run_owned") is original_task


def test_claim_defers_control_cleanup_until_release():
    registry = RunRegistry()
    agent = object()
    task = object()
    registry.set_status("run_owned", "running", session_id="conversation-a")
    registry.register_agent("run_owned", agent)
    registry.register_task("run_owned", task)
    registry.register_approval_session("run_owned", "approval-owned")

    assert registry.claim_stop_target("run_owned") is not None
    registry.remove_control("run_owned")
    assert registry.agent_for("run_owned") is agent
    assert registry.task_for("run_owned") is task
    assert registry.approval_session_for("run_owned") == "approval-owned"

    registry.release_stop_target("run_owned")
    assert registry.agent_for("run_owned") is None
    assert registry.task_for("run_owned") is None
    assert registry.approval_session_for("run_owned") is None


def test_registry_owns_approval_session_lookup_and_cleanup():
    registry = RunRegistry()

    registry.register_approval_session("run_owned", "approval-owned")

    assert registry.approval_session_for("run_owned") == "approval-owned"
    assert registry.approval_session_for("run_missing") is None
    assert dict(registry.approval_sessions) == {
        "run_owned": "approval-owned"
    }

    registry.remove_control("run_owned")

    assert registry.approval_session_for("run_owned") is None
    assert "run_owned" not in registry.approval_sessions


def test_replace_approval_sessions_preserves_legacy_map_compatibility():
    registry = RunRegistry()
    registry.replace_approval_sessions(
        {"run_one": "approval-one", "run_two": "approval-two"}
    )

    assert registry.approval_session_for("run_one") == "approval-one"
    assert registry.approval_session_for("run_two") == "approval-two"
