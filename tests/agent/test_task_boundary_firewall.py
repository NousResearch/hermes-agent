from __future__ import annotations

import importlib


TASK_A_ID = "phase5-task-a"
TASK_B_ID = "phase5-task-b"
TASK_A_CURRENT_PIN = "/tmp/phase5/task-a/current-pin.md"
TASK_A_WORKSPEC = "/tmp/phase5/task-a/WorkSpec.md"
TASK_A_TASK_STATE = "/tmp/phase5/task-a/task-state.json"
RAW_A_NEEDLE = "RAW_CLOSED_TASK_A_PHASE5_NEEDLE"


def _load_firewall_module():
    return importlib.import_module("agent.task_boundary_firewall")


def _registry_snapshot_with_closed_a_and_active_b():
    return {
        "schema": "context_health_task_registry_v1",
        "active_task_id": TASK_B_ID,
        "tasks": {
            TASK_A_ID: {
                "task_id": TASK_A_ID,
                "status": "closed",
                "workspec_path": TASK_A_WORKSPEC,
                "task_state_path": TASK_A_TASK_STATE,
                "current_pin_path": TASK_A_CURRENT_PIN,
                "raw_transcript_excerpt": RAW_A_NEEDLE,
            },
            TASK_B_ID: {
                "task_id": TASK_B_ID,
                "status": "active",
                "workspec_path": "/tmp/phase5/task-b/WorkSpec.md",
            },
        },
    }


def test_firewall_disabled_returns_use_original_pass_through():
    module = _load_firewall_module()

    decision = module.enforce_task_boundary_firewall(
        policy={"enabled": False},
        registry_snapshot=_registry_snapshot_with_closed_a_and_active_b(),
        current_task_id=TASK_B_ID,
        user_message="simple message while firewall disabled",
    )

    assert decision.action == "use_original"
    assert decision.allowed_task_ids in ((), [], None)
    assert decision.excluded_task_ids in ((), [], None)


def test_new_independent_b_excludes_closed_task_a_by_default():
    module = _load_firewall_module()

    decision = module.enforce_task_boundary_firewall(
        policy={
            "enabled": True,
            "runtime_behavior_enabled": True,
            "task_boundary": {"enabled": True, "default_without_clear_continuation": "new_task"},
            "task_boundary_firewall": {"enabled": True},
        },
        registry_snapshot=_registry_snapshot_with_closed_a_and_active_b(),
        current_task_id=TASK_B_ID,
        user_message="start unrelated B task",
    )

    assert decision.action == "allow_new_task"
    assert TASK_B_ID in decision.allowed_task_ids
    assert TASK_A_ID in decision.excluded_task_ids
    assert TASK_A_CURRENT_PIN not in repr(decision)
    assert RAW_A_NEEDLE not in repr(decision)


def test_explicit_continuation_allows_only_safe_a_pointer():
    module = _load_firewall_module()

    decision = module.enforce_task_boundary_firewall(
        policy={"enabled": True, "runtime_behavior_enabled": True, "task_boundary_firewall": {"enabled": True}},
        registry_snapshot=_registry_snapshot_with_closed_a_and_active_b(),
        current_task_id=TASK_A_ID,
        user_message=f"continue {TASK_A_ID} using its WorkSpec",
        explicit_continuation_refs=[TASK_A_ID],
    )

    assert decision.action in {"allow_continue_task", "allow_linked_task_context"}
    assert TASK_A_ID in decision.allowed_task_ids
    assert TASK_A_ID not in decision.excluded_task_ids
    assert TASK_A_CURRENT_PIN in repr(decision.safe_linked_task_pointers)
    assert RAW_A_NEEDLE not in repr(decision)


def test_ambiguous_previous_reference_holds():
    module = _load_firewall_module()

    decision = module.enforce_task_boundary_firewall(
        policy={"enabled": True, "runtime_behavior_enabled": True, "task_boundary_firewall": {"enabled": True}},
        registry_snapshot=_registry_snapshot_with_closed_a_and_active_b(),
        current_task_id=TASK_B_ID,
        user_message="continue the previous one",
    )

    assert decision.action == "hold"
    assert decision.hold_response
    assert RAW_A_NEEDLE not in decision.hold_response


def test_closed_a_paths_quarantined_for_independent_b():
    module = _load_firewall_module()

    decision = module.enforce_task_boundary_firewall(
        policy={"enabled": True, "runtime_behavior_enabled": True, "task_boundary_firewall": {"enabled": True}},
        registry_snapshot=_registry_snapshot_with_closed_a_and_active_b(),
        current_task_id=TASK_B_ID,
        user_message="new unrelated B task",
    )

    rendered = repr(decision.filtered_registry_snapshot)
    assert TASK_A_CURRENT_PIN not in rendered
    assert TASK_A_WORKSPEC not in rendered
    assert TASK_A_TASK_STATE not in rendered
    assert TASK_B_ID in rendered


def test_user_pasted_old_task_marker_without_clear_continuation_holds():
    module = _load_firewall_module()

    decision = module.enforce_task_boundary_firewall(
        policy={"enabled": True, "runtime_behavior_enabled": True, "task_boundary_firewall": {"enabled": True}},
        registry_snapshot=_registry_snapshot_with_closed_a_and_active_b(),
        current_task_id=TASK_B_ID,
        user_message=f"Here is {TASK_A_ID}: {RAW_A_NEEDLE}\nNow do something different",
    )

    assert decision.action == "hold"
    assert RAW_A_NEEDLE not in repr(decision)
