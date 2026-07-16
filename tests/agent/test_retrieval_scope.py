from __future__ import annotations

import importlib


TASK_A_ID = "phase6-task-a"
TASK_B_ID = "phase6-task-b"
TASK_A_SESSION_ID = "session-phase6-a"
TASK_B_SESSION_ID = "session-phase6-b"
TASK_A_NEEDLE = "PHASE6_CLOSED_TASK_A_RETRIEVAL_NEEDLE"


def _load_retrieval_scope_module():
    return importlib.import_module("agent.retrieval_scope")


def _policy(*, enabled: bool = True):
    return {
        "enabled": enabled,
        "runtime_behavior_enabled": enabled,
        "retrieval_scope": {"enabled": enabled},
        "task_registry": {"enabled": enabled},
        "task_boundary_firewall": {"enabled": enabled},
    }


def _registry_snapshot_with_closed_a_and_active_b():
    return {
        "schema": "context_health_task_registry_v1",
        "active_task_id": TASK_B_ID,
        "tasks": {
            TASK_A_ID: {
                "task_id": TASK_A_ID,
                "status": "closed",
                "session_id": TASK_A_SESSION_ID,
                "latest_turn_id": "turn-a",
                "workspec_path": "/tmp/phase6/task-a/WorkSpec.md",
                "task_state_path": "/tmp/phase6/task-a/task-state.json",
                "current_pin_path": "/tmp/phase6/task-a/current-pin.md",
                "raw_transcript_excerpt": TASK_A_NEEDLE,
            },
            TASK_B_ID: {
                "task_id": TASK_B_ID,
                "status": "active",
                "session_id": TASK_B_SESSION_ID,
                "latest_turn_id": "turn-b",
                "workspec_path": "/tmp/phase6/task-b/WorkSpec.md",
            },
        },
    }


def _new_task_decision():
    return type("Decision", (), {"action": "new_task", "effective_task_id": TASK_B_ID})()


def _linked_task_decision():
    return type(
        "Decision",
        (),
        {"action": "link_task", "effective_task_id": TASK_B_ID, "linked_task_id": TASK_A_ID},
    )()


def test_retrieval_scope_disabled_returns_use_original_pass_through():
    module = _load_retrieval_scope_module()

    decision = module.enforce_retrieval_scope(
        policy=_policy(enabled=False),
        tool_name="session_search",
        tool_args={"query": TASK_A_NEEDLE},
        current_task_id=TASK_B_ID,
        registry_snapshot=_registry_snapshot_with_closed_a_and_active_b(),
        registry_decision=_new_task_decision(),
    )

    assert decision.action == "use_original"
    assert decision.rewritten_args in ({}, None)
    assert decision.hold_response in ("", None)
    assert decision.allowed_task_ids in ([], (), None)
    assert decision.excluded_task_ids in ([], (), None)


def test_new_independent_b_excludes_closed_task_a_retrieval():
    module = _load_retrieval_scope_module()

    decision = module.enforce_retrieval_scope(
        policy=_policy(enabled=True),
        tool_name="session_search",
        tool_args={"query": TASK_A_NEEDLE},
        current_task_id=TASK_B_ID,
        registry_snapshot=_registry_snapshot_with_closed_a_and_active_b(),
        registry_decision=_new_task_decision(),
    )

    assert decision.action in {"rewrite_args", "hold", "block"}
    assert decision.reason
    assert TASK_B_ID in decision.allowed_task_ids
    assert TASK_A_ID in decision.excluded_task_ids
    assert TASK_B_SESSION_ID in decision.allowed_session_ids
    assert TASK_A_SESSION_ID in decision.excluded_session_ids
    assert TASK_A_NEEDLE not in repr(decision)


def test_explicit_continuation_allows_only_current_and_linked_task_ids():
    module = _load_retrieval_scope_module()

    decision = module.enforce_retrieval_scope(
        policy=_policy(enabled=True),
        tool_name="session_search",
        tool_args={"session_id": TASK_A_SESSION_ID},
        current_task_id=TASK_B_ID,
        registry_snapshot=_registry_snapshot_with_closed_a_and_active_b(),
        explicit_continuation_refs=[TASK_A_ID],
        registry_decision=_linked_task_decision(),
    )

    assert decision.action in {"allow", "rewrite_args"}
    assert set(decision.allowed_task_ids) == {TASK_B_ID, TASK_A_ID}
    assert set(decision.allowed_session_ids) == {TASK_B_SESSION_ID, TASK_A_SESSION_ID}
    assert TASK_A_ID not in decision.excluded_task_ids
    assert TASK_A_SESSION_ID not in decision.excluded_session_ids
    assert TASK_A_NEEDLE not in repr(decision)


def test_ambiguous_prior_task_retrieval_holds_before_search():
    module = _load_retrieval_scope_module()

    decision = module.enforce_retrieval_scope(
        policy=_policy(enabled=True),
        tool_name="session_search",
        tool_args={"query": "use the previous one"},
        current_task_id=TASK_B_ID,
        registry_snapshot=_registry_snapshot_with_closed_a_and_active_b(),
        registry_decision=type("Decision", (), {"action": "hold", "effective_task_id": TASK_B_ID})(),
        user_message="use the previous one",
    )

    assert decision.action in {"hold", "block"}
    assert decision.hold_response
    assert TASK_A_ID in decision.excluded_task_ids
    assert TASK_A_NEEDLE not in decision.hold_response
    assert TASK_A_NEEDLE not in repr(decision)


def test_enabled_adapter_failure_returns_fail_closed_decision_without_raw_leak():
    module = _load_retrieval_scope_module()

    decision = module.safe_retrieval_scope_failure(
        policy=_policy(enabled=True),
        reason="synthetic adapter failure with token password secret",
        current_task_id=TASK_B_ID,
        registry_snapshot=_registry_snapshot_with_closed_a_and_active_b(),
        registry_decision=_new_task_decision(),
    )

    assert decision.action in {"hold", "block"}
    assert decision.hold_response
    rendered = repr(decision)
    assert TASK_A_NEEDLE not in rendered
    assert "synthetic adapter failure" not in rendered
    assert "password" not in rendered.lower()
    assert "secret" not in rendered.lower()
