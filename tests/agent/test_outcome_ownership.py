"""Behavioral checks for same-run blocker recovery and honest completion."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.outcome_ownership import latest_outcome_blocker, outcome_followthrough_nudge


def _append_round(messages, name, args, result, *, call_id):
    messages.extend([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }],
        },
        {
            "role": "tool",
            "name": name,
            "tool_name": name,
            "tool_call_id": call_id,
            "content": result,
        },
    ])


def _agent(**overrides):
    values = {
        "valid_tool_names": {"terminal", "read_file", "patch"},
        "max_iterations": 20,
        "_outcome_recovery_nudges": 0,
        "_outcome_ownership_failure": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _failed_messages(result='{"success": false, "error": "pytest failed: missing import"}'):
    messages = [{"role": "user", "content": "Repair the runtime and verify it."}]
    _append_round(messages, "patch", {"path": "agent/runtime.py"}, result, call_id="write-1")
    return messages


def test_recoverable_blocker_requires_a_real_followthrough_result():
    messages = _failed_messages()
    agent = _agent()

    nudge = outcome_followthrough_nudge(agent, "I will inspect the import next.", messages)

    assert nudge and "diagnose" in nudge
    assert "pytest failed: missing import" in nudge
    assert "attempt 1" in nudge
    assert agent._outcome_recovery_nudges == 1

    _append_round(messages, "read_file", {"path": "agent/runtime.py"}, "from agent.missing import x", call_id="read-1")
    blocker = latest_outcome_blocker(messages)
    assert blocker is not None
    assert blocker.has_followup_action
    assert blocker.has_successful_followup
    assert outcome_followthrough_nudge(agent, "The missing import is the cause.", messages) is None


def test_actionable_next_step_is_executed_in_the_same_run():
    messages = _failed_messages()
    _append_round(messages, "read_file", {"path": "agent/runtime.py"}, "current source", call_id="read-1")

    nudge = outcome_followthrough_nudge(_agent(), "Next step is to repair the import.", messages)

    assert nudge and "execute it now" in nudge


def test_recommendation_only_stop_is_nudged_for_an_execution_request():
    agent = _agent()
    messages = [{"role": "user", "content": "Implement the repair in the existing runtime."}]

    nudge = outcome_followthrough_nudge(agent, "Next step is to update the shared handler.", messages)

    assert nudge and "perform it now" in nudge
    assert "recommended_action" in nudge
    assert agent._outcome_recovery_nudges == 1

    informational = [{"role": "user", "content": "What does the runtime handler do?"}]
    assert outcome_followthrough_nudge(_agent(), "Next step is to read the documentation.", informational) is None


def test_second_stop_without_a_tool_action_is_an_unsuccessful_outcome():
    agent = _agent()
    messages = _failed_messages()

    assert outcome_followthrough_nudge(agent, "I will use an alternate path.", messages)
    assert agent._outcome_followthrough_pending is True

    assert outcome_followthrough_nudge(agent, "I still cannot proceed.", messages) is None
    assert agent._outcome_ownership_failure.startswith("followthrough_action_not_taken:")


def test_user_only_approval_boundary_is_unsuccessful_not_bypassed():
    agent = _agent()
    messages = [{"role": "user", "content": "Deploy the approved change."}]

    assert outcome_followthrough_nudge(agent, "I need your approval before the deployment.", messages) is None
    assert agent._outcome_ownership_failure.startswith("user_only_decision:")


def test_text_stop_gate_continues_the_existing_turn_after_a_blocker(monkeypatch):
    import agent.turn_stop_gates as gates

    class GateAgent:
        valid_tool_names = {"terminal", "read_file"}
        max_iterations = 20
        _outcome_recovery_nudges = 0

        def _emit_interim_assistant_message(self, _message):
            pass

        def _flush_messages_to_session_db(self, *_args):
            pass

        def _interim_content_was_streamed(self, _content):
            return False

    monkeypatch.setattr(gates, "_verify_on_stop_nudge", lambda _agent: None)
    monkeypatch.setattr(gates, "_pre_verify_nudge", lambda *_args: None)
    monkeypatch.setattr(gates, "_kanban_stop_nudge", lambda *_args: None)

    messages = _failed_messages()
    agent = GateAgent()
    verdict = gates.apply_stop_gates(
        agent,
        {"role": "assistant", "content": "I will fix the missing import next."},
        final_response="I will fix the missing import next.",
        messages=messages,
        conversation_history=[],
        pending_verification_response=None,
        pending_verification_response_previewed=False,
    )

    assert verdict.continue_turn is True
    assert verdict.final_response is None
    assert messages[-1]["_verification_stop_synthetic"] is True
    assert "perform" in messages[-1]["content"]


def test_pending_and_unknown_are_typed_and_reconciled_without_repeating_side_effect():
    pending = [{"role": "user", "content": "Check the existing operation."}]
    _append_round(
        pending, "publish", {"operation_id": "op-7"},
        '{"status":"awaiting_receipt","reason":"receipt delayed","operation_id":"op-7"}',
        call_id="publish-1",
    )
    pending_blocker = latest_outcome_blocker(pending)
    assert pending_blocker and pending_blocker.kind == "pending"
    nudge = outcome_followthrough_nudge(_agent(), "Still waiting.", pending)
    assert nudge and "read/reconciliation" in nudge
    assert "do not" in nudge and "repeat the side effect" in nudge

    unknown = [{"role": "user", "content": "Check the existing operation."}]
    _append_round(
        unknown, "publish", {"operation_id": "op-7"},
        '{"reason":"response lost","operation_id":"op-7"}', call_id="publish-2",
    )
    unknown[-1]["effect_disposition"] = "unknown"
    unknown_blocker = latest_outcome_blocker(unknown)
    assert unknown_blocker and unknown_blocker.kind == "unknown"
    unknown_nudge = outcome_followthrough_nudge(_agent(), "The publish may have worked.", unknown)
    assert unknown_nudge and "read-only action" in unknown_nudge


def test_matching_terminal_readback_closes_a_pending_operation():
    messages = [{"role": "user", "content": "Check the existing operation."}]
    _append_round(
        messages, "publish", {"operation_id": "op-7"},
        '{"status":"pending","operation_id":"op-7","reason":"receipt delayed"}',
        call_id="publish-1",
    )
    _append_round(
        messages, "read_receipt", {"operation_id": "op-7"},
        '{"status":"delivered","operation_id":"op-7","receipt_verified":true,"target_readback":true}',
        call_id="receipt-1",
    )

    assert latest_outcome_blocker(messages) is None


def test_repeated_identical_failure_selects_an_alternate_path():
    messages = _failed_messages()
    failure_result = messages[-1]["content"]
    _append_round(messages, "patch", {"path": "agent/runtime.py"}, failure_result, call_id="write-2")
    blocker = latest_outcome_blocker(messages)

    assert blocker and blocker.repeated_without_progress
    nudge = outcome_followthrough_nudge(_agent(), "The same patch failed again.", messages)
    assert nudge and "different authorized path" in nudge

    messages = _failed_messages()
    _append_round(messages, "read_file", {"path": "agent/runtime.py"}, "new source evidence", call_id="read-1")
    _append_round(messages, "patch", {"path": "agent/runtime.py"}, failure_result, call_id="write-2")
    blocker_with_new_evidence = latest_outcome_blocker(messages)
    assert blocker_with_new_evidence and not blocker_with_new_evidence.repeated_without_progress


def test_true_permission_or_identity_conflict_fails_closed():
    messages = [{"role": "user", "content": "Make the change."}]
    _append_round(
        messages, "patch", {"path": "protected.py"},
        '{"status":"permission_denied","reason":"approval required"}', call_id="write-1",
    )
    agent = _agent()

    assert outcome_followthrough_nudge(agent, "I can try another route.", messages) is None
    assert agent._outcome_ownership_failure.startswith("safety_conflict:")


def test_terminal_uses_exit_status_not_arbitrary_stdout_for_outcome():
    messages = [{"role": "user", "content": "Check the issue state."}]
    _append_round(
        messages, "terminal", {"command": "gh issue view 12"},
        json.dumps({
            "output": '{"body":"provenance conflict; prior test failed"}',
            "exit_code": 0, "error": None,
        }),
        call_id="read-1",
    )

    assert latest_outcome_blocker(messages) is None


def test_nonzero_terminal_result_requires_readback_before_retry():
    messages = [{"role": "user", "content": "Post the authorized comment."}]
    _append_round(
        messages, "terminal", {"command": "gh issue comment 12 --body ..."},
        json.dumps({"output": "issuecomment-123", "exit_code": 1, "error": "connection reset"}),
        call_id="write-1",
    )

    blocker = latest_outcome_blocker(messages)
    assert blocker and blocker.kind == "unknown"
    nudge = outcome_followthrough_nudge(_agent(), "I will retry the comment.", messages)
    assert nudge and "read-only action" in nudge and "never repeat" in nudge

    denied = [{"role": "user", "content": "Write the protected value."}]
    _append_round(
        denied, "terminal", {"command": "write-protected-value"},
        json.dumps({
            "output": "", "exit_code": -1, "status": "permission_denied",
            "error": "approval required",
        }),
        call_id="denied-1",
    )
    assert latest_outcome_blocker(denied).kind == "safety_conflict"


def test_recovery_uses_existing_iteration_budget_not_a_fixed_mutation_threshold():
    messages = _failed_messages()
    _append_round(messages, "patch", {"path": "agent/runtime.py"}, messages[-1]["content"], call_id="write-2")
    agent = _agent(max_iterations=8, _outcome_recovery_nudges=2)

    nudge = outcome_followthrough_nudge(agent, "The repair remains fixable.", messages)

    assert nudge and "different authorized path" in nudge
    assert agent._outcome_ownership_failure is None

    exhausted = _agent(max_iterations=2, _outcome_recovery_nudges=2)
    assert outcome_followthrough_nudge(exhausted, "Still blocked.", messages) is None
    assert exhausted._outcome_ownership_failure
