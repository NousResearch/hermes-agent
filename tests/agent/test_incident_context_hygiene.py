"""Focused regressions for bounded, disposable repair incident context."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.incident_state import (
    FIELD_LABELS,
    INCIDENT_RESET_ACTION,
    build_incident_state,
    incident_state_message,
    update_after_tool_round,
)


def _call(name: str, call_id: str, args: str = '{"path":"config.yaml"}'):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=args),
    )


def _failed_round(call_id: str = "c1"):
    call = _call("patch", call_id)
    assistant = SimpleNamespace(tool_calls=[call])
    messages = [
        {"role": "user", "content": "Make the original Hermes workflow pass."},
        {"role": "assistant", "tool_calls": [call]},
        {
            "role": "tool",
            "name": "patch",
            "tool_call_id": call_id,
            "content": '{"success": false, "error": "regression test failed"}',
        },
    ]
    return assistant, messages


def test_incident_checkpoint_has_only_bounded_fields_and_no_old_narrative():
    old_narrative = "OLD VERBOSE DEBUGGING OUTPUT " * 500
    state = build_incident_state([
        {"role": "user", "content": "Fix the original workflow."},
        {"role": "assistant", "content": old_narrative},
        {"role": "tool", "name": "terminal", "content": '{"exit_code": 1, "error": "test failed"}'},
    ])

    rendered = state.render()
    assert all(f"{label}:" in rendered for label in FIELD_LABELS)
    assert "OLD VERBOSE DEBUGGING OUTPUT" not in rendered
    assert len(rendered) <= 1_500 * 4


def test_second_failed_mutation_requests_first_principles_reset():
    agent = SimpleNamespace(
        incident_context_enabled=True,
        _incident_state=None,
        _incident_failed_repair_attempts=0,
        incident_reset_after_failed_repairs=2,
    )
    assistant, messages = _failed_round()

    first = update_after_tool_round(agent, assistant, messages, 1)
    assert first.failed and first.compact and not first.hard_reset
    assert agent._incident_failed_repair_attempts == 1

    second_assistant, second_tail = _failed_round("c2")
    messages.extend(second_tail[1:])
    second = update_after_tool_round(agent, second_assistant, messages, 3)
    assert second.failed and second.compact and second.hard_reset
    assert INCIDENT_RESET_ACTION in second.state.next_action_current_observed_state


def test_active_incident_keeps_newly_retrieved_read_output_until_next_failure():
    agent = SimpleNamespace(
        incident_context_enabled=True,
        _incident_state={
            "original_failure_requested_outcome": "Fix Hermes.",
            "current_root_cause_hypothesis": "The repair path is stale.",
            "supporting_evidence": "patch failed",
            "files_config_changed": "config.yaml",
            "latest_test_result": "failed",
            "next_action_current_observed_state": "Read current config.",
        },
        _incident_failed_repair_attempts=0,
        incident_reset_after_failed_repairs=2,
    )
    call = _call("read_file", "read1", '{"path":"config.yaml"}')
    assistant = SimpleNamespace(tool_calls=[call])
    messages = [
        incident_state_message(agent._incident_state),
        {"role": "assistant", "tool_calls": [call]},
        {
            "role": "tool",
            "name": "read_file",
            "tool_call_id": "read1",
            "content": "authoritative current config: incident_state enabled",
        },
    ]

    update = update_after_tool_round(agent, assistant, messages, 1)
    assert not update.compact
    assert not update.failed


def test_incident_compressor_returns_state_only_and_respects_cap():
    compressor = ContextCompressor(
        model="test/model",
        quiet_mode=True,
        incident_summary_token_cap=1_500,
    )
    messages = [
        {"role": "user", "content": "Fix Hermes."},
        {"role": "assistant", "content": "OLD NARRATIVE " * 500},
        {"role": "tool", "content": "OLD COMMAND OUTPUT " * 500},
    ]
    state = build_incident_state(messages)
    compressed = compressor.compress(
        messages,
        incident_state=state.as_dict(),
        hard_reset=True,
    )

    assert len(compressed) == 1
    assert compressed[0]["_incident_state"] is True
    assert INCIDENT_RESET_ACTION in compressed[0]["content"]
    assert "OLD NARRATIVE" not in compressed[0]["content"]
    assert "OLD COMMAND OUTPUT" not in compressed[0]["content"]
    assert len(compressed[0]["content"]) <= 1_500 * 4


def test_new_user_request_does_not_resurrect_completed_checkpoint():
    old = incident_state_message({
        "original_failure_requested_outcome": "Old incident",
        "current_root_cause_hypothesis": "Old hypothesis",
        "supporting_evidence": "Old evidence",
        "files_config_changed": "old.py",
        "latest_test_result": "passed",
        "next_action_current_observed_state": "Closed",
    })
    call = _call("patch", "new1")
    messages = [
        old,
        {"role": "assistant", "content": "Old incident resolved."},
        {"role": "user", "content": "New incident: repair the scheduler."},
        {"role": "assistant", "tool_calls": [call]},
        {
            "role": "tool",
            "name": "patch",
            "tool_call_id": "new1",
            "content": '{"success": false, "error": "new test failed"}',
        },
    ]
    agent = SimpleNamespace(
        incident_context_enabled=True,
        _incident_state=None,
        _incident_failed_repair_attempts=0,
        incident_reset_after_failed_repairs=2,
    )

    update = update_after_tool_round(
        agent, SimpleNamespace(tool_calls=[call]), messages, 3,
    )
    assert update.state.original_failure_requested_outcome == "New incident: repair the scheduler."


def test_unresolved_checkpoint_is_recovered_after_process_restart():
    state = {
        "original_failure_requested_outcome": "Resume the repair.",
        "current_root_cause_hypothesis": "The first repair was incomplete.",
        "supporting_evidence": "regression test failed",
        "files_config_changed": "config.yaml",
        "latest_test_result": "failed",
        "next_action_current_observed_state": "Re-read current config.",
    }
    call = _call("patch", "resume1")
    messages = [
        incident_state_message(state),
        {"role": "assistant", "tool_calls": [call]},
        {
            "role": "tool",
            "name": "patch",
            "tool_call_id": "resume1",
            "content": '{"success": false, "error": "second regression failed"}',
        },
    ]
    agent = SimpleNamespace(
        incident_context_enabled=True,
        _incident_state=None,
        _incident_failed_repair_attempts=0,
        incident_reset_after_failed_repairs=2,
    )

    update = update_after_tool_round(
        agent, SimpleNamespace(tool_calls=[call]), messages, 1,
    )
    assert update.hard_reset
    assert update.state.original_failure_requested_outcome == "Resume the repair."


def test_hard_reset_rotates_real_session_to_one_checkpoint(tmp_path):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(db_path=tmp_path / "state.db")
    parent = "INCIDENT_PARENT"
    db.create_session(parent, source="cli")
    db.append_message(parent, "user", "Fix Hermes repair loop.")
    db.append_message(parent, "assistant", "OLD VERBOSE DIAGNOSTIC NARRATIVE " * 100)
    messages = db.get_messages_as_conversation(parent)
    state = build_incident_state(messages)

    try:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                platform="cli",
                quiet_mode=True,
                session_db=db,
                session_id=parent,
                skip_context_files=True,
                skip_memory=True,
            )
        agent._compression_feasibility_checked = True
        agent.compression_in_place = True
        agent.incident_context_enabled = True
        returned, _ = agent._compress_context(
            messages,
            "system prompt",
            approx_tokens=120_000,
            force=True,
            incident_state=state.as_dict(),
            hard_reset=True,
        )

        child = agent.session_id
        child_rows = db.get_messages_as_conversation(child)
        assert child != parent
        assert db.get_session(parent)["ended_at"] is not None
        assert len(returned) == len(child_rows) == 1
        assert "_incident_state" in returned[0]
        assert "OLD VERBOSE DIAGNOSTIC NARRATIVE" not in child_rows[0]["content"]
        assert INCIDENT_RESET_ACTION in child_rows[0]["content"]
    finally:
        db.close()
