import json

import pytest

from agent.continuation_engine import should_use_continuation_engine
from agent.intent_preclassifier import preclassify_intent
from hermes_cli.command_templates import build_command_invocation


def _active_snapshot() -> dict:
    return {
        "outcomeStatus": "interrupted",
        "activeTodos": [
            {"id": "todo-1", "content": "Finish the remaining work", "status": "in_progress"}
        ],
    }


def _sample_contract_payload(runtime_mode: str | None = None) -> dict:
    context = {"ticket": "wave-c-t5"}
    if runtime_mode:
        context["command_runtime"] = {
            "command_name": "handoff",
            "runtime_mode": runtime_mode,
        }
    return {
        "task": "Resume the delegated task",
        "expected_outcome": "Implementation resumed with verification evidence",
        "required_skills": ["python", "testing"],
        "required_tools": ["read_file", "patch", "terminal"],
        "must_do": ["inspect current state before acting"],
        "must_not_do": ["discard the preserved contract"],
        "context": context,
    }


@pytest.mark.parametrize(
    ("command_name", "expected_runtime_mode", "expected_semantics", "should_continue"),
    [
        ("handoff", "default", "blocks", False),
        ("init-deep", "default", "blocks", False),
        ("start-work", "default", "blocks", False),
        ("ralph-loop", "ralph", "injects", True),
        ("ulw-loop", "ultrawork", "injects", True),
    ],
    ids=["handoff", "init-deep", "start-work", "ralph-loop", "ulw-loop"],
)
def test_continuation_matrix_default_entry_surfaces(command_name, expected_runtime_mode, expected_semantics, should_continue):
    invocation = build_command_invocation(
        command_name,
        raw_args="Continue the delegated task",
        session_id="sess-1",
        cwd="/tmp",
    )

    result = preclassify_intent(
        {
            "message": "Continue the delegated task",
            "task_contract": invocation.task_contract,
        }
    )

    assert result.inferred_runtime_mode == expected_runtime_mode
    assert should_use_continuation_engine(result.inferred_runtime_mode, _active_snapshot()) is should_continue

    if expected_semantics == "injects":
        assert invocation.task_contract["context"]["command_runtime"]["runtime_mode"] == expected_runtime_mode


@pytest.mark.parametrize(
    ("command_name", "runtime_mode"),
    [
        ("handoff", "ralph"),
        ("handoff", "ultrawork"),
        ("start-work", "ralph"),
        ("start-work", "ultrawork"),
    ],
    ids=[
        "handoff-preserves-ralph",
        "handoff-preserves-ultrawork",
        "start-work-preserves-ralph",
        "start-work-preserves-ultrawork",
    ],
)
def test_continuation_matrix_preserve_surfaces_keep_explicit_runtime_contracts(command_name, runtime_mode):
    payload = _sample_contract_payload(runtime_mode)

    invocation = build_command_invocation(
        command_name,
        raw_args=json.dumps(payload),
        session_id="sess-2",
        cwd="/tmp",
    )

    result = preclassify_intent(
        {
            "message": payload["task"],
            "task_contract": invocation.task_contract,
        }
    )

    assert invocation.task_contract == payload
    assert result.task_contract is not None
    assert result.task_contract.model_dump() == payload
    assert result.inferred_runtime_mode == runtime_mode
    assert should_use_continuation_engine(result.inferred_runtime_mode, _active_snapshot()) is True
