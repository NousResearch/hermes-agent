import json

import pytest

from hermes_cli.workflows_assistant import (
    AssistantValidationError,
    WorkflowDraftResult,
    build_draft_prompt,
    draft_workflow,
    parse_assistant_payload,
    refine_workflow,
)


def _valid_payload():
    return {
        "spec": {
            "id": "code_review_flow",
            "name": "Code Review Flow",
            "version": 1,
            "triggers": [{"type": "manual", "id": "manual"}],
            "nodes": {
                "implement": {
                    "type": "agent_task",
                    "profile": "implementer",
                    "title": "Implement change",
                    "prompt": "Implement ${ input.request } and return JSON.",
                },
                "done": {"type": "pass", "output": {"status": "ok"}},
            },
            "edges": [{"from": "implement", "to": "done"}],
        },
        "summary": "Implements a change then marks it done.",
        "assumptions": ["Manual trigger for first version."],
        "questions": [],
        "warnings": [],
        "unsupported_requests": [],
    }


def test_parse_assistant_payload_returns_validated_draft_result():
    result = parse_assistant_payload(_valid_payload())

    assert isinstance(result, WorkflowDraftResult)
    assert result.spec.id == "code_review_flow"
    assert result.valid is True
    assert result.validation_errors == []
    assert result.summary.startswith("Implements")
    assert result.spec.nodes["implement"].type == "agent_task"


def test_parse_assistant_payload_rejects_unsupported_runtime_primitives():
    payload = _valid_payload()
    payload["spec"]["nodes"]["notify"] = {"type": "send_message", "output": {}}
    payload["spec"]["edges"].append({"from": "done", "to": "notify"})

    with pytest.raises(AssistantValidationError) as exc:
        parse_assistant_payload(payload)

    assert "unsupported node type" in str(exc.value)
    assert "send_message" in str(exc.value)


def test_parse_assistant_payload_returns_clear_validation_errors():
    payload = _valid_payload()
    del payload["spec"]["nodes"]["implement"]["profile"]

    with pytest.raises(AssistantValidationError) as exc:
        parse_assistant_payload(payload)

    assert "agent_task node implement requires a non-blank profile" in str(exc.value)


def test_draft_workflow_calls_runner_with_plain_goal_and_returns_valid_spec():
    calls = []

    def fake_runner(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(_valid_payload())

    from hermes_cli.workflows_assistant import draft_workflow

    result = draft_workflow(
        "When I ask for a code change, have an implementer do it.",
        runner=fake_runner,
    )

    assert result.spec.id == "code_review_flow"
    assert "When I ask for a code change" in calls[0]
    assert "Return JSON only" in calls[0]
    assert "send_message" in calls[0]  # listed as unsupported, not allowed


def test_refine_workflow_includes_current_spec_and_instruction():
    payload = _valid_payload()
    current = parse_assistant_payload(payload).spec
    calls = []

    def fake_runner(prompt: str) -> str:
        calls.append(prompt)
        payload["summary"] = "Added reviewer step."
        return json.dumps(payload)

    from hermes_cli.workflows_assistant import refine_workflow

    result = refine_workflow(current, "Add a reviewer after implement", runner=fake_runner)

    assert result.summary == "Added reviewer step."
    assert "Add a reviewer after implement" in calls[0]
    assert '"nodes"' in calls[0]


def test_draft_workflow_repairs_once_after_invalid_first_response():
    responses = [
        json.dumps({"summary": "bad", "spec": {"id": "bad", "name": "Bad", "version": 1, "nodes": {}}}),
        json.dumps(_valid_payload()),
    ]

    def fake_runner(prompt: str) -> str:
        return responses.pop(0)

    from hermes_cli.workflows_assistant import draft_workflow

    result = draft_workflow("Build a valid workflow", runner=fake_runner, repair_attempts=1)
    assert result.spec.id == "code_review_flow"


def test_draft_workflow_rejects_blank_goal_without_calling_runner():
    calls = []

    def fake_runner(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(_valid_payload())

    with pytest.raises(AssistantValidationError, match="workflow goal is required"):
        draft_workflow(" \t\n", runner=fake_runner)

    assert calls == []


def test_refine_workflow_rejects_blank_instruction_without_calling_runner():
    current = parse_assistant_payload(_valid_payload()).spec
    calls = []

    def fake_runner(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(_valid_payload())

    with pytest.raises(AssistantValidationError, match="refine instruction is required"):
        refine_workflow(current, " \t\n", runner=fake_runner)

    assert calls == []


def test_draft_workflow_honors_multiple_repair_attempts():
    responses = [
        json.dumps({"summary": "bad", "spec": {"id": "bad", "name": "Bad", "version": 1, "nodes": {}}}),
        json.dumps({"summary": "bad", "spec": {"id": "also_bad", "name": "Also Bad", "version": 1, "nodes": {}}}),
        json.dumps(_valid_payload()),
    ]
    calls = []

    def fake_runner(prompt: str) -> str:
        calls.append(prompt)
        return responses.pop(0)

    result = draft_workflow("Build a valid workflow", runner=fake_runner, repair_attempts=2)

    assert result.spec.id == "code_review_flow"
    assert len(calls) == 3


def test_draft_prompt_uses_valid_empty_edges_example():
    prompt = build_draft_prompt("Build a valid workflow")

    assert '"edges": []' in prompt
    assert "next_node_id" not in prompt
