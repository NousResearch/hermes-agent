import pytest

from hermes_cli.workflows_assistant import (
    AssistantValidationError,
    WorkflowDraftResult,
    parse_assistant_payload,
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
