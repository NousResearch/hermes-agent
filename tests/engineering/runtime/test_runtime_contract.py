from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from uuid import UUID

import pytest

import engineering.runtime.base as runtime_base_module
import engineering.runtime.models as runtime_models_module
from engineering.runtime import (
    AgentRuntime,
    AgentRuntimeError,
    RuntimeUsage,
    TurnRequest,
    TurnResult,
    TurnStatus,
)


def make_request(**overrides: object) -> TurnRequest:
    values: dict[str, object] = {
        "workflow_run_id": "workflow-123",
        "attempt": 1,
        "message": "Inspect the current implementation.",
    }
    values.update(overrides)
    return TurnRequest(**values)  # type: ignore[arg-type]


def make_result(**overrides: object) -> TurnResult:
    values: dict[str, object] = {
        "request_id": "request-123",
        "workflow_run_id": "workflow-123",
        "attempt": 1,
        "status": TurnStatus.RETURNED,
        "response": "The runtime returned normally.",
    }
    values.update(overrides)
    return TurnResult(**values)  # type: ignore[arg-type]


def test_turn_status_represents_runtime_outcomes_only() -> None:
    assert [status.value for status in TurnStatus] == [
        "RETURNED",
        "FAILED",
        "INTERRUPTED",
    ]


def test_turn_request_generates_uuid4_identity() -> None:
    first = make_request()
    second = make_request()

    assert UUID(first.request_id).version == 4
    assert UUID(second.request_id).version == 4
    assert first.request_id != second.request_id


def test_turn_request_preserves_explicit_identity_and_workflow_lineage() -> None:
    request = make_request(
        request_id="request-imported",
        workflow_run_id="workflow-99",
        attempt=3,
    )

    assert request.request_id == "request-imported"
    assert request.workflow_run_id == "workflow-99"
    assert request.attempt == 3


@pytest.mark.parametrize("attempt", [0, -1, True, 1.5, "1"])
def test_turn_request_attempt_must_be_positive(attempt: object) -> None:
    with pytest.raises(
        ValueError, match="attempt must be an integer greater than zero"
    ):
        make_request(attempt=attempt)


@pytest.mark.parametrize("message", ["", "   ", None, 7])
def test_turn_request_rejects_empty_or_invalid_message(message: object) -> None:
    with pytest.raises(ValueError, match="message must be a non-empty string"):
        make_request(message=message)


def test_turn_request_preserves_optional_runtime_correlation() -> None:
    request = make_request(session_id="session-7", task_id="task-8")

    assert request.session_id == "session-7"
    assert request.task_id == "task-8"


def test_turn_request_metadata_is_deeply_immutable_and_detached() -> None:
    items = ["initial"]
    nested = {"items": items}
    request = make_request(metadata={"nested": nested})

    items.append("later")
    nested["extra"] = True

    frozen_nested = request.metadata["nested"]
    assert frozen_nested["items"] == ("initial",)  # type: ignore[index]
    assert "extra" not in frozen_nested  # type: ignore[operator]
    with pytest.raises(TypeError):
        request.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        frozen_nested["new"] = "value"  # type: ignore[index]


def test_turn_request_rejects_runtime_objects_in_metadata() -> None:
    with pytest.raises(TypeError, match="provider-neutral structured data"):
        make_request(metadata={"client": object()})


def test_turn_request_is_immutable() -> None:
    request = make_request()

    with pytest.raises(FrozenInstanceError):
        request.message = "changed"  # type: ignore[misc]


def test_turn_result_generates_uuid4_turn_identity() -> None:
    first = make_result()
    second = make_result()

    assert UUID(first.turn_id).version == 4
    assert UUID(second.turn_id).version == 4
    assert first.turn_id != second.turn_id


def test_turn_result_preserves_explicit_identities_and_lineage() -> None:
    result = make_result(
        turn_id="turn-imported",
        request_id="request-imported",
        workflow_run_id="workflow-99",
        attempt=2,
    )

    assert result.turn_id == "turn-imported"
    assert result.request_id == "request-imported"
    assert result.workflow_run_id == "workflow-99"
    assert result.attempt == 2


@pytest.mark.parametrize("attempt", [0, -1, True, 1.5, "1"])
def test_turn_result_attempt_must_be_positive(attempt: object) -> None:
    with pytest.raises(
        ValueError, match="attempt must be an integer greater than zero"
    ):
        make_result(attempt=attempt)


def test_returned_result_accepts_non_empty_response() -> None:
    result = make_result(status=TurnStatus.RETURNED, response="Returned.")

    assert result.status is TurnStatus.RETURNED
    assert result.response == "Returned."


@pytest.mark.parametrize("response", [None, "", "   "])
def test_returned_result_requires_non_empty_response(response: object) -> None:
    with pytest.raises(ValueError, match="response must be a non-empty string"):
        make_result(status=TurnStatus.RETURNED, response=response)


@pytest.mark.parametrize(
    "status", [TurnStatus.FAILED, TurnStatus.INTERRUPTED]
)
def test_non_returned_result_may_omit_response(status: TurnStatus) -> None:
    result = make_result(status=status, response=None)

    assert result.status is status
    assert result.response is None


def test_turn_result_preserves_optional_runtime_details() -> None:
    result = make_result(
        session_id="session-1",
        task_id="task-2",
        provider="provider-neutral-name",
        model="model-name",
    )

    assert result.session_id == "session-1"
    assert result.task_id == "task-2"
    assert result.provider == "provider-neutral-name"
    assert result.model == "model-name"


def test_usage_is_optional_and_provider_neutral() -> None:
    assert make_result().usage is None

    usage = RuntimeUsage(input_tokens=5, output_tokens=3, total_tokens=8)
    result = make_result(usage=usage)

    assert result.usage == usage
    assert {field.name for field in fields(RuntimeUsage)} == {
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_usage_rejects_invalid_token_counts(value: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer or None"):
        RuntimeUsage(input_tokens=value)  # type: ignore[arg-type]


def test_turn_result_metadata_and_object_are_immutable() -> None:
    result = make_result(metadata={"labels": ["runtime", "turn"]})

    assert result.metadata["labels"] == ("runtime", "turn")
    with pytest.raises(TypeError):
        result.metadata["new"] = True  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.response = "changed"  # type: ignore[misc]


def test_agent_runtime_is_a_minimal_protocol() -> None:
    public_methods = {
        name
        for name, value in inspect.getmembers(
            AgentRuntime, predicate=inspect.isfunction
        )
        if not name.startswith("_")
    }

    assert getattr(AgentRuntime, "_is_protocol", False) is True
    assert public_methods == {"run_turn"}
    signature = inspect.signature(AgentRuntime.run_turn)
    assert signature.return_annotation == "TurnResult"
    assert signature.parameters["request"].annotation == "TurnRequest"


def test_runtime_error_policy_reserves_exceptions_for_contract_failures() -> None:
    assert issubclass(AgentRuntimeError, Exception)
    assert "infrastructure" in inspect.getdoc(AgentRuntimeError).lower()
    assert {"FAILED", "INTERRUPTED"}.issubset(TurnStatus.__members__)


def test_runtime_models_have_no_engineering_completion_or_gate_fields() -> None:
    model_fields = {
        field.name for model in (TurnRequest, TurnResult) for field in fields(model)
    }
    forbidden = {
        "completed",
        "engineering_completed",
        "verified",
        "reviewed",
        "verification",
        "review",
        "workflow_state",
    }

    assert model_fields.isdisjoint(forbidden)
    assert "COMPLETED" not in TurnStatus.__members__


def test_runtime_boundary_has_no_hermes_domain_or_execution_dependencies() -> None:
    forbidden_roots = {
        "agent",
        "hermes_cli",
        "hermes_state",
        "run_agent",
        "subprocess",
        "tools",
    }
    forbidden_imported_names = {
        "EngineeringStore",
        "ReviewResult",
        "VerificationResult",
        "WorkflowRun",
    }
    imported_roots: set[str] = set()
    imported_names: set[str] = set()

    for module in (runtime_base_module, runtime_models_module):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.partition(".")[0] for alias in node.names
                )
                imported_names.update(
                    alias.asname or alias.name for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition(".")[0])
                imported_names.update(alias.name for alias in node.names)

    assert imported_roots.isdisjoint(forbidden_roots)
    assert imported_names.isdisjoint(forbidden_imported_names)
    assert "run_conversation" not in vars(AgentRuntime)
    assert "conversation_loop" not in vars(AgentRuntime)
    assert "finalize_turn" not in vars(AgentRuntime)


def test_runtime_models_do_not_persist_or_execute() -> None:
    source = "\n".join(
        inspect.getsource(module)
        for module in (runtime_base_module, runtime_models_module)
    )
    forbidden_symbols = {
        "append_evidence",
        "save_workflow",
        "read_text",
        "write_text",
        "subprocess",
    }

    assert all(symbol not in source for symbol in forbidden_symbols)
    assert not any(
        isinstance(value, Path)
        for model in (make_request(), make_result())
        for value in model.metadata.values()
    )
