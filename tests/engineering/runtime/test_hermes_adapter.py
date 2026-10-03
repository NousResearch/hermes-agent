from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping
from pathlib import Path
from uuid import UUID

import pytest

import engineering.runtime.hermes as hermes_adapter_module
from engineering.runtime import (
    AgentRuntimeError,
    HermesRuntimeAdapter,
    TurnRequest,
    TurnStatus,
)


class FakeHermesAgent:
    def __init__(
        self,
        result: object,
        *,
        api_mode: str = "chat_completions",
        session_id: str = "session-1",
        provider: str = "openai-compatible",
        model: str = "model-1",
    ) -> None:
        self.result = result
        self.api_mode = api_mode
        self.session_id = session_id
        self.provider = provider
        self.model = model
        self.calls: list[dict[str, object]] = []
        self.error: BaseException | None = None

    def run_conversation(
        self,
        user_message: object,
        *,
        task_id: str | None = None,
    ) -> Mapping[str, object]:
        self.calls.append(
            {"user_message": user_message, "task_id": task_id}
        )
        if self.error is not None:
            raise self.error
        return self.result  # type: ignore[return-value]


def make_request(**overrides: object) -> TurnRequest:
    values: dict[str, object] = {
        "request_id": "request-1",
        "workflow_run_id": "workflow-1",
        "attempt": 2,
        "message": "Inspect the implementation.",
    }
    values.update(overrides)
    return TurnRequest(**values)  # type: ignore[arg-type]


def normal_result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "completed": True,
        "failed": False,
        "interrupted": False,
        "final_response": "Hermes returned normally.",
        "session_id": "session-1",
        "provider": "resolved-provider",
        "model": "resolved-model",
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "api_calls": 2,
        "turn_exit_reason": "text_response(final)",
        "partial": False,
    }
    result.update(overrides)
    return result


def test_adapter_calls_public_run_conversation_surface() -> None:
    agent = FakeHermesAgent(normal_result())
    adapter = HermesRuntimeAdapter(agent)

    result = adapter.run_turn(make_request(task_id="task-7"))

    assert agent.calls == [
        {
            "user_message": "Inspect the implementation.",
            "task_id": "task-7",
        }
    ]
    assert result.request_id == "request-1"
    assert result.workflow_run_id == "workflow-1"
    assert result.attempt == 2
    assert result.task_id == "task-7"


def test_adapter_generates_distinct_runtime_task_id_when_absent() -> None:
    agent = FakeHermesAgent(normal_result())

    result = HermesRuntimeAdapter(agent).run_turn(make_request())

    generated_task_id = agent.calls[0]["task_id"]
    assert isinstance(generated_task_id, str)
    assert UUID(generated_task_id).version == 4
    assert generated_task_id != result.request_id
    assert result.task_id == generated_task_id


def test_adapter_maps_normal_return_without_engineering_completion() -> None:
    result = HermesRuntimeAdapter(
        FakeHermesAgent(normal_result())
    ).run_turn(make_request())

    assert result.status is TurnStatus.RETURNED
    assert result.response == "Hermes returned normally."
    assert result.session_id == "session-1"
    assert result.provider == "resolved-provider"
    assert result.model == "resolved-model"
    assert result.usage is not None
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert result.usage.total_tokens == 18
    assert "completed" not in result.metadata


def test_adapter_maps_allowlisted_diagnostic_metadata_only() -> None:
    raw = normal_result(
        failure_reason="none",
        error="diagnostic",
        interrupt_message="not-used",
        messages=[{"role": "user", "content": "private"}],
        base_url="https://secret-endpoint.invalid",
    )

    result = HermesRuntimeAdapter(FakeHermesAgent(raw)).run_turn(make_request())

    assert dict(result.metadata) == {
        "turn_exit_reason": "text_response(final)",
        "failure_reason": "none",
        "error": "diagnostic",
        "interrupt_message": "not-used",
        "api_calls": 2,
        "partial": False,
    }


@pytest.mark.parametrize(
    ("raw", "expected_status"),
    [
        (
            normal_result(
                completed=False,
                failed=True,
                final_response="Provider failed.",
            ),
            TurnStatus.FAILED,
        ),
        (
            normal_result(
                completed=False,
                interrupted=True,
                final_response="Stopped by user.",
            ),
            TurnStatus.INTERRUPTED,
        ),
        (
            normal_result(completed=False, final_response=None),
            TurnStatus.FAILED,
        ),
    ],
)
def test_adapter_maps_non_returned_outcomes(
    raw: dict[str, object],
    expected_status: TurnStatus,
) -> None:
    result = HermesRuntimeAdapter(FakeHermesAgent(raw)).run_turn(make_request())

    assert result.status is expected_status


def test_interrupt_takes_precedence_over_failed_flag() -> None:
    raw = normal_result(completed=False, failed=True, interrupted=True)

    result = HermesRuntimeAdapter(FakeHermesAgent(raw)).run_turn(make_request())

    assert result.status is TurnStatus.INTERRUPTED


def test_failed_or_interrupted_result_may_omit_response() -> None:
    for flag in ("failed", "interrupted"):
        raw = normal_result(completed=False, final_response=None)
        raw[flag] = True

        result = HermesRuntimeAdapter(FakeHermesAgent(raw)).run_turn(
            make_request()
        )

        assert result.response is None


def test_result_provider_model_and_session_fall_back_to_agent() -> None:
    raw = normal_result()
    for key in ("provider", "model", "session_id"):
        raw.pop(key)
    agent = FakeHermesAgent(
        raw,
        session_id="session-agent",
        provider="provider-agent",
        model="model-agent",
    )

    result = HermesRuntimeAdapter(agent).run_turn(make_request())

    assert result.session_id == "session-agent"
    assert result.provider == "provider-agent"
    assert result.model == "model-agent"


def test_requested_session_must_match_configured_agent() -> None:
    agent = FakeHermesAgent(normal_result(), session_id="session-agent")

    with pytest.raises(AgentRuntimeError, match="session_id does not match"):
        HermesRuntimeAdapter(agent).run_turn(
            make_request(session_id="different-session")
        )

    assert agent.calls == []


def test_matching_requested_session_allows_rotation_in_result() -> None:
    agent = FakeHermesAgent(
        normal_result(session_id="rotated-session"),
        session_id="original-session",
    )

    result = HermesRuntimeAdapter(agent).run_turn(
        make_request(session_id="original-session")
    )

    assert result.session_id == "rotated-session"


def test_codex_app_server_is_rejected_before_execution() -> None:
    agent = FakeHermesAgent(
        normal_result(),
        api_mode="codex_app_server",
    )

    with pytest.raises(
        AgentRuntimeError,
        match="unsupported by the V1 Engineering Surface",
    ):
        HermesRuntimeAdapter(agent).run_turn(make_request())

    assert agent.calls == []


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not-a-mapping",
        {"completed": True, "final_response": None},
        {"completed": True, "final_response": "   "},
        {"completed": "yes", "final_response": "response"},
        {"final_response": "response"},
    ],
)
def test_malformed_hermes_result_is_a_contract_error(raw: object) -> None:
    with pytest.raises(AgentRuntimeError):
        HermesRuntimeAdapter(FakeHermesAgent(raw)).run_turn(make_request())


@pytest.mark.parametrize(
    "field",
    ["input_tokens", "output_tokens", "total_tokens"],
)
def test_invalid_usage_is_a_contract_error(field: str) -> None:
    raw = normal_result(**{field: -1})

    with pytest.raises(AgentRuntimeError, match=field):
        HermesRuntimeAdapter(FakeHermesAgent(raw)).run_turn(make_request())


def test_missing_usage_remains_optional() -> None:
    raw = normal_result()
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        raw.pop(key)

    result = HermesRuntimeAdapter(FakeHermesAgent(raw)).run_turn(make_request())

    assert result.usage is None


def test_runtime_exception_is_wrapped_with_cause() -> None:
    agent = FakeHermesAgent(normal_result())
    agent.error = OSError("provider unavailable")

    with pytest.raises(AgentRuntimeError) as captured:
        HermesRuntimeAdapter(agent).run_turn(make_request())

    assert isinstance(captured.value.__cause__, OSError)


@pytest.mark.parametrize(
    "error",
    [
        InterruptedError(),
        KeyboardInterrupt(),
        type("CancelledError", (BaseException,), {})(),
    ],
)
def test_runtime_interrupt_exceptions_become_interrupted_result(
    error: BaseException,
) -> None:
    agent = FakeHermesAgent(normal_result())
    agent.error = error

    result = HermesRuntimeAdapter(agent).run_turn(make_request(task_id="task-1"))

    assert result.status is TurnStatus.INTERRUPTED
    assert result.response is None
    assert result.task_id == "task-1"


def test_constructor_requires_run_conversation_capability() -> None:
    with pytest.raises(TypeError, match="run_conversation"):
        HermesRuntimeAdapter(object())  # type: ignore[arg-type]


def test_adapter_depends_on_no_vendor_sdk_or_engineering_gate() -> None:
    tree = ast.parse(inspect.getsource(hermes_adapter_module))
    forbidden_roots = {
        "anthropic",
        "hermes_cli",
        "openai",
        "tools",
    }
    forbidden_names = {
        "EngineeringOrchestrator",
        "EngineeringStore",
        "ReviewGate",
        "VerificationEngine",
        "WorkflowRun",
    }
    imported_roots: set[str] = set()
    imported_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.partition(".")[0] for alias in node.names
            )
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.partition(".")[0])
            imported_names.update(alias.name for alias in node.names)

    assert imported_roots.isdisjoint(forbidden_roots)
    assert imported_names.isdisjoint(forbidden_names)


def test_current_aiagent_surface_matches_adapter_dependency() -> None:
    tree = ast.parse(Path("run_agent.py").read_text(encoding="utf-8"))
    agent_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AIAgent"
    )
    run_conversation = next(
        node
        for node in agent_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_conversation"
    )
    positional_names = [argument.arg for argument in run_conversation.args.args]
    task_id_index = positional_names.index("task_id")
    first_default_index = len(positional_names) - len(
        run_conversation.args.defaults
    )
    task_id_default = run_conversation.args.defaults[
        task_id_index - first_default_index
    ]

    assert "user_message" in positional_names
    assert isinstance(task_id_default, ast.Constant)
    assert task_id_default.value is None
