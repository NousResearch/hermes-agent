from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.continuation_engine import (
    DEFAULT_ITERATION_CAP,
    DEFAULT_MAX_RUNTIME_RESUMES,
    PROMISE_DONE_TAG,
    apply_bounded_continuation_engine,
    resolve_max_resumes,
    response_contains_promise_done,
)


@dataclass
class DummyChild:
    responses: list[dict[str, Any]]
    session_id: str = "sess-123"

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    def run_conversation(self, *, user_message: str) -> dict[str, Any]:
        self.calls.append(user_message)
        if not self.responses:
            raise AssertionError("No more queued responses")
        return self.responses.pop(0)


def _result(
    final_response: str = "working",
    *,
    outcome_status: str = "incomplete",
    stop_requested: bool = False,
    retry_requested: bool = False,
    active_todos: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if active_todos is None and outcome_status == "incomplete":
        active_todos = [{"id": "1", "content": "finish task", "status": "in_progress"}]
    return {
        "final_response": final_response,
        "api_calls": 1,
        "completed": outcome_status == "completed",
        "stopRequested": stop_requested,
        "retryRequested": retry_requested,
        "todoItems": active_todos or [],
    }


def test_response_contains_promise_done_accepts_only_exact_tag_marker():
    assert response_contains_promise_done(PROMISE_DONE_TAG) is True
    assert response_contains_promise_done(f"All set. {PROMISE_DONE_TAG}") is True

    assert response_contains_promise_done("DONE") is False
    assert response_contains_promise_done("<promise>done</promise>") is False
    assert response_contains_promise_done("<promise>DONE</promise >") is False
    assert response_contains_promise_done("<promise>DONE</ promise>") is False
    assert response_contains_promise_done("<promise>DONE</promise\n>") is False


def test_resolve_max_resumes_defaults_ralph_to_iteration_cap_but_preserves_ultrawork_default():
    assert resolve_max_resumes(runtime_mode="ralph", max_resumes=None, iteration_cap=7) == 7
    assert resolve_max_resumes(runtime_mode="ralph", max_resumes=None, iteration_cap=None) == DEFAULT_ITERATION_CAP
    assert resolve_max_resumes(runtime_mode="ultrawork", max_resumes=None, iteration_cap=7) == DEFAULT_MAX_RUNTIME_RESUMES
    assert resolve_max_resumes(runtime_mode="ralph", max_resumes=3, iteration_cap=7) == 3


def test_apply_bounded_continuation_engine_stops_ralph_on_exact_done_marker():
    child = DummyChild(
        responses=[
            _result(final_response=f"Finished successfully. {PROMISE_DONE_TAG}", outcome_status="incomplete"),
            _result(final_response="should never run", outcome_status="incomplete"),
        ]
    )

    state = apply_bounded_continuation_engine(
        child,
        _result(final_response="starting", outcome_status="incomplete"),
        runtime_mode="ralph",
        iteration_cap=5,
    )

    assert state["resume_count"] == 1
    assert len(child.calls) == 1
    assert state["exhausted"] is False
    assert state["result"]["final_response"].endswith(PROMISE_DONE_TAG)


def test_apply_bounded_continuation_engine_does_not_treat_malformed_done_as_exit():
    child = DummyChild(
        responses=[
            _result(final_response="Almost there <promise>DONE</promise >", outcome_status="incomplete"),
            _result(final_response=f"Now complete {PROMISE_DONE_TAG}", outcome_status="completed", active_todos=[]),
        ]
    )

    state = apply_bounded_continuation_engine(
        child,
        _result(final_response="starting", outcome_status="incomplete"),
        runtime_mode="ralph",
        max_resumes=2,
        iteration_cap=5,
    )

    assert state["resume_count"] == 2
    assert len(child.calls) == 2
    assert state["result"]["final_response"].startswith("Now complete")


def test_apply_bounded_continuation_engine_stops_when_stop_requested():
    child = DummyChild(
        responses=[
            _result(final_response="stop now", outcome_status="incomplete", stop_requested=True),
            _result(final_response="should never run", outcome_status="incomplete"),
        ]
    )

    state = apply_bounded_continuation_engine(
        child,
        _result(final_response="starting", outcome_status="incomplete"),
        runtime_mode="ralph",
        iteration_cap=5,
    )

    assert state["resume_count"] == 1
    assert len(child.calls) == 1
    assert state["snapshot"]["stopRequested"] is True
    assert state["exhausted"] is False


def test_apply_bounded_continuation_engine_exposes_continuation_state_for_persistence():
    child = DummyChild(
        responses=[
            _result(final_response="still working", outcome_status="incomplete"),
            _result(final_response=f"done {PROMISE_DONE_TAG}", outcome_status="completed", active_todos=[]),
        ]
    )

    state = apply_bounded_continuation_engine(
        child,
        _result(final_response="starting", outcome_status="incomplete"),
        runtime_mode="ralph",
        iteration_cap=2,
    )

    continuation_state = state["continuation_state"]
    assert continuation_state["mode"] == "ralph"
    assert continuation_state["resume_count"] == 2
    assert continuation_state["iteration_cap"] == 2
    assert continuation_state["max_resumes"] == 2
    assert continuation_state["snapshot_count"] == 3
    assert continuation_state["done"] is True
    assert continuation_state["stop_requested"] is False
    assert continuation_state["session_id"] == "sess-123"
    assert state["snapshot"]["continuation_state"] == continuation_state
    assert state["snapshots"][-1]["continuation_state"] == continuation_state


def test_apply_bounded_continuation_engine_uses_metis_gap_check_to_skip_complete_ralph_loop():
    child = DummyChild(
        responses=[
            _result(final_response="should never run", outcome_status="incomplete"),
        ]
    )

    state = apply_bounded_continuation_engine(
        child,
        _result(final_response="Implemented add module and write tests", outcome_status="incomplete"),
        runtime_mode="ralph",
        iteration_cap=5,
        gap_check_plan=["add module", "write tests"],
        gap_check_evidence={"completed": ["add module", "write tests"]},
        gap_check_role="verifier",
    )

    assert state["resume_count"] == 0
    assert child.calls == []
    assert state["gap_check"]["status"] == "complete"
    assert state["continuation_state"]["gap_complete"] is True
    assert state["exhausted"] is False


def test_apply_bounded_continuation_engine_uses_metis_missing_prompt_for_next_iteration():
    child = DummyChild(
        responses=[
            _result(final_response="Ran focused verification", outcome_status="completed", active_todos=[]),
        ]
    )

    state = apply_bounded_continuation_engine(
        child,
        _result(final_response="Created module", outcome_status="incomplete"),
        runtime_mode="ralph",
        max_resumes=1,
        iteration_cap=5,
        gap_check_plan=["Create module", "Run focused verification"],
        gap_check_evidence={"completed": ["Create module"]},
        gap_check_role="reviewer",
    )

    assert state["resume_count"] == 1
    assert "Run focused verification" in state["gap_check"]["missing_items"]
    assert "Run focused verification" in child.calls[0]
