"""Core-facing request builders for plan and final review checkpoints."""

from types import SimpleNamespace

from agent.review_checkpoints import (
    create_review_checkpoint_runtime,
    review_final_checkpoint,
    review_tool_checkpoint,
)
from agent.review_runner import ReviewResult


def _runtime(calls, *, enabled=True):
    def run(request):
        calls.append(request)
        return ReviewResult(
            checkpoint_id=request.checkpoint_id,
            status="completed",
            verdict="PASS",
        )

    return create_review_checkpoint_runtime(
        session_id="session-1",
        provider="openai-codex",
        model="gpt-review",
        enabled=enabled,
        run_review_fn=run,
    )


def _agent(runtime):
    return SimpleNamespace(
        review_checkpoint_runtime=runtime,
        provider="openai-codex",
        model="gpt-economy",
    )


def _tool(name, arguments):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arguments)
    )


def test_tool_checkpoint_sends_names_effects_and_argument_keys_not_values():
    calls = []
    decision = review_tool_checkpoint(
        _agent(_runtime(calls)),
        turn_id="turn-1",
        attempt=0,
        user_message="Update the file",
        assistant_content="I will inspect then update it.",
        tool_calls=[
            _tool("read_file", '{"path":"C:/private/secret.txt"}'),
            _tool("write_file", '{"path":"C:/private/secret.txt","content":"secret"}'),
        ],
    )

    assert decision.action == "continue"
    request = calls[0]
    assert request.phase == "plan"
    assert request.checkpoint_id == "turn-1:plan:0"
    assert request.provider == "openai-codex"
    assert request.model == "gpt-review"
    assert request.main_model == "gpt-economy"
    assert request.candidate["actions"] == [
        {
            "tool": "read_file",
            "effect": "read",
            "argument_keys": ["path"],
            "redacted_arguments": {},
        },
        {
            "tool": "write_file",
            "effect": "state_change",
            "argument_keys": ["content", "path"],
            "redacted_arguments": {},
        },
    ]
    assert "C:/private" not in str(request.candidate)
    assert '"secret"' not in str(request.candidate)


def test_final_checkpoint_holds_bounded_candidate_and_evidence():
    calls = []
    decision = review_final_checkpoint(
        _agent(_runtime(calls)),
        turn_id="turn-1",
        attempt=1,
        user_message="Finish the task",
        final_response="Completed safely.",
        evidence=["36 tests passed"],
    )

    assert decision.action == "continue"
    request = calls[0]
    assert request.phase == "final"
    assert request.attempt == 1
    assert request.candidate == {
        "summary": "Completed safely.",
        "evidence": ["36 tests passed"],
    }


def test_disabled_runtime_is_zero_call_noop_at_both_seams():
    calls = []
    agent = _agent(_runtime(calls, enabled=False))

    plan = review_tool_checkpoint(
        agent,
        turn_id="turn-1",
        attempt=0,
        user_message="Do it",
        assistant_content="",
        tool_calls=[_tool("write_file", "{}")],
    )
    final = review_final_checkpoint(
        agent,
        turn_id="turn-1",
        attempt=0,
        user_message="Do it",
        final_response="Done",
    )

    assert plan.action == "continue"
    assert final.action == "continue"
    assert calls == []
