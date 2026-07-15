from agent.model_authored_tool_args import (
    capture_model_authored_tool_args,
    restore_model_authored_tool_args,
)


def test_todo_payload_is_restored_exactly_without_content_interpretation():
    authored = {
        "todos": [{"id": "step-1", "content": "keep exact", "status": "pending"}],
        "canonical_checkpoint": {"case_id": "case:exact"},
        "goal_outcome": {"status": "continue", "reason": "more work"},
        "plan_approval": {"plan_id": "plan:exact"},
    }
    captured = capture_model_authored_tool_args("todo", authored)

    candidate = {
        "todos": [],
        "canonical_checkpoint": {"case_id": "case:forged"},
        "goal_outcome": {"status": "complete", "reason": "forged"},
        "delivery_outcome": {"action": "suppress", "reason": "injected"},
    }
    restored = restore_model_authored_tool_args("todo", candidate, captured)

    assert restored == authored
    assert restored is not authored
    restored["todos"][0]["content"] = "changed after restore"
    assert authored["todos"][0]["content"] == "keep exact"


def test_non_model_owned_tool_keeps_middleware_payload():
    candidate = {"query": "rewritten by an installed operator plugin"}
    assert capture_model_authored_tool_args("web_search", {"query": "original"}) is None
    assert (
        restore_model_authored_tool_args("web_search", candidate, None)
        is candidate
    )
