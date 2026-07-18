from __future__ import annotations

from agent.chat_completion_helpers import _iteration_limit_summary_request


def test_kanban_iteration_limit_requests_compact_continuation_handoff() -> None:
    prompt = _iteration_limit_summary_request(is_kanban=True)

    assert "PARTIAL_UNVERIFIED_HANDOFF" in prompt
    for heading in (
        "COMPLETED",
        "CHANGED_FILES",
        "TESTS",
        "REMAINING",
        "RESUME",
        "INVARIANTS",
    ):
        assert heading in prompt
    assert "Do not claim PASS" in prompt
    assert "without calling any more tools" in prompt
    assert len(prompt) < 1600


def test_non_kanban_iteration_limit_keeps_general_summary_contract() -> None:
    prompt = _iteration_limit_summary_request(is_kanban=False)

    assert "summarizing what you've found and accomplished so far" in prompt
    assert "PARTIAL_UNVERIFIED_HANDOFF" not in prompt
