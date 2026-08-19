"""transform_llm_output must log the #64714 skipped-transform warning.

Mirrors the sibling ``transform_tool_result``/``transform_terminal_output``
tests: a valid-but-losing replacement (a second registered plugin whose
result would also have won, but the first-registered plugin already claimed
the turn) must surface in logs instead of being silently shadowed — the
gotcha issue #64714 was filed against for this exact hook.
"""

import logging

from agent.turn_finalizer import finalize_turn
from tests.agent.test_turn_finalizer_final_response_persistence import FakeAgent


def _run_finalize(monkeypatch, invoke_hook):
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoke_hook)
    agent = FakeAgent()
    messages = [{"role": "user", "content": "hi"}]

    return finalize_turn(
        agent,
        final_response="original",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="hi",
        original_user_message="hi",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )


def test_skipped_valid_results_log_runtime_warning(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
        result = _run_finalize(
            monkeypatch,
            invoke_hook=lambda name, **kw: (
                [None, "first", "second"] if name == "transform_llm_output" else []
            ),
        )
    assert result["final_response"] == "first"
    warnings = [r.getMessage() for r in caplog.records if "skipped" in r.getMessage()]
    assert len(warnings) == 1
    assert "skipped 1 valid" in warnings[0]

    # A lone winner is not a conflict: no warning.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="agent.conversation_loop"):
        result = _run_finalize(
            monkeypatch,
            invoke_hook=lambda name, **kw: (
                ["only"] if name == "transform_llm_output" else []
            ),
        )
    assert result["final_response"] == "only"
    assert not [r for r in caplog.records if "skipped" in r.getMessage()]
