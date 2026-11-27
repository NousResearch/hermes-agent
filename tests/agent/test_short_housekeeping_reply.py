"""Recovered housekeeping replies remain successful even when short or transformed."""
import pytest

from tests.agent import test_reply_line_before_silence as cases

real_loop = cases.real_loop


@pytest.mark.parametrize("terminal", ["NO_REPLY", ""], ids=["marker", "empty"])
@pytest.mark.parametrize(("raw", "clean"), [("3pm", "3pm"), ("A\ud800B", "A\ufffdB")])
def test_short_recovered_reply_matches_history_and_receipt(real_loop, terminal, raw, clean):
    cases._register_todo(real_loop.agent)
    real_loop.agent._silence_marker_needs_reply = True
    result = real_loop.run([
        cases._response(raw, finish_reason="tool_calls", tool_calls=[
            cases._call("todo_list", {"todos": []}, "todo")]),
        cases._response(terminal),
    ], "reply")
    assert result["completed"] is True
    assert result["final_response"] == clean
    assert cases._stored_final(real_loop) == clean
    assert "final_assistant_row_id" in cases._receipt(real_loop.agent, result)
