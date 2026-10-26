"""``hermes doctor --turn``: opt-in only, and the verdict follows the turn's real outcome."""

from types import SimpleNamespace

import pytest

from hermes_cli import doctor_turn

_TODO_ROW = {"role": "tool", "name": "todo_list", "tool_name": "todo_list",
             "content": '{"todos": [{"content": "doctor check", "status": "pending"}]}'}


class _FakeAgent:
    provider, model, api_mode = "xai", "grok-x", "codex_responses"
    session_input_tokens, session_output_tokens = 100, 5

    def __init__(self, result):
        self._result, self.closed = result, False

    def run_conversation(self, _prompt):
        return self._result

    def close(self):
        self.closed = True


@pytest.mark.parametrize("result, expected", [
    # Reached todo_list through the deferred-tool dispatcher: the result row names the real tool.
    ({"final_response": "HERMES_DOCTOR_OK", "completed": True, "messages": [_TODO_ROW]}, "pass"),
    # Answered, never ran the tool: the route works but tool calling does not.
    ({"final_response": "HERMES_DOCTOR_OK", "completed": True, "messages": []}, "warn"),
    # A failed turn leaves an explanation in final_response; that text is not a pass.
    ({"final_response": "Provider said: HTTP 404 model not found", "failed": True, "completed": False,
      "error": "HTTP 404 model not found", "messages": [_TODO_ROW]}, "fail"),
    ({"final_response": "", "completed": True, "messages": []}, "fail"),
])
def test_turn_verdict_follows_the_turn_outcome(monkeypatch, result, expected):
    agent = _FakeAgent(result)
    monkeypatch.setattr(doctor_turn, "_build_agent", lambda: agent)
    issues: list[str] = []
    assert doctor_turn.run_turn_check(issues) == expected
    assert agent.closed
    assert bool(issues) == (expected == "fail")


def test_turn_check_never_runs_without_the_flag(monkeypatch):
    def _boom():
        raise AssertionError("a model call was made without --turn")

    monkeypatch.setattr(doctor_turn, "_build_agent", _boom)
    for args in (SimpleNamespace(fix=False, ack=None), SimpleNamespace(turn=False, live=True)):
        assert doctor_turn.maybe_run_turn_check(args, []) is None
