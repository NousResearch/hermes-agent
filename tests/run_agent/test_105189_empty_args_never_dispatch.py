"""#105189: empty tool-call arguments must never be dispatched with synthesized `{}`.

A mid-tool-call stream drop can deliver zero argument bytes (routers may still
stamp a non-None ``finish_reason``). ``validate_tool_calls`` used to rewrite
``""`` to ``"{}"`` and return ``"ok"``, so a side-effecting tool
(e.g. ``browser_exec``) executed with fabricated empty args — bypassing the
``_parse_tool_arguments`` invariant (``agent/tool_executor.py``: invalid args
mean "tool was not executed").

Locked behavior: empty/whitespace args take the invalid-JSON path (bounded
retry, then error results surfaced to the model) and the call is never
dispatched with synthesized args.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_tool_validation import validate_tool_calls


def _tc(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _agent(**overrides):
    agent = MagicMock()
    agent.valid_tool_names = {"browser_exec", "read_file"}
    agent._invalid_tool_retries = 0
    agent._invalid_json_retries = 0
    agent._uniquify_tool_call_ids = MagicMock()
    agent._build_assistant_message = MagicMock(
        return_value={"role": "assistant", "content": "x"}
    )
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def _validate(agent, *tool_calls):
    assistant_message = SimpleNamespace(content="", tool_calls=list(tool_calls))
    messages = []
    verdict = validate_tool_calls(
        agent,
        assistant_message,
        "tool_calls",
        messages=messages,
        conversation_history=[],
        api_call_count=1,
        effective_task_id="task-1",
    )
    return verdict, assistant_message, messages


def test_empty_args_are_not_silently_rewritten_to_empty_object():
    agent = _agent()
    verdict, assistant_message, _ = _validate(
        agent, _tc("call-1", "browser_exec", "")
    )
    assert verdict.action != "ok"
    assert assistant_message.tool_calls[0].function.arguments == "", (
        "empty args must reach neither the handler as '{}' nor the transcript "
        "rewritten; they take the retry/error path instead"
    )


def test_whitespace_args_are_not_dispatched():
    agent = _agent()
    verdict, assistant_message, _ = _validate(
        agent, _tc("call-1", "browser_exec", "   ")
    )
    assert verdict.action != "ok"
    assert assistant_message.tool_calls[0].function.arguments == "   "


def test_empty_args_are_retried_not_refused_as_truncation():
    # "" carries no truncation evidence (it does not end mid-object); it must
    # take the bounded-retry path ("continue"), not the outright truncation
    # refusal ("return" with "Response truncated...").
    agent = _agent()
    verdict, _, _ = _validate(agent, _tc("call-1", "browser_exec", ""))
    assert verdict.action == "continue"
    assert verdict.result is None


def test_empty_args_retry_exhaustion_surfaces_error_to_model():
    agent = _agent(_invalid_json_retries=2)
    verdict, _, messages = _validate(agent, _tc("call-1", "browser_exec", ""))
    assert verdict.action == "continue"
    tool_results = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_results) == 1
    assert "empty object" in tool_results[0]["content"]
    assert agent._invalid_json_retries == 0


def test_valid_sibling_still_runs_after_empty_args_eventually_rejected():
    # Mixed-shape batch: the empty-args call must not poison validation of a
    # sibling with complete args — the whole batch takes the retry path and
    # neither call is dispatched with fabricated args yet.
    agent = _agent()
    verdict, assistant_message, _ = _validate(
        agent,
        _tc("call-1", "browser_exec", ""),
        _tc("call-2", "read_file", '{"path": "/tmp/foo"}'),
    )
    assert verdict.action == "continue"
    assert assistant_message.tool_calls[0].function.arguments == ""
    assert (
        assistant_message.tool_calls[1].function.arguments
        == '{"path": "/tmp/foo"}'
    )


def test_genuinely_truncated_args_still_refused_outright():
    # Non-empty args cut off mid-object keep the old outright-refusal path.
    agent = _agent()
    verdict, _, _ = _validate(
        agent, _tc("call-1", "browser_exec", '{"code": "# Fill email')
    )
    assert verdict.action == "return"
    assert "truncated" in (verdict.result or {}).get("final_response", "").lower()
