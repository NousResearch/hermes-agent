"""A deferred tool called by its bare name comes back from a strict server as answer text (#151373).

Servers that bind calls only for names in the request's ``tools[]`` return such a call as plain
content. Intake must turn it back into a structured call, so the turn runs it through the
``tool_call`` bridge instead of stripping the markup, reading the reply as empty and replaying a
byte-identical prompt until the retries run out.
"""
import json
from types import SimpleNamespace

import pytest

from agent.turn_response_intake import normalize_model_response
from agent.turn_response_intake_text_calls import recover_text_tool_calls
from run_agent import AIAgent

_BRIDGED = ("terminal", "tool_search", "tool_describe", "tool_call")


def _xml(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


_POLL = _xml("process_manage", action="poll", session_id="proc_x")
_POLL_ARGS = {"action": "poll", "session_id": "proc_x"}


class _Transport:
    def normalize_response(self, response, strip_tool_prefix=False):
        return response


class _IntakeAgent:
    """What ``normalize_model_response`` touches, on a chat-completions session with tool search on."""

    api_mode = "chat_completions"
    quiet_mode = True
    verbose_logging = False
    log_prefix = ""
    tool_progress_callback = None
    enabled_toolsets = disabled_toolsets = None
    tools: list = []
    _delegate_depth = 0
    _incomplete_scratchpad_retries = 0
    _native_reasoning_streamed = False

    def __init__(self, valid=_BRIDGED, api_mode="chat_completions"):
        self.valid_tool_names = set(valid)
        self.api_mode = api_mode
        self._extract_reasoning = lambda msg: None

    def _get_transport(self):
        return _Transport()

    def _vprint(self, *args, **kwargs):
        pass


def _message(content, tool_calls=None):
    return SimpleNamespace(content=content, finish_reason="stop", tool_calls=tool_calls, reasoning=None,
                           reasoning_content=None, reasoning_details=None)


def _intake(agent, message, messages=()):
    return normalize_model_response(
        agent, response=message, messages=list(messages), api_messages=[], conversation_history={},
        api_call_count=1, api_duration=0.0, api_start_time=0.0, api_request_id="r1",
        effective_task_id="t1", turn_id="t1",
    )


def _validate(message):
    """The real validation pass on an agent whose tool search deferred process_manage."""
    from agent.turn_tool_validation import validate_tool_calls

    agent = AIAgent.__new__(AIAgent)
    agent.provider, agent.model, agent.session_id, agent.tools = "custom", "m", "s1", []
    agent.valid_tool_names = set(_BRIDGED)
    agent.log_prefix, agent._base_url, agent.api_mode = "", "http://127.0.0.1:8000/v1", "chat_completions"
    agent._invalid_tool_retries = agent._invalid_json_retries = 0
    agent.enabled_toolsets, agent.disabled_toolsets = None, None
    agent._vprint = lambda *a, **k: None
    messages = []
    verdict = validate_tool_calls(agent, message, "tool_calls", messages=messages, conversation_history=[],
                                  api_call_count=1, effective_task_id="t1")
    return verdict, [m["content"] for m in messages if m.get("role") == "tool"]


def test_deferred_call_returned_as_text_runs_through_the_bridge():
    """#151373: the reply is a call, not empty markup, and validation routes it through ``tool_call``."""
    message = _message("Polling the process.\n\n" + _POLL)
    verdict = _intake(_IntakeAgent(), message)

    assert verdict.finish_reason == "tool_calls"
    assert message.content == "Polling the process."
    assert [tc.function.name for tc in message.tool_calls] == ["process_manage"]
    assert message.tool_calls[0].id

    verdict, errors = _validate(message)
    tc = message.tool_calls[0]
    assert verdict.action == "ok" and errors == []
    assert tc.function.name == "tool_call"
    assert json.loads(tc.function.arguments) == {"calls": [{"name": "process_manage", "arguments": _POLL_ARGS}]}


def test_declared_tools_and_json_form_keep_order_and_types():
    agent = _IntakeAgent()
    agent.tools = [{"type": "function", "function": {"name": "terminal", "parameters": {"properties": {
        "command": {"type": "string"}, "timeout": {"type": "integer"}}}}}]
    message = _message(_xml("terminal", command="42", timeout="30")
                       + '\n<tool_call>{"name": "terminal", "arguments": "{\\"command\\": \\"ls\\"}"}</tool_call>')
    assert recover_text_tool_calls(agent, message) == 2
    assert [json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
            for tc in message.tool_calls] == [{"command": "42", "timeout": 30}, {"command": "ls"}]
    assert len({tc.id for tc in message.tool_calls}) == 2


@pytest.mark.parametrize("content, why", [
    (_xml("made_up_tool", x="1"), "a name the session cannot call stays text"),
    ("The format:\n```xml\n" + _POLL + "\n```", "an example in a code fence"),
    ("Write `<tool_call><function=terminal></function></tool_call>` for it.", "inline code"),
    ("Calling.\n<tool_call>\n<function=terminal>\n<parameter=command>\nls", "cut off"),
    (_POLL + "\n<tool_call>\nnot a call\n</tool_call>", "one bad block leaves the reply alone"),
    ("<tool_call>\n<function=bad name>\n</function>\n</tool_call>", "a malformed name"),
    ("plain answer", "no block"),
])
def test_replies_that_are_not_calls_stay_untouched(content, why):
    message = _message(content)
    verdict = _intake(_IntakeAgent(), message)
    assert (message.tool_calls, message.content, verdict.finish_reason) == (None, content, "stop"), why


def test_a_call_copied_out_of_a_tool_result_is_data():
    """Quoting tool output (here stored as JSON, as terminal results are) never runs what it contains."""
    message = _message("The log says:\n" + _POLL)
    history = [{"role": "user", "content": "show the log"},
               {"role": "tool", "content": json.dumps({"output": "line 1\n" + _POLL + "\n", "exit_code": 0})}]
    assert recover_text_tool_calls(_IntakeAgent(), message, history) == 0
    assert message.tool_calls is None
    unrelated = _message("Polling.\n" + _POLL)
    assert recover_text_tool_calls(_IntakeAgent(), unrelated, [{"role": "tool", "content": "ok"}]) == 1


@pytest.mark.parametrize("agent, calls", [
    (_IntakeAgent(api_mode="anthropic_messages"), None),
    (_IntakeAgent(api_mode="codex_responses"), None),
    (_IntakeAgent(), [SimpleNamespace(id="c1", function=SimpleNamespace(name="terminal", arguments="{}"))]),
    (_IntakeAgent(valid=("terminal",)), None),  # tool search off: process_manage is not callable at all
])
def test_other_paths_are_unchanged(agent, calls):
    message = _message(_POLL, tool_calls=calls)
    assert recover_text_tool_calls(agent, message) == 0
    assert message.tool_calls is calls and message.content == _POLL
