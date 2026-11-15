"""Client-side recovery of leaked qwen3-XML tool calls for custom (vLLM) providers.

Symptom: a vLLM server running the qwen3_xml tool-call parser + reasoning parser leaks the
tool-call markup into the streamed assistant content as plain text — ``tool_call`` wrappers
containing ``function=NAME`` elements with ``parameter=KEY`` bodies — while returning zero
structured tool_calls. The agent shows the raw XML as visible text, runs the empty-response
ladder, and the turn dies. Real stream captures begin mid-block (dangling closing tags),
then repeat complete blocks.

Contract of ``parse_leaked_tool_call_blocks``:
- every complete ``function`` block (opened, ≥1 parameter pair, closed inside a closed
  wrapper) becomes an OpenAI-shaped tool call dict;
- parameter bodies are raw strings, JSON-parsed when the value is JSON;
- incomplete/dangling markup stays in ``remaining_text`` byte-unchanged;
- text without the markup passes through untouched with an empty list.

The intake wiring (``normalize_model_response`` for provider ``custom``) turns a recovered
round into a real tool round BEFORE the empty-response classification in the turn loop.
"""

from types import SimpleNamespace
from typing import Any
import json

from agent.leaked_tool_calls import parse_leaked_tool_call_blocks
from agent.turn_response_intake import normalize_model_response

# Tag literals are built from parts so this file never carries contiguous tool-call markup.
_OPEN = "<" + "tool_call>"
_CLOSE = "</" + "tool_call" + ">"
_FN_CLOSE = "</" + "function" + ">"
_P_CLOSE = "</" + "parameter" + ">"


def _fn(name: str) -> str:
    return "<" + "function=" + name + ">"


def _param(key: str, body: str) -> str:
    return "<" + "parameter=" + key + ">" + body + "</" + "parameter" + ">"


def _block(name: str, params: list[tuple[str, str]]) -> str:
    """One complete block, byte-shaped like the stream: each body wrapped in newlines."""
    inner = "".join(_param(k, "\n" + v + "\n") for k, v in params)
    return _OPEN + "\n" + _fn(name) + "\n" + inner + "\n" + _FN_CLOSE + "\n" + _CLOSE


_CMD = "uname -a && whoami && pwd && ls /home/ryoto/Entrena"

# The exact paste stream (paste_2_174024.txt): the capture began mid-block, so the text
# starts with a dangling tail; then 4 complete consecutive blocks, the same terminal
# command repeated (block 2 carries an extra timeout parameter).
_DANGLING = _P_CLOSE + "\n" + _param("limit", "\n200\n") + "\n" + _FN_CLOSE + "\n" + _CLOSE
_BLOCK_1 = _block("terminal", [("command", _CMD)])
_BLOCK_2 = _block("terminal", [("command", _CMD + " 2>&1 | head"), ("timeout", "60")])
PASTE = _DANGLING + _BLOCK_1 + _BLOCK_2 + _BLOCK_1 + _BLOCK_1


def test_exact_paste_sample():
    """Dangling head preserved, the 4 complete blocks recovered, no dedupe required."""
    calls, remaining = parse_leaked_tool_call_blocks(PASTE)
    assert len(calls) == 4
    for c in calls:
        assert c["type"] == "function"
        assert c["function"]["name"] == "terminal"
        assert c["id"]
    ids = [c["id"] for c in calls]
    assert len(set(ids)) == 4, "ids must be unique for call/result pairing"
    args = [json.loads(c["function"]["arguments"]) for c in calls]
    assert args[0] == {"command": _CMD}
    assert args[1] == {"command": _CMD + " 2>&1 | head", "timeout": 60}
    assert args[2] == args[3] == {"command": _CMD}
    # The dangling tail is untouched; every complete block is consumed.
    assert remaining == _DANGLING
    assert "<" + "function=" not in remaining


def test_clean_single_block():
    """One complete block: recovered, and no markup survives in the remaining text."""
    calls, remaining = parse_leaked_tool_call_blocks(_block("terminal", [("command", "ls -l")]))
    assert calls == [{
        "id": calls[0]["id"],
        "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps({"command": "ls -l"})},
    }]
    assert remaining == ""


def test_text_only_passthrough():
    """Text without the markup passes through byte-identical with an empty list."""
    text = "The answer is 42.\nNo markup here."
    calls, remaining = parse_leaked_tool_call_blocks(text)
    assert calls == []
    assert remaining == text


def test_malformed_block_preserved():
    """A truncated block (opened, never closed) stays in the text unchanged — never a call."""
    broken = (
        _OPEN + "\n" + _fn("terminal") + "\n" + _param("command", "\nuname -a\n")
        + "\n" + _FN_CLOSE
    )
    calls, remaining = parse_leaked_tool_call_blocks(broken)
    assert calls == []
    assert remaining == broken
    # A complete wrapper with zero parameter pairs is not a function call either.
    empty = _OPEN + "\n" + _fn("terminal") + "\n" + _FN_CLOSE + "\n" + _CLOSE
    calls, remaining = parse_leaked_tool_call_blocks(empty)
    assert calls == []
    assert remaining == empty


def test_parameter_values_json_when_json_string_otherwise():
    """Bodies are raw strings; JSON bodies parse, plain bodies stay strings."""
    calls, _ = parse_leaked_tool_call_blocks(
        _block("web_search", [("query", "\nhello world\n"), ("limit", "\n200\n"),
                              ("flag", "\ntrue\n")])
    )
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"query": "hello world", "limit": 200, "flag": True}


# --- intake wiring -----------------------------------------------------------------------


class _Transport:
    def normalize_response(self, response, strip_tool_prefix=False):
        return response


class _Agent:
    """Minimal stand-in exposing exactly what the intake path touches."""

    api_mode = "openai_chat"
    quiet_mode = True
    verbose_logging = False
    log_prefix = ""
    _delegate_depth = 0
    _incomplete_scratchpad_retries = 0
    _native_reasoning_streamed = False

    def __init__(self, provider: str = "custom", model: str = "qwen3"):
        self.provider = provider
        self.model = model
        self.tool_progress_callback: Any = None

    def _get_transport(self):
        return _Transport()

    def _vprint(self, *args, **kwargs):
        pass


def _run(agent, *, content, tool_calls=None):
    message = SimpleNamespace(
        content=content, finish_reason="stop", tool_calls=tool_calls,
        reasoning=None, reasoning_content=None, reasoning_details=None,
    )
    verdict = normalize_model_response(
        agent, response=message, messages=[], api_messages=[], conversation_history={},
        api_call_count=1, api_duration=0.0, api_start_time=0.0, api_request_id="r1",
        effective_task_id="t1", turn_id="t1",
    )
    return verdict, message


def test_custom_provider_leak_becomes_a_real_tool_round():
    """Provider 'custom' + empty tool_calls + function= marker: recovered as a tool round
    with finish_reason rebound, before the empty-response classification can fire."""
    agent = _Agent(provider="custom")
    verdict, _message = _run(agent, content=_block("terminal", [("command", "ls -l")]))
    assert verdict.action == "fallthrough"
    tcs = verdict.assistant_message.tool_calls
    assert len(tcs) == 1
    assert tcs[0].function.name == "terminal"
    assert json.loads(tcs[0].function.arguments) == {"command": "ls -l"}
    assert tcs[0].id
    assert verdict.assistant_message.content == ""
    assert verdict.finish_reason == "tool_calls"


def test_custom_provider_recovers_whole_paste_stream():
    """The 4-block stream lands as one 4-call tool round; the dangling tail stays visible text."""
    agent = _Agent(provider="custom")
    verdict, _message = _run(agent, content=PASTE)
    tcs = verdict.assistant_message.tool_calls
    assert len(tcs) == 4
    assert all(tc.function.name == "terminal" for tc in tcs)
    assert verdict.assistant_message.content == _DANGLING
    assert verdict.finish_reason == "tool_calls"


def test_non_custom_provider_untouched():
    """The recovery is scoped to provider 'custom'; other routes keep the raw text."""
    agent = _Agent(provider="openrouter")
    verdict, _message = _run(agent, content=_block("terminal", [("command", "ls -l")]))
    assert verdict.assistant_message.tool_calls is None
    assert verdict.assistant_message.content == _block("terminal", [("command", "ls -l")])
    assert verdict.finish_reason == "stop"


def test_custom_provider_with_structured_tool_calls_untouched():
    """When the provider DID emit structured tool_calls, the leak path stays out of it."""
    real = SimpleNamespace(id="call_0", type="function",
                           function=SimpleNamespace(name="terminal", arguments='{"command": "ls"}'))
    content = _block("terminal", [("command", "ls -l")])
    agent = _Agent(provider="custom")
    verdict, _message = _run(agent, content=content, tool_calls=[real])
    assert verdict.assistant_message.tool_calls == [real]
    assert verdict.assistant_message.content == content


def test_custom_provider_dangling_only_stays_text():
    """A stream cut mid-block (dangling markup, no complete block) is not a tool round."""
    agent = _Agent(provider="custom")
    verdict, _message = _run(agent, content=_DANGLING)
    assert verdict.assistant_message.tool_calls is None
    assert verdict.assistant_message.content == _DANGLING
    assert verdict.finish_reason == "stop"
