from agent.conversation_loop import _promote_content_tool_calls
from agent.transports.types import NormalizedResponse


def _msg(content, tool_calls=None, finish="stop"):
    return NormalizedResponse(content=content, tool_calls=tool_calls, finish_reason=finish)


def test_promotes_when_no_structured_calls():
    msg = _msg('<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>')
    _promote_content_tool_calls(msg, {"web_search"})
    assert msg.tool_calls
    assert msg.tool_calls[0].name == "web_search"
    assert msg.finish_reason == "tool_calls"
    assert "<tool_call>" not in (msg.content or "")


def test_noop_when_structured_calls_present():
    existing = [object()]
    msg = _msg("anything", tool_calls=existing, finish="tool_calls")
    _promote_content_tool_calls(msg, {"web_search"})
    assert msg.tool_calls is existing  # untouched → non-breaking


def test_noop_when_nothing_recognised():
    msg = _msg("a normal answer")
    _promote_content_tool_calls(msg, {"web_search"})
    assert not msg.tool_calls
    assert msg.finish_reason == "stop"
    assert msg.content == "a normal answer"


def test_noop_when_content_not_a_string():
    msg = _msg(None)
    _promote_content_tool_calls(msg, {"web_search"})
    assert not msg.tool_calls
    assert msg.finish_reason == "stop"


def test_promotion_disarms_glm_truncation_quirk():
    # run_agent._should_treat_stop_as_truncated fires only on
    # finish_reason=="stop" AND falsy tool_calls. Promotion flips both gates,
    # so a promoted Ollama-GLM content call can never be rewritten to "length".
    msg = _msg('<tool_call>{"name":"web_search","arguments":{}}</tool_call>')
    _promote_content_tool_calls(msg, {"web_search"})
    assert msg.finish_reason != "stop"
    assert msg.tool_calls
