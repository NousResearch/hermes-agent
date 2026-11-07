"""``agent.single_tool_call_models``: one tool call per assistant message on the wire.

gpt-oss's chat template (Harmony) infers a tool result's function from the most recent assistant
tool call, and Azure AI Foundry's gpt-oss deployments reject a history in which one assistant
message carries several calls: HTTP 400 "Message has tool role, but there was no previous
assistant message with a tool call!". A primary model that made parallel calls (DeepSeek here)
then makes a fallback to gpt-oss fail on the first request and the turn is lost.

The replay below has the shape of that failed fallback request (roles, keys and call/result
layout of a real request dump; the contents are placeholders).
"""

import pytest

import agent.transports.chat_completions as cc
from agent.transports import get_transport


@pytest.fixture
def transport():
    return get_transport("chat_completions")


@pytest.fixture
def markers(monkeypatch):
    def _set(value):
        monkeypatch.setattr(cc, "_single_tool_call_models",
                            lambda: tuple(m.lower() for m in value))
    return _set


def _call(cid, name="terminal"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": '{"q": "x"}'}}


def _assistant(*calls, content="", reasoning="thinking"):
    msg = {"role": "assistant", "content": content, "tool_calls": list(calls)}
    if reasoning is not None:
        msg["reasoning"] = reasoning
        msg["reasoning_content"] = reasoning
    return msg


def _tool(cid, content="result"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def _incident_history():
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "re-skin the deck"},
        _assistant(_call("call_a")),
        _tool("call_a"),
        _assistant(_call("call_b", "vision_analyze"), _call("call_c", "vision_analyze")),
        _tool("call_b"),
        _tool("call_c"),
        {"role": "assistant", "content": "done", "reasoning": "r", "reasoning_content": "r"},
        {"role": "user", "content": "next"},
        _assistant(_call("call_d", "search_files"), _call("call_e", "search_files"), content="Looking."),
        _tool("call_d"),
        _tool("call_e"),
        _assistant(_call("call_f"), _call("call_g", "read_file"), content="Reading"),
        _tool("call_f"),
        _tool("call_g"),
    ]


def _assert_one_call_per_assistant_each_answered(messages):
    """The gpt-oss template's invariant: a tool message follows an assistant message with exactly
    one call (or another result of that same call), and answers that call."""
    last_call = None
    for msg in messages:
        if msg["role"] == "assistant":
            calls = msg.get("tool_calls") or []
            assert len(calls) <= 1, msg
            last_call = calls[0]["id"] if calls else None
        elif msg["role"] == "tool":
            assert last_call is not None, "tool message without a preceding assistant tool call"
            assert msg["tool_call_id"] == last_call
        else:
            last_call = None


def test_default_leaves_parallel_calls_as_they_are(transport, markers):
    markers([])
    history = _incident_history()
    out = transport.convert_messages(history, model="gpt-oss-120b")
    assert [len(m.get("tool_calls") or []) for m in out] == [len(m.get("tool_calls") or []) for m in history]


def test_listed_model_gets_one_call_per_assistant_message(transport, markers):
    markers(["gpt-oss"])
    history = _incident_history()
    out = transport.convert_messages(history, model="gpt-oss-120b")
    _assert_one_call_per_assistant_each_answered(out)
    # nothing lost: every call and every result is still there, in order
    ids = [m["tool_calls"][0]["id"] for m in out if m.get("tool_calls")]
    assert ids == [f"call_{c}" for c in "abcdefg"]
    assert [m["tool_call_id"] for m in out if m["role"] == "tool"] == ids
    # the first split keeps content and reasoning, the second is a bare call
    first, second = (m for m in out if m.get("tool_calls") and m["tool_calls"][0]["id"] in ("call_d", "call_e"))
    assert first["content"] == "Looking." and first["reasoning_content"] == "thinking"
    assert second == {"role": "assistant", "content": "", "tool_calls": [_call("call_e", "search_files")]}
    # stored history is never rewritten
    assert len(history[4]["tool_calls"]) == 2


def test_unlisted_model_is_untouched(transport, markers):
    markers(["gpt-oss"])
    history = _incident_history()
    out = transport.convert_messages(history, model="deepseek-v4.1-flash")
    assert any(len(m.get("tool_calls") or []) == 2 for m in out)


def test_fallback_request_is_built_split(transport, markers):
    """build_kwargs (every attempt, including a fallback, goes through it) sends the split copy."""
    markers(["gpt-oss"])
    kwargs = transport.build_kwargs(model="gpt-oss-120b", messages=_incident_history(), tools=None)
    _assert_one_call_per_assistant_each_answered(kwargs["messages"])


def test_result_out_of_order_and_stray_result(transport, markers):
    markers(["gpt-oss"])
    history = [
        {"role": "user", "content": "go"},
        _assistant(_call("call_1"), _call("call_2"), reasoning=None),
        _tool("call_2", "second"),
        _tool("call_1", "first"),
        _tool("call_zz", "stray"),
    ]
    out = transport.convert_messages(history, model="openai/GPT-OSS-20B")
    assert [(m["role"], m.get("tool_call_id")) for m in out] == [
        ("user", None), ("assistant", None), ("tool", "call_1"), ("assistant", None), ("tool", "call_2"),
        ("tool", "call_zz"),
    ]


def test_switch_read_from_config(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config",
                        lambda: {"agent": {"single_tool_call_models": ["GPT-OSS", " "]}})
    assert cc._single_tool_call_models() == ("gpt-oss",)
    assert cc._route_requires_single_tool_calls("gpt-oss-120b")
    assert not cc._route_requires_single_tool_calls("gpt-5.5")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"agent": {}})
    assert cc._single_tool_call_models() == ()
