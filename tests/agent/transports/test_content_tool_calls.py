import dataclasses
from pathlib import Path

import pytest

from agent.transports.content_tool_calls import (
    FORMATS,
    RawCall,
    _deterministic_call_id,
    extract_content_tool_calls,
    find_bare_json_object,
    find_gemma_function,
    find_kimi_k2,
    find_minimax_invoke,
    find_tool_call_json,
)

FIX = Path("tests/fixtures/content_tool_calls")
VALID = {"web_search", "terminal"}


def test_rawcall_is_frozen():
    rc = RawCall(
        name="web_search", arguments={"q": "x"}, span="<tool_call>...</tool_call>"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rc.name = "other"


def test_deterministic_id_stable_and_prefixed():
    a = _deterministic_call_id("web_search", '{"q":"x"}', 0)
    b = _deterministic_call_id("web_search", '{"q":"x"}', 0)
    assert a == b
    assert a.startswith("call_")
    assert len(a) == len("call_") + 12


def test_registry_registered_after_import():
    names = {f.name for f in FORMATS}
    assert {
        "tool_call_json",
        "bare_json_object",
        "kimi_k2",
        "minimax_invoke",
        "gemma_function",
    } <= names


def test_tool_call_json_extracts():
    calls = find_tool_call_json((FIX / "tool_call_json.txt").read_text())
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "hermes nousresearch"}


def test_tool_call_json_ignores_prose():
    assert find_tool_call_json("talking about <tool_call> tags") == []


def test_tool_call_json_malformed_body_ignored():
    assert find_tool_call_json("<tool_call>{not valid json}</tool_call>") == []


def test_bare_json_whole_content_promotes():
    calls = find_bare_json_object((FIX / "bare_json_object.txt").read_text())
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "北京今天的天气"}


def test_bare_json_embedded_rejected():
    assert find_bare_json_object('here: {"name":"web_search","arguments":{}} ok') == []


def test_bare_json_extra_keys_rejected():
    assert (
        find_bare_json_object('{"name":"web_search","arguments":{},"description":"x"}')
        == []
    )


def test_bare_json_oversized_rejected():
    assert (
        find_bare_json_object(
            '{"name":"web_search","arguments":{"q":"' + "x" * 50000 + '"}}'
        )
        == []
    )


def test_kimi_k2_extracts():
    calls = find_kimi_k2((FIX / "kimi_k2_tokens.txt").read_text())
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "hermes nousresearch"}


def test_kimi_k2_absent_returns_empty():
    assert find_kimi_k2("normal answer") == []


def test_kimi_k2_multiple_calls_in_one_section():
    content = (
        "<|tool_calls_section_begin|>"
        '<|tool_call_begin|>functions.web_search:0<|tool_call_argument_begin|>{"query": "a"}<|tool_call_end|>'
        '<|tool_call_begin|>functions.terminal:1<|tool_call_argument_begin|>{"cmd": "ls"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )
    names = [c.name for c in find_kimi_k2(content)]
    assert names == ["web_search", "terminal"]


def test_minimax_invoke_extracts():
    calls = find_minimax_invoke((FIX / "minimax_invoke.txt").read_text())
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "hermes"}


def test_minimax_invoke_absent_returns_empty():
    assert find_minimax_invoke("plain text mentioning invoke") == []


def test_gemma_function_extracts():
    calls = find_gemma_function((FIX / "gemma_function.txt").read_text())
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "hermes"}


def test_gemma_function_prose_rejected():
    assert find_gemma_function("Use <function> in JavaScript to declare.") == []


def test_gemma_function_inline_sentence_rejected():
    # Promotion executes — a mid-sentence narrated call must not fire.
    assert (
        find_gemma_function(
            'First I will call: <function name="web_search">{"q": "x"}</function>'
        )
        == []
    )


def test_promotes_valid_name_only():
    calls, residual = extract_content_tool_calls(
        (FIX / "tool_call_json.txt").read_text(), VALID
    )
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == '{"query": "hermes nousresearch"}'
    assert calls[0].id.startswith("call_")
    assert "<tool_call>" not in residual


def test_promotes_multiple_kimi_calls():
    content = (
        "<|tool_calls_section_begin|>"
        '<|tool_call_begin|>functions.web_search:0<|tool_call_argument_begin|>{"query": "a"}<|tool_call_end|>'
        '<|tool_call_begin|>functions.terminal:1<|tool_call_argument_begin|>{"cmd": "ls"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )
    calls, _ = extract_content_tool_calls(content, VALID)
    assert [c.name for c in calls] == ["web_search", "terminal"]
    assert len({c.id for c in calls}) == 2  # distinct ids


def test_drops_unknown_tool_name():
    calls, residual = extract_content_tool_calls(
        '<tool_call>{"name":"definitely_not_a_tool","arguments":{}}</tool_call>', VALID
    )
    assert calls == []
    assert "<tool_call>" not in residual  # span still removed from residual


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("HERMES_PROMOTE_TOOLCALLS", "false")
    calls, residual = extract_content_tool_calls(
        (FIX / "tool_call_json.txt").read_text(), VALID
    )
    assert calls == []
    assert "<tool_call>" in residual  # disabled → content untouched


def test_bare_json_flag_opt_out(monkeypatch):
    monkeypatch.setenv("HERMES_PROMOTE_BARE_JSON_TOOLCALL", "false")
    calls, _ = extract_content_tool_calls('{"name":"web_search","arguments":{}}', VALID)
    assert calls == []


def test_overlapping_spans_deduped():
    # bare_json and tool_call_json could both fire; the same byte-range must
    # not be removed or counted twice.
    content = '<tool_call>{"name":"web_search","arguments":{}}</tool_call>'
    calls, residual = extract_content_tool_calls(content, VALID)
    assert len(calls) == 1
    assert residual.strip() == ""
