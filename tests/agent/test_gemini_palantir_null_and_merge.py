"""Regression tests for Palantir/Vertex google-proxy Gemini fixes (2026-06-13).

Two independent bugs that made gemini-3-5-flash 422/400 opaquely on the
Foundry google proxy once a real multi-turn tool history was present:

1. NULL values inside functionResponse.response -> HTTP 422 (empty params).
   The proxy rejects any ``null`` (e.g. terminal's ``"error": null``). Native
   Google AI Studio tolerates it; only the Foundry proxy bites.

2. Parallel tool calls split into N separate user turns -> HTTP 400
   "the number of function response parts is equal to the number of function
   call parts of the function call turn". Gemini-native requires a model turn
   with N functionCalls to be answered by ONE user turn with N
   functionResponses.

Both verified empirically against the live proxy by replaying the exact
22-message / 37-tool request that failed in session 20260613_214056_c75da3.
"""
from agent.gemini_native_adapter import build_gemini_request, _strip_nulls


def test_strip_nulls_drops_none_values():
    assert _strip_nulls({"output": "x", "error": None, "exit_code": 0}) == {
        "output": "x",
        "exit_code": 0,
    }


def test_strip_nulls_recurses_into_nested():
    assert _strip_nulls({"a": {"b": None, "c": 1}, "d": [None, 2, {"e": None}]}) == {
        "a": {"c": 1},
        "d": [2, {}],
    }


def test_function_response_has_no_null_values():
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "1", "type": "function",
                 "function": {"name": "terminal", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "1", "name": "terminal",
         "content": '{"output": "ok", "exit_code": 0, "error": null}'},
    ]
    req = build_gemini_request(messages=msgs, tools=None)
    # Assert the invariant structurally, not via the serialized string: a
    # separators change (e.g. json.dumps(..., separators=(",", ":")) emitting
    # ``:null`` without a space) would silently defeat a ``": null"`` substring
    # count. Walk the parsed request and assert no ``None`` survives anywhere.
    def _assert_no_none(node, path="req"):
        if node is None:
            raise AssertionError(f"null leaked into request at {path}")
        if isinstance(node, dict):
            for k, v in node.items():
                _assert_no_none(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _assert_no_none(v, f"{path}[{i}]")

    _assert_no_none(req)
    fr = req["contents"][2]["parts"][0]["functionResponse"]["response"]
    assert "error" not in fr
    assert fr["output"] == "ok"


def test_parallel_tool_results_merge_into_single_user_turn():
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                {"id": "3", "type": "function", "function": {"name": "c", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "1", "name": "a", "content": '{"r": 1}'},
        {"role": "tool", "tool_call_id": "2", "name": "b", "content": '{"r": 2}'},
        {"role": "tool", "tool_call_id": "3", "name": "c", "content": '{"r": 3}'},
    ]
    req = build_gemini_request(messages=msgs, tools=None)
    roles = [c["role"] for c in req["contents"]]
    assert roles == ["user", "model", "user"], roles
    model_turn = req["contents"][1]
    user_turn = req["contents"][2]
    fc = [p for p in model_turn["parts"] if "functionCall" in p]
    fr = [p for p in user_turn["parts"] if "functionResponse" in p]
    assert len(fc) == 3
    assert len(fr) == 3  # balanced: N functionCalls -> N functionResponses in one turn


def test_separate_text_user_turn_not_merged_into_tool_results():
    """A normal text user message must NOT be merged into a functionResponse turn.

    Upstream (gemini-cli#28700 + #55125) now interposes a placeholder model
    turn between the functionResponse content and the human text content —
    two adjacent user contents are HTTP 400-unsafe, and folding the text in
    makes Gemini "continue the tool result" instead of answering.
    """
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "1", "type": "function", "function": {"name": "a", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "name": "a", "content": '{"r": 1}'},
        {"role": "user", "content": "now answer"},
    ]
    req = build_gemini_request(messages=msgs, tools=None)
    roles = [c["role"] for c in req["contents"]]
    # placeholder model turn keeps alternation valid; text stays its own user turn
    assert roles == ["user", "model", "user", "model", "user"], roles
    # last user turn is the text, not merged with the functionResponse turn
    assert any("text" in p for p in req["contents"][4]["parts"])
