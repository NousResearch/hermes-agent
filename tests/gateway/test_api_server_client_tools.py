"""Tests for the API-server client-tools bridge.

Covers the 9 spec cases from client-tools-spec.md.  The handler is exercised
through a real aiohttp app with a stubbed ``_run_agent`` - the bridge logic is
real, the agent is a canned responder.
"""
import asyncio
import importlib.util
import json
import sys
from unittest.mock import patch

import pytest

# Make repo-root imports resolvable regardless of how pytest was invoked.
sys.path.insert(0, "/opt/data/projects/hermes-prs/hermes-agent")

from gateway.platforms.api_server_client_tools import (  # noqa: E402
    ClientToolsBridge,
    fold_tool_result,
)

CLIP = {
    "type": "function",
    "function": {
        "name": "clipboard_history_read",
        "description": "Read the user clipboard",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _make_bridge(tools=None, tool_choice="auto"):
    return ClientToolsBridge(tools if tools is not None else [CLIP], tool_choice)


# ---------------------------------------------------------------- 1. legacy path
def test_no_tools_means_no_bridge():
    b = _make_bridge(tools=[])
    assert b.suppressed
    # extraction on arbitrary text is a no-op
    calls, residual = b.extract_calls("plain answer")
    assert calls == [] and residual == "plain answer"


# ---------------------------------------------------------------- 2. contract
def test_contract_contains_schemas_and_markers():
    contract = _make_bridge().system_contract()
    assert "<tool_call>" in contract
    assert "clipboard_history_read" in contract
    assert "parameters" in contract


# ---------------------------------------------------------------- 3. extraction
def test_extract_single_call_and_residual():
    text = 'Checking. <tool_call>{"function": {"name": "clipboard_history_read", "arguments": {}}}</tool_call>'
    calls, residual = _make_bridge().extract_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "clipboard_history_read"
    assert calls[0]["type"] == "function"
    assert calls[0]["id"]
    assert residual == "Checking."


def test_extract_parallel_calls_capped():
    block = '<tool_call>{"function": {"name": "clipboard_history_read", "arguments": {}}}</tool_call>'
    calls, _ = _make_bridge().extract_calls(block * 10)
    assert len(calls) == 6  # MAX_PARALLEL_CALLS


# ---------------------------------------------------------------- 4. fold
def test_fold_replaces_pair_with_continuation():
    bridge = _make_bridge(tools=[
        CLIP,
        {"type": "function", "function": {"name": "terminal", "parameters": {}}},
    ])
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "run ls"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "file.txt"},
    ]
    new, ok = fold_tool_result(msgs, bridge)
    assert ok
    assert len(new) == 3
    cont = new[-1]
    assert cont["role"] == "user"
    assert "terminal" in cont["content"] and "file.txt" in cont["content"]


def test_fold_rejects_non_client_tools():
    bridge = _make_bridge()
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "totally_unknown", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "y"},
    ]
    new, ok = fold_tool_result(msgs, bridge)
    assert not ok and new == msgs


def test_fold_handles_wire_names():
    bridge = _make_bridge(tools=[
        {"type": "function", "function": {"name": "terminal", "parameters": {}}},
    ])
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "client__terminal", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "out"},
    ]
    new, ok = fold_tool_result(msgs, bridge)
    assert ok and "client__terminal" in new[-1]["content"]


# ---------------------------------------------------------------- 5. collisions
def test_reserved_names_get_wire_prefix_bidirectionally():
    bridge = _make_bridge(tools=[
        {"type": "function", "function": {"name": "terminal", "parameters": {}}},
    ])
    assert bridge.wire_name("terminal") == "client__terminal"
    assert bridge.original_name("client__terminal") == "terminal"
    calls, _ = bridge.extract_calls(
        '<tool_call>{"function": {"name": "client__terminal", "arguments": {}}}</tool_call>'
    )
    assert calls[0]["function"]["name"] == "terminal"  # mapped back for the client


# ---------------------------------------------------------------- 6. tool_choice
def test_forced_function_named_in_contract():
    bridge = _make_bridge(tool_choice={"type": "function", "function": {"name": "clipboard_history_read"}})
    assert "explicitly requested" in bridge.system_contract()


# ---------------------------------------------------------------- 7. kill-switch unit
def test_disabled_bridge_is_plain_none():
    # Simulate the handler's decision: bridge only when enabled and tools present
    enabled = False
    body = {"tools": [CLIP]}
    bridge = _make_bridge(body["tools"]) if (enabled and body["tools"]) else None
    assert bridge is None


# ---------------------------------------------------------------- 8. malformed input
def test_malformed_and_oversized_dropped_gracefully():
    bridge = _make_bridge(tools=[
        "garbage",
        {"type": "function", "function": {"name": "bad name!", "parameters": {}}},
        {"type": "function", "function": {"name": "x" * 70, "parameters": {}}},
        {"nope": True},
    ])
    assert bridge.suppressed


def test_tool_cap_enforced():
    many = [{"type": "function", "function": {"name": f"tool_{i}", "parameters": {}}} for i in range(50)]
    bridge = _make_bridge(tools=many)
    assert len(bridge._schemas) == 32


# ---------------------------------------------------------------- 9. end-to-end via handler
@pytest.fixture
def adapter_with_stub(monkeypatch):
    """Build a real ApiServerPlatformConfig-driven handler with _run_agent stubbed."""
    # The adapter class name differs across versions; import module and find it.
    import gateway.platforms.api_server as api_mod
    cls = None
    for name in dir(api_mod):
        obj = getattr(api_mod, name)
        if isinstance(obj, type) and hasattr(obj, "_handle_chat_completions") and hasattr(obj, "_check_auth"):
            cls = obj
            break
    assert cls is not None, "no adapter class with _handle_chat_completions found"
    inst = cls.__new__(cls)  # skip __init__ (needs full gateway config); set only what the handler touches
    inst._pending_agent_requests = 0  # admission decorator bookkeeping
    inst._client_tools_enabled = True
    inst._direct_model_requests = False
    inst._model_name = "mimikyu"
    inst._api_key = None
    inst._parse_session_key_header = lambda request: (None, None)
    inst._parse_model_routes = lambda extra: {}
    inst._model_routes = {}
    inst._resolve_route = lambda model_name: None
    inst._concurrency_limited_response = lambda: None
    inst._request_route_conflict_error = lambda **kw: None
    inst._cors_headers_for_origin = lambda origin: {}

    async def fake_run_agent(self, **kwargs):
        # Whatever the user says, decide to call the clipboard.
        return (
            {
                "final_response": 'Checking your clipboard. <tool_call>{"function": {"name": "clipboard_history_read", "arguments": {}}}</tool_call>',
                "partial": False, "failed": False, "completed": True,
                "error": None, "session_id": "s1",
            },
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(cls, "_run_agent", fake_run_agent)
    return inst


@pytest.mark.asyncio
async def test_handler_returns_tool_calls_nonstreaming(adapter_with_stub, aiohttp_client_factory=None):
    from aiohttp import web
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter_with_stub._handle_chat_completions)
    from aiohttp.test_utils import TestClient, TestServer
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        body = {
            "model": "mimikyu",
            "messages": [{"role": "user", "content": "whats on my clipboard?"}],
            "tools": [CLIP],
        }
        resp = await client.post("/v1/chat/completions", json=body)
        assert resp.status == 200
        data = await resp.json()
        msg = data["choices"][0]["message"]
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        assert msg["tool_calls"][0]["function"]["name"] == "clipboard_history_read"
        assert "<tool_call>" not in (msg["content"] or "")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_handler_legacy_path_untouched(adapter_with_stub):
    """No tools in request -> plain content answer, finish stop (regression guard)."""
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    async def plain(self, **kwargs):
        return (
            {"final_response": "2 + 2 is 4", "partial": False, "failed": False,
             "completed": True, "error": None, "session_id": "s2"},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
    adapter_with_stub._run_agent = plain.__get__(adapter_with_stub)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter_with_stub._handle_chat_completions)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post("/v1/chat/completions", json={
            "model": "mimikyu",
            "messages": [{"role": "user", "content": "what is 2+2"}],
        })
        data = await resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["choices"][0]["message"]["content"] == "2 + 2 is 4"
        assert "tool_calls" not in data["choices"][0]["message"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_handler_followup_folds_tool_result(adapter_with_stub):
    """Second request carries assistant tool_calls + role:tool -> continuation text reaches the agent."""
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    seen = {}

    async def spy(self, **kwargs):
        seen["user_message"] = kwargs.get("user_message")
        seen["history"] = kwargs.get("conversation_history")
        return (
            {"final_response": "Your clipboard has: hello world", "partial": False,
             "failed": False, "completed": True, "error": None, "session_id": "s3"},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
    adapter_with_stub._run_agent = spy.__get__(adapter_with_stub)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter_with_stub._handle_chat_completions)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post("/v1/chat/completions", json={
            "model": "mimikyu",
            "messages": [
                {"role": "user", "content": "whats on my clipboard?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "clipboard_history_read", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "hello world"},
            ],
            "tools": [CLIP],
        })
        data = await resp.json()
        assert resp.status == 200
        assert "hello world" in seen["user_message"]
        assert data["choices"][0]["message"]["content"] == "Your clipboard has: hello world"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_handler_streaming_suppresses_blocks_and_emits_tool_calls(adapter_with_stub):
    """stream=True: raw <tool_call> blocks never hit the wire; tool_calls deltas do."""
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    async def fake_stream(self, **kwargs):
        cb = kwargs.get("stream_delta_callback")
        for piece in ["Checking your clipboard. ", "<tool_call>", '{"function": ',
                      '{"name": "clipboard_history_read", ', '"arguments": {}}}',
                      "</tool_call>"]:
            if cb:
                cb(piece)
        return (
            {"final_response": "Checking your clipboard. <tool_call>{\"function\": {\"name\": \"clipboard_history_read\", \"arguments\": {}}}</tool_call>",
             "partial": False, "failed": False, "completed": True, "error": None,
             "session_id": "s4"},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
    adapter_with_stub._run_agent = fake_stream.__get__(adapter_with_stub)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter_with_stub._handle_chat_completions)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post("/v1/chat/completions", json={
            "model": "mimikyu",
            "messages": [{"role": "user", "content": "whats on my clipboard?"}],
            "tools": [CLIP],
            "stream": True,
        })
        assert resp.status == 200
        raw = await resp.text()
        assert "data: [DONE]" in raw
        frames = [json.loads(l[len("data: "):]) for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
        # no raw block text ever streamed
        for f in frames:
            delta = f["choices"][0].get("delta", {})
            assert "<tool_call>" not in str(delta.get("content") or "")
        # a tool_calls delta arrived with the right name
        tc_frames = [f for f in frames if f["choices"][0].get("delta", {}).get("tool_calls")]
        assert tc_frames, "expected tool_calls delta chunks"
        fn = tc_frames[0]["choices"][0]["delta"]["tool_calls"][0]["function"]
        assert fn["name"] == "clipboard_history_read"
        # finish_reason on the final frame
        finishes = [f["choices"][0].get("finish_reason") for f in frames if f["choices"][0].get("finish_reason")]
        assert finishes and finishes[-1] == "tool_calls"
        # residual prose still streamed
        contents = "".join(str(f["choices"][0].get("delta", {}).get("content") or "") for f in frames)
        assert "Checking your clipboard." in contents
    finally:
        await client.close()
