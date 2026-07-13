"""Tests for the Hermes MCP bridge and the live-client tool socket.

Covers: OpenAI->MCP schema translation, JSON-RPC request handling, the parent
LiveToolServer <-> bridge ParentToolClient socket round-trip (executing a real
callback), overage guard, usage mapping, and delta/system-prompt extraction.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import hermes_mcp_bridge as bridge
from agent import claude_live_client as lc


# ---------------------------------------------------------------------------
# Schema translation
# ---------------------------------------------------------------------------


def test_translate_openai_tools_to_mcp():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
        {"type": "function", "function": {"name": "aaa", "description": "", "parameters": {}}},
    ]
    out = bridge.translate_openai_tools_to_mcp(tools)
    # Sorted by name for byte-stability.
    assert [t["name"] for t in out] == ["aaa", "read_file"]
    read = next(t for t in out if t["name"] == "read_file")
    assert read["inputSchema"]["properties"]["path"]["type"] == "string"
    assert "description" in read


def test_translate_skips_malformed():
    assert bridge.translate_openai_tools_to_mcp([{"nope": 1}, "x", {"function": {}}]) == []
    assert bridge.translate_openai_tools_to_mcp(None) == []


# ---------------------------------------------------------------------------
# JSON-RPC handling
# ---------------------------------------------------------------------------


def test_handle_initialize():
    resp = bridge.handle_request({"method": "initialize", "id": 1}, tools=[], parent=None)
    assert resp["result"]["serverInfo"]["name"] == "hermes"
    assert resp["result"]["protocolVersion"] == bridge.PROTOCOL_VERSION


def test_handle_initialized_notification_no_response():
    assert bridge.handle_request({"method": "notifications/initialized"}, tools=[], parent=None) is None


def test_handle_tools_list():
    tools = [{"name": "t", "description": "d", "inputSchema": {}}]
    resp = bridge.handle_request({"method": "tools/list", "id": 2}, tools=tools, parent=None)
    assert resp["result"]["tools"] == tools


def test_tools_call_forwards_to_parent():
    class FakeParent:
        def call(self, name, arguments):
            assert name == "read_file"
            assert arguments == {"path": "/x"}
            return {"content": "file body", "is_error": False}

    resp = bridge.handle_request(
        {"method": "tools/call", "id": 3, "params": {"name": "read_file", "arguments": {"path": "/x"}}},
        tools=[],
        parent=FakeParent(),
    )
    assert resp["result"]["content"][0]["text"] == "file body"
    assert resp["result"]["isError"] is False


def test_tools_call_parent_error_becomes_tool_error():
    class BoomParent:
        def call(self, name, arguments):
            raise OSError("socket gone")

    resp = bridge.handle_request(
        {"method": "tools/call", "id": 4, "params": {"name": "x", "arguments": {}}},
        tools=[],
        parent=BoomParent(),
    )
    assert resp["result"]["isError"] is True
    assert "socket gone" in resp["result"]["content"][0]["text"]


def test_tools_call_missing_name():
    resp = bridge.handle_request(
        {"method": "tools/call", "id": 5, "params": {}}, tools=[], parent=object()
    )
    assert resp["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# Socket round-trip: parent LiveToolServer <-> bridge ParentToolClient
# ---------------------------------------------------------------------------


def test_socket_roundtrip_executes_callback():
    calls = []

    def executor(name, arguments):
        calls.append((name, arguments))
        return f"ran {name} with {arguments.get('v')}", False

    server = lc.LiveToolServer(executor)
    server.start()
    try:
        client = bridge.ParentToolClient(server.socket_path, server.token)
        out = client.call("do_thing", {"v": 7})
        assert out["content"] == "ran do_thing with 7"
        assert out["is_error"] is False
        assert calls == [("do_thing", {"v": 7})]
    finally:
        server.close()


def test_socket_rejects_bad_token():
    server = lc.LiveToolServer(lambda n, a: ("ok", False))
    server.start()
    try:
        client = bridge.ParentToolClient(server.socket_path, "wrong-token")
        out = client.call("x", {})
        assert out["is_error"] is True
        assert "unauthorized" in out["content"]
    finally:
        server.close()


def test_socket_executor_exception_is_tool_error():
    def boom(name, arguments):
        raise ValueError("kaboom")

    server = lc.LiveToolServer(boom)
    server.start()
    try:
        client = bridge.ParentToolClient(server.socket_path, server.token)
        out = client.call("x", {})
        assert out["is_error"] is True
        assert "kaboom" in out["content"]
    finally:
        server.close()


# ---------------------------------------------------------------------------
# Overage guard
# ---------------------------------------------------------------------------


def test_check_overage_detects_using_overage():
    events = [
        {"type": "rate_limit_event", "rate_limit_info": {"isUsingOverage": False}},
        {"type": "rate_limit_event", "rate_limit_info": {"isUsingOverage": True, "rateLimitType": "tokens"}},
    ]
    info = lc.check_overage(events)
    assert info is not None
    assert info["rateLimitType"] == "tokens"


def test_check_overage_none_when_not_overage():
    assert lc.check_overage([{"rate_limit_info": {"isUsingOverage": False}}]) is None
    assert lc.check_overage([]) is None


# ---------------------------------------------------------------------------
# Usage mapping
# ---------------------------------------------------------------------------


def test_build_usage_maps_cache_tokens():
    usage = lc.ClaudeLiveClient._build_usage(
        {
            "input_tokens": 10,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 900,
            "output_tokens": 20,
        }
    )
    assert usage.prompt_tokens == 915
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 935
    assert usage.prompt_tokens_details.cached_tokens == 900


# ---------------------------------------------------------------------------
# Delta + system prompt extraction
# ---------------------------------------------------------------------------


def test_system_prompt_has_cache_boundary():
    text = lc._system_prompt_text([{"role": "system", "content": "be nice"}])
    assert "be nice" in text
    assert text.rstrip().endswith(lc._CACHE_BOUNDARY_MARKER.strip())


def test_delta_user_text_only_after_last_assistant():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply1"},
        {"role": "user", "content": "second"},
    ]
    delta = lc._delta_user_text(messages)
    assert delta == "second"
    assert "first" not in delta
    assert "sys" not in delta


def test_delta_first_turn_excludes_system():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    assert lc._delta_user_text(messages) == "hello"


# ---------------------------------------------------------------------------
# Auth scrub
# ---------------------------------------------------------------------------


def test_build_live_subprocess_env_scrubs_api_keys(monkeypatch):
    for key in lc._AUTH_SCRUB_KEYS:
        monkeypatch.setenv(key, "leak")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "keep-me")
    env = lc.build_live_subprocess_env()
    for key in lc._AUTH_SCRUB_KEYS:
        assert key not in env, f"{key} must be scrubbed"
    assert env.get("CLAUDE_CODE_OAUTH_TOKEN") == "keep-me"
    assert env.get("HOME")


# ---------------------------------------------------------------------------
# Live-mode toggle
# ---------------------------------------------------------------------------


def test_live_mode_default_on(monkeypatch):
    monkeypatch.delenv("HERMES_CLAUDE_CLI_LIVE", raising=False)
    assert lc.live_mode_enabled() is True


def test_live_mode_off_switch(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_LIVE", "0")
    assert lc.live_mode_enabled() is False
    monkeypatch.setenv("HERMES_CLAUDE_CLI_LIVE", "false")
    assert lc.live_mode_enabled() is False


# ---------------------------------------------------------------------------
# Full stdio bridge integration (real bridge subprocess; NO claude, NO network)
# ---------------------------------------------------------------------------


def test_bridge_subprocess_end_to_end(tmp_path):
    """Spawn the real hermes_mcp_bridge.py and drive the MCP handshake over
    stdio, with tools/call forwarded to a live LiveToolServer socket."""
    tools_file = tmp_path / "tools.json"
    tools_file.write_text(json.dumps([{"name": "echo", "description": "e", "inputSchema": {}}]))

    executed = []

    def executor(name, arguments):
        executed.append((name, arguments))
        return json.dumps({"echoed": arguments}), False

    server = lc.LiveToolServer(executor)
    server.start()
    bridge_path = str(Path(bridge.__file__))
    env = dict(os.environ)
    env["HERMES_MCP_BRIDGE_SOCKET"] = server.socket_path
    env["HERMES_MCP_BRIDGE_TOKEN"] = server.token
    env["HERMES_MCP_BRIDGE_TOOLS"] = str(tools_file)

    proc = subprocess.Popen(
        [sys.executable, bridge_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        def rpc(obj):
            proc.stdin.write(json.dumps(obj) + "\n")
            proc.stdin.flush()
            return json.loads(proc.stdout.readline())

        init = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert init["result"]["serverInfo"]["name"] == "hermes"

        listed = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert [t["name"] for t in listed["result"]["tools"]] == ["echo"]

        called = rpc(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"x": 1}},
            }
        )
        text = called["result"]["content"][0]["text"]
        assert json.loads(text) == {"echoed": {"x": 1}}
        assert executed == [("echo", {"x": 1})]
    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait(timeout=5)
        server.close()
