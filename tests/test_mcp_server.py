"""
Tests for gateway/mcp_server.py

Run with:
    python -m pytest tests/test_mcp_server.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gateway.mcp_server import HermesMCPServer, HermesToolBridge, HERMES_MCP_TOOLS


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

class TestToolSchema:
    def test_all_tools_have_required_fields(self):
        for tool in HERMES_MCP_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"].get("type") == "object"

    def test_tool_names_are_unique(self):
        names = [t["name"] for t in HERMES_MCP_TOOLS]
        assert len(names) == len(set(names))

    def test_expected_tools_present(self):
        names = {t["name"] for t in HERMES_MCP_TOOLS}
        expected = {
            "terminal", "read_file", "write_file",
            "web_search", "web_extract",
            "memory_read", "memory_write",
            "list_skills", "run_agent",
        }
        assert expected.issubset(names)

    def test_required_fields_are_lists(self):
        for tool in HERMES_MCP_TOOLS:
            if "required" in tool["inputSchema"]:
                assert isinstance(tool["inputSchema"]["required"], list)


# ---------------------------------------------------------------------------
# MCP protocol
# ---------------------------------------------------------------------------

class TestMCPProtocol:
    def setup_method(self):
        self.server = HermesMCPServer()

    def test_initialize(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = run(self.server.handle_request(req))
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "protocolVersion" in resp["result"]
        assert "capabilities" in resp["result"]
        assert resp["result"]["serverInfo"]["name"] == "hermes-agent"

    def test_tools_list(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        resp = run(self.server.handle_request(req))
        tools = resp["result"]["tools"]
        assert isinstance(tools, list)
        assert len(tools) > 0
        for t in tools:
            assert "name" in t
            assert "description" in t
            assert "inputSchema" in t

    def test_tools_list_matches_constant(self):
        req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
        resp = run(self.server.handle_request(req))
        assert resp["result"]["tools"] == HERMES_MCP_TOOLS

    def test_unknown_method_returns_error(self):
        req = {"jsonrpc": "2.0", "id": 4, "method": "foobar", "params": {}}
        resp = run(self.server.handle_request(req))
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_notification_returns_none(self):
        req = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        resp = run(self.server.handle_request(req))
        assert resp is None

    def test_ping(self):
        req = {"jsonrpc": "2.0", "id": 5, "method": "ping"}
        resp = run(self.server.handle_request(req))
        assert resp["result"] == {}

    def test_tools_call_terminal(self):
        req = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "terminal",
                "arguments": {"command": "echo hello-hermes"},
            },
        }
        resp = run(self.server.handle_request(req))
        assert resp["result"]["content"][0]["type"] == "text"
        assert "hello-hermes" in resp["result"]["content"][0]["text"]

    def test_tools_call_unknown_tool(self):
        req = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
        resp = run(self.server.handle_request(req))
        text = resp["result"]["content"][0]["text"]
        assert "error" in text.lower() or "unknown" in text.lower()


# ---------------------------------------------------------------------------
# Tool bridge
# ---------------------------------------------------------------------------

class TestHermesToolBridge:
    def setup_method(self):
        self.bridge = HermesToolBridge()

    def test_terminal_basic(self):
        result = run(self.bridge.call("terminal", {"command": "echo mcp-test"}))
        assert "mcp-test" in result

    def test_terminal_exit_code(self):
        result = run(self.bridge.call("terminal", {"command": "bash -c 'exit 1'"}))
        assert "exit 1" in result

    def test_terminal_timeout(self):
        result = run(self.bridge.call("terminal", {"command": "sleep 60", "timeout": 1}))
        assert "timed out" in result.lower()

    def test_write_and_read_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            write_result = run(self.bridge.call("write_file", {"path": path, "content": "hello MCP\n"}))
            assert "Wrote" in write_result
            read_result = run(self.bridge.call("read_file", {"path": path}))
            assert "hello MCP" in read_result
        finally:
            os.unlink(path)

    def test_read_file_not_found(self):
        result = run(self.bridge.call("read_file", {"path": "/nonexistent/path/file.txt"}))
        assert "error" in result.lower() or "not found" in result.lower()

    def test_write_file_append(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            run(self.bridge.call("write_file", {"path": path, "content": "line1\n"}))
            run(self.bridge.call("write_file", {"path": path, "content": "line2\n", "append": True}))
            content = run(self.bridge.call("read_file", {"path": path}))
            assert "line1" in content
            assert "line2" in content
        finally:
            os.unlink(path)

    def test_read_file_line_range(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\nline4\nline5\n")
            path = f.name
        try:
            result = run(self.bridge.call("read_file", {
                "path": path, "start_line": 2, "end_line": 3
            }))
            assert "line2" in result
            assert "line3" in result
            assert "line1" not in result
            assert "line4" not in result
        finally:
            os.unlink(path)

    def test_memory_read_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.bridge._hermes_home = tmpdir
            result = run(self.bridge.call("memory_read", {"section": "both"}))
            data = json.loads(result)
            assert "MEMORY.md" in data
            assert "USER.md" in data

    def test_memory_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.bridge._hermes_home = tmpdir
            run(self.bridge.call("memory_write", {
                "section": "memory",
                "key": "test_key",
                "value": "test_value",
                "action": "add",
            }))
            result = run(self.bridge.call("memory_read", {"section": "memory"}))
            assert "test_value" in json.loads(result)["MEMORY.md"]

    def test_memory_write_replace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.bridge._hermes_home = tmpdir
            run(self.bridge.call("memory_write", {
                "section": "memory", "key": "mykey", "value": "old_value", "action": "add",
            }))
            run(self.bridge.call("memory_write", {
                "section": "memory", "key": "mykey", "value": "new_value", "action": "replace",
            }))
            result = run(self.bridge.call("memory_read", {"section": "memory"}))
            d = json.loads(result)
            assert "new_value" in d["MEMORY.md"]
            assert "old_value" not in d["MEMORY.md"]

    def test_list_skills_no_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.bridge._hermes_home = tmpdir
            result = run(self.bridge.call("list_skills", {}))
            assert "no skills" in result.lower()

    def test_list_skills_with_skills(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.bridge._hermes_home = tmpdir
            skill_dir = os.path.join(tmpdir, "skills", "test-skill")
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
                f.write("---\nname: test-skill\ndescription: A test skill\nversion: 1.0.0\n---\n# Test\n")
            result = run(self.bridge.call("list_skills", {}))
            data = json.loads(result)
            assert len(data) == 1
            assert data[0]["name"] == "test-skill"
            assert data[0]["description"] == "A test skill"

    def test_unknown_tool(self):
        result = run(self.bridge.call("does_not_exist", {}))
        assert "unknown tool" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------------
# JSON-RPC format
# ---------------------------------------------------------------------------

class TestJSONRPCFormat:
    def setup_method(self):
        self.server = HermesMCPServer()

    def test_response_has_jsonrpc_field(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
        resp = run(self.server.handle_request(req))
        assert resp.get("jsonrpc") == "2.0"

    def test_response_id_matches_request(self):
        for req_id in [1, "abc", 42]:
            req = {"jsonrpc": "2.0", "id": req_id, "method": "ping"}
            resp = run(self.server.handle_request(req))
            assert resp["id"] == req_id

    def test_response_is_json_serializable(self):
        req = {"jsonrpc": "2.0", "id": 99, "method": "tools/list"}
        resp = run(self.server.handle_request(req))
        reparsed = json.loads(json.dumps(resp))
        assert reparsed["id"] == 99
