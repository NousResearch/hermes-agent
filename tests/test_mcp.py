"""
Tests for MCP (Model Context Protocol) client integration.

Covers:
  - Tool name conversion and sanitization
  - MCP inputSchema -> hermes parameters conversion
  - MCPServerConfig parsing (stdio, http, disabled, unknown)
  - MCPClient protocol (mock transport: connect, list_tools, call_tool, errors)
  - MCPManager config loading and tool registration
  - Handler routing (success, error, disconnected server)
  - Meta-tool actions (status, reconnect, list_tools, unknown)

Run with: python -m pytest tests/test_mcp.py -v
"""

import json
import os
import sys
import types
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies that tools/__init__.py eagerly imports.
# ---------------------------------------------------------------------------
_OPTIONAL_DEPS = [
    "firecrawl", "fal_client", "browserbase", "playwright",
    "agent", "agent.auxiliary_client", "agent.display",
]
for _dep in _OPTIONAL_DEPS:
    if _dep not in sys.modules:
        sys.modules[_dep] = types.ModuleType(_dep)

_agent_aux = sys.modules.setdefault("agent.auxiliary_client", types.ModuleType("agent.auxiliary_client"))
_agent_aux.get_text_auxiliary_client = lambda: (None, "stub-model")
_agent_aux.get_vision_auxiliary_client = lambda: (None, "stub-vision-model")

_firecrawl = sys.modules.setdefault("firecrawl", types.ModuleType("firecrawl"))
_firecrawl.Firecrawl = MagicMock

_debug_mod = types.ModuleType("tools.debug_helpers")
class _StubDebugSession:
    def __init__(self, *a, **kw):
        self.active = False
        self.session_id = "test"
        self.log_dir = "/tmp"
    def log_call(self, *a, **kw): pass
    def save(self, *a, **kw): pass
    def get_session_info(self): return {}
_debug_mod.DebugSession = _StubDebugSession
sys.modules.setdefault("tools.debug_helpers", _debug_mod)

# Now safe to import
from tools.mcp_client import (
    MCPClient,
    StdioTransport,
    HttpTransport,
    MCPTransportError,
    MCPProtocolError,
    MCP_PROTOCOL_VERSION,
)
from tools.mcp_manager import (
    MCPManager,
    MCPServerConfig,
    MCPServerConnection,
    make_tool_name,
    _sanitize_name,
    _convert_mcp_schema,
)


# ===========================================================================
# Tool name conversion
# ===========================================================================

class TestToolNameConversion(unittest.TestCase):
    """Test tool name generation and sanitization."""

    def test_basic_name(self):
        self.assertEqual(make_tool_name("github", "create_issue"), "mcp_github_create_issue")

    def test_hyphenated_name(self):
        self.assertEqual(make_tool_name("my-server", "list-items"), "mcp_my_server_list_items")

    def test_special_chars(self):
        self.assertEqual(make_tool_name("srv.1", "tool@2"), "mcp_srv_1_tool_2")

    def test_uppercase(self):
        self.assertEqual(make_tool_name("GitHub", "CreateIssue"), "mcp_github_createissue")

    def test_sanitize_empty(self):
        self.assertEqual(_sanitize_name(""), "")

    def test_sanitize_consecutive_underscores(self):
        self.assertEqual(_sanitize_name("a__b___c"), "a_b_c")

    def test_sanitize_leading_trailing(self):
        self.assertEqual(_sanitize_name("__name__"), "name")


# ===========================================================================
# Schema conversion
# ===========================================================================

class TestSchemaConversion(unittest.TestCase):
    """Test MCP inputSchema to hermes parameters conversion."""

    def test_basic_conversion(self):
        mcp_tool = {
            "name": "create_issue",
            "description": "Create a GitHub issue",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["title"],
            },
        }
        schema = _convert_mcp_schema(mcp_tool, "github")
        self.assertEqual(schema["name"], "mcp_github_create_issue")
        self.assertIn("[MCP:github]", schema["description"])
        self.assertEqual(schema["parameters"]["type"], "object")
        self.assertIn("title", schema["parameters"]["properties"])

    def test_empty_input_schema(self):
        mcp_tool = {
            "name": "ping",
            "description": "Ping server",
            "inputSchema": {},
        }
        schema = _convert_mcp_schema(mcp_tool, "test")
        self.assertEqual(schema["parameters"]["type"], "object")
        self.assertEqual(schema["parameters"]["properties"], {})

    def test_missing_input_schema(self):
        mcp_tool = {
            "name": "no_schema",
            "description": "Tool without schema",
        }
        schema = _convert_mcp_schema(mcp_tool, "test")
        self.assertEqual(schema["parameters"]["type"], "object")

    def test_description_fallback(self):
        mcp_tool = {"name": "foo"}
        schema = _convert_mcp_schema(mcp_tool, "bar")
        self.assertIn("foo", schema["description"])


# ===========================================================================
# MCPServerConfig parsing
# ===========================================================================

class TestMCPServerConfig(unittest.TestCase):
    """Test config parsing for different server types."""

    def test_stdio_config(self):
        cfg = MCPServerConfig.from_dict("fs", {
            "command": "npx",
            "args": ["-y", "@mcp/server-filesystem", "/tmp"],
            "env": {"NODE_ENV": "production"},
        })
        self.assertEqual(cfg.name, "fs")
        self.assertEqual(cfg.transport_type, "stdio")
        self.assertEqual(cfg.command, "npx")
        self.assertEqual(len(cfg.args), 3)
        self.assertTrue(cfg.enabled)

    def test_http_config(self):
        cfg = MCPServerConfig.from_dict("api", {
            "url": "https://mcp.example.com/api",
            "headers": {"Authorization": "Bearer token"},
        })
        self.assertEqual(cfg.transport_type, "http")
        self.assertEqual(cfg.url, "https://mcp.example.com/api")
        self.assertIn("Authorization", cfg.headers)

    def test_disabled_config(self):
        cfg = MCPServerConfig.from_dict("disabled", {
            "command": "npx",
            "args": [],
            "enabled": False,
        })
        self.assertFalse(cfg.enabled)

    def test_unknown_config(self):
        cfg = MCPServerConfig.from_dict("bad", {"foo": "bar"})
        self.assertEqual(cfg.transport_type, "unknown")
        self.assertFalse(cfg.enabled)

    def test_auto_connect_default(self):
        cfg = MCPServerConfig.from_dict("test", {"command": "echo"})
        self.assertTrue(cfg.auto_connect)

    def test_auto_connect_disabled(self):
        cfg = MCPServerConfig.from_dict("test", {
            "command": "echo",
            "auto_connect": False,
        })
        self.assertFalse(cfg.auto_connect)


# ===========================================================================
# MCPClient protocol (mock transport)
# ===========================================================================

class MockTransport:
    """Mock transport that returns preset responses."""

    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self._sent = []
        self._started = False
        self._connected = True

    @property
    def is_connected(self):
        return self._connected

    def start(self):
        self._started = True

    def stop(self):
        self._connected = False

    def send(self, message):
        self._sent.append(message)

    def receive(self, timeout=60):
        if self._responses:
            return self._responses.pop(0)
        raise MCPTransportError("No more mock responses")


class TestMCPClientProtocol(unittest.TestCase):
    """Test MCPClient connect/list_tools/call_tool with mock transport."""

    def test_connect(self):
        transport = MockTransport([
            # initialize response
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "test-server", "version": "1.0"},
                },
            },
        ])
        client = MCPClient(transport)
        result = client.connect()
        self.assertTrue(client.is_connected)
        self.assertEqual(result["serverInfo"]["name"], "test-server")
        self.assertEqual(client.server_name, "test-server")

    def test_list_tools(self):
        transport = MockTransport([
            # initialize
            {"jsonrpc": "2.0", "id": 1, "result": {
                "capabilities": {}, "serverInfo": {"name": "srv"},
            }},
            # list_tools
            {"jsonrpc": "2.0", "id": 2, "result": {
                "tools": [
                    {"name": "add", "description": "Add numbers", "inputSchema": {}},
                    {"name": "sub", "description": "Subtract", "inputSchema": {}},
                ],
            }},
        ])
        client = MCPClient(transport)
        client.connect()
        tools = client.list_tools()
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["name"], "add")

    def test_call_tool(self):
        transport = MockTransport([
            # initialize
            {"jsonrpc": "2.0", "id": 1, "result": {
                "capabilities": {}, "serverInfo": {"name": "srv"},
            }},
            # call_tool
            {"jsonrpc": "2.0", "id": 2, "result": {
                "content": [{"type": "text", "text": "42"}],
                "isError": False,
            }},
        ])
        client = MCPClient(transport)
        client.connect()
        result = client.call_tool("add", {"a": 1, "b": 2})
        self.assertEqual(result["content"][0]["text"], "42")
        self.assertFalse(result["isError"])

    def test_protocol_error(self):
        transport = MockTransport([
            # initialize
            {"jsonrpc": "2.0", "id": 1, "result": {
                "capabilities": {}, "serverInfo": {"name": "srv"},
            }},
            # error response
            {"jsonrpc": "2.0", "id": 2, "error": {
                "code": -32601,
                "message": "Method not found",
            }},
        ])
        client = MCPClient(transport)
        client.connect()
        with self.assertRaises(MCPProtocolError) as ctx:
            client.list_tools()
        self.assertEqual(ctx.exception.code, -32601)

    def test_call_without_connect(self):
        transport = MockTransport()
        transport._connected = False
        client = MCPClient(transport)
        with self.assertRaises(MCPTransportError):
            client.list_tools()

    def test_disconnect(self):
        transport = MockTransport([
            {"jsonrpc": "2.0", "id": 1, "result": {
                "capabilities": {}, "serverInfo": {"name": "srv"},
            }},
        ])
        client = MCPClient(transport)
        client.connect()
        self.assertTrue(client.is_connected)
        client.disconnect()
        self.assertFalse(client.is_connected)

    def test_server_name_default(self):
        transport = MockTransport()
        client = MCPClient(transport)
        self.assertEqual(client.server_name, "unknown")


# ===========================================================================
# MCPManager
# ===========================================================================

class TestMCPManager(unittest.TestCase):
    """Test MCPManager config loading and status."""

    def test_empty_config(self):
        manager = MCPManager()
        with patch.object(manager, '_load_config', return_value={}):
            result = manager.initialize()
        self.assertEqual(result["servers"], 0)

    def test_load_config_parses_servers(self):
        manager = MCPManager()
        raw = {
            "github": {
                "command": "npx",
                "args": ["-y", "@mcp/server-github"],
                "env": {"GITHUB_TOKEN": "xxx"},
            },
            "api": {
                "url": "https://mcp.example.com/api",
            },
        }
        with patch.object(manager, '_load_config', return_value=raw):
            with patch.object(manager, '_connect_server', return_value=False):
                result = manager.initialize()
        self.assertEqual(result["servers"], 2)
        self.assertIn("github", manager.servers)
        self.assertIn("api", manager.servers)
        self.assertEqual(manager.servers["github"].config.transport_type, "stdio")
        self.assertEqual(manager.servers["api"].config.transport_type, "http")

    def test_disabled_server_not_connected(self):
        manager = MCPManager()
        raw = {
            "disabled_srv": {
                "command": "echo",
                "enabled": False,
            },
        }
        with patch.object(manager, '_load_config', return_value=raw):
            with patch.object(manager, '_connect_server') as mock_connect:
                manager.initialize()
        mock_connect.assert_not_called()

    def test_get_status_empty(self):
        manager = MCPManager()
        status = manager.get_status()
        self.assertEqual(status["total_servers"], 0)

    def test_shutdown_all(self):
        manager = MCPManager()
        mock_client = MagicMock()
        conn = MCPServerConnection(MCPServerConfig("test", "stdio", command="echo"))
        conn.client = mock_client
        conn.connected = True
        manager._servers["test"] = conn
        manager.shutdown_all()
        mock_client.disconnect.assert_called_once()
        self.assertFalse(conn.connected)


# ===========================================================================
# Handler routing
# ===========================================================================

class TestHandlerRouting(unittest.TestCase):
    """Test MCP tool call routing through MCPManager."""

    def _setup_manager_with_mock_server(self):
        manager = MCPManager()
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {
            "content": [{"type": "text", "text": "hello"}],
            "isError": False,
        }
        mock_client.is_connected = True

        config = MCPServerConfig("test", "stdio", command="echo")
        conn = MCPServerConnection(config)
        conn.client = mock_client
        conn.connected = True
        conn.tools = [{"name": "greet"}]
        manager._servers["test"] = conn
        return manager

    def test_successful_call(self):
        manager = self._setup_manager_with_mock_server()
        result = json.loads(manager._handle_tool_call("test", "greet", {"name": "world"}))
        self.assertEqual(result["result"], "hello")
        self.assertEqual(result["mcp_server"], "test")

    def test_unknown_server(self):
        manager = MCPManager()
        result = json.loads(manager._handle_tool_call("nonexistent", "foo", {}))
        self.assertIn("error", result)
        self.assertIn("not found", result["error"])

    def test_disconnected_server_reconnect_fail(self):
        manager = MCPManager()
        config = MCPServerConfig("dead", "stdio", command="echo")
        conn = MCPServerConnection(config)
        conn.connected = False
        conn.error = "Connection refused"
        manager._servers["dead"] = conn

        with patch.object(manager, '_connect_server', return_value=False):
            result = json.loads(manager._handle_tool_call("dead", "foo", {}))
        self.assertIn("error", result)
        self.assertIn("disconnected", result["error"])

    def test_tool_error_response(self):
        manager = self._setup_manager_with_mock_server()
        manager._servers["test"].client.call_tool.return_value = {
            "content": [{"type": "text", "text": "something went wrong"}],
            "isError": True,
        }
        result = json.loads(manager._handle_tool_call("test", "fail", {}))
        self.assertIn("error", result)

    def test_protocol_error_during_call(self):
        manager = self._setup_manager_with_mock_server()
        manager._servers["test"].client.call_tool.side_effect = MCPProtocolError(
            code=-32602, message="Invalid params"
        )
        result = json.loads(manager._handle_tool_call("test", "bad", {}))
        self.assertIn("error", result)
        self.assertIn("-32602", str(result.get("code")))

    def test_transport_error_marks_disconnected(self):
        manager = self._setup_manager_with_mock_server()
        manager._servers["test"].client.call_tool.side_effect = MCPTransportError("broken pipe")
        result = json.loads(manager._handle_tool_call("test", "err", {}))
        self.assertIn("error", result)
        self.assertFalse(manager._servers["test"].connected)


# ===========================================================================
# Meta-tool actions
# ===========================================================================

class TestMetaToolActions(unittest.TestCase):
    """Test the mcp management tool handler."""

    def test_status_action(self):
        from tools.mcp_tool import _handle_mcp
        result = json.loads(_handle_mcp({"action": "status"}))
        self.assertIn("total_servers", result)

    def test_list_tools_action(self):
        from tools.mcp_tool import _handle_mcp
        result = json.loads(_handle_mcp({"action": "list_tools"}))
        self.assertIn("total", result)
        self.assertIn("tools", result)

    def test_reconnect_without_server_name(self):
        from tools.mcp_tool import _handle_mcp
        result = json.loads(_handle_mcp({"action": "reconnect"}))
        self.assertIn("error", result)

    def test_unknown_action(self):
        from tools.mcp_tool import _handle_mcp
        result = json.loads(_handle_mcp({"action": "invalid_action"}))
        self.assertIn("error", result)
        self.assertIn("Unknown action", result["error"])

    def test_default_action_is_status(self):
        from tools.mcp_tool import _handle_mcp
        result = json.loads(_handle_mcp({}))
        self.assertIn("total_servers", result)


# ===========================================================================
# Schema validation
# ===========================================================================

class TestMCPSchemaRegistration(unittest.TestCase):
    """Test that the MCP tool schema is properly formed."""

    def test_schema_structure(self):
        from tools.mcp_tool import MCP_SCHEMA
        self.assertEqual(MCP_SCHEMA["name"], "mcp")
        self.assertIn("parameters", MCP_SCHEMA)
        self.assertIn("properties", MCP_SCHEMA["parameters"])
        self.assertIn("action", MCP_SCHEMA["parameters"]["properties"])
        self.assertIn("required", MCP_SCHEMA["parameters"])

    def test_schema_actions(self):
        from tools.mcp_tool import MCP_SCHEMA
        actions = MCP_SCHEMA["parameters"]["properties"]["action"]["enum"]
        self.assertIn("status", actions)
        self.assertIn("reconnect", actions)
        self.assertIn("list_tools", actions)


# ===========================================================================
# StdioTransport unit tests
# ===========================================================================

class TestStdioTransport(unittest.TestCase):
    """Test StdioTransport without actually spawning processes."""

    def test_init(self):
        t = StdioTransport(command="npx", args=["-y", "server"])
        self.assertEqual(t.command, "npx")
        self.assertEqual(t.args, ["-y", "server"])
        self.assertFalse(t.is_connected)

    def test_send_without_connect_raises(self):
        t = StdioTransport(command="echo")
        with self.assertRaises(MCPTransportError):
            t.send({"test": True})

    def test_receive_timeout(self):
        t = StdioTransport(command="echo")
        t._running = True
        with self.assertRaises(MCPTransportError):
            t.receive(timeout=0.1)


# ===========================================================================
# HttpTransport unit tests
# ===========================================================================

class TestHttpTransport(unittest.TestCase):
    """Test HttpTransport without actual HTTP calls."""

    def test_init(self):
        t = HttpTransport(url="https://example.com/mcp")
        self.assertEqual(t.url, "https://example.com/mcp")
        self.assertFalse(t.is_connected)

    def test_start_stop(self):
        t = HttpTransport(url="https://example.com/mcp")
        t.start()
        self.assertTrue(t.is_connected)
        t.stop()
        self.assertFalse(t.is_connected)

    def test_send_and_receive_not_connected(self):
        t = HttpTransport(url="https://example.com/mcp")
        with self.assertRaises(MCPTransportError):
            t.send_and_receive({"test": True})


# ===========================================================================
# Notification Routing
# ===========================================================================

class TestNotificationRouting(unittest.TestCase):
    """Test notification dispatch in StdioTransport."""

    def test_register_and_dispatch_notification(self):
        t = StdioTransport(command="echo")
        received = []
        t.on_notification("notifications/progress", lambda p: received.append(p))
        t._dispatch_notification({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"progress": 50},
        })
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["progress"], 50)

    def test_multiple_handlers_same_method(self):
        t = StdioTransport(command="echo")
        results = []
        t.on_notification("test/event", lambda p: results.append("a"))
        t.on_notification("test/event", lambda p: results.append("b"))
        t._dispatch_notification({"method": "test/event", "params": {}})
        self.assertEqual(results, ["a", "b"])

    def test_handler_error_isolation(self):
        t = StdioTransport(command="echo")
        results = []
        t.on_notification("test/ev", lambda p: (_ for _ in ()).throw(ValueError("boom")))
        t.on_notification("test/ev", lambda p: results.append("ok"))
        t._dispatch_notification({"method": "test/ev", "params": {}})
        # Second handler should still fire despite first raising
        self.assertEqual(results, ["ok"])

    def test_remove_notification_handler(self):
        t = StdioTransport(command="echo")
        results = []
        handler = lambda p: results.append("called")
        t.on_notification("test/rm", handler)
        t.remove_notification_handler("test/rm", handler)
        t._dispatch_notification({"method": "test/rm", "params": {}})
        self.assertEqual(results, [])

    def test_client_on_notification_delegates_to_transport(self):
        transport = MockTransport([
            {"jsonrpc": "2.0", "id": 1, "result": {
                "capabilities": {}, "serverInfo": {"name": "srv"},
            }},
        ])
        # MockTransport doesn't have on_notification, so it should be no-op
        client = MCPClient(transport)
        client.connect()
        # Should not raise even though transport lacks the method
        client.on_notification("test/method", lambda p: None)


# ===========================================================================
# MCP Resources
# ===========================================================================

class TestMCPResources(unittest.TestCase):
    """Test MCP resource listing and reading."""

    def _make_connected_client(self, extra_responses=None, caps=None):
        if caps is None:
            caps = {"resources": {"subscribe": True}, "tools": {}}
        responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {
                "capabilities": caps,
                "serverInfo": {"name": "res-srv"},
            }},
        ]
        if extra_responses:
            responses.extend(extra_responses)
        transport = MockTransport(responses)
        client = MCPClient(transport)
        client.connect()
        return client, transport

    def test_list_resources(self):
        client, _ = self._make_connected_client([
            {"jsonrpc": "2.0", "id": 2, "result": {
                "resources": [
                    {"uri": "file:///tmp/a.txt", "name": "a.txt"},
                    {"uri": "file:///tmp/b.txt", "name": "b.txt"},
                ],
            }},
        ])
        resources = client.list_resources()
        self.assertEqual(len(resources), 2)
        self.assertEqual(resources[0]["uri"], "file:///tmp/a.txt")

    def test_list_resources_no_capability(self):
        client, _ = self._make_connected_client(caps={"tools": {}})
        resources = client.list_resources()
        self.assertEqual(resources, [])

    def test_read_resource(self):
        client, _ = self._make_connected_client([
            {"jsonrpc": "2.0", "id": 2, "result": {
                "contents": [{"uri": "file:///tmp/a.txt", "text": "hello"}],
            }},
        ])
        result = client.read_resource("file:///tmp/a.txt")
        self.assertIn("contents", result)

    def test_subscribe_resource(self):
        client, transport = self._make_connected_client([
            {"jsonrpc": "2.0", "id": 2, "result": {}},
        ])
        client.subscribe_resource("file:///tmp/a.txt")
        # Verify request was sent
        sent = transport._sent
        self.assertTrue(any(m.get("method") == "resources/subscribe" for m in sent))

    def test_subscribe_no_capability(self):
        client, _ = self._make_connected_client(caps={"resources": {}, "tools": {}})
        with self.assertRaises(MCPProtocolError):
            client.subscribe_resource("file:///tmp/a.txt")

    def test_manager_resource_read(self):
        manager = MCPManager()
        mock_client = MagicMock()
        mock_client.read_resource.return_value = {
            "contents": [{"uri": "file:///a", "text": "data"}],
        }
        config = MCPServerConfig("res", "stdio", command="echo")
        conn = MCPServerConnection(config)
        conn.client = mock_client
        conn.connected = True
        manager._servers["res"] = conn
        result = json.loads(manager._handle_resource_read("res", "file:///a"))
        self.assertIn("contents", result)


# ===========================================================================
# MCP Prompts
# ===========================================================================

class TestMCPPrompts(unittest.TestCase):
    """Test MCP prompt listing and rendering."""

    def _make_connected_client(self, extra_responses=None, caps=None):
        if caps is None:
            caps = {"prompts": {}, "tools": {}}
        responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {
                "capabilities": caps,
                "serverInfo": {"name": "prompt-srv"},
            }},
        ]
        if extra_responses:
            responses.extend(extra_responses)
        transport = MockTransport(responses)
        client = MCPClient(transport)
        client.connect()
        return client, transport

    def test_list_prompts(self):
        client, _ = self._make_connected_client([
            {"jsonrpc": "2.0", "id": 2, "result": {
                "prompts": [
                    {"name": "summarize", "description": "Summarize text"},
                ],
            }},
        ])
        prompts = client.list_prompts()
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]["name"], "summarize")

    def test_list_prompts_no_capability(self):
        client, _ = self._make_connected_client(caps={"tools": {}})
        prompts = client.list_prompts()
        self.assertEqual(prompts, [])

    def test_get_prompt(self):
        client, _ = self._make_connected_client([
            {"jsonrpc": "2.0", "id": 2, "result": {
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": "Summarize: hello"}},
                ],
            }},
        ])
        result = client.get_prompt("summarize", {"text": "hello"})
        self.assertIn("messages", result)

    def test_manager_prompt_get(self):
        manager = MCPManager()
        mock_client = MagicMock()
        mock_client.get_prompt.return_value = {
            "messages": [{"role": "user", "content": {"type": "text", "text": "Hi"}}],
        }
        config = MCPServerConfig("p", "stdio", command="echo")
        conn = MCPServerConnection(config)
        conn.client = mock_client
        conn.connected = True
        manager._servers["p"] = conn
        result = json.loads(manager._handle_prompt_get("p", "greet"))
        self.assertIn("messages", result)


# ===========================================================================
# Progress Notifications
# ===========================================================================

class TestMCPProgress(unittest.TestCase):
    """Test progress notification handling."""

    def test_call_tool_with_progress(self):
        transport = MockTransport([
            {"jsonrpc": "2.0", "id": 1, "result": {
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "srv"},
            }},
            {"jsonrpc": "2.0", "id": 2, "result": {
                "content": [{"type": "text", "text": "done"}],
                "isError": False,
            }},
        ])
        client = MCPClient(transport)
        client.connect()
        progress_updates = []
        result = client.call_tool_with_progress(
            "slow_task", {"x": 1},
            progress_callback=lambda p, t, m: progress_updates.append((p, t, m)),
        )
        self.assertEqual(result["content"][0]["text"], "done")

    def test_progress_token_in_request(self):
        transport = MockTransport([
            {"jsonrpc": "2.0", "id": 1, "result": {
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "srv"},
            }},
            {"jsonrpc": "2.0", "id": 2, "result": {
                "content": [], "isError": False,
            }},
        ])
        client = MCPClient(transport)
        client.connect()
        client.call_tool_with_progress("t", None, progress_callback=lambda p, t, m: None)
        # The tools/call request should have _meta.progressToken
        call_msg = transport._sent[-1]
        self.assertIn("_meta", call_msg.get("params", {}))
        self.assertIn("progressToken", call_msg["params"]["_meta"])

    def test_progress_callback_dispatch(self):
        transport = MockTransport([
            {"jsonrpc": "2.0", "id": 1, "result": {
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "srv"},
            }},
        ])
        client = MCPClient(transport)
        client.connect()
        received = []
        client._progress_token_counter = 0
        client._ensure_progress_handler()
        # Manually register a callback
        with client._progress_lock:
            client._progress_token_counter += 1
            token = f"hermes-progress-{client._progress_token_counter}"
            client._progress_callbacks[token] = lambda p, t, m: received.append((p, t, m))
        # Simulate progress notification
        client._on_progress({"progressToken": token, "progress": 50, "total": 100, "message": "half"})
        self.assertEqual(received, [(50, 100, "half")])

    def test_progress_unknown_token_ignored(self):
        transport = MockTransport([
            {"jsonrpc": "2.0", "id": 1, "result": {
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "srv"},
            }},
        ])
        client = MCPClient(transport)
        client.connect()
        # Should not raise
        client._on_progress({"progressToken": "unknown", "progress": 10})

    def test_manager_on_progress_logs(self):
        manager = MCPManager()
        # Should not raise -- just logs
        manager._on_progress("srv", {"progressToken": "t1", "progress": 5, "total": 10, "message": "ok"})
        manager._on_progress("srv", {"progressToken": "t2", "progress": 3, "message": "no total"})


# ===========================================================================
# list_changed Notifications
# ===========================================================================

class TestListChangedNotifications(unittest.TestCase):
    """Test tools/resources/prompts list_changed handling."""

    def _setup_manager_with_mock_server(self):
        manager = MCPManager()
        mock_client = MagicMock()
        mock_client.list_tools.return_value = [
            {"name": "new_tool", "description": "New", "inputSchema": {}},
        ]
        mock_client.list_resources.return_value = [
            {"uri": "file:///new", "name": "new"},
        ]
        mock_client.list_prompts.return_value = [
            {"name": "new_prompt", "description": "New prompt"},
        ]
        mock_client.is_connected = True
        config = MCPServerConfig("test", "stdio", command="echo")
        conn = MCPServerConnection(config)
        conn.client = mock_client
        conn.connected = True
        conn.tools = [{"name": "old_tool"}]
        conn.resources = []
        conn.prompts = []
        manager._servers["test"] = conn
        return manager

    def test_tools_list_changed(self):
        manager = self._setup_manager_with_mock_server()
        with patch.object(manager, '_register_all_tools'):
            manager._on_tools_list_changed("test")
        self.assertEqual(len(manager._servers["test"].tools), 1)
        self.assertEqual(manager._servers["test"].tools[0]["name"], "new_tool")

    def test_resources_list_changed(self):
        manager = self._setup_manager_with_mock_server()
        manager._on_resources_list_changed("test")
        self.assertEqual(len(manager._servers["test"].resources), 1)
        self.assertEqual(manager._servers["test"].resources[0]["uri"], "file:///new")

    def test_prompts_list_changed(self):
        manager = self._setup_manager_with_mock_server()
        manager._on_prompts_list_changed("test")
        self.assertEqual(len(manager._servers["test"].prompts), 1)
        self.assertEqual(manager._servers["test"].prompts[0]["name"], "new_prompt")

    def test_list_changed_disconnected_server_noop(self):
        manager = MCPManager()
        config = MCPServerConfig("dead", "stdio", command="echo")
        conn = MCPServerConnection(config)
        conn.connected = False
        manager._servers["dead"] = conn
        # Should not raise
        manager._on_tools_list_changed("dead")
        manager._on_resources_list_changed("dead")
        manager._on_prompts_list_changed("dead")

    def test_list_changed_unknown_server_noop(self):
        manager = MCPManager()
        # Should not raise
        manager._on_tools_list_changed("nonexistent")
        manager._on_resources_list_changed("nonexistent")
        manager._on_prompts_list_changed("nonexistent")


# ===========================================================================
# Structured Logging
# ===========================================================================

class TestMCPLogging(unittest.TestCase):
    """Test MCP structured logging."""

    def test_set_log_level(self):
        transport = MockTransport([
            {"jsonrpc": "2.0", "id": 1, "result": {
                "capabilities": {"logging": {}, "tools": {}},
                "serverInfo": {"name": "srv"},
            }},
            {"jsonrpc": "2.0", "id": 2, "result": {}},
        ])
        client = MCPClient(transport)
        client.connect()
        client.set_log_level("warning")
        sent = transport._sent
        self.assertTrue(any(
            m.get("method") == "logging/setLevel" and m.get("params", {}).get("level") == "warning"
            for m in sent
        ))

    def test_log_level_mapping(self):
        from tools.mcp_client import _MCP_LOG_LEVELS
        import logging as py_logging
        self.assertEqual(_MCP_LOG_LEVELS["debug"], py_logging.DEBUG)
        self.assertEqual(_MCP_LOG_LEVELS["error"], py_logging.ERROR)
        self.assertEqual(_MCP_LOG_LEVELS["emergency"], py_logging.CRITICAL)

    def test_manager_on_log_message(self):
        manager = MCPManager()
        # Should not raise
        manager._on_log_message("srv", {
            "level": "info",
            "data": "Server started",
            "logger": "main",
        })

    def test_manager_handle_set_log_level(self):
        manager = MCPManager()
        mock_client = MagicMock()
        config = MCPServerConfig("log_srv", "stdio", command="echo")
        conn = MCPServerConnection(config)
        conn.client = mock_client
        conn.connected = True
        manager._servers["log_srv"] = conn
        result = json.loads(manager._handle_set_log_level("log_srv", "debug"))
        self.assertEqual(result["status"], "ok")
        mock_client.set_log_level.assert_called_once_with("debug")

    def test_manager_handle_set_log_level_not_connected(self):
        manager = MCPManager()
        result = json.loads(manager._handle_set_log_level("unknown", "debug"))
        self.assertIn("error", result)

    def test_meta_tool_set_log_level(self):
        from tools.mcp_tool import _handle_mcp
        result = json.loads(_handle_mcp({"action": "set_log_level"}))
        self.assertIn("error", result)  # Missing server_name and log_level


# ===========================================================================
# Extended Meta-tool Actions
# ===========================================================================

class TestExtendedMetaToolActions(unittest.TestCase):
    """Test new meta-tool actions for resources, prompts, logging."""

    def test_list_resources_action(self):
        from tools.mcp_tool import _handle_mcp
        result = json.loads(_handle_mcp({"action": "list_resources"}))
        self.assertIn("total", result)
        self.assertIn("resources", result)

    def test_read_resource_missing_params(self):
        from tools.mcp_tool import _handle_mcp
        result = json.loads(_handle_mcp({"action": "read_resource"}))
        self.assertIn("error", result)

    def test_list_prompts_action(self):
        from tools.mcp_tool import _handle_mcp
        result = json.loads(_handle_mcp({"action": "list_prompts"}))
        self.assertIn("total", result)
        self.assertIn("prompts", result)

    def test_get_prompt_missing_params(self):
        from tools.mcp_tool import _handle_mcp
        result = json.loads(_handle_mcp({"action": "get_prompt"}))
        self.assertIn("error", result)

    def test_schema_has_new_actions(self):
        from tools.mcp_tool import MCP_SCHEMA
        actions = MCP_SCHEMA["parameters"]["properties"]["action"]["enum"]
        self.assertIn("list_resources", actions)
        self.assertIn("read_resource", actions)
        self.assertIn("list_prompts", actions)
        self.assertIn("get_prompt", actions)
        self.assertIn("set_log_level", actions)

    def test_schema_has_new_properties(self):
        from tools.mcp_tool import MCP_SCHEMA
        props = MCP_SCHEMA["parameters"]["properties"]
        self.assertIn("uri", props)
        self.assertIn("prompt_name", props)
        self.assertIn("arguments", props)
        self.assertIn("log_level", props)


if __name__ == "__main__":
    unittest.main()
