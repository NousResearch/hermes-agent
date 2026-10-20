"""Portable unittest suite: policy fakes, actual SDK HTTP, disposable loopback.

No live Rive editor connection. Python 3.11+ plus requirements.txt; pytest is not
needed to run `python -m unittest discover -s scripts -p test_rive_mcp.py`.
"""
import contextlib
import importlib
import io
import json
import unittest
import logging
import os
import warnings
import subprocess
import sys
import tempfile
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from pathlib import Path
from mcp.types import CallToolResult, TextContent
import httpx2
from types import SimpleNamespace
import rive_mcp
import rive_doctor


class FakeSession:
    """Test-only MCP boundary; no socket or Rive editor access."""
    def __init__(self, tools=None, result=None):
        self.tools = tools or [
            {"name": "session_info", "inputSchema": {"type": "object", "properties": {}}}
        ]
        self.events = []
        self.result = result

    async def call_tool(self, name, arguments):
        self.events.append((name, arguments))
        return self.result if self.result is not None else CallToolResult(content=[TextContent(type="text", text='{"activeFileId":null}')])

    async def initialize(self):
        self.events.append("initialize")

    async def list_tools(self, **kwargs):
        self.events.append("list")
        return SimpleNamespace(tools=self.tools, next_cursor=None)


@contextlib.asynccontextmanager
async def fake_connection(session):
    yield session


class CliTests(unittest.TestCase):
    def invoke(self, argv, session=None, connect=None):
        module = importlib.import_module("rive_mcp")
        session = session or FakeSession()
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = module.main(argv, connect=connect or (lambda: fake_connection(session)), stdout=out, stderr=err)
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue(), session

    def args_file(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "arguments.json"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_optional_expected_file_guards_read_calls_without_apply(self):
        tools = FakeSession().tools + [{"name": "list_artboards", "inputSchema": {"type": "object"}}]
        for active, expected, allowed in [("wanted", "wanted", True), ("wrong", "wanted", False),
                                           (None, "wanted", False), ("wanted", "", False)]:
            with self.subTest(active=active, expected=expected):
                result = {"structuredContent": {"activeFileId": active}}
                code, out, err, session = self.invoke(
                    ["call", "list_artboards", "--args", self.args_file("{}"), "--expected-file", expected],
                    FakeSession(tools, result))
                self.assertEqual(code, 0 if allowed else 2, err)
                self.assertEqual(session.events, ["initialize", "list", ("session_info", {})]
                                 + ([("list_artboards", {})] if allowed else []))

    def test_help_and_missing_dependency_diagnostic_without_site_packages(self):
        script = str(Path(__file__).with_name("rive_mcp.py"))
        help_result = subprocess.run([sys.executable, "-S", script, "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("requirements.txt", help_result.stdout)
        missing = subprocess.run([sys.executable, "-S", script, "list"], capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("requirements.txt", missing.stderr)
        self.assertNotIn("Traceback", missing.stderr)
        self.assertEqual(missing.stdout, "")
        with patch.object(rive_mcp, "version", return_value="1.0.0"):
            code, out, err, session = self.invoke(["list"])
        self.assertEqual(code, 2)
        self.assertIn("SDK 2.x", err)
        self.assertEqual(session.events, [])

    def test_ambiguous_json_response_cannot_hide_failure_flags(self):
        for text in ['{"success":false,"success":true}', '{"success":false,"value":NaN}', '{"success":true,"value":1e999}']:
            with self.subTest(text=text):
                body = {"content": [{"type": "text", "text": text}], "isError": False}
                code, out, err, _ = self.invoke(["call", "session_info", "--args", self.args_file("{}")], FakeSession(result=body))
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
        body = {"content": [{"type": "text", "text": '{ sample code, not a JSON document }'}]}
        self.assertEqual(self.invoke(["call", "session_info", "--args", self.args_file("{}")], FakeSession(result=body))[0], 0)

    def test_schema_validation_forbids_external_resolution_and_unknown_dialects(self):
        tools = [
            {"name": "session_info", "inputSchema": {"type": "object"}, "outputSchema": {"$ref": "https://credential-secret-value.invalid/schema"}},
            {"name": "session_info", "inputSchema": {"$schema": "https://credential-secret-value.invalid/dialect", "type": "object"}},
            {"name": "session_info", "inputSchema": {"$ref": "https://credential-secret-value.invalid/schema"}},
        ]
        for tool in tools:
            with self.subTest(tool=tool), warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                code, out, err, session = self.invoke(["call", "session_info", "--args", self.args_file("{}")], FakeSession([tool]))
                self.assertEqual(code, 2)
                self.assertEqual(session.events, ["initialize", "list"])
                self.assertEqual(captured, [])
                self.assertNotIn("credential-secret-value", out + err)
        local = {"name": "session_info", "inputSchema": {"$defs": {"args": {"type": "object"}}, "$ref": "#/$defs/args"}}
        self.assertEqual(self.invoke(["call", "session_info", "--args", self.args_file("{}")], FakeSession([local]))[0], 0)

    def test_invalid_output_schema_is_refused_before_target_call(self):
        tool = {"name": "session_info", "inputSchema": {"type": "object"},
                "outputSchema": {"type": "not-a-json-schema-type"}}
        code, out, err, session = self.invoke(
            ["call", "session_info", "--args", self.args_file("{}")], FakeSession([tool]))
        self.assertEqual(code, 2)
        self.assertEqual(session.events, ["initialize", "list"])
        self.assertEqual(out, "")
        self.assertIn("schema", err)

    def test_actual_sdk_mutation_failure_never_retries_or_follows_redirects(self):
        for failure in (307, 404, 503, "timeout"):
            with self.subTest(failure=failure):
                events = []

                async def handle(request):
                    self.assertEqual(str(request.url), rive_mcp.ENDPOINT)
                    if request.method == "GET":
                        return httpx2.Response(405)
                    body = json.loads(request.content)
                    method = body["method"]
                    events.append(body)
                    if method == "notifications/initialized":
                        return httpx2.Response(202)
                    result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                              "serverInfo": {"name": "fixture", "version": "1"}}
                    if method == "tools/list":
                        result = {"tools": FakeSession().tools + [{"name": "run_tests", "inputSchema": {"type": "object"}}]}
                    if method == "tools/call":
                        if body["params"]["name"] == "run_tests":
                            if failure == "timeout":
                                raise httpx2.ReadTimeout("secret-transport-detail")
                            return httpx2.Response(int(failure), headers={"location": "https://secret.invalid/mcp"})
                        result = {"content": [{"type": "text", "text": '{"activeFileId":"wanted"}'}]}
                    return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

                code, out, err, _ = self.invoke(
                    ["call", "run_tests", "--args", self.args_file("{}"), "--apply", "--expected-file", "wanted"],
                    connect=lambda: rive_mcp.sdk_connection(transport=httpx2.MockTransport(handle)))
                self.assertEqual(code, 3)
                self.assertEqual(out, "")
                self.assertNotIn("secret", err)
                self.assertIn("not retried", err)
                self.assertEqual([e["params"]["name"] for e in events if e["method"] == "tools/call"],
                                 ["session_info", "run_tests"])

    def test_actual_sdk_http200_errors_fail_but_query_and_source_payloads_pass(self):
        import rive_mcp
        cases = [
            ({"error": {"code": -32602, "message": "credential-secret-value"}}, 3),
            ({"result": {"isError": True, "content": [{"type": "text", "text": "credential-secret-value"}]}}, 2),
            ({"result": {"content": [], "structuredContent": {"success": False, "errors": ["credential-secret-value"]}}}, 2),
            ({"result": {"content": [{"type": "text", "text": 'local x = { success = false, errors = {"example"} }'}]}}, 0),
            ({"result": {"content": [{"type": "text", "text": '{"objects":[{"success":false,"errors":["ordinary property"]}],"entries":[{"type":"error","message":"diagnostic"}]}'}]}}, 0),
            ({"result": {"content": [], "structuredContent": {"success": True, "errors": [], "content": "plain reference"}}}, 0),
        ]
        for response, wanted in cases:
            with self.subTest(wanted=wanted, response=response):
                calls = []
                async def handle(request):
                    if request.method == "GET":
                        return httpx2.Response(405)
                    body = json.loads(request.content)
                    method = body["method"]
                    if method == "notifications/initialized":
                        return httpx2.Response(202)
                    payload = {"result": {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "synthetic", "version": "1"}}}
                    if method == "tools/list":
                        payload = {"result": {"tools": FakeSession().tools}}
                    if method == "tools/call":
                        calls.append(body["params"])
                        payload = response
                    return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], **payload})
                code, out, err, _ = self.invoke(["call", "session_info", "--args", self.args_file("{}")],
                    connect=lambda: rive_mcp.sdk_connection(transport=httpx2.MockTransport(handle)))
                self.assertEqual(code, wanted, err)
                self.assertNotIn("credential-secret-value", out + err)
                self.assertEqual(len(calls), 1)
                if wanted == 0:
                    self.assertEqual(json.loads(out)["content"], response["result"]["content"])

    def test_list_paginates_or_refuses_duplicate_names_and_cursor_cycles(self):
        class Pages(FakeSession):
            async def list_tools(self, *, params=None):
                self.events.append(("list", None if params is None else params.cursor))
                if params is None:
                    return SimpleNamespace(tools=[{"name": "first", "inputSchema": {"type": "object"}}], next_cursor="page2")
                return SimpleNamespace(tools=[{"name": "second", "inputSchema": {"type": "object"}}], next_cursor=None)
        code, out, err, session = self.invoke(["list"], Pages())
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["count"], 2)
        self.assertEqual(session.events, ["initialize", ("list", None), ("list", "page2")])
        class Repeated(Pages):
            async def list_tools(self, *, params=None):
                return SimpleNamespace(tools=[], next_cursor="same")
        code, out, err, _ = self.invoke(["list"], Repeated())
        self.assertEqual(code, 2)
        class Duplicates(Pages):
            async def list_tools(self, *, params=None):
                return SimpleNamespace(tools=[{"name": "duplicate"}, {"name": "duplicate"}], next_cursor=None)
        code, out, err, _ = self.invoke(["list"], Duplicates())
        self.assertEqual(code, 2)

    def test_cli_help_entrypoint_and_exact_flags_without_secret_echo(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("rive_mcp.py")), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("{list,schema,call}", result.stdout)
        for argv in [["list", "--endpoint", "https://credential-secret-value.invalid/mcp"],
                     ["call", "session_info"],
                     ["call", "session_info", "--args", self.args_file("{}"), "--app"],
                     ["credential-secret-value"]]:
            with self.subTest(argv=argv):
                code, out, err, session = self.invoke(argv)
                self.assertEqual(code, 2)
                self.assertEqual(session.events, [])
                self.assertNotIn("credential-secret-value", out + err)

    def test_cli_protocol_failure_is_safe_and_never_retries_mutation(self):
        class BrokenMutation(FakeSession):
            async def call_tool(self, name, arguments):
                self.events.append((name, arguments))
                if name == "session_info":
                    return {"structuredContent": {"success": True, "content": '{"activeFileId":"wanted"}'}}
                logging.getLogger("mcp.synthetic").error("credential-secret-value")
                raise RuntimeError("credential-secret-value")
        session = BrokenMutation(FakeSession().tools + [{"name": "run_tests", "inputSchema": {"type": "object"}}])
        capture = io.StringIO()
        handler = logging.StreamHandler(capture)
        logging.getLogger().addHandler(handler)
        self.addCleanup(logging.getLogger().removeHandler, handler)
        try:
            code, out, err, session = self.invoke(["call", "run_tests", "--args", self.args_file("{}"), "--apply", "--expected-file", "wanted"], session)
        except Exception as exc:
            self.fail("protocol exception escaped instead of safe CLI failure: " + type(exc).__name__)
        self.assertEqual(code, 3)
        self.assertEqual(out, "")
        self.assertNotIn("credential-secret-value", err + capture.getvalue())
        self.assertIn("not retried", err)
        self.assertEqual(session.events, ["initialize", "list", ("session_info", {}), ("run_tests", {})])

    def test_dry_run_validates_policy_without_invoking_target(self):
        for name, readonly in [("session_info", True), ("run_tests", False)]:
            with self.subTest(name=name):
                tools = FakeSession().tools + [{"name": "run_tests", "inputSchema": {"type": "object"}}]
                code, out, err, session = self.invoke(["call", name, "--args", self.args_file("{}"), "--dry-run"], FakeSession(tools))
                self.assertEqual(code, 0, err)
                report = json.loads(out)
                self.assertTrue(report["dry_run"])
                self.assertEqual(report["classification"], "read-only" if readonly else "mutation-class")
                self.assertFalse(report["target_called"])
                self.assertFalse(report["active_file_checked"])
                self.assertEqual(session.events, ["initialize", "list"])
        code, out, err, session = self.invoke(["call", "session_info", "--args", self.args_file("{}"), "--dry-run", "--apply"])
        self.assertEqual(code, 2)
        self.assertEqual(session.events, [])

    def test_exact_read_whitelist_never_trusts_names_annotations_or_new_commands(self):
        readonly = {
            "session_info": {}, "query_property_keys": {"objectIds": []}, "query_objects": {},
            "query_property_values": {"propertyKeys": []}, "find_objects": {}, "get_artboard_hierarchy": {},
            "list_artboards": {}, "get_selection": {}, "script_diagnostics": {}, "get_scripts": {},
            "grep": {"pattern": "x"}, "read_console": {}, "get_scripting_reference": {"topic": "example_test"},
        }
        commands = {
            "animation_editor": ["listStateMachines", "listLinearAnimations", "queryStateMachine", "queryStateMachineLayer", "queryKeyFrames"],
            "assets_tool": ["listAssets", "queryAsset"], "tag_editor": ["queryTags"],
            "open_file_editor": ["getCurrentFile", "listArtboards", "getSelectedArtboard"],
            "viewmodel_editor": ["listViewModels", "listViewModelInstances", "listDataBinds", "listConverters", "listDataEnums"],
            "mesh_rigging_tool": ["querySkin"], "text_editor": ["view"],
        }
        cases = [(name, args, True) for name, args in readonly.items()]
        cases += [(name, {"command": command, **({"path": "x"} if name == "text_editor" else {})}, True)
                  for name, values in commands.items() for command in values]
        cases += [(name, args, False) for name, args in [
            ("animation_editor", {"command": "simulateStateMachine"}), ("assets_tool", {"command": "addImageInstance"}),
            ("open_file_editor", {"command": "focusArtboard"}), ("text_editor", {"command": "insert", "path": "x"}),
            ("assets_tool", {"command": "listNewReadLookingCommand"}), ("get_new_tool", {}),
            ("run_tests", {"path": "x"}), ("recompile_all_scripts", {}), ("export_file", {}), ("select_objects", {}),
            ("capture_artboard", {}),
            ("session_info", {"command": "erase"}), ("text_editor", {"command": "view", "path": "x", "text": "code"}),
            ("assets_tool", {"command": "listAssets", "data": {"addImageInstance": {}}}),
        ]]
        for name, args, allowed in cases:
            with self.subTest(name=name, args=args):
                tool = {"name": name, "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True}}
                code, out, err, session = self.invoke(["call", name, "--args", self.args_file(json.dumps(args))], FakeSession([tool]))
                self.assertEqual(code, 0 if allowed else 2, err)
                self.assertEqual(session.events, ["initialize", "list"] + ([(name, args)] if allowed else []))

    def test_mutation_requires_apply_and_fresh_exact_file_before_single_call(self):
        tools = FakeSession().tools + [{"name": "run_tests", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
        args = ["call", "run_tests", "--args", self.args_file('{"path":"tests/example"}')]
        cases = [([], "wanted", False, False), (["--expected-file", "wanted"], "wanted", False, False),
                 (["--apply"], "wanted", False, False),
                 (["--apply", "--expected-file", "wanted"], None, True, False),
                 (["--apply", "--expected-file", "wanted"], "wrong", True, False),
                 (["--apply", "--expected-file", "wanted"], "wanted", True, True)]
        for flags, active, check_file, allowed in cases:
            with self.subTest(flags=flags, active=active):
                result = {"structuredContent": {"success": True, "errors": [], "content": json.dumps({"activeFileId": active})}}
                code, out, err, session = self.invoke(args + flags, FakeSession(tools, result))
                self.assertEqual(code, 0 if allowed else 2, err)
                expected = ["initialize", "list"]
                if check_file:
                    expected.append(("session_info", {}))
                if allowed:
                    expected.append(("run_tests", {"path": "tests/example"}))
                self.assertEqual(session.events, expected)
                if not allowed:
                    self.assertEqual(out, "")

    def test_call_fails_error_envelopes_at_all_rive_response_layers(self):
        bodies = [
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "secret-value"}},
            {"isError": True, "content": []},
            {"content": [{"type": "text", "text": '{"success":false,"errors":["secret-value"]}'}]},
            {"structuredContent": {"success": False, "errors": [], "content": "{}"}},
            {"structuredContent": {"success": True, "errors": ["secret-value"]}},
            {"structuredContent": {"success": True, "errors": [], "content": '{"success":false}'}},
            {"content": [{"type": "text", "text": '{"result":{"errors":["secret-value"]}}'}]},
            {"structuredContent": {"content": '[{"success":true},{"success":false}]'}},
        ]
        for body in bodies:
            with self.subTest(body=body):
                code, out, err, session = self.invoke(["call", "session_info", "--args", self.args_file("{}")], FakeSession(result=body))
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertIn("server reported", err)
                self.assertNotIn("secret-value", err)

    def test_call_validates_current_session_schema_before_tool_call(self):
        schema = {"type": "object", "properties": {"command": {"enum": ["read"]}, "count": {"type": "integer", "minimum": 1}},
                  "required": ["command"], "additionalProperties": False}
        for args in [{}, {"command": "new"}, {"command": "read", "count": "secret-value"},
                     {"command": "read", "count": 0}, {"command": "read", "extra": 1}]:
            with self.subTest(args=args):
                session = FakeSession([{"name": "session_info", "inputSchema": schema}])
                code, out, err, session = self.invoke(["call", "session_info", "--args", self.args_file(json.dumps(args))], session)
                self.assertEqual(code, 2)
                self.assertEqual(session.events, ["initialize", "list"])
                self.assertIn("schema", err)
                self.assertNotIn("secret-value", err)
        code, out, err, session = self.invoke(["call", "missing_tool", "--args", self.args_file("{}")])
        self.assertEqual(code, 2)
        self.assertIn("unknown tool", err)
        self.assertEqual(session.events, ["initialize", "list"])

    def test_call_rejects_non_strict_json_without_network_or_echo(self):
        for text in ['{bad secret-value', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}',
                     '{"a":1,"a":2}', '{"a":{"b":1,"b":2}}', '[]', 'null', '"secret-value"']:
            with self.subTest(text=text):
                session = FakeSession()
                try:
                    code, out, err, _ = self.invoke(["call", "session_info", "--args", self.args_file(text)], session)
                except Exception as exc:
                    self.fail("invalid JSON escaped CLI instead of safe refusal: " + type(exc).__name__)
                self.assertEqual(code, 2)
                self.assertEqual(session.events, [])
                self.assertEqual(out, "")
                self.assertNotIn("secret-value", err)

    def test_call_reads_arguments_file_and_returns_sdk_result(self):
        code, out, err, session = self.invoke(["call", "session_info", "--args", self.args_file("{}")])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["content"][0]["text"], '{"activeFileId":null}')
        self.assertEqual(session.events[-1], ("session_info", {}))

    def test_schema_cli_resolves_exact_tool_or_refuses_unknown(self):
        code, out, err, session = self.invoke(["schema", "session_info"])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), session.tools[0])
        code, out, err, session = self.invoke(["schema", "unknown_tool"])
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("unknown tool", err)

    def test_list_cli_initializes_and_lists_live_session_schema(self):
        self.assertIsNotNone(importlib.util.find_spec("rive_mcp"), "CLI module missing")
        code, out, err, session = self.invoke(["list"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["count"], 1)
        self.assertEqual(json.loads(out)["tools"], session.tools)
        self.assertEqual(session.events, ["initialize", "list"])
        self.assertEqual(err, "")


class SdkBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Actual SDK over synthetic HTTP transport; NOT live Rive integrations."""
    async def test_official_sdk_connection_fixed_loopback_handshake_and_list(self):
        import rive_mcp
        self.assertTrue(hasattr(rive_mcp, "sdk_connection"), "official connection missing")
        requests = []
        async def handle(request):
            requests.append((request.method, str(request.url)))
            self.assertEqual(str(request.url), "http://127.0.0.1:9791/mcp")
            if request.method == "GET":
                return httpx2.Response(405)
            body = json.loads(request.content)
            if body["method"] == "notifications/initialized":
                return httpx2.Response(202)
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "synthetic-test", "version": "1"}}
            if body["method"] == "tools/list":
                result = {"tools": [{"name": "session_info", "inputSchema": {"type": "object"}}]}
            return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
        async with rive_mcp.sdk_connection(transport=httpx2.MockTransport(handle)) as session:
            await session.initialize()
            listed = await session.list_tools()
            self.assertEqual(listed.tools[0].name, "session_info")
        self.assertNotIn("DELETE", [r[0] for r in requests])


@contextlib.contextmanager
def loopback_server(*, sse=False, failure=None):
    """A real local HTTP peer, never the user's editor or its fixed port."""
    events = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def respond(self, status, data=b"", content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.respond(405)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            events.append(body)
            method = body["method"]
            if failure:
                self.respond(failure)
                return
            if method == "notifications/initialized":
                self.respond(202)
                return
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "disposable-test-peer", "version": "1"}}
            if method == "tools/list":
                result = {"tools": FakeSession().tools, "nextCursor": "second"}
                if body.get("params", {}).get("cursor") == "second":
                    result = {"tools": [{"name": "list_artboards", "inputSchema": {"type": "object"}}]}
            if method == "tools/call":
                result = {"content": [{"type": "text", "text": '{"activeFileId":"fixture-file"}'}]}
            data = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode()
            self.respond(200, b"event: message\ndata: " + data + b"\n\n" if sse else data,
                         "text/event-stream" if sse else "application/json")

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            with patch.object(rive_mcp, "ENDPOINT", f"http://127.0.0.1:{port}/mcp"), \
                    patch.object(rive_doctor, "RIVE_PORT", port):
                yield events
        finally:
            server.shutdown()
            thread.join(timeout=5)
            if thread.is_alive():
                raise AssertionError("disposable server failed to stop")


class DoctorTests(unittest.TestCase):
    def invoke(self, flags):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rive_doctor.main(flags)
        self.assertEqual(err.getvalue(), "")
        return code, json.loads(out.getvalue())["official_rive"]

    def test_closed_and_tcp_only_never_claim_functional_mcp(self):
        # Reserve a port without listening: no race with unrelated processes.
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            with patch.object(rive_doctor, "RIVE_PORT", reserved.getsockname()[1]):
                code, state = self.invoke(["--json"])
                self.assertEqual(code, 1)
                self.assertEqual(state["status"], "closed")
                self.assertFalse(state["tcp_reachable"])
                self.assertFalse(state["handshake_verified"])
        with loopback_server(failure=503) as events:
            code, state = self.invoke(["--json"])
            self.assertEqual(code, 2)
            self.assertEqual(state["status"], "tcp-only")
            self.assertTrue(state["tcp_reachable"])
            self.assertFalse(state["handshake_verified"])
            self.assertEqual(events, [])
            # Stock Python still diagnoses the listener without dependencies.
            script_dir = str(Path(__file__).parent)
            probe = ("import sys; sys.path.insert(0, sys.argv[1]); import rive_doctor; "
                     "rive_doctor.RIVE_PORT=int(sys.argv[2]); "
                     "raise SystemExit(rive_doctor.main(sys.argv[3:]))")
            for flags, wanted in [(["--json"], 2), (["--json", "--handshake"], 1)]:
                result = subprocess.run([sys.executable, "-S", "-c", probe, script_dir,
                                         str(rive_doctor.RIVE_PORT), *flags], capture_output=True, text=True)
                self.assertEqual(result.returncode, wanted, result.stderr)
                self.assertEqual(result.stderr, "")
                if "--handshake" in flags:
                    self.assertIn("requirements.txt", json.loads(result.stdout)["official_rive"]["detail"])
            self.assertEqual(events, [])
            code, state = self.invoke(["--json", "--handshake"])
            self.assertEqual(code, 1)
            self.assertEqual(state["status"], "handshake-failed")
            self.assertFalse(state["handshake_verified"])

    def test_sdk_live_handshake_is_not_file_authoring_or_export_verification(self):
        for sse in (False, True):
            with self.subTest(sse=sse), socket.socket() as proxy, contextlib.ExitStack() as stack:
                # A reserved, non-listening proxy guarantees a failure if the
                # production client accidentally honors proxy environment vars.
                proxy.bind(("127.0.0.1", 0))
                proxy_url = f"http://127.0.0.1:{proxy.getsockname()[1]}"
                stack.enter_context(patch.dict(os.environ, {"HTTP_PROXY": proxy_url, "http_proxy": proxy_url,
                                                            "ALL_PROXY": proxy_url, "all_proxy": proxy_url,
                                                            "NO_PROXY": "", "no_proxy": ""}))
                events = stack.enter_context(loopback_server(sse=sse))
                code, state = self.invoke(["--json", "--handshake"])
                self.assertEqual(code, 0)
                self.assertEqual(state["status"], "mcp-verified")
                self.assertTrue(state["handshake_verified"])
                self.assertEqual(state["tool_count"], 2)
                self.assertFalse(state["active_file_checked"])
                self.assertIn("not", state["scope"])
                self.assertEqual([event["method"] for event in events],
                                 ["initialize", "notifications/initialized", "tools/list", "tools/list"])
                # The production HTTP stack also exercises an actual guarded read.
                args = Path(__file__).with_name("empty-args.json")
                out, err = io.StringIO(), io.StringIO()
                code = rive_mcp.main(["call", "list_artboards", "--args", str(args),
                                      "--expected-file", "fixture-file"], stdout=out, stderr=err)
                self.assertEqual(code, 0, err.getvalue())
                self.assertEqual([event["params"]["name"] for event in events if event["method"] == "tools/call"],
                                 ["session_info", "list_artboards"])


if __name__ == "__main__":
    unittest.main()
