"""Runs -> registry -> MCP SDK, using a local HTTP API and real in-memory MCP peers."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from mcp import types
from mcp.client.session import ClientSession
from mcp.server import Server

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tools import mcp_tool, mcp_tool_loop
from tools.mcp_tool_handlers import _make_tool_handler
from tools.registry import ToolRegistry


@pytest_asyncio.fixture
async def api(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-api-key"}))
    factory = MagicMock()
    monkeypatch.setattr(adapter, "_create_agent", factory)
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    async with TestClient(TestServer(app), headers={"Authorization": "Bearer test-api-key"}) as client:
        yield adapter, factory, client
    await asyncio.gather(*list(adapter._active_run_tasks.values()), return_exceptions=True)
    adapter._run_idempotency_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [
    [], "credential-must-not-be-echoed", {"primary": "credential-must-not-be-echoed"},
    {"primary": {"token": "x" * 16384}}, {"primary": {"token": "é" * 8192}},
    {"primary": {"nested": [[[[[[[[[0]]]]]]]]]}},
    {"primary": {"bad": float("nan")}}, {"primary": {"bad": "\ud800"}},
    {"": {}}, {"bad\nname": {}},
    {"primary": {"progressToken": "cannot-override"}},
    {"primary": {"modelcontextprotocol.io/reserved": "cannot-override"}},
    {"primary": {"tools.mcp.com/reserved": "cannot-override"}},
])
async def test_invalid_metadata_never_allocates_a_run(api, invalid):
    adapter, factory, client = api
    response = await client.post("/v1/runs", json={"input": "hello", "mcp_meta": invalid})
    payload = await response.json()
    assert response.status == 400
    assert payload["error"]["code"] == "invalid_mcp_meta"
    assert "credential-must-not-be-echoed" not in json.dumps(payload)
    factory.assert_not_called()
    assert not adapter._run_statuses and not adapter._active_run_tasks
    assert not adapter._run_streams and not adapter._run_owners


@pytest.fixture
def wire(monkeypatch):
    """The same SDK session serves concurrent callers on Hermes' dedicated MCP loop."""
    records = []
    ready = threading.Event()
    stop = None
    mcp_tool_loop._ensure_mcp_loop()
    registry = ToolRegistry()

    async def call_tool(_ctx, params):
        records.append(params.model_dump(by_alias=True, exclude_none=True))
        return types.CallToolResult(content=[types.TextContent(type="text", text="ok")])

    async def list_tools(_ctx, _params):
        return types.ListToolsResult(tools=[types.Tool(
            name="probe", input_schema={"type": "object", "additionalProperties": True})])

    async def serve():
        nonlocal stop
        stop = asyncio.Event()
        c_send, s_read = anyio.create_memory_object_stream(8)
        s_send, c_read = anyio.create_memory_object_stream(8)
        peer = Server("metadata-test", on_call_tool=call_tool, on_list_tools=list_tools)
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(peer.run, s_read, s_send, peer.create_initialization_options())
                async with ClientSession(c_read, c_send) as session:
                    await session.initialize()
                    await session.list_tools()
                    for name in ("primary", "other"):
                        server = mcp_tool.MCPServerTask(name)
                        server.session = session
                        monkeypatch.setitem(mcp_tool._servers, name, server)
                        registry.register(
                            name=name, toolset="test-mcp", schema={"name": name, "parameters": {"type": "object"}},
                            handler=_make_tool_handler(name, "probe", 10))
                    ready.set()
                    await stop.wait()
                group.cancel_scope.cancel()
        finally:
            ready.set()

    future = asyncio.run_coroutine_threadsafe(serve(), mcp_tool._mcp_loop)
    try:
        if not ready.wait(15):
            future.result(timeout=1)  # Surface SDK setup errors rather than hiding them behind a timeout.
            pytest.fail("MCP peer did not start")
        if future.done():
            future.result()
        yield records, registry
    finally:
        if stop is not None:
            mcp_tool._mcp_loop.call_soon_threadsafe(stop.set)
        try:
            future.result(timeout=15)
        finally:
            mcp_tool_loop._stop_mcp_loop()


async def finish(adapter, run_id):
    task = adapter._active_run_tasks.get(run_id)
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), 20)
    return adapter._run_statuses[run_id]


@pytest.mark.asyncio
async def test_concurrent_runs_and_delegates_keep_server_scoped_metadata(api, wire, monkeypatch, tmp_path, caplog):
    from tools.delegate_tool_child_run import _ChildRun
    from tools.delegate_tool_dispatch import _run_children_parallel

    adapter, factory, client = api
    records, registry = wire
    capabilities = await client.get("/v1/capabilities")
    assert capabilities.status == 200
    assert (await capabilities.json())["features"]["runs_mcp_meta"] == {"format": "per_server", "max_bytes": 16384}
    rendezvous = threading.Barrier(2, timeout=15)
    cancelled = threading.Event()
    release = threading.Event()
    assertions = {name: {"example.com/assertion": f"secret-{name}", "com.example.mcp/vendor": True,
                         "nested": {"user": name}}
                  for name in ("alice", "bob", "failure", "cancel")}

    def invoke(server, label):
        # A model-supplied lookalike remains an ordinary argument, never transport metadata.
        result = registry.get_entry(server).handler({"label": label, "_meta": {"example.com/assertion": "spoof"}})
        assert json.loads(result) == {"result": "ok"}

    def run(user_message, **_kwargs):
        label = user_message
        if label in ("alice", "bob"):
            rendezvous.wait()
        invoke("primary", label)
        invoke("other", label)
        if label in ("alice", "bob"):
            # Exercise both actual delegation executor hops, without making an LLM call.
            def run_child(i, _task, child):
                work = _ChildRun(child, None, i, label, None, None)
                result, error, deferred = work.await_child()
                assert error is None and not deferred
                return {"task_index": i, "status": "completed", "summary": result["final_response"]}

            def child_run(**_kw):
                invoke("primary", label + "-child")
                return {"final_response": "done"}

            children = [(i, {"goal": "probe"}, SimpleNamespace(session_id=f"child-{label}-{i}", run_conversation=child_run))
                        for i in range(2)]
            batch = SimpleNamespace(parent_agent=None, task_list=[t for _, t, _ in children], children=children,
                                    max_children=2, live_deleg_id=None, run_child=run_child)
            outcomes = []
            _run_children_parallel(batch, outcomes, honor_parent_interrupt=True)
            assert len(outcomes) == 2 and all(r["status"] == "completed" for r in outcomes)
        if label == "failure":
            raise RuntimeError("planned test failure")
        if label == "cancel":
            cancelled.set()
            assert release.wait(15)
            return {"final_response": "stopped", "interrupted": True}
        return {"final_response": "done"}

    agent = MagicMock()
    agent.run_conversation.side_effect = run
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    agent.interrupt.side_effect = lambda *_a, **_kw: release.set()
    factory.return_value = agent

    async def start(label, **extra):
        payload = {"input": label, "session_id": "test-" + label, **extra}
        response = await client.post("/v1/runs", json=payload, headers={"Idempotency-Key": "test-" + label})
        assert response.status == 202, await response.text()
        return (await response.json())["run_id"]

    run_ids = await asyncio.gather(*(start(name, mcp_meta={"primary": assertions[name]}) for name in ("alice", "bob")))
    for run_id in run_ids:
        assert (await finish(adapter, run_id))["status"] == "completed"
    # Idempotency replay cannot substitute credentials or cause a second execution.
    assert await start("alice", mcp_meta={"primary": assertions["alice"]}) == run_ids[0]
    conflict = await client.post("/v1/runs", json={"input": "alice", "session_id": "test-alice",
                                 "mcp_meta": {"primary": assertions["bob"]}},
                                 headers={"Idempotency-Key": "test-alice"})
    assert conflict.status == 409
    failed = await start("failure", mcp_meta={"primary": assertions["failure"]})
    assert (await finish(adapter, failed))["status"] == "failed"
    stopped = await start("cancel", mcp_meta={"primary": assertions["cancel"]})
    assert await asyncio.to_thread(cancelled.wait, 15)
    assert (await client.post(f"/v1/runs/{stopped}/stop")).status == 200
    assert (await finish(adapter, stopped))["status"] == "cancelled"
    plain = await start("plain")
    assert (await finish(adapter, plain))["status"] == "completed"
    # Empty values keep existing behavior; unknown/case-mismatched names cannot
    # accidentally select a different destination after tool-name normalization.
    for label, value in (("null", None), ("empty", {}), ("target-empty", {"primary": {}}),
                         ("unknown", {"Primary": assertions["alice"]})):
        run_id = await start(label, mcp_meta=value)
        assert (await finish(adapter, run_id))["status"] == "completed"
        calls = [r for r in records if r["arguments"]["label"] == label]
        assert len(calls) == 2 and all(not r.get("_meta") for r in calls)

    # Simulate one expired transport at the SDK boundary. The real recovery
    # dispatcher retries with a fresh copy, even if that attempt mutated its input.
    sdk_session = mcp_tool._servers["primary"].session
    sdk_call = sdk_session.call_tool
    attempts = []

    async def expire_once(name, arguments, **kwargs):
        if arguments["label"] == "retry" and kwargs.get("meta"):
            attempts.append(json.loads(json.dumps(kwargs["meta"])))
            if len(attempts) == 1:
                kwargs["meta"]["nested"]["user"] = "must-not-survive"
                raise RuntimeError("Session expired")
        return await sdk_call(name, arguments=arguments, **kwargs)

    with monkeypatch.context() as recovery:
        recovery.setattr(sdk_session, "call_tool", expire_once)
        recovery.setattr(mcp_tool_loop, "_signal_reconnect_and_wait", lambda *_a, **_kw: True)
        retried = await start("retry", mcp_meta={"primary": assertions["alice"]})
        assert (await finish(adapter, retried))["status"] == "completed"
    assert attempts == [assertions["alice"], assertions["alice"]]
    retry_calls = [r for r in records if r["arguments"]["label"] == "retry"]
    assert len(retry_calls) == 2 and retry_calls[0]["_meta"] == assertions["alice"]
    assert not retry_calls[1].get("_meta")

    for label, assertion in assertions.items():
        matching = [r for r in records if r["arguments"]["label"] == label]
        assert len(matching) == 2
        assert matching[0]["_meta"] == assertion
        assert not matching[1].get("_meta")  # Other server never receives this credential.
    for label in ("alice", "bob"):
        children = [r for r in records if r["arguments"]["label"] == label + "-child"]
        assert len(children) == 2 and all(r["_meta"] == assertions[label] for r in children)
    assert all(not r.get("_meta") for r in records if r["arguments"]["label"] == "plain")

    events = await client.get(f"/v1/runs/{run_ids[0]}/events")
    exposed = await events.text() + json.dumps(adapter._run_statuses) + caplog.text
    exposed += repr(factory.call_args_list)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            exposed += path.read_bytes().decode("utf-8", errors="ignore")
    assert all(assertion["example.com/assertion"] not in exposed for assertion in assertions.values())

    # A legacy transport cannot silently turn an authenticated call into an anonymous one.
    from tools.mcp_run_meta import parse_mcp_run_meta, reset_mcp_run_meta, set_mcp_run_meta

    class LegacySession:
        calls = 0

        async def call_tool(self, name, arguments):
            self.calls += 1
            return types.CallToolResult(content=[])

    legacy = mcp_tool.MCPServerTask("legacy")
    legacy.session = LegacySession()
    monkeypatch.setitem(mcp_tool._servers, "legacy", legacy)
    token = set_mcp_run_meta(parse_mcp_run_meta({"legacy": assertions["alice"]}))
    try:
        result = await asyncio.to_thread(_make_tool_handler("legacy", "probe", 10), {})
        assert "error" in json.loads(result)
        assert legacy.session.calls == 0
    finally:
        reset_mcp_run_meta(token)
