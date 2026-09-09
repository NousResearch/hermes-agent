"""Credential-bearing calls fail closed on the current bare-name MCP registry.

Two real SDK peers stay connected throughout. Until the profile-qualified registry
in #99594 lands, only its owning profile may use a colliding name, not both at once.
"""

import asyncio
import json
import threading
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from mcp import types
from mcp.client.session import ClientSession
from mcp.server import Server

from agent import secret_scope
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import _profile_runtime_scope
from hermes_cli import profiles
from tools import mcp_tool, mcp_tool_discovery, mcp_tool_handlers, mcp_tool_loop
from tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_profile_collision_never_sends_assertions_to_a_foreign_peer(monkeypatch, tmp_path):
    from tools.delegate_tool_child_run import _ChildRun
    from tools.delegate_tool_dispatch import _run_children_parallel

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    homes = {name: profiles._get_profiles_root() / name for name in ("alpha", "beta")}
    keys = {name: ("a" if name == "alpha" else "b") * 32 for name in homes}
    for name, home in homes.items():
        assert home.is_relative_to(tmp_path)
        home.mkdir(parents=True)
        (home / ".env").write_text(f"API_SERVER_KEY={keys[name]}\n", encoding="utf-8")
        (home / "config.yaml").write_text("mcp_servers: {}\n", encoding="utf-8")

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = SimpleNamespace(config=SimpleNamespace(
        multiplex_profiles=True, multiplex_profile_allowlist=list(homes)))
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post("/p/{profile}/v1/runs", adapter._handle_runs)
    records, servers, results = {name: [] for name in homes}, {}, {}
    ready, acquired = threading.Event(), threading.Event()
    stop = None
    registry = ToolRegistry()
    registry.register(name="github", toolset="mcp-test", schema={"name": "github", "parameters": {"type": "object"}},
                      handler=mcp_tool_handlers._make_tool_handler("github", "probe", 10))

    def invoke(label):
        results[label] = json.loads(registry.get_entry("github").handler({"label": label}))
        return {"final_response": "done"}

    def run(user_message, **_kwargs):
        invoke(user_message)
        if "-owns-" in user_message:
            def run_child(i, _task, child):
                work = _ChildRun(child, None, i, user_message, None, None)
                result, error, deferred = work.await_child()
                assert error is None and not deferred
                return {"task_index": i, "status": "completed", "summary": result["final_response"]}

            children = [(i, {"goal": "probe"}, SimpleNamespace(session_id=f"{user_message}-child-{i}",
                         run_conversation=lambda i=i, **_kw: invoke(f"{user_message}-child-{i}"))) for i in range(2)]
            batch = SimpleNamespace(parent_agent=None, task_list=[t for _, t, _ in children], children=children,
                                    max_children=2, live_deleg_id=None, run_child=run_child)
            outcomes = []
            _run_children_parallel(batch, outcomes, honor_parent_interrupt=True)
            assert len(outcomes) == 2 and all(r["status"] == "completed" for r in outcomes)
        return {"final_response": "done"}

    agent = MagicMock()
    agent.run_conversation.side_effect = run
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kw: agent)

    async def serve():
        nonlocal stop
        stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                group = await stack.enter_async_context(anyio.create_task_group())
                for profile in homes:
                    async def call(_ctx, params, owner=profile):
                        records[owner].append(params.model_dump(by_alias=True, exclude_none=True))
                        return types.CallToolResult(content=[types.TextContent(type="text", text="ok")])

                    async def list_tools(_ctx, _params):
                        return types.ListToolsResult(tools=[types.Tool(name="probe", input_schema={"type": "object"})])

                    # Independent endpoints, sessions and wire queues; never one shared fake call_tool.
                    c_send, s_read = anyio.create_memory_object_stream(8)
                    s_send, c_read = anyio.create_memory_object_stream(8)
                    peer = Server(profile, on_call_tool=call, on_list_tools=list_tools)
                    group.start_soon(peer.run, s_read, s_send, peer.create_initialization_options())
                    session = await stack.enter_async_context(ClientSession(c_read, c_send))
                    await session.initialize()
                    with _profile_runtime_scope(homes[profile]):
                        server = mcp_tool.MCPServerTask("github")
                        server.session = session
                        servers[profile] = server
                ready.set()
                await stop.wait()
                group.cancel_scope.cancel()
        finally:
            ready.set()

    async def publish(profile):
        with _profile_runtime_scope(homes[profile]):
            mcp_tool_discovery._adopt_server("github", servers[profile])

    async def on_loop(coro):
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, mcp_tool._mcp_loop))

    mcp_tool_loop._ensure_mcp_loop()
    peer_future = asyncio.run_coroutine_threadsafe(serve(), mcp_tool._mcp_loop)
    try:
        assert await asyncio.to_thread(ready.wait, 15)
        if peer_future.done():
            peer_future.result()
        # Keep dictionary cleanup hermetic even though adoption uses the real publishing path.
        monkeypatch.setitem(mcp_tool._servers, "github", None)
        monkeypatch.setitem(mcp_tool._server_scope_keys, "github", None)
        async with TestClient(TestServer(app)) as client:
            wrong_key = await client.post("/p/beta/v1/runs", headers={"Authorization": f"Bearer {keys['alpha']}"},
                                          json={"input": "must-not-run"})
            assert wrong_key.status == 401 and not results
            async def start(profile, label):
                response = await client.post(f"/p/{profile}/v1/runs", headers={"Authorization": f"Bearer {keys[profile]}"},
                    json={"input": label, "session_id": label,
                          "mcp_meta": {"github": {"com.example.mcp/assertion": "credential-" + profile}}})
                assert response.status == 202, await response.text()
                return (await response.json())["run_id"]

            async def finish(run_id):
                task = adapter._active_run_tasks.get(run_id)
                if task is not None:
                    await asyncio.wait_for(asyncio.shield(task), 20)
                assert adapter._run_statuses[run_id]["status"] == "completed"

            # Both profiles are really served by /p/<profile>, with independent API keys.
            for owner in homes:
                await on_loop(publish(owner))
                ids = await asyncio.gather(*(start(profile, f"{owner}-owns-{profile}") for profile in homes))
                await asyncio.gather(*(finish(run_id) for run_id in ids))
                for profile in homes:
                    for suffix in ("", "-child-0", "-child-1"):
                        result = results[f"{owner}-owns-{profile}{suffix}"]
                        assert ("error" in result) == (profile != owner)
            for profile in homes:
                assert len(records[profile]) == 3
                assert all(call["_meta"] == {"com.example.mcp/assertion": "credential-" + profile}
                           for call in records[profile])

            # Admission/acquisition is not enough: ownership can change while queued on _rpc_lock.
            await on_loop(publish("alpha"))
            await on_loop(servers["alpha"]._rpc_lock.acquire())
            acquire_server = mcp_tool_handlers._acquire_call_server

            def observe_acquisition(*args):
                result = acquire_server(*args)
                acquired.set()
                return result

            with monkeypatch.context() as queue:
                queue.setattr(mcp_tool_handlers, "_acquire_call_server", observe_acquisition)
                queued = await start("alpha", "queued")
                assert await asyncio.to_thread(acquired.wait, 15)
                await on_loop(publish("beta"))
                mcp_tool._mcp_loop.call_soon_threadsafe(servers["alpha"]._rpc_lock.release)
                await finish(queued)
            assert "error" in results["queued"]

            # A foreign lazy entry must be refused before reconnect/spawn, not after it.
            with monkeypatch.context() as lazy:
                lazy.delitem(mcp_tool._servers, "github")
                lazy.setitem(mcp_tool._lazy_server_configs, "github", {"url": "https://beta.invalid/mcp"})
                spawn = MagicMock(return_value=False)
                lazy.setattr(mcp_tool_discovery, "_ensure_lazy_server_connected", spawn)
                await finish(await start("alpha", "foreign-lazy"))
                spawn.assert_not_called()
                assert "error" in results["foreign-lazy"]

            # Recovery must not touch a replacement owned by another profile either.
            await on_loop(publish("alpha"))
            async def expired(*_args, **_kwargs):
                await publish("beta")
                raise RuntimeError("Session expired")

            with monkeypatch.context() as retry:
                retry.setattr(servers["alpha"].session, "call_tool", expired)
                reconnect = MagicMock(return_value=True)
                retry.setattr(mcp_tool_loop, "_signal_reconnect_and_wait", reconnect)
                await finish(await start("alpha", "foreign-retry"))
                reconnect.assert_not_called()
                assert "error" in results["foreign-retry"]
            with monkeypatch.context() as unknown:
                unknown.delitem(mcp_tool._server_scope_keys, "github")
                await finish(await start("beta", "unknown-owner"))
                assert "error" in results["unknown-owner"]

            # Even a matching default-home lookup is not authority when the request lost its scope.
            from hermes_constants import reset_hermes_home_override, set_hermes_home_override
            from tools.mcp_run_meta import parse_mcp_run_meta, reset_mcp_run_meta, set_mcp_run_meta

            with _profile_runtime_scope(homes["alpha"]):
                meta_token = set_mcp_run_meta(parse_mcp_run_meta({"github": {"com.example.mcp/assertion": "alpha-bound"}}))
                try:
                    # Merely switching the ambient profile must not retarget inherited credentials.
                    with _profile_runtime_scope(homes["beta"]):
                        switched = await asyncio.to_thread(registry.get_entry("github").handler, {"label": "switched"})
                        assert "error" in json.loads(switched)
                finally:
                    reset_mcp_run_meta(meta_token)

            scope_token = set_hermes_home_override(None)
            try:
                with pytest.raises(ValueError, match="explicit profile scope"):
                    set_mcp_run_meta(parse_mcp_run_meta({"github": {"com.example.mcp/assertion": "unscoped"}}))
            finally:
                reset_hermes_home_override(scope_token)
            assert all(len(calls) == 3 for calls in records.values())
    finally:
        if stop is not None:
            mcp_tool._mcp_loop.call_soon_threadsafe(stop.set)
        try:
            await asyncio.wait_for(asyncio.wrap_future(peer_future), 20)
        finally:
            mcp_tool_loop._stop_mcp_loop()
            await adapter.disconnect()
