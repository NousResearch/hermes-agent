"""Real HTTP approval round trips through the registry and plugin policy gate."""

import asyncio
import json
import itertools
import time

import httpx
import pytest

import mcp_serve
from hermes_cli import plugins
from tools import approval
from tools.registry import registry

_request_ids = itertools.count(1)


@pytest.fixture
def guarded_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "approvals:\n  mode: manual\n  timeout: 2\n", encoding="utf-8"
    )
    manager = plugins.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    executed = []
    name = "mcp_test_guarded"
    manager._hooks["pre_tool_call"] = [
        lambda tool_name, **kwargs: (
            {"action": "approve", "message": "Human required"}
            if tool_name == name
            else None
        )
    ]
    registry.register(
        name=name,
        toolset="mcp-test",
        schema={
            "name": name,
            "description": "Approval test",
            "parameters": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            },
        },
        handler=lambda args, **kwargs: (
            executed.append(args["label"]) or json.dumps({"ran": args["label"]})
        ),
    )
    bridge = mcp_serve.EventBridge()
    server = mcp_serve.create_mcp_server(event_bridge=bridge, expose_tools=[name])
    try:
        yield server, bridge, executed
    finally:
        bridge.stop()
        registry.deregister(name)


async def _call(client, name, arguments=None):
    response = await asyncio.wait_for(
        client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": next(_request_ids),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments or {}},
            },
        ),
        5,
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    return json.loads("".join(block.get("text", "") for block in result["content"]))


async def _pending(client, count):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        result = await _call(client, "permissions_list_open")
        if result["count"] == count:
            return result["approvals"]
        await asyncio.sleep(0.02)
    pytest.fail(f"Expected {count} real pending approvals; last result: {result}")


@pytest.mark.asyncio
async def test_http_approval_resolves_only_the_selected_guarded_call(guarded_server):
    server, bridge, executed = guarded_server
    app = mcp_serve.create_streamable_http_app(
        server,
        auth_config=mcp_serve.McpHttpAuthConfig(psk="test-secret"),
        json_response=True,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8666",
            headers={
                "Authorization": "Bearer test-secret",
                "Accept": "application/json, text/event-stream",
            },
        ) as client:
            initialized = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "approval-test", "version": "1"},
                    },
                },
            )
            assert initialized.status_code == 200
            client.headers["Mcp-Session-Id"] = initialized.headers["Mcp-Session-Id"]
            client.headers["MCP-Protocol-Version"] = "2025-06-18"
            calls = [
                asyncio.create_task(_call(client, "mcp_test_guarded", {"label": label}))
                for label in ["one", "two"]
            ]
            try:
                pending = await _pending(client, 2)
                assert len({item["session_key"] for item in pending}) == 2
                assert executed == []
                events = await _call(client, "events_poll")
                assert (
                    len([
                        e for e in events["events"] if e["type"] == "approval_requested"
                    ])
                    == 2
                )
                invalid = await _call(
                    client,
                    "permissions_respond",
                    {"id": "unknown", "decision": "allow-once"},
                )
                assert "error" in invalid
                selected = pending[0]
                assert "error" in mcp_serve.EventBridge().respond_to_approval(
                    selected["id"], "allow-once"
                )
                response = await _call(
                    client,
                    "permissions_respond",
                    {"id": selected["id"], "decision": "allow-once"},
                )
                assert response["resolved"] is True
                remaining = await _pending(client, 1)
                assert remaining[0]["id"] != selected["id"]
                replay = await _call(
                    client,
                    "permissions_respond",
                    {"id": selected["id"], "decision": "allow-once"},
                )
                assert "error" in replay
                denied = await _call(
                    client,
                    "permissions_respond",
                    {"id": remaining[0]["id"], "decision": "deny"},
                )
                assert denied["resolved"] is True
                results = await asyncio.wait_for(asyncio.gather(*calls), 4)
                assert sum("ran" in result for result in results) == 1
                assert len(executed) == 1
                assert (await _call(client, "permissions_list_open"))["count"] == 0
                assert all(
                    not approval.list_gateway_approvals(item["session_key"])
                    for item in pending
                )
            finally:
                bridge.stop()
                for call in calls:
                    if not call.done():
                        call.cancel()
                await asyncio.gather(*calls, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ["timeout", "shutdown"])
async def test_approval_end_fails_closed_and_rejects_stale_responses(
    guarded_server, end
):
    server, bridge, executed = guarded_server

    def decode(result):
        return json.loads("".join(block.text for block in result.content))

    call = asyncio.create_task(server.call_tool("mcp_test_guarded", {"label": "never"}))
    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            pending = decode(await server.call_tool("permissions_list_open", {}))[
                "approvals"
            ]
            if pending:
                break
            await asyncio.sleep(0.02)
        assert pending, "Guarded call never published an approval"
        if end == "shutdown":
            bridge.stop()
        result = decode(await asyncio.wait_for(call, 4))
        assert "error" in result
        assert executed == []
        assert decode(await server.call_tool("permissions_list_open", {}))["count"] == 0
        assert not approval.list_gateway_approvals(pending[0]["session_key"])
        stale = decode(
            await server.call_tool(
                "permissions_respond",
                {"id": pending[0]["id"], "decision": "allow-once"},
            )
        )
        assert "error" in stale
    finally:
        bridge.stop()
        await asyncio.gather(call, return_exceptions=True)
