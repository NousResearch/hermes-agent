"""Wire-level opt-in delegation lifecycle contracts for #137069; no model calls."""

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import acp
import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from tools.delegate_tool_progress import _build_child_progress_callback


class WireClient:
    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer
        self.sequence = 0
        self.updates = []

    async def request(self, method, params):
        self.sequence += 1
        request_id = self.sequence
        self.writer.write((json.dumps({"jsonrpc": "2.0", "id": request_id,
                                      "method": method, "params": params}) + "\n").encode())
        await self.writer.drain()
        while True:
            line = await asyncio.wait_for(self.reader.readline(), timeout=10)
            assert line, "ACP transport closed before response"
            message = json.loads(line)
            if message.get("method") == "session/update":
                self.updates.append(message["params"])
            if message.get("id") == request_id:
                return message


@asynccontextmanager
async def connected(agent):
    finished = asyncio.get_running_loop().create_future()

    async def serve(reader, writer):
        try:
            await acp.run_agent(agent, writer, reader, use_unstable_protocol=True,
                                **(agent.subagent_events.connection_options() if hasattr(agent, "subagent_events") else {}))
        except Exception as exc:
            finished.set_exception(exc)
        else:
            finished.set_result(None)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
    try:
        yield WireClient(reader, writer)
    finally:
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(finished, timeout=10)
        server.close()
        await server.wait_closed()


def emit_delegation(model):
    """Actual delegate relay fed at the model execution boundary."""
    callback = model.tool_progress_callback
    callback("tool.started", "delegate_task", "research", {"goal": "research"})
    first = _build_child_progress_callback(0, "Research", model, task_count=2, subagent_id="sa-first")
    second = _build_child_progress_callback(1, "Verify", model, task_count=2, subagent_id="sa-second")
    first("subagent.start")
    second("subagent.start")
    first("tool.started", "terminal", "checking", {"command": "pwd"})
    nested_parent = MagicMock()
    nested_parent.tool_progress_callback = first
    nested_parent._delegate_spinner = None
    nested = _build_child_progress_callback(0, "Nested", nested_parent,
                                            subagent_id="sa-nested", parent_id="sa-first")
    nested("subagent.start")
    nested("tool.started", "web_search", "checking nested work", {"query": "test"})
    nested("subagent.complete", status="completed")
    second("subagent.complete", status="interrupted")
    first("subagent.complete", status="completed")
    callback("tool.completed", "delegate_task", "done", {"result": "done"})
    return {"final_response": "done", "messages": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("capabilities", [{}, {"subagents": None}, {"subagents": False}, {"subagents": {}}])
async def test_delegation_lifecycle_over_sdk_transport(capabilities, tmp_path):
    model = MagicMock()
    model._delegate_spinner = None
    model.model, model.provider = "test-model", "openrouter"
    model.run_conversation.side_effect = lambda **kwargs: emit_delegation(model)
    agent = HermesACPAgent(SessionManager(agent_factory=lambda: model))
    async with connected(agent) as client:
        initialized = await client.request("initialize", {"protocolVersion": 1, "clientCapabilities": capabilities})
        assert "result" in initialized
        created = await client.request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
        root = created["result"]["sessionId"]
        result = await client.request("session/prompt", {"sessionId": root,
                                      "prompt": [{"type": "text", "text": "research"}]})
        assert result["result"]["stopReason"] == "end_turn"
        associations = [p for p in client.updates if p["update"]["sessionUpdate"] == "subagent_update"]
        parent_tools = [p for p in client.updates if p["update"]["sessionUpdate"] in ("tool_call", "tool_call_update")]
        assert parent_tools and all(p["sessionId"] == root for p in parent_tools)
        if capabilities != {"subagents": {}}:
            assert associations == []
            return
        starts = [p for p in associations if "capabilities" in p["update"]]
        assert len(starts) == 3
        first, second, nested = [p["update"]["sessionId"] for p in starts]
        assert len({root, first, second, nested}) == 4
        assert [p["sessionId"] for p in starts] == [root, root, first]
        assert all(p["update"]["capabilities"] == {} for p in starts)
        assert all(p["update"]["state"]["state"] == "running" for p in starts)
        finishes = {p["update"]["sessionId"]: p["update"]["state"] for p in associations
                    if p["update"]["state"]["state"] == "idle"}
        assert finishes == {first: {"state": "idle", "stopReason": "end_turn"},
                            second: {"state": "idle", "stopReason": "cancelled"},
                            nested: {"state": "idle", "stopReason": "end_turn"}}
        progress = [p for p in associations if p["update"].get("_meta", {}).get("hermes", {}).get("tool")]
        assert progress[0]["update"]["sessionId"] == first
        assert progress[0]["update"]["_meta"]["hermes"]["tool"] == "terminal"
        assert progress[1]["sessionId"] == first
        assert progress[1]["update"]["sessionId"] == nested
        assert progress[1]["update"]["_meta"]["hermes"]["tool"] == "web_search"
        with pytest.raises(acp.RequestError):
            await agent.cancel(first)
        for method, extra in (("session/prompt", {"prompt": [{"type": "text", "text": "bad"}]}),
                              ("session/resume", {"cwd": str(tmp_path), "mcpServers": []}),
                              ("session/set_mode", {"modeId": "default"}),
                              ("session/load", {"cwd": str(tmp_path), "mcpServers": []}),
                              ("session/fork", {"cwd": str(tmp_path), "mcpServers": []}),
                              ("session/set_model", {"modelId": "test-model"}),
                              ("session/set_config_option", {"configId": "approval_mode", "value": "auto"})):
            rejected = await client.request(method, {"sessionId": first, **extra})
            assert rejected["error"]["code"] == -32602
        assert model.run_conversation.call_count == 1


@pytest.mark.asyncio
async def test_failed_announcement_retries_without_guessing_parentage():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from acp.connection import StreamDirection
    from acp_adapter.subagents import SubagentEvents

    events = SubagentEvents()
    events.observe(SimpleNamespace(direction=StreamDirection.INCOMING, message={
        "method": "initialize", "params": {"clientCapabilities": {"subagents": {}}}}))
    events._sender = SimpleNamespace(send=AsyncMock(side_effect=[RuntimeError("offline"), None, None, None]))
    loop = asyncio.get_running_loop()
    callback = events.wrap_progress("root", loop, lambda *args, **kwargs: None)
    await asyncio.to_thread(callback, "subagent.start", subagent_id="nested", parent_id="missing")
    events._sender.send.assert_not_awaited()
    await asyncio.to_thread(callback, "subagent.start", subagent_id="child", goal="Retry")
    await asyncio.to_thread(callback, "subagent.start", subagent_id="child", goal="Retry")
    first, retry = [c.args[0]["params"]["update"] for c in events._sender.send.await_args_list]
    assert first == retry
    assert retry["capabilities"] == {}
    await asyncio.to_thread(callback, "subagent.complete", subagent_id="child", status="timeout")
    timed_out = events._sender.send.await_args.args[0]["params"]["update"]
    assert timed_out["state"] == {"state": "unknown"}
    assert timed_out["_meta"]["hermes"]["status"] == "timeout"
    await asyncio.to_thread(callback, "subagent.complete", subagent_id="child", status="failed")
    failed = events._sender.send.await_args.args[0]["params"]["update"]
    assert failed["state"] == {"state": "idle"}
    assert failed["sessionId"] == retry["sessionId"] == timed_out["sessionId"]
    assert "capabilities" not in failed
