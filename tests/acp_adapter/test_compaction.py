"""ACP compaction lifecycle wire contracts for #137752, without model calls."""

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import acp
import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from gateway.warning_notifications import DiagnosticText
from agent.compaction_lifecycle import _CompactionLifecycle
from agent.conversation_compression import (
    COMPACTION_STATUS, COMPACTION_HEARTBEAT_STATUS,
)


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
                                **(agent.session_updates.connection_options() if hasattr(agent, "session_updates") else {}))
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


@pytest.mark.asyncio
@pytest.mark.parametrize("capabilities", [{}, {"session": None}, {"session": {}},
    {"session": {"compaction": None}}, {"session": {"compaction": True}},
    {"session": {"compaction": {}}}])
async def test_compaction_opt_in_and_stable_entity(capabilities, tmp_path):
    model = MagicMock()
    model.model, model.provider = "test-model", "openrouter"
    model.status_callback = model.compaction_callback = None

    def run(**kwargs):
        callback = model.status_callback
        if callable(callback):
            callback("lifecycle", COMPACTION_STATUS)
            callback("lifecycle", COMPACTION_HEARTBEAT_STATUS)
            callback("warning", "Compression is currently blocked")
            callback("lifecycle", DiagnosticText("Compression failed earlier; cached warning"))
        cycle = _CompactionLifecycle(model, True)
        cycle.commit_status = "committed"
        cycle.complete()
        cycle.complete()  # terminal callback is one-shot
        if callable(callback):
            callback("lifecycle", COMPACTION_HEARTBEAT_STATUS)
        if callable(callback):
            callback("lifecycle", COMPACTION_STATUS)
        failed = _CompactionLifecycle(model, True)
        failed.complete()
        return {"final_response": "done", "messages": []}

    model.run_conversation.side_effect = run
    agent = HermesACPAgent(SessionManager(agent_factory=lambda: model))
    async with connected(agent) as client:
        await client.request("initialize", {"protocolVersion": 1, "clientCapabilities": capabilities})
        created = await client.request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
        root = created["result"]["sessionId"]
        response = await client.request("session/prompt", {"sessionId": root,
                                       "prompt": [{"type": "text", "text": "compress"}]})
        assert response["result"]["stopReason"] == "end_turn"
        updates = [p for p in client.updates if p["update"]["sessionUpdate"] == "compaction_update"]
        if capabilities != {"session": {"compaction": {}}}:
            assert updates == []
            return
        assert [p["update"]["status"] for p in updates] == [
            "in_progress", "in_progress", "completed", "in_progress", "failed"]
        first, second = updates[0]["update"]["compactionId"], updates[3]["update"]["compactionId"]
        assert first != second
        assert all(p["sessionId"] == root for p in updates)
        assert [p["update"]["compactionId"] for p in updates] == [first, first, first, second, second]
        assert all("summary" not in p["update"] for p in updates)
        assert model.status_callback is None
        assert model.compaction_callback is None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancelled", "unfinished", "interrupted"])
async def test_turn_exit_closes_announced_compaction(outcome, tmp_path):
    model = MagicMock()
    model.model, model.provider = "test-model", "openrouter"
    model.status_callback = model.compaction_callback = None
    model._interrupt_requested = False
    late_callbacks = []

    def run(**kwargs):
        late_callbacks.append(model.status_callback)
        model.status_callback("lifecycle", COMPACTION_STATUS)
        if outcome == "cancelled":
            _CompactionLifecycle(model, True).complete(force_terminal=True)
        elif outcome == "interrupted":
            model._interrupt_requested = True
        return {"final_response": "done", "messages": []}

    model.run_conversation.side_effect = run
    agent = HermesACPAgent(SessionManager(agent_factory=lambda: model))
    async with connected(agent) as client:
        await client.request("initialize", {"protocolVersion": 1,
                            "clientCapabilities": {"session": {"compaction": {}}}})
        created = await client.request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
        await client.request("session/prompt", {"sessionId": created["result"]["sessionId"],
                            "prompt": [{"type": "text", "text": "compress"}]})
        await asyncio.to_thread(late_callbacks[0], "lifecycle", COMPACTION_HEARTBEAT_STATUS)
        await client.request("session/list", {})  # drain the same ordered stream
        updates = [p["update"] for p in client.updates if p["update"]["sessionUpdate"] == "compaction_update"]
        assert [u["status"] for u in updates] == ["in_progress", "failed" if outcome == "unfinished" else "cancelled"]
        assert updates[0]["compactionId"] == updates[1]["compactionId"]
        assert model.status_callback is None and model.compaction_callback is None


@pytest.mark.asyncio
@pytest.mark.parametrize("save_fails", [False, True])
async def test_manual_compress_uses_same_wire_lifecycle(monkeypatch, tmp_path, save_fails):
    from types import SimpleNamespace

    model = MagicMock()
    model.model, model.provider = "test-model", "openrouter"
    model.status_callback = model.compaction_callback = None
    model._interrupt_requested = False

    def compress(agent, history, request, **kwargs):
        agent.status_callback("lifecycle", COMPACTION_STATUS)
        cycle = _CompactionLifecycle(agent, True)
        cycle.commit_status = "committed"
        cycle.complete()
        return SimpleNamespace(status="compressed", before_messages=history, after_messages=[],
                               before_tokens=100, after_tokens=20)

    monkeypatch.setattr("agent.conversation_compression_manual.compress_now", compress)
    agent = HermesACPAgent(SessionManager(agent_factory=lambda: model))
    async with connected(agent) as client:
        await client.request("initialize", {"protocolVersion": 1,
                            "clientCapabilities": {"session": {"compaction": {}}}})
        created = await client.request("session/new", {"cwd": str(tmp_path), "mcpServers": []})
        root = created["result"]["sessionId"]
        agent.session_manager.get_session(root).history = [{"role": "user", "content": "earlier"}]
        if save_fails:
            def failed_save(session_id):
                raise RuntimeError("persistence failed")
            monkeypatch.setattr(agent.session_manager, "save_session", failed_save)
        response = await client.request("session/prompt", {"sessionId": root,
                                       "prompt": [{"type": "text", "text": "/compress"}]})
        assert response["result"]["stopReason"] == "end_turn"
        updates = [p["update"] for p in client.updates if p["update"]["sessionUpdate"] == "compaction_update"]
        assert [u["status"] for u in updates] == ["in_progress", "failed" if save_fails else "completed"]
        assert updates[0]["compactionId"] == updates[1]["compactionId"]
        assert model.status_callback is None and model.compaction_callback is None
