"""Subagent visibility contracts through real relay, callbacks, journal and ACP updates."""
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from acp.schema import ClientCapabilities

from acp_adapter.events import make_tool_progress_cb
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionState
from acp_adapter.subagents import SubagentProgress
from tools.delegate_tool_progress import _ChildProgressRelay


class RecordingClient:
    def __init__(self):
        self.updates = []

    async def session_update(self, session_id, update):
        self.updates.append((session_id, update.model_dump(by_alias=True, exclude_none=True)))

    async def request_permission(self, *args, **kwargs):
        raise AssertionError("Visibility must not request control permissions")


def snapshots(client):
    return [update["_meta"]["hermes"]["subagentProgress"] for _, update in client.updates if "_meta" in update]


@pytest.mark.asyncio
async def test_parallel_nested_children_keep_identity_across_turns_and_replay(tmp_path):
    client = RecordingClient()
    server = HermesACPAgent()
    server.on_connect(client)
    await server.initialize(client_capabilities=ClientCapabilities(field_meta={"hermes": {"subagentProgress": 1}}))
    state = SessionState("root", SimpleNamespace(), cwd=str(tmp_path))
    loop = asyncio.get_running_loop()
    first = server._wire_turn_callbacks(state, "root", client, loop).tool_progress_cb
    alpha = _ChildProgressRelay(0, "DO NOT SEND PROMPT", None, first, 2, "alpha", None, 1, None, None, {})
    beta = _ChildProgressRelay(1, "DO NOT SEND PROMPT", None, first, 2, "beta", None, 1, None, None, {})
    nested = _ChildProgressRelay(0, "DO NOT SEND PROMPT", None, alpha, 1, "nested", "alpha", 2, None, None, {})

    def run(relay):
        relay("subagent.start")
        relay("tool.started", "terminal", "DO NOT SEND PREVIEW", {"secret": "DO NOT SEND ARG"})
        relay("subagent.text", preview="Public output. Bearer abcdefghijklmnopqrstuvwxyz123456")
        relay("reasoning.available", preview="DO NOT SEND REASONING")

    await asyncio.gather(*(asyncio.to_thread(run, relay) for relay in (alpha, beta, nested)))
    server._wire_turn_callbacks(state, "root", client, loop)
    for relay, status in ((alpha, "completed"), (beta, "failed"), (nested, "interrupted")):
        await asyncio.to_thread(relay, "subagent.complete", status=status, summary="DO NOT SEND SUMMARY")
    if journal := getattr(state, "subagent_progress", None):
        await journal.drain()
    latest = {node["id"]: node for node in snapshots(client)}
    assert {key: node["status"] for key, node in latest.items()} == {
        "alpha": "completed", "beta": "failed", "nested": "canceled"}
    assert latest["nested"]["parentId"] == "alpha" and latest["nested"]["depth"] == 2
    assert all(node["tools"] == ["terminal"] and "Public output" in node["text"] for node in latest.values())
    encoded = json.dumps(client.updates)
    assert "DO NOT SEND" not in encoded and "abcdefghijklmnopqrstuvwxyz123456" not in encoded
    assert {session_id for session_id, _ in client.updates} == {"root"}
    sequences = [node["sequence"] for node in snapshots(client)]
    assert sequences == sorted(set(sequences))

    # A reconnect replaces the connection; retained old-turn callbacks use the new binding.
    reconnected = RecordingClient()
    server.on_connect(reconnected)
    await server._subagent_progress_for(state).replay()
    assert {node["id"]: node for node in snapshots(reconnected)} == latest


@pytest.mark.asyncio
async def test_journal_is_bounded_private_and_cold_restore_does_not_claim_live_children(tmp_path):
    journal = SubagentProgress("../../untrusted-id", tmp_path)
    client = RecordingClient()
    journal.bind(asyncio.get_running_loop(), lambda: client, metadata=True)
    callback = make_tool_progress_cb(client, "root", asyncio.get_running_loop(), {}, {}, subagent_progress=journal.record)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda index: callback("subagent.start", subagent_id=f"child-{index}", depth=1), range(70)))
    callback("subagent.text", preview="x" * 20000, subagent_id="child-0")
    for _ in range(40):
        callback("subagent.tool", "read_file", args={"private": "DO NOT SEND"}, subagent_id="child-0")
    await journal.drain()
    rows = json.loads(journal.path.read_text(encoding="utf-8-sig"))
    assert journal.path.parent == tmp_path and len(rows) == 64
    child = next(node for node in rows if node["id"] == "child-0")
    assert len(child["text"]) == 16384 and len(child["tools"]) == 32
    restored = SubagentProgress("../../untrusted-id", tmp_path)
    replay = RecordingClient()
    restored.bind(asyncio.get_running_loop(), lambda: replay, metadata=True)
    await restored.replay()
    assert len(snapshots(replay)) == 64
    assert all(node["status"] == "failed" for node in snapshots(replay))
    assert "DO NOT SEND" not in journal.path.read_text(encoding="utf-8-sig")


@pytest.mark.asyncio
async def test_legacy_clients_receive_standard_tool_lifecycle_without_extension(tmp_path):
    journal = SubagentProgress("legacy", tmp_path)
    client = RecordingClient()
    journal.bind(asyncio.get_running_loop(), lambda: client, metadata=False)
    journal.record("subagent.start", subagent_id="child", depth=1)
    await journal.drain()
    journal.record("subagent.complete", subagent_id="child", status="timeout")
    await journal.drain()
    assert [update["sessionUpdate"] for _, update in client.updates] == ["tool_call", "tool_call_update"]
    assert [update["status"] for _, update in client.updates] == ["in_progress", "failed"]
    assert all("_meta" not in update and "rawInput" not in update for _, update in client.updates)


@pytest.mark.asyncio
async def test_slow_connection_gets_latest_cumulative_state_without_a_chunk_backlog(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowClient(RecordingClient):
        async def session_update(self, session_id, update):
            entered.set()
            await release.wait()
            await super().session_update(session_id, update)

    client = SlowClient()
    journal = SubagentProgress("slow", tmp_path)
    journal.bind(asyncio.get_running_loop(), lambda: client, metadata=True)
    journal.record("subagent.start", subagent_id="child", depth=1)
    await asyncio.wait_for(entered.wait(), timeout=1)
    for _ in range(100):
        journal.record("subagent.text", preview="x", subagent_id="child")
    journal.record("subagent.complete", subagent_id="child", status="completed")
    await asyncio.sleep(0)
    release.set()
    await journal.drain()
    assert len(client.updates) == 2
    assert snapshots(client)[-1]["text"] == "x" * 100
    assert snapshots(client)[-1]["status"] == "completed"
