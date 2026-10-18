"""Live reasoning for editors on the gateway path (#134889 ACP-4).

The in-process server (``acp_adapter/server.py``) gave every editor turn a reasoning sink and streamed it as
``agent_thought_chunk``. On the gateway path the owner publishes ``reasoning.delta`` only for a live session
whose ``show_reasoning`` flag a viewer set (``gateway/local_viewer_events.py``), so the editor asks for it on
every attach. The end-to-end test runs an actual ``hermes acp`` stdio process against an ordinary isolated
gateway and a provider that reasons before it answers.
"""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from acp.schema import AllowedOutcome
from websockets.exceptions import ConnectionClosed

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.fakes.fake_llm_provider import FakeLLMServer, Text

REASONING = "REASONING-BEFORE-THE-ANSWER"
ANSWER = "THE-ANSWER"
# A card the owner still lists for the session: an attach that resumes the session offers it to the editor again.
CARD = {"kind": "approval", "prompt_id": "p-1", "execution_generation": 3, "command": "rm -r build",
        "description": "remove the build directory", "choices": ["once", "deny"]}


@pytest.fixture
def model_peer():
    """The daemon's provider: each of the test's three turns streams its reasoning before its answer."""
    with FakeLLMServer([Text(ANSWER, reasoning=REASONING) for _ in range(3)]) as provider:
        yield SimpleNamespace(server_port=provider.port)


async def _turn(acp, sid):
    """One prompt; the editor must get the reasoning live, as thought chunks before the answer starts."""
    start = len(acp.frames)
    response = await acp.rpc("session/prompt", sessionId=sid, prompt=[{"type": "text", "text": "think first"}])
    assert response.get("result", {}).get("stopReason") == "end_turn", response
    updates = [frame["params"]["update"] for frame in acp.frames[start:]
               if frame.get("method") == "session/update" and frame["params"].get("sessionId") == sid]
    chunks = [(update["sessionUpdate"], update["content"]["text"]) for update in updates
              if update.get("sessionUpdate") in ("agent_thought_chunk", "agent_message_chunk")]
    kinds = [kind for kind, _ in chunks]
    assert "".join(text for kind, text in chunks if kind == "agent_thought_chunk") == REASONING, chunks
    assert "".join(text for kind, text in chunks if kind == "agent_message_chunk") == ANSWER, chunks
    # The provider streams the reasoning first: a projection that held it back would land after the answer.
    last_thought = max(index for index, kind in enumerate(kinds) if kind == "agent_thought_chunk")
    assert last_thought < kinds.index("agent_message_chunk"), chunks


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_every_session_the_editor_attaches_streams_its_reasoning_live(daemon, tmp_path):
    async with viewer(daemon) as ws, editor(daemon, tmp_path) as acp:
        await acp.rpc("initialize", protocolVersion=1, clientCapabilities={})
        created = await acp.rpc("session/new", cwd=str(tmp_path), mcpServers=[])
        await _turn(acp, created["result"]["sessionId"])
        # Sessions another client opened over a raw viewer socket, which never asks for reasoning.
        to_load, to_fork = [
            (await ws.rpc("session.create", request_id=name, source="cli", cwd=str(tmp_path)))["session_id"]
            for name in ("to-load", "to-fork")]
        loaded = await acp.rpc("session/load", sessionId=to_load, cwd=str(tmp_path), mcpServers=[])
        assert "result" in loaded, loaded
        await _turn(acp, to_load)
        forked = await acp.rpc("session/fork", sessionId=to_fork, cwd=str(tmp_path), mcpServers=[])
        await _turn(acp, forked["result"]["sessionId"])


@pytest.mark.asyncio
@pytest.mark.parametrize("attach", ["new", "load", "resume", "fork", "reconnect"])
@pytest.mark.parametrize("failure", [None, GatewayRPCError("not_found"), GatewayClientError("Gateway disconnected"),
                                     ConnectionClosed(None, None), TimeoutError()],
                         ids=["set", "refused", "reply-lost", "socket-closed", "unanswered"])
async def test_an_attach_asks_for_live_reasoning_before_it_replies_and_never_fails_on_it(monkeypatch, attach, failure):
    # The owner runs the conversation as "logical"; a load or resume may hold another id for it (a compressed
    # chain's), and a fork holds its child.
    monkeypatch.setattr("acp_adapter.catalog.logical_session_id", lambda db, held: "logical")
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    agent._conn.request_permission.side_effect = lambda session_id, tool_call, options: SimpleNamespace(
        outcome=AllowedOutcome(outcome="selected", option_id=options[0].option_id))  # answers every card at once
    asked = []
    done = {"request": False}  # the flag request has finished (answered, refused or given up)

    async def rpc(method, **params):
        asked.append((method, params))
        if method == "config.set":
            try:
                for _ in range(3):  # the owner's round trip: a card offered before it is answered meanwhile
                    await asyncio.sleep(0)
                if failure is not None:
                    raise failure
            finally:
                done["request"] = True
        return {"runtime.describe": {"session_create": {"sources": ["acp"]}, "capabilities": ["acp-editor-policy-v1"],
                                     "profile_id": str(agent._home)},
                "session.create": {"session_id": "logical", "info": {"cwd": "/"}},
                "session.info": {"cwd": "/"},
                "session.resume": {"session_id": params.get("session_id"), "messages": [], "prompts": [CARD],
                                   "pending": [], "execution_generation": 3, "revision": 1},
                "session.mutate": {"status": "committed", "branched_session_id": "child"},
                "config.set": {"key": "show_reasoning", "value": "1", "scope": "session"},
                "approval.respond": {"status": "resolved"}}[method]

    owner = AsyncMock(events=asyncio.Queue())
    owner.rpc.side_effect = rpc
    agent._gateway = owner
    session_fields = agent._session_fields

    async def reply_fields(*args, **kwargs):  # every attach but the re-attach builds its reply here
        done.setdefault("at_reply", done["request"])
        return await session_fields(*args, **kwargs)

    agent._session_fields = reply_fields
    if attach == "new":
        held = (await agent.new_session(cwd="/")).session_id
    elif attach in ("load", "resume"):
        held = "held"
        await getattr(agent, f"{attach}_session")(cwd="/", session_id=held)
    elif attach == "fork":
        held = (await agent.fork_session(cwd="/", session_id="parent")).session_id
    else:  # the gateway restarted: the editor's next call reconnects and re-attaches the session it holds
        agent._gateway, agent._snapshots["logical"], held = None, {}, "logical"

        @asynccontextmanager
        async def connect_gateway():
            yield owner

        monkeypatch.setattr("acp_adapter.gateway_server.connect_gateway", connect_gateway)
        await agent._client()
    owner_id = "child" if attach == "fork" else "logical"
    try:
        # Asked for the session the owner runs, waiting at most the CLI's 10 s, and finished before the attach
        # replied; at worst a failure costs the live reasoning, never the session.
        assert ("config.set", {"session_id": owner_id, "key": "show_reasoning", "value": "1", "_timeout": 10}) in asked
        assert done.get("at_reply", True), "the attach replied before its request finished"
        assert owner_id in agent._snapshots, "the attach dropped the session"
        for _ in range(5):  # the cards the attach offered settle
            await asyncio.gather(*agent._permissions.values(), return_exceptions=True)
            await asyncio.sleep(0)
        offered = 1 if attach in ("load", "resume", "reconnect") else 0  # the owner's pending card comes back
        assert agent._conn.request_permission.await_count == offered, "the card was not offered"
        answers = [method for method, _ in asked if method == "approval.respond"]
        assert len(answers) == offered, "the editor's answer to a card was lost"
        # Thoughts, a lone line break too, reach the id the editor holds, outside the reply, even between two
        # parts of the answer.
        for kind, payload in [("message.delta", {"text": "AN"}), ("reasoning.delta", {"text": "thinking"}),
                              ("reasoning.delta", {"text": "\n"}), ("message.delta", {"text": "SWER"}),
                              ("message.complete", {"text": "ANSWER", "outcome": "completed"})]:
            await agent._project({"session_id": owner_id, "type": kind, "admission_id": "a1", "payload": payload})
        updates = [(call.kwargs["session_id"], call.kwargs["update"])
                   for call in agent._conn.session_update.await_args_list]
        reply = "".join(update.content.text for _, update in updates if update.session_update == "agent_message_chunk")
        thoughts = [(sid, update.content.text) for sid, update in updates
                    if update.session_update == "agent_thought_chunk"]
        assert (reply, thoughts) == ("ANSWER", [(held, "thinking"), (held, "\n")])
    finally:
        if agent._event_task:
            agent._event_task.cancel()
            await asyncio.gather(agent._event_task, return_exceptions=True)
