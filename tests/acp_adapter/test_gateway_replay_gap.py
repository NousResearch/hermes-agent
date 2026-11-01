"""ACP across a ``session.replay_gap``: the pump re-attaches that session and survives; a cut prompt settles once,
live or from one receipt read the pump owns; terminal evidence beats a recovery error; an interim stays whole;
a reused reply is hidden after a gap or a receipt only when all of it was shown. (Refused re-attach:
test_gateway_terminal_projection.)"""
import asyncio
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from acp.schema import TextContentBlock

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError
from tests.gateway.fixtures.replay_gap_owner import FINAL, Owner, frame_cap  # frame_cap: autouse
from tests.waiting import async_wait_until as until

def resumed(*pending):
    return {"replay_epoch": "e2", "last_sequence": 0, "pending": list(pending), "prompts": [], "resumed": True}


def shown(agent, session="s"):
    return "".join(c.kwargs["update"].content.text for c in agent._conn.session_update.await_args_list
                   if c.kwargs["session_id"] == session and hasattr(getattr(c.kwargs["update"], "content", None), "text"))


def frame(kind, payload, *, session="s", admission=None, seq=None):
    params = {"type": kind, "session_id": session, "payload": payload, "admission_id": admission}
    return {"method": "event", "params": {**params, **({"replay_epoch": "e", "seq": seq} if seq else {})}}


def terminal(admission, reply, **result):
    return {"admission_id": admission, "status": "terminal", "outcome": "completed",
            "result": {"final_response": reply, "completed": True, **result}}


@pytest_asyncio.fixture
async def rig():
    """``rig(answer)``: an agent whose gateway ``rpc`` is ``answer``, with its pump running; torn down after."""
    agents = []

    def make(answer):
        agent = GatewayACPAgent()
        agent._snapshots.update(s={"replay_epoch": "e", "last_sequence": 0}, t={"replay_epoch": "e", "last_sequence": 0})
        agent._gateway, agent._conn, agent.asked = AsyncMock(events=asyncio.Queue()), AsyncMock(), []
        agent._gateway.rpc.side_effect = answer
        agent._event_task = asyncio.create_task(agent._events())
        agents.append(agent)
        return agent
    yield make
    tasks = [task for agent in agents for task in (*agent.asked, agent._event_task)]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def ask(agent, session="s"):
    agent.asked.append(asyncio.create_task(agent.prompt([TextContentBlock(type="text", text="q")], session)))
    return agent.asked[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ended", "live"])
async def test_pump_survives_and_the_editor_sees_the_reply_once(tmp_path, mode):
    owner = Owner(tmp_path / "state.db", pause=mode)
    agent = GatewayACPAgent()
    agent._gateway, agent._conn, agent.asked = owner, AsyncMock(), []
    agent._snapshots["s"] = await owner.conn.resume(owner.ref, {})
    pump = agent._event_task = asyncio.create_task(agent._events())
    try:
        response = await asyncio.wait_for(ask(agent), 30)
        await asyncio.to_thread(owner.turns[-1].join, 10)
        await until(lambda: owner.events.empty())
        assert owner.gaps, "probe did not overflow the fanout (vacuous)"
        assert response.stop_reason == "end_turn" and not pump.done() and agent._failure is None
        assert shown(agent) == FINAL, (len(shown(agent)), len(FINAL))
    finally:
        for task in (*agent.asked, pump):
            task.cancel()
        await asyncio.gather(*agent.asked, pump, return_exceptions=True)
        await asyncio.to_thread(owner.finish)


@pytest.mark.asyncio
async def test_completion_in_delivery_beats_a_recovery_error(rig):
    delivering, release = asyncio.Event(), asyncio.Event()

    async def answer(method, **params):
        if method == "prompt.submit":
            agent._gateway.events.put_nowait(frame("session.replay_gap", {}))
            await until(lambda: agent._snapshots["s"].get("resumed"))
            # Admitted after the snapshot: its completion is live, and in delivery when the ack lands.
            agent._gateway.events.put_nowait(frame("message.complete", {"text": "done", "outcome": "completed"},
                                                   admission="a", seq=1))
            await delivering.wait()
            return {"admission_id": "a", "status": "queued"}
        if method == "session.resume":
            return resumed()
        raise GatewayClientError("read_failed")

    agent = rig(answer)

    async def update(**kwargs):
        delivering.set()
        await release.wait()

    agent._conn.session_update.side_effect = update
    task = ask(agent)
    await asyncio.wait_for(delivering.wait(), 5)
    await until(lambda: task.done() or agent._admissions.get("s") == "a")
    release.set()
    assert (await asyncio.wait_for(task, 2)).stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_one_recovery_read_settles_a_cancellation_and_the_pump_survives(rig):
    """Acked during the re-attach, a turn that ended in the hole is read once (the pump owns its recovery) and
    that fetched cancellation is final (a second read never answers). The other session streams on."""
    attaching, reads = asyncio.Event(), []

    async def answer(method, **params):
        if method == "prompt.submit" and params["session_id"] == "t":
            return {"admission_id": "t-a", "status": "queued"}
        if method == "prompt.submit":
            agent._gateway.events.put_nowait(frame("session.replay_gap", {}))
            await attaching.wait()
            return {"admission_id": "a", "status": "queued"}
        if method == "session.resume":
            attaching.set()
            await until(lambda: agent._admissions.get("s") == "a")
            return resumed()
        reads.append(params["admission_id"])
        if len(reads) > 1:
            await asyncio.Event().wait()
        return {"admission_id": "a", "status": "terminal", "outcome": "cancelled"}

    agent = rig(answer)
    other = ask(agent, "t")
    await until(lambda: agent._admissions.get("t") == "t-a")
    first = ask(agent)
    assert (await asyncio.wait_for(asyncio.shield(first), 2)).stop_reason == "cancelled"
    agent._gateway.events.put_nowait(frame("message.complete", {"text": "other", "outcome": "completed"},
                                           session="t", admission="t-a", seq=1))
    assert (await asyncio.wait_for(other, 2)).stop_reason == "end_turn"
    assert reads == ["a"] and agent._failure is None


@pytest.mark.asyncio
async def test_retry_after_a_lost_ack_completes_a_partial_reused_reply(rig):
    submissions = []

    async def answer(method, **params):
        if method == "prompt.submit":
            submissions.append(params["input_id"])
            if len(submissions) == 1:
                agent._gateway.events.put_nowait(frame("message.delta", {"text": "Hel"}, admission="a", seq=1))
                agent._gateway.events.put_nowait(frame("session.replay_gap", {}))
                raise GatewayClientError("submit timed out")
            return {"admission_id": "a", "status": "terminal", "outcome": "completed"}
        return resumed() if method == "session.resume" else terminal("a", "Hello", response_reused=True)

    agent = rig(answer)
    with pytest.raises(GatewayClientError):
        await ask(agent)
    await until(lambda: agent._snapshots["s"].get("resumed"))
    assert (await asyncio.wait_for(ask(agent), 2)).stop_reason == "end_turn"
    assert submissions[0] == submissions[1] and shown(agent) == "Hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["", "ANSWER"])
async def test_interim_after_the_gap_shows_the_reply_once(rig, completion):
    """An unstreamed interim may be the reply's only copy (a codex app-server final; the completion is empty)."""
    async def answer(method, **params):
        if method == "prompt.submit":
            return {"admission_id": "a", "status": "queued"}
        return resumed({"admission_id": "a", "status": "started"})

    agent = rig(answer)
    task = ask(agent)
    await until(lambda: agent._admissions.get("s") == "a")
    agent._gateway.events.put_nowait(frame("message.delta", {"text": "Let me "}, admission="a", seq=1))
    agent._gateway.events.put_nowait(frame("session.replay_gap", {}))
    await until(lambda: agent._snapshots["s"].get("resumed"))
    for seq, (kind, text) in enumerate([("message.interim", "ANSWER"), ("message.complete", completion)], 1):
        agent._gateway.events.put_nowait(frame(kind, {"text": text, "already_streamed": False,
                                                      "outcome": "completed"}, admission="a", seq=seq))
    assert (await asyncio.wait_for(task, 2)).stop_reason == "end_turn"
    assert shown(agent).count("ANSWER") == 1, shown(agent)


@pytest.mark.asyncio
async def test_without_a_gap_a_streamed_reused_reply_is_not_repainted(rig):
    """The owner may append to a reused final (auto media tags); without a gap the streamed copy stays the one."""
    agent = rig(AsyncMock(return_value={"admission_id": "a", "status": "queued"}))
    task = ask(agent)
    await until(lambda: agent._admissions.get("s") == "a")
    for seq, (kind, payload) in enumerate([
            ("message.delta", {"text": "FINAL"}), ("message.delta", {"text": None}),
            ("tool.start", {"tool_call_id": "t1", "tool_name": "memory", "args": {}}),
            ("tool.complete", {"tool_call_id": "t1", "tool_name": "memory", "result": "ok"}),
            ("message.complete", {"text": "FINAL\nMEDIA:/tmp/x.png", "outcome": "completed",
                                  "response_reused": True})], 1):
        agent._gateway.events.put_nowait(frame(kind, payload, admission="a", seq=seq))
    assert (await asyncio.wait_for(task, 2)).stop_reason == "end_turn"
    assert shown(agent).count("FINAL") == 1, shown(agent)
