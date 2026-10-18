"""Classic CLI (`hermes`, `hermes chat` -> GatewayAgent) across a ``session.replay_gap``: it re-attaches, shows
the settled reply once and never leaves a later turn waiting on a retired subscription. A recovery belongs
to its turn: a session switch, a later turn or Ctrl-C's settle bound ends it before it sends or paints."""
import asyncio
import contextlib
import threading

import pytest

from tests.gateway.fixtures.replay_gap_owner import FINAL, Owner, frame_cap  # frame_cap: autouse
from tests.hermes_cli.test_gateway_agent import SID, FakeClient, connect_with, frame

GAP = frame("session.replay_gap", {}, admission=None)


def run_turn(agent, text, timeout):
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault("result", agent.run_conversation(text)), daemon=True)
    worker.start()
    worker.join(timeout)
    return out.get("result")


def test_turn_ended_in_the_hole_reports_the_answer_and_the_next_turn_streams(tmp_path):
    from hermes_cli.gateway_agent import GatewayAgent
    from hermes_cli.gateway_connection import GatewayConnection
    owners, streamed = [], []

    @contextlib.asynccontextmanager
    async def connect():
        owners.append(Owner(tmp_path / "state.db"))
        yield owners[-1]

    connection = GatewayConnection(connect=connect)
    connection.start(timeout=30)
    try:
        connection.state("s").adopt_snapshot(connection.call("session.resume", session_id="s"))
        agent = GatewayAgent(connection=connection, session_id="s")
        agent.stream_delta_callback = streamed.append
        first = run_turn(agent, "write a long answer", 30)
        assert owners[0].gaps, "probe did not overflow the fanout (vacuous)"
        assert first is not None and not first["failed"] and first["final_response"] == FINAL, first
        assert "".join(t for t in streamed if t) == FINAL, "hole or duplicate"
        streamed.clear()
        second = run_turn(agent, "and a short one", 15)
        assert second is not None, "turn 2 hangs: the gap detached the viewer and nothing re-attached"
        assert not second["failed"] and "".join(t for t in streamed if t) == FINAL
    finally:
        owners[0].finish()
        connection.close()


@pytest.fixture
def agents():
    from hermes_cli.gateway_agent import GatewayAgent
    from hermes_cli.gateway_connection import GatewayConnection
    made = []

    def build(script, **kw):
        client = FakeClient(script, result=kw.pop("result", None))
        made.append(GatewayConnection(connect=connect_with(client)))
        made[-1].start(5)
        made[-1].state(SID)  # gaps are counted per attached session
        return GatewayAgent(connection=made[-1], session_id=SID, **kw), client
    yield build
    for connection in made:
        connection.close()


@pytest.mark.parametrize("change", ["session", "interrupt"])
def test_idle_gap_reattach_cannot_submit_abandoned_input(agents, monkeypatch, change):
    entered, release = threading.Event(), threading.Event()
    agent, client = agents([])
    agent.connection.state(SID).gaps = 1  # a gap between turns retired the subscription
    original = client.rpc

    async def rpc(method, **params):
        if method == "session.resume" and params["session_id"] == SID:
            entered.set()
            await asyncio.to_thread(release.wait, 5)
            return {"pending": []}
        return await original(method, **params)

    monkeypatch.setattr(client, "rpc", rpc)
    worker = threading.Thread(target=agent.run_conversation, args=("old question",), daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        if change == "session":
            agent.session_id = "new-session"
            agent.reset_session_state()
        else:
            monkeypatch.setattr("hermes_cli.gateway_agent.INTERRUPT_SETTLE_SECONDS", 0)
            agent.interrupt()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert not [params for method, params in client.calls if method == "prompt.submit"]


def test_a_turn_past_its_interrupt_settle_bound_still_reads_its_receipt_and_stats(agents, monkeypatch):
    """Ctrl-C's settle bound ends the turn's wait for frames; the end-of-turn ``prompt.receipt`` and
    cumulative ``session.stats`` reads still run (main made both), since the turn still owns the terminal."""
    agent, client = agents([frame("message.delta", {"text": "partial"}), "wait", "wait", "wait", "wait"])
    monkeypatch.setattr("hermes_cli.gateway_agent.INTERRUPT_SETTLE_SECONDS", 0)
    worker = threading.Thread(target=lambda: agent.run_conversation("q"), daemon=True)
    worker.start()
    try:
        deadline = threading.Event()
        while not any(method == "prompt.submit" for method, _ in client.calls) and not deadline.wait(0.01):
            pass
        agent.interrupt()
    finally:
        worker.join(5)
    assert not worker.is_alive()
    methods = [method for method, _ in client.calls]
    assert "prompt.receipt" in methods and "session.stats" in methods, methods


def test_a_long_streamed_reply_keeps_its_delta_buffers_linear():
    """``turn.shown += text`` on an attribute copied the segment per delta: 200k deltas took ~50 s."""
    import time
    from hermes_cli.gateway_agent import GatewayAgent, _Turn
    agent = GatewayAgent.__new__(GatewayAgent)
    agent.stream_delta_callback, agent._turn = (lambda _text: None), _Turn(SID)
    started = time.monotonic()
    for _ in range(200_000):
        agent._on_delta({"text": "token-text "}, lambda _text: None, {})
    assert time.monotonic() - started < 10
    assert agent._turn.shown == agent._turn.spoken == "token-text " * 200_000


def test_tool_start_closes_a_prefix_whose_separator_fell_in_the_hole(agents, monkeypatch):
    output, final = [], "The answer is 42."
    agent, client = agents([frame("message.delta", {"text": "The answer"}), GAP],
                           result={"final_response": final, "completed": True},
                           stream_delta_callback=output.append,
                           tool_progress_callback=lambda *a, **k: output.append("<tool>"))
    original = client.rpc

    async def rpc(method, **params):
        if method == "session.resume":
            await client.events.put(frame("tool.start", {"name": "read_file", "tool_id": "t", "args": {}}))
            await client.events.put(frame("message.complete", {"text": final, "outcome": "completed"}))
            return {"pending": [{"admission_id": "adm-1", "status": "started"}]}
        return await original(method, **params)

    monkeypatch.setattr(client, "rpc", rpc)
    assert run_turn(agent, "q", 5) is not None
    assert "".join(x for x in output[output.index("<tool>") + 1:] if isinstance(x, str)) == final


def test_gap_completes_the_voice_stream_without_terminal_streaming(agents):
    spoken, final = [], "First sentence. Second sentence."
    agent, _ = agents([frame("message.delta", {"text": "First sentence. "}), GAP],
                      result={"final_response": final, "completed": True})
    result = agent.run_conversation("q", stream_callback=spoken.append)
    assert not result["failed"] and result["final_response"] == final
    assert "".join(spoken) == final
