"""Classic CLI on a session whose FIFO a gateway restart paused (#139532).

The restarted owner fences the turn it died running ``unknown``; nothing queued behind it runs
until it is acknowledged. The classic CLI used to submit into that FIFO and wait forever on a
``message.complete`` that never comes. It must refuse with the recovery instead, and ``/discard``
must acknowledge the lost turn from the chat itself.
"""
import asyncio

import pytest

from hermes_cli.gateway_agent import GatewayAgent
from hermes_cli.gateway_connection import GatewayConnection
from tests.hermes_cli.test_gateway_agent import SID, FakeClient, connect_with

# Without the fix these turns wait on a completion that never comes: fail, never hang the file.
pytestmark = pytest.mark.timeout(20)

LOST = {"admission_id": "lost-1", "status": "unknown", "execution_generation": 4, "text": "old"}


class _PausedClient(FakeClient):
    """A gateway whose session snapshot carries a lost (``unknown``) head."""

    def __init__(self, pending, *, fence_after_submit=False):
        super().__init__([])
        self.pending = list(pending)
        self.fence_after_submit = fence_after_submit

    async def rpc(self, method, *, _timeout=None, **params):
        if method == "session.resume":
            self.calls.append((method, params))
            return {"session_id": SID, "stored_session_id": SID, "revision": 1, "execution_generation": 7,
                    "messages": [], "pending": list(self.pending)}
        if method == "prompt.resolve_unknown":
            self.calls.append((method, params))
            self.pending = [r for r in self.pending if r["admission_id"] != params["admission_id"]]
            return {"admission_id": params["admission_id"], "status": "terminal"}
        if method == "prompt.submit" and self.fence_after_submit:
            self.calls.append((method, params))
            # The owner restarted between the CLI's check and its submit: the new row queues
            # behind a lost head, and the owner says so in its pending snapshot.
            self.pending = [LOST, {"admission_id": "adm-1", "status": "queued", "execution_generation": None}]
            asyncio.get_running_loop().call_soon(lambda: self.events.put_nowait({
                "jsonrpc": "2.0", "method": "event", "params": {
                    "type": "session.info", "session_id": SID, "payload": {"pending": list(self.pending)}}}))
            return {"admission_id": "adm-1", "status": "queued", "execution_generation": None}
        return await super().rpc(method, _timeout=_timeout, **params)


def _agent(client):
    conn = GatewayConnection(connect=connect_with(client))
    conn.start(5)
    conn.state(SID).adopt_snapshot(conn.call("session.resume", session_id=SID))
    return conn, GatewayAgent(connection=conn, session_id=SID)


def test_prompt_on_a_paused_session_is_refused_with_the_recovery_not_queued_forever():
    client = _PausedClient([LOST])
    conn, agent = _agent(client)
    try:
        result = agent.run_conversation("continue")
    finally:
        conn.close()
    assert result["failed"] and result["failure_reason"] == "unknown_execution"
    assert "/discard" in result["error"]
    assert "prompt.submit" not in [m for m, _ in client.calls]  # nothing left queued behind it


def test_prompt_fenced_after_submit_is_withdrawn_instead_of_waiting_forever():
    client = _PausedClient([], fence_after_submit=True)
    conn, agent = _agent(client)
    try:
        result = agent.run_conversation("continue")
    finally:
        conn.close()
    assert result["failed"] and result["failure_reason"] == "unknown_execution"
    assert ("prompt.cancel", {"session_id": SID, "admission_id": "adm-1"}) in client.calls


def test_prompt_queued_behind_another_running_turn_says_so_once_then_gets_its_own_reply():
    """Another admission runs (an automation turn after a restart): its frames are not this
    turn's, so the CLI tells the user once that the message is queued, then settles on its own."""
    from tests.hermes_cli.test_gateway_agent import frame
    printed = []
    other = {"admission_id": "other", "status": "started", "execution_generation": 6}

    class _Busy(_PausedClient):
        async def rpc(self, method, *, _timeout=None, **params):
            if method != "prompt.submit":
                return await super().rpc(method, _timeout=_timeout, **params)
            self.calls.append((method, params))
            queued = [other, {"admission_id": "adm-1", "status": "queued", "execution_generation": None}]

            async def play():
                await self.events.put({"jsonrpc": "2.0", "method": "event", "params": {
                    "type": "session.info", "session_id": SID, "payload": {"pending": queued}}})
                await asyncio.sleep(0.3)  # health: allow HX013 -- the other turn runs a while
                await self.events.put(frame("message.complete", {"text": "theirs"}, admission="other"))
                await self.events.put(frame("message.complete", {"text": "mine", "outcome": "completed"}))
            asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(play()))
            return {"admission_id": "adm-1", "status": "queued", "execution_generation": None}

    client = _Busy([])
    conn, agent = _agent(client)
    agent._print_fn = printed.append
    try:
        result = agent.run_conversation("continue")
    finally:
        conn.close()
    assert result["final_response"] == "mine"
    assert sum("Queued: another turn is still running" in line for line in printed) == 1, printed


def test_discard_command_acknowledges_the_lost_turn_and_unblocks_the_next_prompt(monkeypatch, capsys):
    from cli import HermesCLI
    client = _PausedClient([LOST])
    conn, agent = _agent(client)
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: conn)
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = SID
    try:
        assert cli.process_command("/discard lost-1") is True
        assert ("prompt.resolve_unknown", {"session_id": SID, "admission_id": "lost-1",
                                           "execution_generation": 4}) in client.calls
        assert "Discarded lost turn lost-1" in capsys.readouterr().out
        assert agent._lost_turns() == []  # the next prompt is submitted, not refused
    finally:
        conn.close()


def test_bare_discard_asks_before_acknowledging_every_lost_turn(monkeypatch, capsys):
    """A bare /discard covers every lost turn, so it confirms first as ``hermes sessions discard``
    does; a declined prompt leaves the turn untouched."""
    from cli import HermesCLI
    client = _PausedClient([LOST])
    conn, agent = _agent(client)
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: conn)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = SID
    try:
        assert cli.process_command("/discard") is True
        assert ("prompt.resolve_unknown", {"session_id": SID, "admission_id": "lost-1",
                                           "execution_generation": 4}) not in client.calls
        assert "Nothing discarded" in capsys.readouterr().out
        assert agent._lost_turns() == ["lost-1"]
    finally:
        conn.close()


def test_bare_discard_without_a_tty_names_the_admission_ids_instead_of_discarding(monkeypatch, capsys):
    from cli import HermesCLI
    client = _PausedClient([LOST])
    conn, agent = _agent(client)
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: conn)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = SID
    try:
        assert cli.process_command("/discard") is True
        assert not [c for c in client.calls if c[0] == "prompt.resolve_unknown"]
        out = capsys.readouterr().out
        assert "lost-1" in out and "/discard <id>" in out
        assert agent._lost_turns() == ["lost-1"]
    finally:
        conn.close()

