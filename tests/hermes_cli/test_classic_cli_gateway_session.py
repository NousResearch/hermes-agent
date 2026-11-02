"""The classic CLI's gateway session across /new and a gateway restart: the reasoning flag follows the
session, and a turn lost in the restart is shown and discarded instead of blocking the next prompt."""
import asyncio
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hermes_cli.gateway_agent import GatewayAgent
from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.gateway_connection import GatewayConnection

SID = "local-abc"
LOST = {"admission_id": "adm-lost", "status": "unknown", "execution_generation": 4, "text": "first ask"}


class FakeClient:
    def __init__(self):
        self.events = asyncio.Queue()
        self.calls = []
        self.pending = []

    async def rpc(self, method, *, _timeout=None, **params):
        self.calls.append((method, params))
        if method == "session.resume":
            return {"session_id": params["session_id"], "subscription_id": "sub-1", "revision": 1,
                    "execution_generation": 7, "messages": [], "pending": list(self.pending)}
        if method == "prompt.resolve_unknown":
            self.pending = [row for row in self.pending if row["admission_id"] != params["admission_id"]]
        if method == "prompt.submit":
            if not self.pending:  # the owner's FIFO: nothing runs behind an unknown turn
                done = {"type": "message.complete", "session_id": SID, "admission_id": "adm-1",
                        "payload": {"text": "ok", "outcome": "completed"}}
                self.events.put_nowait({"jsonrpc": "2.0", "method": "event", "params": done})
            return {"admission_id": "adm-1", "status": "queued", "execution_generation": 7}
        return {}

    def methods(self):
        return [method for method, _ in self.calls]

    def reasoning_sets(self):
        return [(p["session_id"], p["value"]) for m, p in self.calls
                if m == "config.set" and p.get("key") == "show_reasoning"]


@pytest.fixture
def gateway():
    client = FakeClient()

    @asynccontextmanager
    async def connect():
        client.events = asyncio.Queue()  # each gateway life is a new socket
        yield client
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    conn.reattach(SID)  # the session store's startup attach: a redial resubscribes it
    yield conn, client
    conn.close()


def run_bounded(agent, text):
    """``run_conversation`` on a thread: a prompt queued behind a lost turn would wait forever."""
    out = []
    thread = threading.Thread(target=lambda: out.append(agent.run_conversation(text)), daemon=True)
    thread.start()
    thread.join(10)
    agent._turn = None  # ends a pump still waiting
    return out[0] if out else {"failed": False, "error": "still waiting"}


def test_a_new_session_gets_the_show_reasoning_flag(gateway):
    conn, client = gateway
    agent = GatewayAgent(connection=conn, session_id=SID, reasoning_callback=print)
    agent.session_id = "local-new"  # /new, /resume, /branch
    assert client.reasoning_sets() == [(SID, "1"), ("local-new", "1")]


def test_a_hidden_viewer_resuming_a_session_turns_its_reasoning_flag_off(gateway):
    conn, client = gateway
    agent = GatewayAgent(connection=conn, session_id=SID)
    agent.session_id = "local-shown"  # /resume of a session another viewer left showing reasoning
    assert ("local-shown", "0") in client.reasoning_sets()


def test_a_restarted_gateway_gets_the_flag_again_before_the_next_turn(gateway):
    conn, client = gateway
    agent = GatewayAgent(connection=conn, session_id=SID, reasoning_callback=print)
    conn.loop.call_soon_threadsafe(client.events.put_nowait, GatewayClientError("Gateway disconnected"))
    conn._thread.join(5)  # the owner hung up; the next call redials a fresh gateway
    assert run_bounded(agent, "after the restart")["final_response"] == "ok"
    assert client.reasoning_sets() == [(SID, "1"), (SID, "1")]
    methods = client.methods()
    assert methods.count("session.resume") == 2  # attach, then the redial's resubscribe
    assert max(i for i, m in enumerate(methods) if m == "config.set") < methods.index("prompt.submit")


def test_a_prompt_behind_a_lost_turn_shows_the_discard_banner_and_discard_lets_it_run(gateway, monkeypatch):
    conn, client = gateway
    agent = GatewayAgent(connection=conn, session_id=SID)
    client.pending = [LOST]  # the owner restarted and fenced the interrupted turn unknown
    conn.loop.call_soon_threadsafe(client.events.put_nowait, GatewayClientError("Gateway disconnected"))
    conn._thread.join(5)  # only the redial's resubscribe can tell this viewer
    refused = run_bounded(agent, "second ask")
    assert refused["failed"] and "/discard adm-lost" in refused["error"]
    assert "prompt.submit" not in client.methods()  # never queued behind the lost turn

    from cli import HermesCLI
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: conn)
    method, _ = HermesCLI._slash_handler("discard") or (None, None)
    assert method is not None
    getattr(HermesCLI, method)(SimpleNamespace(session_id=SID), "/discard adm-lost")
    assert ("prompt.resolve_unknown", {"session_id": SID, "admission_id": "adm-lost",
                                       "execution_generation": 4}) in client.calls
    assert run_bounded(agent, "second ask")["final_response"] == "ok"


def test_a_bare_discard_prints_usage_and_calls_nothing(gateway, monkeypatch):
    conn, client = gateway
    from cli import HermesCLI
    printed = []
    monkeypatch.setattr("cli._cprint", printed.append)
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: conn)
    calls = len(client.calls)
    entry = HermesCLI._slash_handler("discard")
    assert entry is not None
    getattr(HermesCLI, entry[0])(SimpleNamespace(session_id=SID), "/discard")
    assert printed == ["  Usage: /discard <admission_id>"]
    assert len(client.calls) == calls
