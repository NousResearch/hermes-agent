"""CLI-12: the Ctrl+T monitor and dock read the gateway session's subagents.

Rewired, delegate_task children run in the gateway, so the CLI process's own
``tools.delegate_tool_registry`` is empty. The monitor lists, tails, steers and stops them through
the session's ``subagent.list`` / ``subagent.tail`` / ``subagent.steer`` / ``subagent.interrupt``
verbs (scripted fake gateway, no live gateway).
"""
import asyncio
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_tools
from hermes_cli.gateway_agent import GatewayAgent
from hermes_cli.gateway_connection import GatewayConnection

SID = "local-abc"


class FakeClient:
    def __init__(self):
        self.events = asyncio.Queue()
        self.calls = []
        self.children = [{"subagent_id": "sa-0-1", "goal": "Audit the parser", "status": "running",
                          "started_at": time.time() - 7, "last_tool": "read_file", "accepting_steer": True}]
        self.steered = []

    async def rpc(self, method, *, _timeout=None, **params):
        self.calls.append((method, params))
        if method == "subagent.list":
            return {"subagents": self.children, "delegations": []}
        if method == "subagent.tail":
            return {"subagent_id": params["subagent_id"], "available": True, "text": "step 1\nstep 2\n",
                    "truncated": False}
        if method == "subagent.steer":
            self.steered.append(params["text"])
            return {"status": "queued", "subagent_id": params["subagent_id"], "text": params["text"]}
        if method == "subagent.interrupt":
            found = any(c["subagent_id"] == params["subagent_id"] for c in self.children)
            self.children = [c for c in self.children if c["subagent_id"] != params["subagent_id"]]
            return {"found": found, "subagent_id": params["subagent_id"]}
        return {}


@pytest.fixture
def gateway(monkeypatch):
    from tools import delegate_tool_registry as registry
    monkeypatch.setattr(registry, "_active_subagents", {})  # the CLI process runs no children
    client = FakeClient()

    @asynccontextmanager
    async def connect():
        yield client
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    conn.state(SID)
    agent = GatewayAgent(connection=conn, session_id=SID)
    gateway_tools.bind(agent)
    yield agent, client
    conn.close()
    gateway_tools._view = None


def test_monitor_lists_tails_and_controls_the_gateway_sessions_children(gateway):
    from hermes_cli import cli_subagent_monitor
    agent, client = gateway
    monitor = cli_subagent_monitor.SubagentMonitor(SimpleNamespace(agent=agent, _agent_running=True))

    monitor.refresh()
    assert [r["subagent_id"] for r in monitor.entries] == ["sa-0-1"]
    assert monitor.entries[0]["last_tool"] == "read_file" and monitor.entries[0]["elapsed"] >= 6
    assert ("subagent.list", {"session_id": SID}) in client.calls
    assert cli_subagent_monitor.subagent_tail(monitor.selected) == "step 1\nstep 2\n"

    assert "note" in monitor.control("steer", "Focus on the tokenizer")
    assert client.steered == ["Focus on the tokenizer"]
    assert "note" in monitor.control("stop")
    assert "error" in monitor.control("stop")  # already gone: the gateway found nothing
    assert monitor.refresh()
    assert monitor.entries == []


def test_idle_session_without_children_does_not_poll_the_roster(gateway):
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    agent, client = gateway
    client.children = []
    monitor = SubagentMonitor(SimpleNamespace(agent=agent, _agent_running=False))
    monitor.refresh()
    assert not any(method == "subagent.list" for method, _ in client.calls)
