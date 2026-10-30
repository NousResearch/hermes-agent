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
        self.steer_status = "queued"
        self.tail_error = None
        self.list_error = None
        self.tail_text = "step 1\nstep 2\n"

    async def rpc(self, method, *, _timeout=None, **params):
        self.calls.append((method, params))
        if method == "subagent.list":
            if self.list_error:
                raise self.list_error
            return {"subagents": self.children, "delegations": []}
        if method == "subagent.tail":
            if self.tail_error:
                raise self.tail_error
            return {"subagent_id": params["subagent_id"], "available": True, "text": self.tail_text,
                    "truncated": False}
        if method == "subagent.steer":
            self.steered.append(params["text"])
            return {"status": self.steer_status, "subagent_id": params["subagent_id"], "text": params["text"]}
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
    monitor = cli_subagent_monitor.SubagentMonitor(SimpleNamespace(agent=agent, _agent_running=True,
                                                                   _invalidate=lambda: None))

    monitor.refresh()
    assert [r["subagent_id"] for r in monitor.entries] == ["sa-0-1"]
    assert monitor.entries[0]["last_tool"] == "read_file" and monitor.entries[0]["elapsed"] >= 6
    assert ("subagent.list", {"session_id": SID}) in client.calls
    monitor.detail = True  # the open detail view: the spinner thread's tick fetches its tail
    client.tail_text = "step 1\r\n\x1b[2Jstep 2\n"  # a child's terminal control codes
    monitor.tick()
    assert cli_subagent_monitor.subagent_tail(monitor, monitor.selected) == "step 1\n[2Jstep 2\n"  # stripped

    assert "note" in monitor.control("steer", "Focus on the tokenizer")
    assert client.steered == ["Focus on the tokenizer"]
    client.steer_status = "rejected"  # the child no longer accepts steering
    assert "error" in monitor.control("steer", "Too late")
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


def test_failing_roster_reads_age_out_the_cached_children(gateway):
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from hermes_cli.gateway_client import GatewayRPCError
    agent, client = gateway
    monitor = SubagentMonitor(SimpleNamespace(agent=agent, _agent_running=False))
    view = gateway_tools.process_view()
    view.subagents = list(client.children)  # children were listed before the gateway stopped answering
    view._subagents_seen = time.time()
    client.list_error = GatewayRPCError("unavailable")
    monitor.refresh()
    assert [r["subagent_id"] for r in monitor.entries] == ["sa-0-1"]  # one failed read keeps the roster
    view._subagents_polled, view._subagents_seen = 0.0, time.time() - 61
    monitor.refresh()
    assert monitor.entries == []  # no answer for a minute: the ghost rows go
    view._subagents_polled = 0.0
    calls = len(client.calls)
    monitor.refresh()
    assert len(client.calls) == calls  # and an idle session stops polling


def test_detail_view_paints_the_cached_tail_and_only_the_tick_asks_the_gateway(gateway):
    """The detail view repaints every frame on the UI loop: it must never block on subagent.tail.
    The spinner thread's tick fetches the tail; a failing gateway is asked once per poll."""
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from hermes_cli import cli_subagent_monitor
    from hermes_cli.gateway_client import GatewayRPCError
    agent, client = gateway
    monitor = cli_subagent_monitor.SubagentMonitor(SimpleNamespace(agent=agent, _agent_running=True,
                                                                   _invalidate=lambda: None))
    monitor.refresh()
    tails = lambda: sum(method == "subagent.tail" for method, _ in client.calls)

    async def run():
        with create_pipe_input() as pipe:
            output = DummyOutput()
            output.get_size = lambda: Size(rows=14, columns=80)
            monitor.app = app = cli_subagent_monitor.build_monitor_application(monitor, input=pipe, output=output)
            rendered = asyncio.Event()
            app.after_render += lambda app: rendered.set()
            task = asyncio.create_task(app.run_async())
            await asyncio.wait_for(rendered.wait(), 3)
            rendered.clear()
            pipe.send_text("\r")  # open the detail view: several frames, none of them asks the gateway
            await asyncio.wait_for(rendered.wait(), 3)
            for _ in range(3):
                rendered.clear()
                app.invalidate()
                await asyncio.wait_for(rendered.wait(), 3)
            assert tails() == 0 and app.current_buffer.text != "step 1\nstep 2\n"
            rendered.clear()
            monitor.tick()  # the spinner thread's tick fetches the tail and repaints from the cache
            await asyncio.wait_for(rendered.wait(), 3)
            assert tails() == 1 and app.current_buffer.text == "step 1\nstep 2\n"
            pipe.send_text("\x1b")
            pipe.send_text("q")
            await asyncio.wait_for(task, 3)
    asyncio.run(run())

    view = gateway_tools.process_view()
    client.tail_error = GatewayRPCError("unavailable")
    view._tail = None
    assert view.subagent_tail("sa-0-1", fetch=True) is None
    assert view.subagent_tail("sa-0-1", fetch=True) is None  # the failure is cached for the poll interval
    assert tails() == 2


def test_a_failing_tick_keeps_the_spinner_thread_alive(gateway, monkeypatch):
    """The spinner thread runs ``tick`` in a bare loop: one raise would end it, and with it the
    spinner and the dock, for the rest of the session. A failed poll skips a beat instead."""
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    agent, _client = gateway
    repaints = []
    monitor = SubagentMonitor(SimpleNamespace(agent=agent, _agent_running=True,
                                              _invalidate=lambda: repaints.append(1)))
    view = gateway_tools.process_view()
    real_rows = view.subagent_rows

    def dead(**_kw):
        raise OSError("socket gone")
    monkeypatch.setattr(view, "subagent_rows", dead)
    monitor.tick()  # must not raise
    assert repaints == []
    monkeypatch.setattr(view, "subagent_rows", real_rows)
    monitor._last_poll = 0
    monitor.tick()
    assert [r["subagent_id"] for r in monitor.entries] == ["sa-0-1"] and repaints == [1]
