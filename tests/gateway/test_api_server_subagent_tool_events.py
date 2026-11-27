"""``platforms.api_server.subagent_tool_events``: opt-in ``subagent.tool.started`` /
``subagent.tool.completed`` on ``/v1/runs/{id}/events``, carrying a child's tool steps without
its arguments, preview or result."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_runs import _RunStream
from tools.delegate_tool_progress import _build_child_progress_callback

MARKER = "tool-content-marker-7f3a"
RUN_ID = "run_subagent_tools"


def _adapter(**extra) -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _child_relay(adapter, loop):
    """A real child relay whose parent callback is the run's SSE callback, plus the run's stream."""
    adapter._run_streams[RUN_ID] = stream = _RunStream()
    parent = MagicMock()
    parent._delegate_spinner = None
    parent.tool_progress_callback = adapter._make_run_event_callback(RUN_ID, loop)
    session_ref = {"session_id": "child-sess-1", "delegation_id": "deleg-1"}
    relay = _build_child_progress_callback(
        1, f"goal mentioning {MARKER}", parent, task_count=2,
        subagent_id="sa-1", parent_id="sa-0", depth=1, model="m", session_ref=session_ref)
    return relay, stream


def _events(stream):
    return [event for _, event in stream.backlog if event is not None]


def _run_child_tool(relay, name, *, is_error=False):
    relay("tool.started", name, f"preview {MARKER}", {"command": MARKER})
    relay("tool.completed", name, None, None, duration=0.12345, is_error=is_error, result=f"output {MARKER}")


@pytest.mark.asyncio
async def test_child_tool_steps_reach_the_run_stream_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(subagent_tool_events=True)
    relay, stream = _child_relay(adapter, asyncio.get_running_loop())

    _run_child_tool(relay, "terminal")
    await asyncio.sleep(0)  # the callback hops onto the loop via call_soon_threadsafe

    started, completed = _events(stream)
    assert [started["event"], completed["event"]] == ["subagent.tool.started", "subagent.tool.completed"]
    for event in (started, completed):
        assert event["run_id"] == RUN_ID and event["timestamp"] > 0
        assert event["tool"] == "terminal"
        assert (event["subagent_id"], event["parent_id"], event["task_index"], event["depth"]) == ("sa-1", "sa-0", 1, 1)
        assert (event["child_session_id"], event["delegation_id"]) == ("child-sess-1", "deleg-1")
    assert started["tool_index"] == completed["tool_index"] == 1
    assert completed["duration"] == 0.123 and completed["error"] is False
    assert "duration" not in started and "error" not in started


@pytest.mark.asyncio
async def test_child_tool_failure_is_reported_on_the_completion(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(subagent_tool_events=True)
    relay, stream = _child_relay(adapter, asyncio.get_running_loop())

    _run_child_tool(relay, "terminal", is_error=True)
    await asyncio.sleep(0)

    assert _events(stream)[-1]["error"] is True


@pytest.mark.asyncio
async def test_child_tool_events_carry_no_tool_content(monkeypatch, tmp_path):
    """The goal, the tool's arguments, its preview and its result all hold MARKER; none may reach the wire."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(subagent_tool_events=True)
    relay, stream = _child_relay(adapter, asyncio.get_running_loop())

    _run_child_tool(relay, "terminal")
    await asyncio.sleep(0)

    events = _events(stream)
    assert [event["event"] for event in events] == ["subagent.tool.started", "subagent.tool.completed"]
    assert MARKER not in json.dumps(events)
    assert not {"preview", "args", "result", "goal"} & set().union(*events)


@pytest.mark.asyncio
async def test_parallel_same_name_tools_finishing_out_of_order_pair_by_call_id(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(subagent_tool_events=True)
    relay, stream = _child_relay(adapter, asyncio.get_running_loop())

    relay("tool.started", "terminal", "a", {}, tool_call_id="call-a")
    relay("tool.started", "terminal", "b", {}, tool_call_id="call-b")
    relay("tool.completed", "terminal", None, None, duration=2.0, is_error=True, tool_call_id="call-b")
    relay("tool.completed", "terminal", None, None, duration=1.0, is_error=False, tool_call_id="call-a")
    await asyncio.sleep(0)

    events = _events(stream)
    assert [(e["tool_index"], e.get("tool_call_id")) for e in events] == [
        (1, "call-a"), (2, "call-b"), (2, "call-b"), (1, "call-a")]
    assert [(e["event"], e["error"]) for e in events[2:]] == [
        ("subagent.tool.completed", True), ("subagent.tool.completed", False)]


@pytest.mark.asyncio
async def test_child_tool_events_are_off_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter()
    relay, stream = _child_relay(adapter, asyncio.get_running_loop())

    relay("subagent.start", preview="goal")
    _run_child_tool(relay, "terminal")
    await asyncio.sleep(0)

    # Lifecycle events still flow; the child's tool steps do not.
    assert [event["event"] for event in _events(stream)] == ["subagent.start"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra", "enabled"),
    [({}, False), ({"subagent_tool_events": True}, True), ({"subagent_tool_events": "false"}, False)])
async def test_capabilities_advertise_the_flag(monkeypatch, tmp_path, extra, enabled):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter(**extra)
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    async with TestClient(TestServer(app)) as client:
        data = await (await client.get("/v1/capabilities")).json()
    assert data["features"]["subagent_tool_events"] is enabled
