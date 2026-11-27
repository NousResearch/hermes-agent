"""delegate_task children (``subagent.*``) projected as one ACP tool call per child.

Regression for #85436 and #149110: both the in-process ACP server and the gateway viewer dropped
``subagent.*`` events, so an ACP client never saw a delegated child run.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from acp_adapter.events import SUBAGENT_EVENTS, subagent_updates


def _kw(**extra):
    return {"subagent_id": "sa-1", "task_index": 0, "task_count": 3, "goal": "Leia o AGENTS.md", "model": "m", **extra}


def _fields(update):
    return update.model_dump(by_alias=False, exclude_none=True)


def test_start_opens_one_in_progress_call_with_the_goal():
    state = {}
    out = subagent_updates(state, "subagent.start", None, "Leia o AGENTS.md", _kw())
    assert len(out) == 1
    f = _fields(out[0])
    assert f["session_update"] == "tool_call"
    assert f["status"] == "in_progress"
    # "think" would render as a reasoning row (bb's provider-acp); a child must show as a tool row.
    assert f["kind"] == "other"
    assert f["title"].startswith("Subagent 1/3 (m): Leia o AGENTS.md")
    assert "sa-1" in state


def test_tool_refreshes_title_and_tail_on_the_same_call():
    state = {}
    start = _fields(subagent_updates(state, "subagent.start", None, None, _kw())[0])
    out = subagent_updates(state, "subagent.tool", "read_file", "AGENTS.md", _kw(tool_count=1))
    assert len(out) == 1
    f = _fields(out[0])
    assert f["session_update"] == "tool_call_update"
    assert f["tool_call_id"] == start["tool_call_id"]
    assert f["title"].endswith("· 1 tool")
    assert "read_file: AGENTS.md" in f["content"][0]["content"]["text"]


def test_complete_closes_with_summary_and_forgets_the_child():
    state = {}
    tc = _fields(subagent_updates(state, "subagent.start", None, None, _kw())[0])["tool_call_id"]
    out = subagent_updates(state, "subagent.complete", None, None, _kw(status="completed", summary="ok 1", duration_seconds=2.5))
    f = _fields(out[-1])
    assert f["tool_call_id"] == tc and f["status"] == "completed"
    assert f["content"][0]["content"]["text"].startswith("ok 1")
    assert state == {}


def test_failed_child_closes_failed_and_late_tool_without_start_still_opens():
    state = {}
    out = subagent_updates(state, "subagent.tool", "terminal", "ls", _kw(subagent_id="sa-2"))
    assert [_fields(u)["session_update"] for u in out] == ["tool_call", "tool_call_update"]
    f = _fields(subagent_updates(state, "subagent.complete", None, None, _kw(subagent_id="sa-2", status="timeout"))[-1])
    assert f["status"] == "failed"


def test_parallel_children_never_share_a_call():
    state = {}
    a = _fields(subagent_updates(state, "subagent.start", None, None, _kw(subagent_id="a", task_index=0))[0])
    b = _fields(subagent_updates(state, "subagent.start", None, None, _kw(subagent_id="b", task_index=1))[0])
    assert a["tool_call_id"] != b["tool_call_id"]
    assert b["title"].startswith("Subagent 2/3")


def test_other_subagent_events_and_orphan_complete_emit_nothing():
    state = {}
    assert "subagent.thinking" not in SUBAGENT_EVENTS
    assert subagent_updates(state, "subagent.thinking", None, "x", _kw()) == []
    assert subagent_updates(state, "subagent.complete", None, None, {"subagent_id": "z"}) == []
    assert state == {}


def test_gateway_viewer_projects_subagent_frames():
    from acp_adapter.gateway_server import GatewayACPAgent

    agent = GatewayACPAgent.__new__(GatewayACPAgent)
    agent._subagents = {}
    agent._conn = MagicMock()
    agent._conn.session_update = AsyncMock()
    payload = {"subagent_id": "sa-9", "goal": "g", "task_index": 0, "task_count": 1}
    asyncio.run(agent._project_subagent("subagent.start", "s1", "ed1", payload))
    asyncio.run(agent._project_subagent("subagent.tool", "s1", "ed1", {**payload, "tool_name": "terminal", "text": "ls"}))
    asyncio.run(agent._project_subagent("subagent.complete", "s1", "ed1", {**payload, "status": "completed", "summary": "done"}))
    calls = agent._conn.session_update.await_args_list
    assert [c.kwargs["session_id"] for c in calls] == ["ed1", "ed1", "ed1"]
    assert _fields(calls[-1].kwargs["update"])["status"] == "completed"
    assert agent._subagents == {"s1": {}}


def test_gateway_viewer_survives_a_failing_send():
    from acp_adapter.gateway_server import GatewayACPAgent

    agent = GatewayACPAgent.__new__(GatewayACPAgent)
    agent._subagents = {}
    agent._conn = MagicMock()
    agent._conn.session_update = AsyncMock(side_effect=RuntimeError("gone"))
    asyncio.run(agent._project_subagent("subagent.start", "s1", "ed1", {"subagent_id": "x", "goal": "g"}))
    agent._conn.session_update.assert_awaited_once()


def test_in_process_callback_relays_subagents_without_touching_tool_queues(monkeypatch):
    from acp_adapter import events

    sent = []
    monkeypatch.setattr(events, "_send_update", lambda conn, sid, loop, update: sent.append((sid, update)))
    ids, meta = {}, {}
    cb = events.make_tool_progress_cb(MagicMock(), "sess", MagicMock(), ids, meta)
    cb("subagent.start", None, "g", None, **_kw())
    cb("subagent.tool", "terminal", "ls", None, **_kw(tool_count=1))
    cb("subagent.complete", None, None, None, **_kw(status="completed", summary="ok"))
    assert [s for s, _ in sent] == ["sess"] * 3
    assert _fields(sent[-1][1])["status"] == "completed"
    assert ids == {} and meta == {}
