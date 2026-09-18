"""Live signals on the /v1/runs event stream.

A messaging turn (Slack, Discord) gets reasoning, lifecycle status, notices, and tool results as
they happen. These tests pin the same signals onto /v1/runs, so an API client can show what the
agent is doing between tool calls instead of a spinner until the reply lands.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agent.credits_tracker import AgentNotice
from gateway.platforms.api_server_runs import _tool_completed_preview, _tool_started_args
from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter


def _events(body: str) -> list[dict]:
    """Decode the ``data:`` frames of an SSE body, skipping comments."""
    return [
        json.loads(line[len("data: "):])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _mock_agent():
    agent = MagicMock()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    return agent


async def _run(script) -> list[dict]:
    adapter = _make_adapter()
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as create:
            agent = _mock_agent()

            def _turn(**_kwargs):
                progress = create.call_args.kwargs["tool_progress_callback"]
                script(agent, progress)
                return {"final_response": "Done."}

            agent.run_conversation.side_effect = _turn
            create.return_value = agent
            resp = await cli.post("/v1/runs", json={"input": "go"})
            run_id = (await resp.json())["run_id"]
            events_resp = await cli.get(f"/v1/runs/{run_id}/events")
            return _events(await events_resp.text())


@pytest.mark.asyncio
async def test_reasoning_streams_as_deltas():
    def script(agent, _progress):
        agent.reasoning_callback("Weighing ")
        agent.reasoning_callback("options")
        agent.reasoning_callback("")  # empty chunks are not worth a frame

    events = await _run(script)
    deltas = [e["text"] for e in events if e["event"] == "reasoning.delta"]
    assert deltas == ["Weighing ", "options"]


@pytest.mark.asyncio
async def test_status_and_notices_reach_the_stream_redacted():
    secret = "sk-ant-api03-" + "a" * 40

    def script(agent, _progress):
        agent.status_callback("lifecycle", "🗜️ Compacting context")
        agent.status_callback("warn", f"Retrying with key {secret}")
        agent.notice_callback(AgentNotice(text="Credits at 90%", level="warn", key="credits_90"))

    events = await _run(script)
    statuses = [e for e in events if e["event"] == "status"]
    assert [s["kind"] for s in statuses] == ["lifecycle", "warn"]
    assert statuses[0]["text"] == "🗜️ Compacting context"
    assert secret not in json.dumps(events)
    [notice] = [e for e in events if e["event"] == "notice"]
    assert notice["text"] == "Credits at 90%"
    assert notice["level"] == "warn"
    assert notice["key"] == "credits_90"


@pytest.mark.asyncio
async def test_tool_completed_carries_a_result_preview():
    def script(_agent, progress):
        progress("tool.started", "terminal", "ls -la", {"command": "ls -la"})
        progress("tool.completed", "terminal", None, None, duration=0.52, is_error=False,
                 result={"output": "README.md\nsrc/", "exit_code": 0})

    events = await _run(script)
    [done] = [e for e in events if e["event"] == "tool.completed"]
    assert done["tool"] == "terminal"
    assert done["duration"] == 0.52
    assert done["error"] is False
    assert json.loads(done["preview"]) == {"output": "README.md\nsrc/", "exit_code": 0}


@pytest.mark.asyncio
async def test_live_events_arrive_in_the_order_the_agent_produced_them():
    def script(agent, progress):
        agent.reasoning_callback("Plan: list files.")
        progress("tool.started", "terminal", "ls", {})
        agent.status_callback("lifecycle", "Tool output is large; summarising")
        progress("tool.completed", "terminal", None, None, duration=0.1, is_error=False, result="ok")

    events = await _run(script)
    names = [e["event"] for e in events if e["event"] != "run.started"]
    assert names[:4] == ["reasoning.delta", "tool.started", "status", "tool.completed"]
    assert names[-1] == "run.completed"


def test_preview_is_redacted_before_it_is_cut():
    secret = "sk-ant-api03-" + "b" * 40
    from agent.redact import redact_sensitive_text as redact

    # The secret straddles the 500-char cut: truncating first would leak its prefix.
    preview = _tool_completed_preview("x" * 480 + " key=" + secret, redact)
    assert "sk-ant-api03-bbbb" not in preview
    assert len(preview) <= 501
    assert _tool_completed_preview(None, redact) == ""


@pytest.mark.asyncio
async def test_tool_started_carries_the_full_arguments():
    command = "curl -sS --max-time 3 http://127.0.0.1:9222/json/version && echo " + "x" * 120

    def script(_agent, progress):
        progress("tool.started", "terminal", "curl -sS --max-time 3 http://127.0.0.1…", {"command": command, "timeout": 30})

    events = await _run(script)
    [started] = [e for e in events if e["event"] == "tool.started"]
    assert started["preview"].endswith("…")
    assert started["args"] == {"command": command, "timeout": 30}


def test_started_args_are_redacted_bounded_and_never_leak_through_structures():
    from agent.redact import redact_sensitive_text as redact

    secret = "sk-ant-api03-" + "c" * 40
    out = _tool_started_args(
        {"command": f"export KEY={secret}", "env": {"KEY": secret}, "body": "y" * 5000, "n": 3},
        redact,
    )
    assert secret not in json.dumps(out)
    assert isinstance(out["env"], str)  # a structure that needed redaction travels as its redacted text
    assert len(out["body"]) == 2001
    assert out["n"] == 3
    assert _tool_started_args(None, redact) is None
    assert _tool_started_args({}, redact) is None
