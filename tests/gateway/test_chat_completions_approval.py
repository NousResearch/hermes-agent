"""Legacy /v1/chat/completions streams surface approval requests that the client can resolve
through ``POST /v1/runs/{completion_id}/approval`` (#51871)."""
import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server import _create_app, _make_adapter


def _approval_agent(decisions):
    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0

    def run_conversation(**_kw):
        # The real blocking wait the terminal guard uses, reached through the turn's contextvar.
        from tools.approval import _gateway_notify_cb
        from tools.approval_context import get_current_session_key
        from tools.approval_gateway_wait import _await_gateway_decision
        key = get_current_session_key(default="")
        decisions.append(_await_gateway_decision(
            key, _gateway_notify_cb(key), {"command": "rm -rf /tmp/x", "description": "dangerous",
                                           "pattern_key": "rm"}))
        return {"final_response": "ok", "messages": []}
    agent.run_conversation.side_effect = run_conversation
    return agent


@pytest.mark.asyncio
async def test_streamed_completion_approval_resolves_via_runs_endpoint(monkeypatch):
    # Bound the agent thread's blocking wait so a regression fails fast instead of holding teardown.
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 5)
    adapter, decisions = _make_adapter(), []
    app = _create_app(adapter)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", side_effect=lambda **_k: _approval_agent(decisions)):
            resp = await cli.post("/v1/chat/completions", json={
                "model": "hermes-agent", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            assert resp.status == 200
            async def _first_approval():
                event = None
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if line == "event: approval.request":
                        event = "next"
                    elif event == "next" and line.startswith("data: "):
                        return json.loads(line[6:])
                return None
            event = await asyncio.wait_for(_first_approval(), 10)
            assert event and event["run_id"].startswith("chatcmpl") and event["command"]
            approval = await cli.post(f"/v1/runs/{event['run_id']}/approval", json={"choice": "once"})
            assert approval.status == 200, await approval.text()
            assert "[DONE]" in (await asyncio.wait_for(resp.text(), 10))
    assert decisions and decisions[0]["choice"] == "once"
    assert event["run_id"] not in adapter._run_approval_sessions
    assert adapter._run_statuses[event["run_id"]]["status"] == "completed"


def _canonical_approval_turn(owner, decisions):
    """A controlled canonical turn: the real blocking gateway wait, published through the
    authority's shared controls exactly as ``TurnRunner._approval_notify_sync`` does."""
    async def handle(event):
        from gateway.session_results import execution_result
        from tools.approval_gateway_wait import _await_gateway_decision
        sid = event.source.chat_id
        live, generation = owner.sessions[sid], owner.db.get_session(sid)['runtime_generation']

        def notify(data):
            owner.register_approval(sid, generation, live.route, data)
        decisions.append(await asyncio.to_thread(_await_gateway_decision, live.route, notify, {
            "command": "rm -rf /tmp/x", "description": "dangerous", "pattern_key": "rm"}))
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    return handle


async def _sse_event(resp, name):
    pending = None
    async for raw in resp.content:
        line = raw.decode().strip()
        if line == f"event: {name}":
            pending = True
        elif pending and line.startswith("data: "):
            return json.loads(line[6:])
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["chat_completions", "session_stream"])
async def test_canonical_streamed_approval_reaches_sse_and_resolves_via_runs_endpoint(api, owner, monkeypatch, surface):
    """Under the session authority the advertised stream run id is bound to its exact durable
    admission: the shared approval prompt reaches the SSE, and the generation-fenced reply to
    ``POST /v1/runs/{run_id}/approval`` resolves the same waiter WS viewers see."""
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 5)
    decisions = []
    owner.runner._handle_message = _canonical_approval_turn(owner, decisions)
    app = _create_app(api)
    app.router.add_post("/v1/runs/{run_id}/approval", api._handle_run_approval)
    async with TestClient(TestServer(app)) as cli:
        if surface == "chat_completions":
            resp = await cli.post("/v1/chat/completions", json={
                "model": "hermes-agent", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            prefix = "chatcmpl"
        else:
            owner.db.create_session("api-stream", source="api_server")
            resp = await cli.post("/api/sessions/api-stream/chat/stream", json={"message": "hi"})
            prefix = "run_"
        assert resp.status == 200, await resp.text()
        event = await asyncio.wait_for(_sse_event(resp, "approval.request"), 10)
        assert event and event["run_id"].startswith(prefix) and event["command"], event
        assert set(event["choices"]) == {"once", "session", "always", "deny"}
        stale = await cli.post(f"/v1/runs/{event['run_id']}/approval", json={
            "request_id": event["request_id"], "execution_generation": event["execution_generation"] + 1,
            "choice": "once"})
        assert stale.status == 409 and (await stale.json())["error"]["code"] == "stale_generation"
        approval = await cli.post(f"/v1/runs/{event['run_id']}/approval", json={
            "request_id": event["request_id"], "execution_generation": event["execution_generation"],
            "choice": "once"})
        assert approval.status == 200, await approval.text()
        assert (await approval.json())["status"] == "resolved"
        await asyncio.wait_for(resp.text(), 10)
    assert decisions and decisions[0]["choice"] == "once"


@pytest.mark.asyncio
async def test_exact_streamed_retry_replays_the_already_pending_approval(api, owner, monkeypatch):
    """A client that disconnects while an approval is pending and retries the unchanged request
    (same session, Idempotency-Key, auth scope) reuses the admission under a new completion id.
    The prompt was published before the retry subscribed, so the retry stream must replay it from
    the authority's pending controls, or an SSE-only client waits on a prompt it never sees."""
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 15)
    decisions = []
    owner.runner._handle_message = _canonical_approval_turn(owner, decisions)
    api._api_key = "pending-retry-key-long-enough"
    headers = {"Authorization": "Bearer " + api._api_key, "X-Hermes-Session-Id": "retry-sse",
               "Idempotency-Key": "retry-once"}
    body = {"model": "hermes-agent", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    app = _create_app(api)
    app.router.add_post("/v1/runs/{run_id}/approval", api._handle_run_approval)
    async with TestClient(TestServer(app)) as cli:
        first = await cli.post("/v1/chat/completions", json=body, headers=headers)
        assert first.status == 200, await first.text()
        original = await asyncio.wait_for(_sse_event(first, "approval.request"), 10)
        assert original, "first stream never saw the approval"
        first.close()  # disconnect while the prompt is still pending
        retry = await cli.post("/v1/chat/completions", json=body, headers=headers)
        assert retry.status == 200, await retry.text()
        replayed = await asyncio.wait_for(_sse_event(retry, "approval.request"), 5)
        assert replayed["request_id"] == original["request_id"]
        assert replayed["execution_generation"] == original["execution_generation"]
        assert replayed["run_id"].startswith("chatcmpl") and replayed["run_id"] != original["run_id"]
        approval = await cli.post(f"/v1/runs/{replayed['run_id']}/approval", json={
            "request_id": replayed["request_id"], "execution_generation": replayed["execution_generation"],
            "choice": "once"}, headers={"Authorization": headers["Authorization"]})
        assert approval.status == 200, await approval.text()
        assert (await approval.json())["status"] == "resolved"
        text = await asyncio.wait_for(retry.text(), 10)
    assert "[DONE]" in text and text.count("event: approval.request") == 0  # replayed exactly once
    assert [d["choice"] for d in decisions] == ["once"]
    with owner.db._read_ctx() as conn:
        rows = conn.execute("SELECT status FROM session_admissions WHERE principal_id='api'").fetchall()
    assert [r[0] for r in rows] == ["terminal"]
