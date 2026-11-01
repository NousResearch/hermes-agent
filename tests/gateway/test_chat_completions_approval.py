"""Legacy /v1/chat/completions streams surface approval requests that the client can resolve
through ``POST /v1/runs/{completion_id}/approval`` (#51871)."""
import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from tests.waiting import async_wait_until
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
    adapter._approval_events = True  # opt-in: the structured frame on an OpenAI-compatible stream
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


async def _drain_stream(resp, frames, prompts):
    """Collect every SSE line of one stream and note each assistant content delta."""
    async for raw in resp.content:
        line = raw.decode().strip()
        frames.append(line)
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        try:
            chunk = json.loads(line[6:])
        except ValueError:
            continue
        choices = chunk.get("choices") or [{}]
        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            prompts.append(delta["content"])


@pytest.mark.asyncio
async def test_default_chat_stream_renders_approval_as_text_and_a_chat_reply_answers_it(monkeypatch):
    """The OpenAI-compatible surface must stay OpenAI-shaped: no named frame, ever.

    An approval reaches the client as assistant text (a named ``approval.request`` data frame is a
    schema error to strict decoders), and the client's next turn — the whole message being one
    approval word — answers that prompt instead of starting an unrelated turn.
    """
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 20)
    adapter, decisions = _make_adapter(), []
    app = _create_app(adapter)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    async with TestClient(TestServer(app)) as cli:
        adapter._host, adapter._port = "127.0.0.1", cli.server.port  # the reply path self-posts here
        with patch.object(adapter, "_create_agent", side_effect=lambda **_k: _approval_agent(decisions)):
            resp = await cli.post("/v1/chat/completions", json={
                "model": "hermes-agent", "stream": True,
                "messages": [{"role": "user", "content": "hi"}]})
            assert resp.status == 200
            frames, deltas = [], []
            reader = asyncio.ensure_future(_drain_stream(resp, frames, deltas))

            def _prompt_now() -> str:
                text = "".join(deltas)
                return text if "Approval needed" in text or reader.done() else ""
            prompt = await async_wait_until(
                _prompt_now, timeout=15, what="the approval prompt on the stream")
            assert "Approval needed" in prompt, f"no approval text on the stream: {frames}"
            assert "rm -rf /tmp/x" in prompt and "deny" in prompt
            assert not any(line == "event: approval.request" for line in frames)
            assert decisions == [], "the agent must still be parked on the approval"

            ack = await cli.post("/v1/chat/completions", json={
                "model": "hermes-agent", "stream": True,
                "messages": [{"role": "user", "content": "approve"}]})
            assert ack.status == 200, await ack.text()
            ack_text = await asyncio.wait_for(ack.text(), 15)
            assert "Approval recorded: once" in ack_text and "[DONE]" in ack_text
            await asyncio.wait_for(reader, 15)
    assert decisions and decisions[0]["choice"] == "once"
    assert "data: [DONE]" in frames, frames[-5:]
    assert adapter._chat_approval_books()["pending"] == {}


@pytest.mark.asyncio
async def test_prompt_closes_the_turn_and_the_answer_turn_streams_the_continuation(monkeypatch):
    """A chat UI cannot type while a response is open, so the prompt must END that turn.

    The agent stays parked (its task is untouched); the turn that answers the prompt carries the
    resumed work on its own stream, because nobody is watching the closed one.
    """
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 20)
    adapter, decisions = _make_adapter(), []
    app = _create_app(adapter)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    async with TestClient(TestServer(app)) as cli:
        adapter._host, adapter._port = "127.0.0.1", cli.server.port  # the reply path self-posts here
        with patch.object(adapter, "_create_agent", side_effect=lambda **_k: _approval_agent(decisions)):
            resp = await cli.post("/v1/chat/completions", json={
                "model": "hermes-agent", "stream": True,
                "messages": [{"role": "user", "content": "hi"}]})
            assert resp.status == 200
            frames, deltas = [], []
            await asyncio.wait_for(_drain_stream(resp, frames, deltas), 15)  # must complete: turn over
            prompt = "".join(deltas)
            assert "Approval needed" in prompt and "rm -rf /tmp/x" in prompt, prompt
            assert decisions == [], "the agent must still be parked on the approval"
            chunks = [json.loads(f[6:]) for f in frames
                      if f.startswith("data: ") and f[6:] != "[DONE]"]
            assert chunks[-1]["choices"][0]["finish_reason"] == "stop", frames[-3:]
            assert "data: [DONE]" in frames[-3:], frames[-3:]

            ack = await cli.post("/v1/chat/completions", json={
                "model": "hermes-agent", "stream": True,
                "messages": [{"role": "user", "content": "approve"}]})
            assert ack.status == 200
            ack_text = await asyncio.wait_for(ack.text(), 15)
    assert "data: [DONE]" in ack_text
    contents = []
    for line in ack_text.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            chunk = json.loads(line[6:])
            contents.append(((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content") or "")
    joined = "".join(contents)
    assert "Approval recorded: once" in joined, joined
    assert joined.rstrip().endswith("ok"), joined  # the resumed turn's own final text
    assert decisions and decisions[0]["choice"] == "once"


@pytest.mark.asyncio
async def test_non_streaming_answer_turn_returns_the_continuation_as_one_message(monkeypatch):
    """A non-streaming client that answers the prompt gets the resumed turn's text in one message."""
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 20)
    adapter, decisions = _make_adapter(), []
    app = _create_app(adapter)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    async with TestClient(TestServer(app)) as cli:
        adapter._host, adapter._port = "127.0.0.1", cli.server.port
        with patch.object(adapter, "_create_agent", side_effect=lambda **_k: _approval_agent(decisions)):
            resp = await cli.post("/v1/chat/completions", json={
                "model": "hermes-agent", "stream": True,
                "messages": [{"role": "user", "content": "hi"}]})
            await asyncio.wait_for(resp.text(), 15)  # prompt surfaced, turn closed, agent parked
            assert decisions == []
            ack = await cli.post("/v1/chat/completions", json={
                "model": "hermes-agent", "messages": [{"role": "user", "content": "approve"}]})
            assert ack.status == 200
            payload = await asyncio.wait_for(ack.json(), 15)
    content = payload["choices"][0]["message"]["content"]
    assert "Approval recorded: once" in content, content
    assert content.rstrip().endswith("ok"), content
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert decisions and decisions[0]["choice"] == "once"


@pytest.mark.asyncio
async def test_chat_reply_answers_a_pending_approval_under_session_authority(api, owner, monkeypatch):
    """Under the session authority the reply must resolve through the generation-fenced door."""
    monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 20)
    decisions = []
    owner.runner._handle_message = _canonical_approval_turn(owner, decisions)
    api._api_key = "chat-reply-authority-key-long"
    headers = {"Authorization": "Bearer " + api._api_key}
    app = _create_app(api)
    app.router.add_post("/v1/runs/{run_id}/approval", api._handle_run_approval)
    async with TestClient(TestServer(app)) as cli:
        api._host, api._port = "127.0.0.1", cli.server.port
        resp = await cli.post("/v1/chat/completions", json={
            "model": "hermes-agent", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}, headers=headers)
        assert resp.status == 200, await resp.text()
        frames, deltas = [], []
        reader = asyncio.ensure_future(_drain_stream(resp, frames, deltas))

        def _prompt_now() -> str:
            text = "".join(deltas)
            return text if "Approval needed" in text or reader.done() else ""
        prompt = await async_wait_until(
            _prompt_now, timeout=15, what="the approval prompt on the stream")
        assert "Approval needed" in prompt, f"no approval text: {frames}"
        ack = await cli.post("/v1/chat/completions", json={
            "model": "hermes-agent", "stream": False,
            "messages": [{"role": "user", "content": "deny"}]}, headers=headers)
        assert ack.status == 200, await ack.text()
        assert "Approval recorded: deny" in (await ack.json())["choices"][0]["message"]["content"]
        await asyncio.wait_for(reader, 15)
    assert [d["choice"] for d in decisions] == ["deny"]


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
    if surface == "chat_completions":
        api._approval_events = True  # opt-in: the structured frame on an OpenAI-compatible stream
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
    api._approval_events = True  # opt-in: the structured frame on an OpenAI-compatible stream
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
