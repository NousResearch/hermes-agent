"""WorkerClient: POST /v1/runs, relay SSE deltas + approval round-trip."""

import pytest

from gateway.worker_client import WorkerClient, WorkerRunError


class StubConsumer:
    def __init__(self):
        self.deltas = []

    def on_delta(self, text):
        self.deltas.append(text)


def _client(events, *, posts):
    """Build a client whose SSE yields *events* and records POSTs into *posts*."""

    async def fake_post(url, body):
        posts.append((url, body))
        return {"run_id": "run_1"} if url.endswith("/v1/runs") else {}

    async def fake_sse(url):
        for ev in events:
            yield ev

    return WorkerClient("http://127.0.0.1:5000", "key", post=fake_post, sse=fake_sse)


@pytest.mark.asyncio
async def test_deltas_forwarded_and_completes():
    posts = []
    events = [
        {"event": "message.delta", "delta": "Hello "},
        {"event": "message.delta", "delta": "world"},
        {"event": "run.completed", "output": "Hello world", "usage": {}},
    ]
    consumer = StubConsumer()
    result = await _client(events, posts=posts).dispatch(input="hi", consumer=consumer)
    assert consumer.deltas == ["Hello ", "world"]
    assert result["output"] == "Hello world"
    assert posts[0][0].endswith("/v1/runs")


@pytest.mark.asyncio
async def test_approval_round_trip():
    posts = []
    events = [
        {"event": "approval.request", "run_id": "run_1", "choices": ["once", "deny"]},
        {"event": "run.completed", "output": "done", "usage": {}},
    ]

    async def approver(event):
        assert "once" in event["choices"]
        return "once"

    await _client(events, posts=posts).dispatch(input="do it", consumer=StubConsumer(), approval_handler=approver)
    approval_posts = [p for p in posts if p[0].endswith("/approval")]
    assert approval_posts and approval_posts[0][1]["choice"] == "once"


@pytest.mark.asyncio
async def test_run_failed_raises():
    events = [{"event": "run.failed", "error": "boom"}]
    with pytest.raises(WorkerRunError, match="boom"):
        await _client(events, posts=[]).dispatch(input="x", consumer=StubConsumer())


@pytest.mark.asyncio
async def test_continue_session_sets_body_flag():
    posts = []
    events = [{"event": "run.completed", "output": "ok", "usage": {}}]
    await _client(events, posts=posts).dispatch(
        input="hi", consumer=StubConsumer(), session_id="agent:coder:tg:dm:1", continue_session=True,
    )
    start_body = posts[0][1]
    assert start_body["continue_session"] is True
    assert start_body["session_id"] == "agent:coder:tg:dm:1"


@pytest.mark.asyncio
async def test_continue_session_omitted_when_no_session():
    posts = []
    events = [{"event": "run.completed", "output": "ok", "usage": {}}]
    await _client(events, posts=posts).dispatch(
        input="hi", consumer=StubConsumer(), continue_session=True,
    )
    assert "continue_session" not in posts[0][1]


class _Status404(Exception):
    status = 404


@pytest.mark.asyncio
async def test_reset_session_tolerates_404():
    """/new before the first routed turn → no session yet → 404 is success."""
    async def deleter(_url):
        raise _Status404()

    client = WorkerClient("http://127.0.0.1:5000", "key", delete=deleter)
    await client.reset_session("agent:coder:tg:dm:1")  # must not raise


@pytest.mark.asyncio
async def test_reset_session_reraises_non_404():
    async def deleter(_url):
        raise RuntimeError("connection refused")

    client = WorkerClient("http://127.0.0.1:5000", "key", delete=deleter)
    with pytest.raises(RuntimeError, match="connection refused"):
        await client.reset_session("agent:coder:tg:dm:1")
