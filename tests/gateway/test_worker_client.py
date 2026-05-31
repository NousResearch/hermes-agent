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
