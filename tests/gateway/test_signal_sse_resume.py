"""A reconnecting Signal SSE stream resumes from the last handled event id (signal-cli replays the gap)."""

import asyncio
import json

import httpx
import pytest

from gateway.config import PlatformConfig


def _event(i: int) -> str:
    env = {"envelope": {"source": "+15550000001", "timestamp": i, "dataMessage": {"message": f"m{i}"}}}
    return f"id:{i}\nevent:receive\ndata:{json.dumps(env)}\n\n"


@pytest.mark.asyncio
async def test_reconnect_sends_last_event_id_of_the_last_handled_event(monkeypatch):
    import plugins.platforms.signal.adapter as sa

    monkeypatch.setattr(sa, "SSE_RETRY_DELAY_INITIAL", 0.01)
    cfg = PlatformConfig()
    cfg.enabled = True
    cfg.extra = {"http_url": "http://signal.test", "account": "+15551234567"}
    adapter = sa.SignalAdapter(cfg)
    seen_headers, delivered = [], []

    def respond(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("last-event-id"))
        if len(seen_headers) == 1:  # an id whose data line never arrived must not become the resume point
            return httpx.Response(200, text=_event(1) + "id:2\n")
        adapter._running = False
        return httpx.Response(200, text=_event(2))

    async def record(envelope):
        delivered.append(envelope["envelope"]["dataMessage"]["message"])

    adapter.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    adapter._handle_envelope = record
    adapter._running = True
    await asyncio.wait_for(adapter._sse_listener(), timeout=5)

    assert seen_headers == [None, "1"]
    assert delivered == ["m1"]  # the listener stopped before reading the replay on the second stream
