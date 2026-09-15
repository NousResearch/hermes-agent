"""Regression coverage for prompt SSE activity during long preflight work."""

import asyncio

import pytest

from gateway.platforms import api_server
from gateway.platforms.api_server_openai_routes import _iter_stream_items


class _RecordingResponse:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.writes.append(data)


@pytest.mark.asyncio
async def test_idle_stream_writes_keepalive_before_remote_client_timeout(monkeypatch):
    """A compression-bound agent must produce a byte within a 20s client window.

    ``_iter_stream_items`` is shared by OpenAI chat and Responses streaming;
    native session streaming uses the same API-server keepalive constant.
    """
    stream_q = asyncio.Queue()
    agent_task = asyncio.get_running_loop().create_future()
    response = _RecordingResponse()
    clock = [0.0, 20.0, 20.0]

    def monotonic() -> float:
        return clock.pop(0) if clock else 20.0

    monkeypatch.setattr("gateway.platforms.api_server_openai_routes.time.monotonic", monotonic)

    waits = 0

    async def elapsed_wait_for(*_args, **_kwargs):
        nonlocal waits
        waits += 1
        if waits == 2:
            agent_task.set_result(None)
        raise asyncio.TimeoutError

    monkeypatch.setattr(
        "gateway.platforms.api_server_openai_routes.asyncio.wait_for", elapsed_wait_for
    )

    assert api_server.CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS <= 20.0
    assert [item async for item in _iter_stream_items(stream_q, agent_task, response)] == []
    assert response.writes == [b": keepalive\n\n"]
