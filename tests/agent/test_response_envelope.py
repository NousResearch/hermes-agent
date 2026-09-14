"""Real OpenAI SDK parsing over the shared client's response normalization."""

import gzip
import json

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from agent import process_bootstrap as bootstrap


_COMPLETION = {
    "id": "completion", "object": "chat.completion", "created": 1, "model": "test",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
}


@pytest.fixture
def client_factory(monkeypatch):
    bootstrap.close_shared_transports()
    monkeypatch.setattr(bootstrap, "_get_proxy_for_base_url", lambda _: None)

    def build(handler, *, async_mode=False, host="api.cline.bot"):
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(httpx, "AsyncHTTPTransport" if async_mode else "HTTPTransport", lambda **kwargs: transport)
        client = bootstrap.build_keepalive_http_client(f"https://{host}/v1", async_mode=async_mode)
        assert client is not None
        return client

    yield build
    bootstrap.close_shared_transports()


class _SSE(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, reads):
        self.reads = reads

    def __iter__(self):
        self.reads.append(True)
        chunk = {"id": "chunk", "object": "chat.completion.chunk", "created": 1, "model": "test",
                 "choices": [{"index": 0, "delta": {"content": "streamed"}, "finish_reason": None}]}
        yield b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n"

    async def __aiter__(self):
        for chunk in self:
            yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_sdk_completion_and_lazy_stream_keep_client_ownership(client_factory, async_mode):
    reads = []

    def handler(request):
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_SSE(reads))
        content = gzip.compress(json.dumps({"data": _COMPLETION, "success": True}).encode())
        return httpx.Response(200, content=content,
                              headers={"content-type": "application/json", "content-encoding": "gzip"})

    client = client_factory(handler, async_mode=async_mode)
    sibling = client_factory(handler, async_mode=async_mode)
    if not async_mode:
        assert client._transport._inner is sibling._transport._inner
    sdk = (AsyncOpenAI if async_mode else OpenAI)(
        base_url="https://api.cline.bot/v1", api_key="test", http_client=client, max_retries=0)
    kwargs = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    try:
        pending = sdk.chat.completions.create(**kwargs)
        completion = await pending if async_mode else pending
        assert completion.choices[0].message.content == "hello"
        pending = sdk.chat.completions.create(**kwargs, stream=True)
        stream = await pending if async_mode else pending
        assert not reads
        chunks = [chunk async for chunk in stream] if async_mode else list(stream)
        assert chunks[0].choices[0].delta.content == "streamed"
        if async_mode:
            await sdk.close()
            response = await sibling.post("https://api.cline.bot/v1/chat/completions", json={})
        else:
            sdk.close()
            response = sibling.post("https://api.cline.bot/v1/chat/completions", json={})
        assert response.json() == _COMPLETION
        assert int(response.headers["content-length"]) == len(response.content)
        assert "content-encoding" not in response.headers
    finally:
        if async_mode:
            await sdk.close()
            await sibling.aclose()
        else:
            sdk.close()
            sibling.close()


@pytest.mark.parametrize("body,status,host", [
    ({"data": _COMPLETION, "success": False}, 200, "api.cline.bot"),
    ({"data": _COMPLETION, "error": {"message": "denied"}}, 200, "api.cline.bot"),
    ({"data": _COMPLETION, "success": True}, 401, "api.cline.bot"),
    ({"data": _COMPLETION, "success": True}, 200, "api.cline.bot.example"),
    (_COMPLETION, 200, "api.cline.bot"),
])
def test_errors_standard_bodies_and_other_hosts_stay_unchanged(client_factory, body, status, host):
    raw = json.dumps(body).encode()
    with client_factory(lambda request: httpx.Response(status, content=raw,
                        headers={"content-type": "application/json"}), host=host) as client:
        response = client.post(f"https://{host}/v1/chat/completions", json={})
        assert response.status_code == status
        assert response.content == raw
