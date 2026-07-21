from __future__ import annotations

import pytest
import httpx

from gateway.platforms.qqbot import onboard


class _ChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks


def _stream_response(
    chunks: list[bytes],
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        headers=headers or {},
        stream=_ChunkStream(chunks),
        request=httpx.Request("POST", "https://q.qq.com/onboard"),
    )


def test_onboard_response_rejects_oversized_content_length():
    response = _stream_response([], headers={"content-length": "11"})

    with pytest.raises(RuntimeError, match="exceeds 10 bytes"):
        onboard._read_onboard_response_with_limit(response, body_limit=10)


def test_onboard_response_rejects_streamed_body_over_limit():
    response = _stream_response([b"a" * 6, b"b" * 6])

    with pytest.raises(RuntimeError, match="exceeds 10 bytes"):
        onboard._read_onboard_response_with_limit(response, body_limit=10)


def test_onboard_response_preserves_json_body():
    response = _stream_response([b'{"retcode":0,', b'"data":{"task_id":"t"}}'])

    bounded = onboard._read_onboard_response_with_limit(response, body_limit=100)

    assert bounded.json() == {"retcode": 0, "data": {"task_id": "t"}}


def test_post_onboard_json_uses_streamed_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api"
        return httpx.Response(
            200,
            json={"retcode": 0, "data": {"task_id": "task-1"}},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = onboard._post_onboard_json(
            client,
            "https://q.qq.com/api",
            {"key": "k"},
        )

    assert response.json()["data"]["task_id"] == "task-1"
