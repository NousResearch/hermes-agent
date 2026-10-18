"""Transcript payload contracts: gzip on buffered text, and the lean/capped list projection.

Why these exist: a long session's transcript page ships multi-megabyte JSON. Measured on a
25k-message session, ``GET /api/sessions/{id}/messages?limit=120&order=latest`` was 1.80 MB
and ``limit=500`` was 5.87 MB, uncompressed even when the caller sent
``Accept-Encoding: gzip``. The client that pages that history is the same one holding the
chat socket, so the transfer cost is a liveness problem, not just a bandwidth one.

The contracts pinned here are the ones the API promises:

* compression happens only when the caller asked for it, only for buffered text/JSON, and
  never for media or a server-sent stream;
* ``include_reasoning`` and ``max_content_chars`` default to today's shape, so no existing
  caller changes behaviour by upgrading;
* a capped field names itself in ``truncated_fields`` instead of shrinking silently.
"""
import asyncio
import gzip

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server import (
    APIServerAdapter,
    _cap_message_fields,
    _gzip_accepts,
    _gzip_allows_type,
    gzip_middleware,
)


def test_gzip_accepts_honours_quality_and_wildcards():
    assert _gzip_accepts("gzip") is True
    assert _gzip_accepts("deflate, gzip;q=0.5") is True
    assert _gzip_accepts("*") is True
    assert _gzip_accepts("gzip;q=0") is False
    assert _gzip_accepts("identity") is False
    assert _gzip_accepts("") is False
    assert _gzip_accepts(None) is False


def test_gzip_allows_type_excludes_media_streams_and_archives():
    assert _gzip_allows_type("application/json") is True
    assert _gzip_allows_type("text/plain; charset=utf-8") is True
    assert _gzip_allows_type("image/png") is False
    assert _gzip_allows_type("audio/mpeg") is False
    assert _gzip_allows_type("text/event-stream") is False
    assert _gzip_allows_type("application/zip") is False
    assert _gzip_allows_type("") is False


def test_cap_message_fields_truncates_and_names_what_it_cut():
    capped = _cap_message_fields({"content": "x" * 40, "role": "assistant"}, 10)
    assert capped["content"].startswith("x" * 10)
    assert "30 characters omitted" in capped["content"]
    assert capped["truncated_fields"] == ["content"]
    assert capped["role"] == "assistant"

    untouched = _cap_message_fields({"content": "short"}, 10)
    assert untouched == {"content": "short"}


def test_message_response_defaults_keep_reasoning_and_full_content():
    message = {"id": "m1", "role": "assistant", "content": "hi", "reasoning": "why", "session_id": "s"}
    default = APIServerAdapter._message_response(message)
    assert default["reasoning"] == "why"
    assert default["content"] == "hi"
    assert "truncated_fields" not in default


def test_message_response_drops_reasoning_only_when_asked():
    message = {"id": "m1", "role": "assistant", "content": "hi", "reasoning": "why",
               "reasoning_content": "why2", "session_id": "s"}
    lean = APIServerAdapter._message_response(message, include_reasoning=False)
    assert "reasoning" not in lean and "reasoning_content" not in lean
    assert lean["content"] == "hi"

    capped = APIServerAdapter._message_response(
        {"id": "m1", "role": "assistant", "content": "y" * 50, "reasoning": "why"},
        max_content_chars=8)
    assert capped["truncated_fields"] == ["content"]
    assert "42 characters omitted" in capped["content"]


def _run_middleware_case(headers, response):
    async def serve():
        app = web.Application(middlewares=[gzip_middleware])

        async def handler(_request):
            return response()

        app.router.add_get("/x", handler)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/x", headers=headers)
            body = await resp.read()
            return resp.status, dict(resp.headers), body

    return asyncio.run(serve())


def test_middleware_compresses_a_buffered_json_page_when_asked():
    payload = {"data": ["z" * 400 for _ in range(10)]}
    status, headers, body = _run_middleware_case(
        {"Accept-Encoding": "gzip"}, lambda: web.json_response(payload))
    assert status == 200
    assert headers.get("Content-Encoding") == "gzip"
    assert "Accept-Encoding" in headers.get("Vary", "")
    # The test client decodes the body for us; the contract here is that the content survives
    # the round trip and the response really is marked encoded (wire size is proven by the
    # deployment verify script, which measures real bytes).
    assert body.startswith(b'{"data"') and b"z" * 400 in body


def test_middleware_leaves_identity_small_and_streaming_responses_alone():
    payload = {"data": ["z" * 400 for _ in range(10)]}
    status, headers, body = _run_middleware_case(
        {"Accept-Encoding": "identity"}, lambda: web.json_response(payload))
    assert status == 200 and not headers.get("Content-Encoding")
    assert len(body) > 1000

    status, headers, body = _run_middleware_case(
        {"Accept-Encoding": "gzip"}, lambda: web.Response(text="ok"))
    assert status == 200 and not headers.get("Content-Encoding") and body == b"ok"

    status, headers, body = _run_middleware_case(
        {"Accept-Encoding": "gzip"},
        lambda: web.Response(body=b"\x89PNG" + b"0" * 4096, content_type="image/png"))
    assert status == 200 and not headers.get("Content-Encoding")

    status, headers, body = _run_middleware_case(
        {"Accept-Encoding": "gzip"},
        lambda: web.Response(body=b"data: 1\n\n" * 200, content_type="text/event-stream"))
    assert status == 200 and not headers.get("Content-Encoding")
