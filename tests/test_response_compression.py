"""Tests for selective HTTP response compression."""

import asyncio
import gzip

import pytest
from starlette.datastructures import Headers

from hermes_cli.response_compression import (
    SelectiveGZipMiddleware,
    _is_excluded_path,
)


_LARGE_BODY = b'{"payload":"' + (b"compressible-value " * 2_000) + b'"}'


def _run_asgi_response(
    *,
    path="/large",
    accept_encoding=(b"gzip",),
    content_types=(b"application/json",),
    response_headers=(),
    streaming=False,
    body=_LARGE_BODY,
    status_code=200,
):
    """Run the real middleware and return raw ASGI response events."""

    async def app(scope, receive, send):
        headers = [(b"content-type", value) for value in content_types]
        headers.extend(response_headers)
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": headers,
            }
        )
        if streaming:
            midpoint = len(body) // 2
            await send(
                {
                    "type": "http.response.body",
                    "body": body[:midpoint],
                    "more_body": True,
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": body[midpoint:],
                    "more_body": False,
                }
            )
        else:
            await send(
                {
                    "type": "http.response.body",
                    "body": body,
                    "more_body": False,
                }
            )

    request_headers = [
        (b"accept-encoding", value) for value in accept_encoding
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": request_headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    events = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        events.append(message)

    async def run():
        middleware = SelectiveGZipMiddleware(app, minimum_size=1024)
        await middleware(scope, receive, send)

    asyncio.run(run())
    return events


def _response_parts(events):
    start = next(event for event in events if event["type"] == "http.response.start")
    body = b"".join(
        event.get("body", b"")
        for event in events
        if event["type"] == "http.response.body"
    )
    return Headers(raw=start["headers"]), body


@pytest.mark.parametrize(
    "content_type",
    [
        b"Application/JSON; Charset=UTF-8",
        b"application/javascript",
        b"application/xml",
        b"application/problem+json; version=1",
        b"image/svg+xml",
        b"text/css",
        b"text/plain; charset=utf-8",
    ],
)
def test_compresses_only_exact_supported_media_types(content_type):
    headers, body = _response_parts(
        _run_asgi_response(content_types=(content_type,))
    )

    assert headers["content-encoding"] == "gzip"
    assert gzip.decompress(body) == _LARGE_BODY


def test_small_json_stays_uncompressed_below_threshold():
    small_body = b'{"ok":true}'
    headers, body = _response_parts(_run_asgi_response(body=small_body))

    assert "content-encoding" not in headers
    assert body == small_body


def test_large_html_stays_uncompressed():
    html_body = b"<html>" + (b"safe-content " * 2_000) + b"</html>"
    headers, body = _response_parts(
        _run_asgi_response(content_types=(b"text/html",), body=html_body)
    )

    assert "content-encoding" not in headers
    assert body == html_body


@pytest.mark.parametrize(
    "content_types",
    [
        (b"application/jsonfoo",),
        (b"text/plain-extra",),
        (),
        (b"application/json", b"application/json"),
        (b"application/json, text/plain",),
    ],
)
def test_rejects_invalid_missing_or_multiple_content_types(content_types):
    headers, body = _response_parts(
        _run_asgi_response(content_types=content_types)
    )

    assert "content-encoding" not in headers
    assert body == _LARGE_BODY


@pytest.mark.parametrize(
    "values",
    [
        (b"gzip",),
        (b"br, GZip; Q=0.5",),
        (b"br", b"gzip"),
        (b"br, *;q=0.5",),
    ],
)
def test_accept_encoding_negotiation_allows_gzip(values):
    headers, body = _response_parts(
        _run_asgi_response(accept_encoding=values)
    )

    assert headers["content-encoding"] == "gzip"
    assert gzip.decompress(body) == _LARGE_BODY


@pytest.mark.parametrize(
    "values",
    [
        (),
        (b"br",),
        (b"gzip;q=0",),
        (b"gzip;q=bogus",),
        (b"gzip;q=1.001",),
        (b"gzip;q=-1",),
        (b"gzip;foo=bar",),
        (b"gzip;q=1;foo=bar",),
        (b"gzip;foo",),
        (b"gzip;=bar",),
        (b"gzip;q = 1",),
        (b"gzip;q= 1",),
        (b"*;q = 1",),
        (b"*;q= 1",),
        (b"*;q=0",),
        (b"gzip;q=0, *;q=1",),
    ],
)
def test_accept_encoding_negotiation_rejects_disallowed_or_malformed_gzip(values):
    headers, body = _response_parts(
        _run_asgi_response(accept_encoding=values)
    )

    assert "content-encoding" not in headers
    assert body == _LARGE_BODY


@pytest.mark.parametrize(
    "path",
    [
        "/api/config",
        "/api/config/raw",
        "/api/env/reveal",
        "/api/auth",
        "/api/auth/ws-ticket",
        "/auth",
        "/auth/native/token",
        "/api/providers/oauth/anthropic/start",
        "/api/mcp/oauth/flows/flow-id",
        "/api/webhooks",
        "/api/webhooks/example",
        "/api/pairing",
        "/api/files/read",
        "/api/files/download",
        "/api/fs/read-text",
        "/api/fs/read-data-url",
        "/api/fs/download",
        "/api/logs",
        "/api/media",
        "/api/ops/backup/download",
        "/api/actions/hermes-update/status",
        "/api/messaging/whatsapp/onboarding/start",
        "/api/messaging/whatsapp/onboarding/pairing-id",
        "/api/messaging/whatsapp/onboarding/pairing-id/apply",
        "/api/messaging/telegram/onboarding/start",
        "/api/messaging/telegram/onboarding/pairing-id",
        "/api/messaging/telegram/onboarding/pairing-id/apply",
    ],
)
def test_sensitive_dashboard_routes_are_not_compressed_even_when_streaming(path):
    headers, body = _response_parts(
        _run_asgi_response(path=path, streaming=True)
    )

    assert "content-encoding" not in headers
    assert body == _LARGE_BODY


@pytest.mark.parametrize(
    "path",
    [
        "/api/configuration",
        "/api/env/revealer",
        "/api/authorize",
        "/authentic",
        "/api/webhooksmith",
        "/api/files/reader",
        "/api/files/downloader",
        "/api/fs/read-texture",
        "/api/fs/downloader",
        "/api/ops/backup/downloader",
        "/api/actionsmith/hermes-update/status",
        "/api/actions-extra/hermes-update/status",
        "/api/messaging/whatsapp/onboardings/start",
        "/api/messaging/telegram/onboarding-extra/start",
    ],
)
def test_sensitive_route_lookalikes_remain_compressible(path):
    headers, body = _response_parts(_run_asgi_response(path=path))

    assert headers["content-encoding"] == "gzip"
    assert gzip.decompress(body) == _LARGE_BODY


def test_sensitive_route_census_is_registered_and_excluded():
    from hermes_cli.web_server import app

    expected_paths = {
        "/api/files/download",
        "/api/fs/download",
        "/api/ops/backup/download",
        "/api/actions/{name}/status",
        "/api/messaging/whatsapp/onboarding/start",
        "/api/messaging/whatsapp/onboarding/{pairing_id}",
        "/api/messaging/whatsapp/onboarding/{pairing_id}/apply",
        "/api/messaging/telegram/onboarding/start",
        "/api/messaging/telegram/onboarding/{pairing_id}",
        "/api/messaging/telegram/onboarding/{pairing_id}/apply",
    }
    registered_paths = {route.path for route in app.routes}

    assert expected_paths <= registered_paths
    assert all(_is_excluded_path(path) for path in expected_paths)


def test_nonstreaming_wire_headers_match_compressed_body():
    headers, body = _response_parts(
        _run_asgi_response(
            response_headers=((b"content-length", str(len(_LARGE_BODY)).encode()),)
        )
    )

    assert headers["vary"] == "Accept-Encoding"
    assert int(headers["content-length"]) == len(body)
    assert gzip.decompress(body) == _LARGE_BODY


def test_streaming_wire_headers_omit_length_and_body_is_valid_gzip():
    headers, body = _response_parts(
        _run_asgi_response(
            streaming=True,
            response_headers=((b"content-length", str(len(_LARGE_BODY)).encode()),),
        )
    )

    assert headers["content-encoding"] == "gzip"
    assert headers["vary"] == "Accept-Encoding"
    assert "content-length" not in headers
    assert gzip.decompress(body) == _LARGE_BODY


def test_preexisting_content_encoding_is_preserved_without_recompression():
    headers, body = _response_parts(
        _run_asgi_response(
            response_headers=(
                (b"content-encoding", b"br"),
                (b"content-length", str(len(_LARGE_BODY)).encode()),
            )
        )
    )

    assert headers["content-encoding"] == "br"
    assert int(headers["content-length"]) == len(_LARGE_BODY)
    assert "vary" not in headers
    assert body == _LARGE_BODY


@pytest.mark.parametrize(
    ("status_code", "response_headers"),
    [
        (206, ()),
        (200, ((b"content-range", b"bytes 0-99/1000"),)),
    ],
)
def test_range_responses_bypass_compression(status_code, response_headers):
    headers, body = _response_parts(
        _run_asgi_response(
            status_code=status_code,
            response_headers=response_headers,
        )
    )

    assert "content-encoding" not in headers
    assert "vary" not in headers
    assert body == _LARGE_BODY


@pytest.mark.parametrize("etag", [b'"strong-validator"', b'W/"weak-validator"'])
def test_etag_bearing_responses_bypass_compression(etag):
    headers, body = _response_parts(
        _run_asgi_response(response_headers=((b"etag", etag),))
    )

    assert headers["etag"] == etag.decode()
    assert "content-encoding" not in headers
    assert "vary" not in headers
    assert body == _LARGE_BODY
