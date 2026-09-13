"""Behavior contracts for selective dashboard response compression."""

import gzip
import json

import pytest

from hermes_cli.response_compression import (
    _accepts_gzip,
    _is_compressible_content_type,
    _is_excluded_path,
    SelectiveGZipMiddleware,
)
from agent.trajectory import save_trajectory


def test_gzip_acceptance_honors_explicit_quality_values():
    assert _accepts_gzip(["br, gzip;q=0.8"]) is True
    assert _accepts_gzip(["gzip;q=0"]) is False
    assert _accepts_gzip(["*;q=0.5"]) is True


def test_compression_allowlist_accepts_json_variants_only():
    assert _is_compressible_content_type(["application/json; charset=utf-8"]) is True
    assert _is_compressible_content_type(["application/problem+json"]) is True
    assert _is_compressible_content_type(["text/html"]) is False
    assert _is_compressible_content_type(["application/json", "text/plain"]) is False


def test_sensitive_dashboard_routes_are_never_compressed():
    assert _is_excluded_path("/api/config/raw") is True
    assert _is_excluded_path("/api/auth/session") is True
    assert _is_excluded_path("/api/audio/voice-config") is True
    assert _is_excluded_path("/api/mcp/servers/example/auth") is True
    assert _is_excluded_path("/api/configuration") is False
    assert _is_excluded_path("/api/status") is False


def test_trajectory_defaults_to_readable_gzip_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_trajectory([{"from": "human", "value": "hello"}], "test-model", True)
    output = tmp_path / "trajectory_samples.jsonl.gz"
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        assert json.loads(stream.readline())["model"] == "test-model"


@pytest.mark.asyncio
async def test_excluded_response_bypasses_compression_in_asgi_path():
    body = b"<html>" + (b"sensitive response" * 100) + b"</html>"

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/html"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})

    messages = []

    async def send(message):
        messages.append(message)

    middleware = SelectiveGZipMiddleware(app, minimum_size=0)
    await middleware(
        {"type": "http", "path": "/api/status", "headers": [(b"accept-encoding", b"gzip")]},
        lambda: None,
        send,
    )

    start, response = messages
    headers = dict(start["headers"])
    assert b"content-encoding" not in headers
    assert response["body"] == body
