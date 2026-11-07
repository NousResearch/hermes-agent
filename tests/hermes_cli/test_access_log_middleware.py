"""`AccessLogMiddleware` — one JSON line per HTTP request, opt-in via `HERMES_ACCESS_LOG`."""

from __future__ import annotations

import asyncio
import json
import logging

from hermes_cli import web_server_lifecycle as lifecycle


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_enabled_only_for_truthy_values(monkeypatch):
    for raw, expected in (("1", True), ("true", True), ("on", True), ("", False), ("0", False), ("no", False)):
        monkeypatch.setenv("HERMES_ACCESS_LOG", raw)
        assert lifecycle.access_log_enabled() is expected, raw


def test_non_http_scopes_pass_through_untouched():
    calls = []

    async def inner(scope, receive, send):
        calls.append(scope["type"])

    seen = []
    middleware = lifecycle.AccessLogMiddleware(inner)
    _run(middleware({"type": "websocket"}, None, None))
    _run(middleware({"type": "lifespan"}, None, None))
    assert calls == ["websocket", "lifespan"]
    assert seen == []


def test_http_request_emits_one_json_line(caplog):
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"hello"})

    sent = []

    async def send(message):
        sent.append(message)

    middleware = lifecycle.AccessLogMiddleware(inner)
    scope = {
        "type": "http", "method": "GET", "path": "/api/skills", "query_string": b"profile=default",
        "client": ("100.69.41.80", 51234), "headers": [(b"host", b"127.0.0.1:9119")],
    }
    with caplog.at_level(logging.INFO, logger="hermes_cli.access"):
        _run(middleware(scope, None, send))

    lines = [r.getMessage() for r in caplog.records if r.name == "hermes_cli.access"]
    assert len(lines) == 1, lines
    record = json.loads(lines[0])
    assert record["path"] == "/api/skills"
    assert record["status"] == 200
    assert "dur_ms" in record
    # the wrapped app still wrote through untouched
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]
