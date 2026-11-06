"""The Teams aiohttp bridge refuses a body the SDK cannot take, with a 400 and no body bytes logged."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

pytest.importorskip("microsoft_teams")
web = pytest.importorskip("aiohttp.web")
from aiohttp.test_utils import TestClient, TestServer

from plugins.platforms.teams.adapter import _AiohttpBridgeAdapter, check_teams_requirements

MARKER = "zz-body-marker-zz"


def _post(raw: bytes, content_type: str = "application/json"):
    """POST ``raw`` to a bridge whose SDK handler records what it was given."""
    assert check_teams_requirements()  # binds the SDK's HttpRequest into the adapter module
    seen: list = []

    async def _sdk_handler(request):
        seen.append(request["body"])
        return {"status": 200, "body": {"ok": True}}

    async def _go():
        app = web.Application()
        _AiohttpBridgeAdapter(app).register_route("POST", "/api/messages", _sdk_handler)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/api/messages", data=raw, headers={"Content-Type": content_type})
            return resp.status, await resp.text()

    status, text = asyncio.run(_go())
    return status, text, seen


@pytest.mark.parametrize(
    "raw",
    [b"[]", f'["{MARKER}"]'.encode(), f'"{MARKER}"'.encode(), b"42", b"true", b"null"],
)
def test_valid_json_that_is_not_an_object_is_refused(raw, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        status, text, seen = _post(raw)
    assert status == 400
    assert json.loads(text) == {"error": "invalid JSON body"}
    assert seen == []
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "not a JSON object" in logged
    assert MARKER not in logged


@pytest.mark.parametrize(
    ("raw", "content_type", "reason"),
    [
        (f'{{"a": "{MARKER}"'.encode(), "application/json", "Expecting"),
        (b"\xff\xfe" + MARKER.encode(), "application/json", "invalid start byte"),
        (b"{}", f"application/json; charset={MARKER}", "LookupError"),
        (b"[" * 100_000 + b"]" * 100_000, "application/json", "nested too deeply"),
        (f'{{"a": "{MARKER}"'.encode(), "text/plain", "Expecting"),
    ],
    ids=["truncated", "undecodable", "unknown-charset", "too-deep", "truncated-text-plain"],
)
def test_unparsable_body_is_refused_without_echoing_it(raw, content_type, reason, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        status, text, seen = _post(raw, content_type)
    assert status == 400
    assert json.loads(text) == {"error": "invalid JSON body"}
    assert seen == []
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "could not be parsed as JSON" in logged
    assert reason in logged
    assert MARKER not in logged


def test_json_object_reaches_the_sdk() -> None:
    status, text, seen = _post(b'{"type": "message"}')
    assert status == 200
    assert json.loads(text) == {"ok": True}
    assert seen == [{"type": "message"}]


@pytest.mark.parametrize("content_type", ["text/plain", "application/octet-stream"])
def test_json_object_with_a_non_json_content_type_still_reaches_the_sdk(content_type) -> None:
    # request.json() on the pinned aiohttp never checked Content-Type; the guard keeps that.
    status, text, seen = _post(b'{"type": "message"}', content_type)
    assert status == 200
    assert json.loads(text) == {"ok": True}
    assert seen == [{"type": "message"}]


def test_declared_charset_is_used_to_decode_the_body() -> None:
    status, _, seen = _post('{"text": "café"}'.encode("latin-1"), "application/json; charset=latin-1")
    assert status == 200
    assert seen == [{"text": "café"}]
