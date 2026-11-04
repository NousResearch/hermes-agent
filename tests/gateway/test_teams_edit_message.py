"""Teams adapter ``edit_message`` — Bot Framework activity update for streamed responses."""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional

import pytest

from gateway.platforms.base import PlatformConfig
from plugins.platforms.teams.adapter import TeamsAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch, *, requests: Optional[List[Dict[str, Any]]] = None) -> TeamsAdapter:
    config = PlatformConfig(
        enabled=True,
        extra={
            "client_id": "client-id",
            "client_secret": "client-secret",
            "tenant_id": "tenant-id",
        },
    )
    adapter = TeamsAdapter(config)
    adapter._app = object()

    async def _fake_token() -> str:
        return "bearer-token"

    monkeypatch.setattr(adapter, "_get_botframework_token", _fake_token)

    captured: List[Dict[str, Any]] = requests if requests is not None else []

    class _Response:
        def __init__(self, status_code: int):
            self.status_code = status_code

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def put(self, url: str, json: Any = None, headers: Any = None):
            captured.append({"method": "PUT", "url": url, "json": json, "headers": dict(headers or {})})
            return _Response(200)

    fake_httpx = types.SimpleNamespace(AsyncClient=lambda timeout=15.0: _Client())
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    return adapter


class TestTeamsEditMessage:
    def test_missing_ids_rejected(self, monkeypatch):
        import asyncio

        adapter = _make_adapter(monkeypatch)
        result = asyncio.run(adapter.edit_message("", "mid", "hello"))
        assert not result.success
        result = asyncio.run(adapter.edit_message("chat", "", "hello"))
        assert not result.success

    def test_id_charset_rejected(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        import asyncio

        result = asyncio.run(adapter.edit_message("bad/id with spaces", "mid", "hello"))
        assert not result.success
        assert "Bot Framework ID set" in result.error

    def test_puts_activity_update(self, monkeypatch):
        requests: List[Dict[str, Any]] = []
        adapter = _make_adapter(monkeypatch, requests=requests)
        import asyncio

        result = asyncio.run(adapter.edit_message("19:chat-id", "12345", "streamed text"))
        assert result.success
        assert result.message_id == "12345"
        assert len(requests) == 1
        assert requests[0]["method"] == "PUT"
        assert requests[0]["url"].endswith("/v3/conversations/19:chat-id/activities/12345")
        assert requests[0]["json"] == {"type": "message", "text": "streamed text"}
        assert requests[0]["headers"]["Authorization"] == "Bearer bearer-token"
