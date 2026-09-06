"""Compatibility seams for extracted RoomLink dispatch handling."""

import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms import api_server
from gateway.platforms import api_server_room_dispatch as room_dispatch


def test_api_server_keeps_room_dispatch_methods_on_the_adapter_class():
    assert {
        "_ensure_hosted_member_session",
        "_normalize_room_dispatch",
    } <= api_server.APIServerAdapter.__dict__.keys()


@pytest.mark.asyncio
async def test_hidden_member_session_method_delegates(monkeypatch):
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    dispatch = object()
    implementation = AsyncMock(return_value="room_session")
    monkeypatch.setattr(
        room_dispatch,
        "_ensure_hosted_member_session",
        implementation,
    )

    assert await adapter._ensure_hosted_member_session(dispatch) == "room_session"
    implementation.assert_awaited_once_with(adapter, dispatch)


@pytest.mark.asyncio
async def test_room_dispatch_normalizer_method_delegates(monkeypatch):
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    request = object()
    body = {"input": "hello"}
    expected = ({"input": "normalized"}, None)
    implementation = AsyncMock(return_value=expected)
    monkeypatch.setattr(room_dispatch, "_normalize_room_dispatch", implementation)

    assert await adapter._normalize_room_dispatch(request, body) == expected
    implementation.assert_awaited_once_with(
        adapter,
        request,
        body,
        _api_server=sys.modules[api_server.__name__],
    )


@pytest.mark.asyncio
async def test_non_room_run_body_passes_through_unchanged():
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._room_grant_token = MagicMock(return_value="")
    request = object()
    body = {"input": "ordinary run"}

    normalized, error = await adapter._normalize_room_dispatch(request, body)

    assert normalized is body
    assert error is None
    adapter._room_grant_token.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_room_dispatch_rejects_extra_fields_before_grant_verification():
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._room_grant_token = MagicMock(return_value="room-grant")
    request = object()
    body = {
        "input": "room prompt",
        "hosted_room_dispatch": {},
        "unexpected": True,
    }

    normalized, error = await adapter._normalize_room_dispatch(request, body)

    assert normalized is body
    assert error.status == 400
    assert json.loads(error.text)["error"]["code"] == "invalid_room_dispatch"
