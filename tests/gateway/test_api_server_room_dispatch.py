"""Compatibility seams for extracted RoomLink dispatch handling."""

import json
import sys
from types import SimpleNamespace
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


@pytest.mark.asyncio
async def test_title_conflict_keeps_guidance_without_merging_sessions(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._ensure_session_db_async = AsyncMock(return_value=db)
    dispatch = SimpleNamespace(room_id="private-room", home_install_id="home",
                               member_id="member", target_profile="default")
    db._execute_write(lambda conn: conn.execute(
        "INSERT INTO sessions(id, source, title, started_at) VALUES(?, ?, ?, ?)",
        ("existing", "bot_room", "Group: private-room", 1)))
    try:
        with pytest.raises(RuntimeError) as caught:
            await adapter._ensure_hosted_member_session(dispatch)
        response = room_dispatch._room_dispatch_error(
            caught.value, _openai_error=api_server._openai_error)
        error = json.loads(response.text)["error"]
        assert response.status == 403
        assert error["code"] == "room_session_title_conflict"
        assert "Rename or migrate" in error["message"]
        assert "private-room" not in response.text
        assert db.get_session("existing")["title"] == "Group: private-room"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_dispatch_failure_logs_safe_diagnostics_not_exception_data(monkeypatch, caplog):
    from gateway.hosted_room_peer import HostedMemberDispatch

    secret = "/private/token=do-not-log\nforged log entry"

    def broken_mapping(_value):
        raise OSError(secret)

    monkeypatch.setattr(HostedMemberDispatch, "from_mapping", broken_mapping)
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._room_grant_token = MagicMock(return_value=secret)
    body = {"hosted_room_dispatch": {}}
    _, response = await adapter._normalize_room_dispatch(object(), body)
    assert response.status == 403
    assert json.loads(response.text)["error"]["code"] == "invalid_room_dispatch"
    assert "OSError" in caplog.text
    assert "broken_mapping" in caplog.text
    assert secret not in caplog.text + response.text
    assert all(record.exc_info is None for record in caplog.records)
