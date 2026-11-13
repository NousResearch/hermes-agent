"""Focused tests for Matrix cron delivery (#107907).

Live lane: mautrix rejects non-create events when the room version is unknown.
Standalone lane: aiohttp's default ClientTimeout uses asyncio.timeout() and
raises ``Timeout context manager should be used inside a task`` when cron
invokes send via asyncio.run() / a thread with no running task.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    return MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
            },
        )
    )


def _configured_pconfig(*, token="syt_test_token"):
    return SimpleNamespace(
        token=token,
        extra={"homeserver": "https://matrix.example.org"},
    )


class _FakePutResponse:
    def __init__(self, event_id="$e"):
        self.status = 200
        self._event_id = event_id

    async def json(self):
        return {"event_id": self._event_id}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class _FakeSession:
    def __init__(self):
        self.put = MagicMock(return_value=_FakePutResponse())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class TestStandaloneClientSessionTimeout:
    @pytest.mark.asyncio
    async def test_client_session_uses_disabled_timeout(self, monkeypatch):
        captured = {}
        fake_session = _FakeSession()

        def _session_factory(*args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return fake_session

        monkeypatch.setattr(aiohttp, "ClientSession", _session_factory)
        monkeypatch.setattr(
            "plugins.platforms.matrix.adapter.get_secret",
            lambda *a, **k: "",
        )

        from plugins.platforms.matrix.adapter import _standalone_send

        result = await _standalone_send(_configured_pconfig(), "!room:hs", "hello")

        assert result["success"] is True
        timeout = captured.get("timeout")
        assert timeout is not None
        assert timeout.total is None
        assert timeout.connect is None
        assert timeout.sock_connect is None
        assert timeout.sock_read is None

    @pytest.mark.asyncio
    async def test_missing_token_does_not_create_session(self, monkeypatch):
        created = {"count": 0}

        def _session_factory(*args, **kwargs):
            created["count"] += 1
            return _FakeSession()

        monkeypatch.setattr(aiohttp, "ClientSession", _session_factory)
        monkeypatch.setattr(
            "plugins.platforms.matrix.adapter.get_secret",
            lambda *a, **k: "",
        )

        from plugins.platforms.matrix.adapter import _standalone_send

        result = await _standalone_send(
            _configured_pconfig(token=""), "!room:hs", "hello"
        )

        assert created["count"] == 0
        assert result.get("success") is not True
        assert "error" in result
        assert "configured" in result["error"].lower()


class TestLiveUnknownRoomVersionHydrate:
    @pytest.mark.asyncio
    async def test_unknown_version_fetches_create_and_retries(self):
        from plugins.platforms.matrix.adapter import EventType, RoomID

        adapter = _make_adapter()
        adapter._encryption = False
        fake = MagicMock()
        fake.send_message_event = AsyncMock(
            side_effect=[
                RuntimeError(
                    "non-create event for room of unknown version in !r:hs"
                ),
                "$ok",
            ]
        )
        fake.get_state_event = AsyncMock(
            return_value=SimpleNamespace(content={"room_version": "10"})
        )
        fake.crypto = None
        adapter._client = fake

        result = await adapter.send("!r:hs", "hello")

        assert result.success is True
        assert result.message_id == "$ok"
        assert fake.send_message_event.await_count == 2
        create_type = getattr(EventType, "ROOM_CREATE", "m.room.create")
        fake.get_state_event.assert_awaited()
        called_room, called_type = fake.get_state_event.await_args.args[:2]
        assert str(called_room) == str(RoomID("!r:hs"))
        assert called_type == create_type

    @pytest.mark.asyncio
    async def test_other_send_error_does_not_hydrate(self):
        adapter = _make_adapter()
        adapter._encryption = False
        fake = MagicMock()
        fake.send_message_event = AsyncMock(
            side_effect=RuntimeError("401 unauthorized")
        )
        fake.get_state_event = AsyncMock(
            return_value=SimpleNamespace(content={"room_version": "10"})
        )
        fake.crypto = None
        adapter._client = fake

        result = await adapter.send("!r:hs", "hello")

        assert result.success is False
        fake.get_state_event.assert_not_awaited()
        assert fake.send_message_event.await_count == 1
