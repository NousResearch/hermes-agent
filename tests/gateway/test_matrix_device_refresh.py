"""Tests for the Matrix peer device-list refresh before encrypted sends.

mautrix only re-fetches a peer's device list when the homeserver reports the
peer in ``device_lists.changed`` during /sync (OlmMachine.handle_device_lists);
there is no catch-up or invalidation path. If that signal is missed (gateway
downtime during a peer's device rotation, crypto-store recreation, or a
Conduit-family homeserver that fails to emit it), outbound megolm sessions are
shared only to the peer's dead device forever and the peer can never decrypt.

These tests cover ``MatrixAdapter._refresh_encrypted_room_devices()`` and the
``_send_room_event()`` chokepoint that routes every encrypted send path
(text, media, edits, reactions, emotes/notices) through the refresh.
"""
import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig


ROOM = "!room:example.org"
BOT = "@bot:example.org"
PEER_TRACKED = "@alice:example.org"
PEER_UNTRACKED = "@fresh-login:example.org"


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------

def _make_adapter(extra_overrides: dict | None = None):
    """Create a MatrixAdapter wired to a fake encrypted-room client."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    extra = {
        "homeserver": "https://matrix.example.org",
        "user_id": BOT,
    }
    if extra_overrides:
        extra.update(extra_overrides)
    config = PlatformConfig(enabled=True, token="syt_test_token", extra=extra)
    adapter = MatrixAdapter(config)

    client = MagicMock()
    client.send_message_event = AsyncMock(return_value="$sent:example.org")
    client.upload_media = AsyncMock(return_value="mxc://example.org/media123")
    client.state_store.is_encrypted = AsyncMock(return_value=True)
    # One member mautrix already tracks and one it has never queried —
    # include_untracked=True must requery BOTH (plus never our own user).
    client.state_store.get_members = AsyncMock(
        return_value=[BOT, PEER_TRACKED, PEER_UNTRACKED]
    )
    crypto = MagicMock()
    crypto._fetch_keys = AsyncMock(return_value={})
    crypto._fetch_keys_lock = asyncio.Lock()
    crypto.share_keys = AsyncMock()
    client.crypto = crypto

    adapter._client = client
    adapter._encryption = True
    return adapter


def _mark_interval_elapsed(adapter, room_id: str = ROOM):
    """Pretend the last refresh for *room_id* happened over an interval ago.

    ``time.monotonic()`` can be smaller than the default 300s interval on a
    freshly booted machine, which would make the first-ever send look
    throttled — pin the last-refresh timestamp instead of relying on uptime.
    """
    adapter._device_refresh_ts[room_id] = (
        time.monotonic() - adapter._device_refresh_interval - 1
    )


# ---------------------------------------------------------------------------
# Refresh before encrypted sends
# ---------------------------------------------------------------------------

class TestMatrixDeviceRefresh:
    @pytest.mark.asyncio
    async def test_refresh_called_before_encrypted_send_when_interval_elapsed(self):
        adapter = _make_adapter()
        _mark_interval_elapsed(adapter)

        result = await adapter.send(ROOM, "hello")

        assert result.success
        adapter._client.crypto._fetch_keys.assert_awaited_once()
        adapter._client.send_message_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refresh_throttled_within_interval(self):
        adapter = _make_adapter()
        _mark_interval_elapsed(adapter)

        await adapter.send(ROOM, "first")
        await adapter.send(ROOM, "second")

        # Second send is within the interval — no second /keys/query.
        adapter._client.crypto._fetch_keys.assert_awaited_once()
        assert adapter._client.send_message_event.await_count == 2

    @pytest.mark.asyncio
    async def test_refresh_disabled_when_config_zero(self):
        adapter = _make_adapter({"device_refresh_seconds": 0})
        _mark_interval_elapsed(adapter)

        result = await adapter.send(ROOM, "hello")

        assert result.success
        adapter._client.crypto._fetch_keys.assert_not_awaited()
        adapter._client.send_message_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_keys_failure_does_not_break_send(self):
        adapter = _make_adapter()
        _mark_interval_elapsed(adapter)
        adapter._client.crypto._fetch_keys = AsyncMock(
            side_effect=RuntimeError("keys/query exploded")
        )

        result = await adapter.send(ROOM, "hello")

        assert result.success
        adapter._client.send_message_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_untracked_and_tracked_members_both_requeried(self):
        adapter = _make_adapter()
        _mark_interval_elapsed(adapter)

        await adapter.send(ROOM, "hello")

        args, kwargs = adapter._client.crypto._fetch_keys.await_args
        queried = set(args[0])
        # include_untracked=True is load-bearing: without it mautrix skips
        # users it never tracked (fresh logins after crypto-store recreation).
        assert kwargs.get("include_untracked") is True
        assert queried == {PEER_TRACKED, PEER_UNTRACKED}
        assert BOT not in queried

    @pytest.mark.asyncio
    async def test_unencrypted_room_skips_fetch(self):
        adapter = _make_adapter()
        _mark_interval_elapsed(adapter)
        adapter._client.state_store.is_encrypted = AsyncMock(return_value=False)

        result = await adapter.send(ROOM, "hello")

        assert result.success
        adapter._client.crypto._fetch_keys.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fetch_keys_lock_held_around_fetch(self):
        adapter = _make_adapter()
        _mark_interval_elapsed(adapter)
        crypto = adapter._client.crypto
        lock = crypto._fetch_keys_lock

        async def _assert_locked(*args, **kwargs):
            assert lock.locked(), "_fetch_keys must run under _fetch_keys_lock"
            return {}

        crypto._fetch_keys = AsyncMock(side_effect=_assert_locked)

        await adapter.send(ROOM, "hello")

        crypto._fetch_keys.assert_awaited_once()
        assert not lock.locked()


# ---------------------------------------------------------------------------
# Chokepoint coverage — every encrypted send path routes through the refresh
# ---------------------------------------------------------------------------

class TestMatrixSendChokepoint:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.refresh = AsyncMock()
        self.adapter._refresh_encrypted_room_devices = self.refresh

    @pytest.mark.asyncio
    async def test_send_text_routes_through_refresh(self):
        await self.adapter.send(ROOM, "hello")
        self.refresh.assert_awaited_with(ROOM)

    @pytest.mark.asyncio
    async def test_edit_message_routes_through_refresh(self):
        await self.adapter.edit_message(ROOM, "$orig:example.org", "edited")
        self.refresh.assert_awaited_with(ROOM)

    @pytest.mark.asyncio
    async def test_media_upload_routes_through_refresh(self):
        # Plaintext room here so the test doesn't depend on a real
        # mautrix.crypto.attachments install; the routing through
        # _send_room_event is identical either way.
        self.adapter._client.state_store.is_encrypted = AsyncMock(return_value=False)
        result = await self.adapter._upload_and_send(
            ROOM, b"data", "photo.png", "image/png", "m.image"
        )
        assert result.success
        self.refresh.assert_awaited_with(ROOM)

    @pytest.mark.asyncio
    async def test_reaction_routes_through_refresh(self):
        await self.adapter._send_reaction(ROOM, "$msg:example.org", "👀")
        self.refresh.assert_awaited_with(ROOM)

    @pytest.mark.asyncio
    async def test_simple_message_routes_through_refresh(self):
        await self.adapter._send_simple_message(ROOM, "waves", "m.emote")
        self.refresh.assert_awaited_with(ROOM)

    @pytest.mark.asyncio
    async def test_e2ee_retry_path_routes_through_refresh(self):
        # First attempt fails, retry after share_keys() must also pass
        # through the chokepoint.
        self.adapter._client.send_message_event = AsyncMock(
            side_effect=[Exception("no session"), "$retried:example.org"]
        )
        result = await self.adapter.send(ROOM, "hello")
        assert result.success
        assert self.refresh.await_count == 2


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

class TestMatrixDeviceRefreshConfig:
    def test_default_interval_is_300(self):
        adapter = _make_adapter()
        assert adapter._device_refresh_interval == 300.0

    def test_extra_overrides_interval(self):
        adapter = _make_adapter({"device_refresh_seconds": 60})
        assert adapter._device_refresh_interval == 60.0

    def test_env_bridge_used_when_extra_absent(self, monkeypatch):
        monkeypatch.setenv("MATRIX_DEVICE_REFRESH_SECONDS", "120")
        adapter = _make_adapter()
        assert adapter._device_refresh_interval == 120.0

    def test_invalid_value_falls_back_to_default(self):
        adapter = _make_adapter({"device_refresh_seconds": "not-a-number"})
        assert adapter._device_refresh_interval == 300.0

    def test_yaml_config_bridges_to_env(self, monkeypatch):
        import os

        from plugins.platforms.matrix.adapter import _apply_yaml_config

        # setenv-to-empty records the prior state for teardown restore while
        # still counting as "unset" for the env-precedence check in the bridge.
        monkeypatch.setenv("MATRIX_DEVICE_REFRESH_SECONDS", "")
        _apply_yaml_config({}, {"device_refresh_seconds": 45})

        assert os.environ.get("MATRIX_DEVICE_REFRESH_SECONDS") == "45"
