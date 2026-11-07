"""Matrix sync-cursor recovery: a rejected ``since`` token is cleared and retried within a
consecutive budget; token failures and forbidden snapshots without a cursor stay fatal."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter
    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    return MatrixAdapter(config)


def _sync_error(message, **attrs):
    """Shape of mautrix's MatrixRequestError: message text + structured attrs."""
    exc = Exception(message)
    for k, v in attrs.items():
        setattr(exc, k, v)
    return exc


class TestMatrixSyncCursorRecovery:

    async def _run_sync_loop_with_cursor(self, *, first_exc, later=None, max_errors=1, cursor="s-stale"):
        """Drive _sync_loop starting from a persisted since token."""
        adapter = _make_adapter()
        adapter._closing = False
        adapter._joined_rooms = {"!old:example.org"}
        adapter._dm_rooms = {"!old:example.org": True}
        calls = {"n": 0}
        since_seen = []

        async def _sync_side_effect(**kwargs):
            calls["n"] += 1
            since_seen.append(kwargs.get("since"))
            if calls["n"] <= max_errors:
                raise first_exc
            adapter._closing = True
            if later is not None:
                return later
            return {
                "next_batch": "s-fresh",
                "rooms": {"join": {"!keep:example.org": {}}},
            }

        fake_client = MagicMock()
        fake_client.sync = AsyncMock(side_effect=_sync_side_effect)
        fake_client.sync_store = MagicMock()
        fake_client.sync_store.get_next_batch = AsyncMock(return_value=cursor)
        fake_client.sync_store.put_next_batch = AsyncMock()
        fake_client.handle_sync = MagicMock(return_value=[])
        adapter._client = fake_client
        with patch("asyncio.sleep", new=AsyncMock()):
            await adapter._sync_loop()
        return adapter, fake_client, since_seen

    @pytest.mark.asyncio
    async def test_sync_loop_resets_forbidden_incremental_sync_cursor(self):
        """403/M_FORBIDDEN with a live since token clears the cursor and retries."""
        adapter, client, since_seen = await self._run_sync_loop_with_cursor(
            first_exc=_sync_error("Forbidden", errcode="M_FORBIDDEN", http_status=403),
        )
        assert since_seen[0] == "s-stale"
        assert None in since_seen[1:]
        client.sync_store.put_next_batch.assert_any_await(None)
        assert "!old:example.org" not in adapter._joined_rooms
        assert "!keep:example.org" in adapter._joined_rooms

    @pytest.mark.asyncio
    async def test_sync_loop_resets_unknown_pos_cursor(self):
        _adapter, client, since_seen = await self._run_sync_loop_with_cursor(
            first_exc=_sync_error("unknown since", errcode="M_UNKNOWN_POS", http_status=400),
        )
        assert since_seen[0] == "s-stale"
        assert None in since_seen[1:]
        client.sync_store.put_next_batch.assert_any_await(None)

    @pytest.mark.asyncio
    async def test_sync_loop_stops_on_unknown_token_even_with_cursor(self):
        _adapter, client, since_seen = await self._run_sync_loop_with_cursor(
            first_exc=_sync_error("Invalid access token", errcode="M_UNKNOWN_TOKEN", http_status=401),
            max_errors=8,
        )
        assert client.sync.await_count == 1
        assert since_seen == ["s-stale"]
        cleared = [
            c for c in client.sync_store.put_next_batch.await_args_list
            if c.args and c.args[0] is None
        ]
        assert cleared == []

    @pytest.mark.asyncio
    async def test_sync_loop_stops_on_forbidden_after_cursor_cleared(self):
        """Once the since token is gone, 403 is permanent auth — not another reset."""
        _adapter, client, since_seen = await self._run_sync_loop_with_cursor(
            first_exc=_sync_error("Forbidden", errcode="M_FORBIDDEN", http_status=403),
            max_errors=8,
        )
        assert client.sync.await_count == 2
        assert since_seen == ["s-stale", None]
        client.sync_store.put_next_batch.assert_any_await(None)

    @pytest.mark.asyncio
    async def test_sync_loop_cursor_reset_budget_survives_recovery_snapshots(self):
        """A snapshot after a rejected cursor does not reset the consecutive budget."""
        adapter = _make_adapter()
        adapter._closing = False
        adapter._joined_rooms = set()
        adapter._dm_rooms = {}
        since_seen = []

        async def _sync_side_effect(**kwargs):
            since_seen.append(kwargs.get("since"))
            n = len(since_seen)
            if kwargs.get("since"):
                raise _sync_error("Forbidden", errcode="M_FORBIDDEN", http_status=403)
            if n >= 12:
                adapter._closing = True
            return {"next_batch": f"s{n}", "rooms": {"join": {}}}

        fake_client = MagicMock()
        fake_client.sync = AsyncMock(side_effect=_sync_side_effect)
        fake_client.sync_store = MagicMock()
        fake_client.sync_store.get_next_batch = AsyncMock(return_value="s-stale")
        fake_client.sync_store.put_next_batch = AsyncMock()
        fake_client.handle_sync = MagicMock(return_value=[])
        fake_client.get_account_data = AsyncMock(side_effect=Exception("no m.direct"))
        adapter._client = fake_client
        with patch("asyncio.sleep", new=AsyncMock()):
            await adapter._sync_loop()
        # Four incremental rejects, each followed by a snapshot, stop at the cap.
        assert since_seen[0] == "s-stale"
        assert sum(1 for since in since_seen if since) == 4
        assert since_seen.count(None) == 3

    @pytest.mark.asyncio
    async def test_sync_loop_cursor_reset_budget_clears_after_incremental_sync(self):
        """An incremental sync with since set clears the budget. A snapshot does not."""
        adapter = _make_adapter()
        adapter._closing = False
        adapter._joined_rooms = set()
        adapter._dm_rooms = {}
        since_seen = []

        async def _sync_side_effect(**kwargs):
            since_seen.append(kwargs.get("since"))
            n = len(since_seen)
            if n == 1:
                raise _sync_error("Forbidden", errcode="M_FORBIDDEN", http_status=403)
            if n == 3:
                return {"next_batch": "s-ok", "rooms": {"join": {}}}
            if n >= 4 and kwargs.get("since"):
                raise _sync_error("Forbidden", errcode="M_FORBIDDEN", http_status=403)
            if n >= 12:
                adapter._closing = True
            return {"next_batch": f"s{n}", "rooms": {"join": {}}}

        fake_client = MagicMock()
        fake_client.sync = AsyncMock(side_effect=_sync_side_effect)
        fake_client.sync_store = MagicMock()
        fake_client.sync_store.get_next_batch = AsyncMock(return_value="s-stale")
        fake_client.sync_store.put_next_batch = AsyncMock()
        fake_client.handle_sync = MagicMock(return_value=[])
        fake_client.get_account_data = AsyncMock(side_effect=Exception("no m.direct"))
        adapter._client = fake_client
        with patch("asyncio.sleep", new=AsyncMock()):
            await adapter._sync_loop()
        assert fake_client.sync.await_count == 10

    @pytest.mark.asyncio
    async def test_absorb_sync_passes_join_map_to_encrypted_drop_warning(self):
        """Join payloads stay maps. A set would raise in the encrypted-drop warning."""
        adapter = _make_adapter()
        seen = {}

        def _warn(rooms_join, client):
            seen["value"] = rooms_join

        adapter._warn_encrypted_drops = _warn
        adapter._dispatch_sync = AsyncMock()
        adapter._refresh_dm_cache = AsyncMock()
        adapter._schedule_pending_invite_joins = MagicMock()
        client = MagicMock()
        client.sync_store.put_next_batch = AsyncMock()
        room = "!room:example.org"
        await adapter._absorb_sync(client, {
            "next_batch": "s1",
            "rooms": {"join": {room: {"timeline": {"events": []}}}},
        })
        assert seen["value"] == {room: {"timeline": {"events": []}}}

    @pytest.mark.asyncio
    async def test_html_403_digits_do_not_reset_cursor(self):
        """Unstructured 502 HTML that happens to contain '403' is still a retry, not a reset."""
        _adapter, client, since_seen = await self._run_sync_loop_with_cursor(
            first_exc=_sync_error(
                '502: <!DOCTYPE html><svg><path d="M17.4517 1403.2C12.7214 1403.2"/></svg>',
                http_status=502,
            ),
        )
        assert client.sync.await_count == 2
        assert since_seen == ["s-stale", "s-stale"]
        cleared = [
            c for c in client.sync_store.put_next_batch.await_args_list
            if c.args and c.args[0] is None
        ]
        assert cleared == []


    @pytest.mark.asyncio
    async def test_sync_loop_stops_on_forbidden_without_cursor(self):
        """Structured 403 with no persisted cursor is still permanent auth."""
        _adapter, client, since_seen = await self._run_sync_loop_with_cursor(
            first_exc=_sync_error("Forbidden", errcode="M_FORBIDDEN", http_status=403),
            cursor=None,
        )
        assert since_seen == [None]
        client.sync_store.put_next_batch.assert_not_awaited()
