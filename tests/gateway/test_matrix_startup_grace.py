"""Matrix startup grace (#133265): offline-backlog drops are logged once per connect, and an
in-process reconnect delivers the outage window instead of dropping it like a cold boot."""

import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.helpers import carry_inbound_dedup, inbound_dedup_caches
from tests.gateway.test_matrix import _make_fake_mautrix

_LOGGER = "plugins.platforms.matrix.adapter"


def _adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter
    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="syt_test_token", extra={
        "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}))
    adapter._encryption = False
    adapter._handle_text_message = AsyncMock()
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    return adapter


def _event(event_id, ts):
    ev = MagicMock()
    ev.room_id, ev.sender, ev.event_id = "!room:example.org", "@alice:example.org", event_id
    ev.timestamp = ev.server_timestamp = int(ts * 1000)
    ev.content = {"msgtype": "m.text", "body": "hi"}
    return ev


async def _connect(adapter, timeline, *, is_reconnect=False):
    """Run the real connect(); the initial sync delivers *timeline* through _on_room_message."""
    mods = _make_fake_mautrix()
    client = MagicMock()
    client.mxid, client.device_id, client.crypto = "@bot:example.org", "DEV", None
    client.whoami = AsyncMock(return_value=MagicMock(user_id="@bot:example.org", device_id="DEV"))
    client.sync = AsyncMock(return_value={"rooms": {"join": {}}})
    client.handle_sync = lambda _data: [adapter._on_room_message(ev) for ev in timeline]
    client.sync_store.put_next_batch = AsyncMock()
    client.api.session.close = AsyncMock()
    mods["mautrix.client"].Client = MagicMock(return_value=client)
    with patch.dict("sys.modules", mods), patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
            patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)):
        assert await adapter.connect(is_reconnect=is_reconnect) is True


@pytest.mark.asyncio
async def test_cold_boot_grace_drop_logs_one_summary(caplog, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter()
    now = time.time()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await _connect(adapter, [_event("$a", now - 1200), _event("$b", now - 360)])
    adapter._handle_text_message.assert_not_called()  # default cold-boot behaviour preserved
    summaries = [r for r in caplog.records if r.name == _LOGGER and "initial sync skipped" in r.getMessage()]
    assert len(summaries) == 1 and summaries[0].levelname == "WARNING"
    assert "skipped 2 message(s)" in summaries[0].getMessage()
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_reconnect_delivers_outage_window_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old = _adapter()
    await _connect(old, [])
    t = time.time() - 120
    old._startup_ts = t - 10  # the bot had been up when the event at T arrived live
    await old._on_room_message(_event("$t", t))
    assert old._handle_text_message.await_count == 1
    await old.disconnect()
    # The runner's reconnect builds a fresh adapter, carries the dedup over, and connects with is_reconnect.
    new = _adapter()
    carry_inbound_dedup(inbound_dedup_caches(old), new)
    await _connect(new, [_event("$t", t), _event("$t60", t + 60)], is_reconnect=True)
    delivered = [c.args[2] for c in new._handle_text_message.await_args_list]
    assert delivered == ["$t60"]
    await new.disconnect()
