"""Matrix startup grace (#133265): offline-backlog drops are logged once per connect (plus once for
E2EE backlog that is decrypted after the initial sync)."""

import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway.test_matrix import _make_adapter, _make_fake_mautrix

_LOGGER = "plugins.platforms.matrix.adapter"


def _adapter():
    adapter = _make_adapter()
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
            patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)), \
            patch.object(adapter, "_connect_setup_e2ee", AsyncMock(return_value=True)):
        assert await adapter.connect(is_reconnect=is_reconnect) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("encryption,quiet_room", [(True, False), (True, True), (False, False)],
                         ids=["e2ee-live-message", "e2ee-quiet-room", "plaintext"])
async def test_cold_boot_grace_drop_logs_one_summary(caplog, tmp_path, monkeypatch, encryption, quiet_room):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter()
    adapter._encryption = encryption
    now = time.time()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await _connect(adapter, [_event("$a", now - 1200), _event("$b", now - 360)])
        adapter._handle_text_message.assert_not_called()  # default cold-boot behaviour preserved
        # E2EE: mautrix decrypts backlog in background tasks, after the initial-sync summary.
        await adapter._on_room_message(_event("$enc", now - 600))
        if quiet_room:  # nobody speaks: shutdown still reports the late-decrypted drop
            await adapter.disconnect()
        else:
            for fresh in ("$live1", "$live2"):  # the first message past the gate closes the backlog window
                await adapter._on_room_message(_event(fresh, time.time()))
        if not encryption:  # a drop after the window closed must not leak into the next connect's summary
            await _connect(adapter, [], is_reconnect=True)
    assert adapter._handle_text_message.await_count == (0 if quiet_room else 2)
    summaries = [r.getMessage() for r in caplog.records if r.name == _LOGGER and "skipped" in r.getMessage()]
    phases = ["Matrix: initial sync", "Matrix: decrypted backlog"] if encryption else ["Matrix: initial sync"]
    assert [m.split(" skipped ")[0] for m in summaries] == phases
    assert "skipped 2 message(s)" in summaries[0] and (not encryption or "skipped 1 message(s)" in summaries[1])
    await adapter.disconnect()
