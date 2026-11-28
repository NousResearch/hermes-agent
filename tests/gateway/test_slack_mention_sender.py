"""Slack ``mention_sender_in_channels``: tag the person being answered on shared surfaces.

Contract:
  * Opt-in (default off): with the flag unset, replies are byte-identical.
  * A reply anchored on a live inbound event in a channel / group DM is prefixed with
    ``<@sender>``; the gateway stamps ``mention_user_id`` + ``mention_message_id`` only
    when the send carries a reply anchor.
  * 1:1 DMs, interim sends, and anchorless sends (cron, notices, synthetic wakes)
    are never tagged, so a stale origin user is never pinged.
  * Native streaming: the mention is in the first frame and the final send seals the
    same stream instead of posting a duplicate.
  * An answer that already tags the sender is not tagged twice.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import _thread_metadata_for_source
from plugins.platforms.slack.adapter import SlackAdapter


def _make_adapter(enabled=True):
    extra = {"mention_sender_in_channels": True} if enabled else {}
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra=extra))
    adapter._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "999.111"})
    client.chat_update = AsyncMock(return_value={"ts": "999.111"})
    client.chat_startStream = AsyncMock(return_value={"ok": True, "ts": "123.456"})
    client.chat_appendStream = AsyncMock(return_value={"ok": True})
    client.chat_stopStream = AsyncMock(return_value={"ok": True})
    adapter._get_client = MagicMock(return_value=client)
    adapter.stop_typing = AsyncMock()
    adapter._running = True
    return adapter, client


def _source(**kw):
    base = dict(platform="slack", chat_id="C123", thread_id="111.000", scope_id="T1",
                user_id="U123ABC", message_id="222.333", chat_type="group", profile=None)
    base.update(kw)
    return SimpleNamespace(**base)


ANCHORED = {"thread_id": "111.000", "mention_user_id": "U123ABC", "mention_message_id": "222.333"}


def _posted_text(client):
    return client.chat_postMessage.await_args.kwargs["text"]


class TestMetadataStamp:
    def test_reply_anchor_stamps_sender(self):
        md = _thread_metadata_for_source(_source(), "222.333") or {}
        assert md["mention_user_id"] == "U123ABC"
        assert md["mention_message_id"] == "222.333"

    def test_anchorless_send_carries_no_sender(self):
        # Notices / shutdown / cold-review sends build metadata from a bare source.
        md = _thread_metadata_for_source(_source()) or {}
        assert "mention_user_id" not in md and "mention_message_id" not in md

    def test_other_platforms_untouched(self):
        md = _thread_metadata_for_source(_source(platform="discord"), "222.333") or {}
        assert "mention_user_id" not in md


class TestSend:
    @pytest.mark.asyncio
    async def test_channel_reply_tags_sender(self):
        adapter, client = _make_adapter()
        result = await adapter.send("C123", "The relay is live.", metadata=dict(ANCHORED))
        assert result.success
        assert _posted_text(client).startswith("<@U123ABC> The relay is live.")

    @pytest.mark.asyncio
    async def test_off_by_default(self):
        adapter, client = _make_adapter(enabled=False)
        await adapter.send("C123", "The relay is live.", metadata=dict(ANCHORED))
        assert "<@U123ABC>" not in _posted_text(client)

    @pytest.mark.asyncio
    async def test_dm_not_tagged(self):
        adapter, client = _make_adapter()
        await adapter.send("D123", "The relay is live.", metadata=dict(ANCHORED))
        assert "<@U123ABC>" not in _posted_text(client)

    @pytest.mark.asyncio
    async def test_anchorless_send_not_tagged(self):
        adapter, client = _make_adapter()
        await adapter.send("C123", "Scheduled report.",
                           metadata={"thread_id": "111.000", "user_id": "U123ABC"})
        assert "<@U123ABC>" not in _posted_text(client)

    @pytest.mark.asyncio
    async def test_interim_send_not_tagged(self):
        adapter, client = _make_adapter()
        await adapter.send("C123", "Working on it…", metadata={**ANCHORED, "_interim_send": True})
        assert "<@U123ABC>" not in _posted_text(client)

    @pytest.mark.asyncio
    async def test_existing_mention_not_duplicated(self):
        adapter, client = _make_adapter()
        await adapter.send("C123", "<@U123ABC> Already tagged.", metadata=dict(ANCHORED))
        assert _posted_text(client).count("<@U123ABC") == 1


class TestNativeStreaming:
    @pytest.mark.asyncio
    async def test_stream_carries_mention_and_final_seals_it(self):
        adapter, client = _make_adapter()
        await adapter.send_draft("C123", 7, "The relay", metadata=dict(ANCHORED))
        result = await adapter.send("C123", "The relay is live.", metadata=dict(ANCHORED))
        assert result.success
        assert client.chat_startStream.await_args.kwargs["markdown_text"] == "<@U123ABC> The relay"
        client.chat_postMessage.assert_not_awaited()
