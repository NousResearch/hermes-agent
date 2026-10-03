"""A Slack IM is the sender and the Bot alone; a group DM (MPIM) or a channel is not.

Owner-only ``/group`` commands (continuing a Group Chat on another computer) need the
difference: they work only in a chat that is provably private
(``gateway.group_chat_identity.is_private_source``).
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.group_chat_identity import is_private_source
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def adapter():
    a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake-token"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache")


@pytest.mark.asyncio
@pytest.mark.parametrize(("channel", "channel_type", "private"), [
    ("D_IM", "im", True), ("G_MPIM", "mpim", False), ("C_CHAN", "channel", False)])
async def test_only_an_im_message_is_private(adapter, channel, channel_type, private):
    event = {"channel": channel, "channel_type": channel_type, "user": "U_USER",
             "text": "<@U_BOT> /group", "ts": "1700000000.000001"}
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="Alice")):
        await adapter._handle_slack_message(event)
    source = adapter.handle_message.call_args[0][0].source
    assert source.chat_type == ("dm" if channel_type != "channel" else "group")
    assert is_private_source(source) is private


@pytest.mark.asyncio
@pytest.mark.parametrize(("channel", "private"), [("D_IM", True), ("G_MPIM", False), ("C_CHAN", False)])
async def test_only_a_slash_command_from_an_im_is_private(adapter, channel, private):
    await adapter._handle_slash_command({"command": "/group", "text": "1", "user_id": "U_USER",
                                         "channel_id": channel})
    source = adapter.handle_message.call_args[0][0].source
    assert is_private_source(source) is private
