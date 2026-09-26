"""Tests for SlackAdapter.send_or_update_status (issue #30045, Slack).

The status-update path must:
  1. Send a fresh message on the first call for a (channel, thread, key).
  2. Edit that same message on subsequent calls with the same key.
  3. Fall back to sending fresh when the cached message edit fails.
  4. Keep distinct keys and distinct threads independent.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    a = SlackAdapter(config)
    a._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(
        side_effect=lambda **kw: {"ok": True, "ts": f"ts_{client.chat_postMessage.call_count}"}
    )
    client.chat_update = AsyncMock(return_value={"ok": True})
    a._get_client = MagicMock(return_value=client)
    a._bot_user_id = "U_BOT"
    a._running = True
    a.stop_typing = AsyncMock()
    return a


METADATA = {"thread_id": "1784585355.415219"}




@pytest.mark.asyncio
async def test_distinct_keys_do_not_crosstalk(adapter):
    await adapter.send_or_update_status(
        "C_CHAN", "context_pressure", "compressing", metadata=METADATA
    )
    await adapter.send_or_update_status(
        "C_CHAN", "model_fallback", "falling back", metadata=METADATA
    )
    client = adapter._get_client.return_value
    assert client.chat_postMessage.call_count == 2
    assert client.chat_update.call_count == 0




class _NotFound(Exception):
    def __init__(self):
        super().__init__("message_not_found")
        self.response = {"ok": False, "error": "message_not_found"}


@pytest.mark.asyncio
async def test_deleted_status_bubble_is_forgotten_and_reposted_quietly(adapter, caplog):
    """A cleaned-up status bubble must not be edited on the next turn; if Slack says it is gone,
    the adapter posts fresh without an ERROR log."""
    await adapter.send_or_update_status("C_CHAN", "lifecycle", "working", metadata=METADATA)
    client = adapter._get_client.return_value
    client.chat_delete = AsyncMock(return_value={"ok": True})
    assert await adapter.delete_message("C_CHAN", "ts_1")
    await adapter.send_or_update_status("C_CHAN", "lifecycle", "working again", metadata=METADATA)
    assert client.chat_update.call_count == 0  # never edits the deleted bubble
    assert client.chat_postMessage.call_count == 2

    client.chat_update = AsyncMock(side_effect=_NotFound())
    caplog.set_level("INFO")
    await adapter.send_or_update_status("C_CHAN", "lifecycle", "third", metadata=METADATA)
    assert client.chat_postMessage.call_count == 3
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
