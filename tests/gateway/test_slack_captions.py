"""Slack file captions get the same formatting as messages.

A caption rides an upload as ``files_upload_v2``'s ``initial_comment``. It used to be
sent raw, so markdown showed literally and ``<!channel>``/``<!here>`` were not escaped,
while every message goes through ``format_message``.
"""

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
        ("slack_bolt.adapter.socket_mode.async_handler",
         slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod

_slack_mod.SLACK_AVAILABLE = True

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture()
def adapter():
    a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.files_upload_v2 = AsyncMock(return_value={"ok": True})
    a._running = True
    return a


@pytest.fixture()
def report(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4 fake content")
    return str(path)


@pytest.mark.asyncio
async def test_document_caption_is_formatted_like_a_message(adapter, report):
    caption = "# Report\n\n**Total:** $10 <!channel>"

    result = await adapter.send_document(chat_id="C123", file_path=report, caption=caption)

    assert result.success
    sent = adapter._app.client.files_upload_v2.call_args[1]["initial_comment"]
    assert sent == adapter.format_message(caption)
    assert "**" not in sent
    assert "<!channel>" not in sent


@pytest.mark.asyncio
async def test_video_caption_is_formatted_like_a_message(adapter, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video")

    result = await adapter.send_video(chat_id="C123", video_path=str(clip), caption="**Look** <!here>")

    assert result.success
    sent = adapter._app.client.files_upload_v2.call_args[1]["initial_comment"]
    assert sent == adapter.format_message("**Look** <!here>")
    assert "<!here>" not in sent


@pytest.mark.asyncio
async def test_upload_without_caption_sends_empty_comment(adapter, report):
    result = await adapter.send_document(chat_id="C123", file_path=report)

    assert result.success
    assert adapter._app.client.files_upload_v2.call_args[1]["initial_comment"] == ""
