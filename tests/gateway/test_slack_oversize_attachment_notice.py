"""A Slack document the 20 MB size gate skips is still named in the turn, beside the user's text,
so the agent can tell the user why "see attached" arrived with nothing attached."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_slack import SlackAdapter


@pytest.fixture()
def adapter():
    a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(return_value={"user": {
        "is_bot": False, "profile": {"display_name": "Test User"}, "real_name": "Test User"}})
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.mark.asyncio
async def test_document_skipped_by_size_gate_is_named_to_the_agent(adapter):
    event = {
        "text": "summarize the attached report", "user": "U_USER", "channel": "D123", "channel_type": "im",
        "ts": "1234567890.000001", "blocks": [], "attachments": [],
        "files": [{"mimetype": "application/pdf", "name": "q3-report.pdf",
                   "url_private_download": "https://files.slack.com/q3-report.pdf", "size": 25 * 1024 * 1024}],
    }
    with patch.object(adapter, "_download_slack_file_bytes", new_callable=AsyncMock) as dl:
        await adapter._handle_slack_message(event)

    dl.assert_not_called()
    msg_event = adapter.handle_message.call_args[0][0]
    assert msg_event.media_urls == []
    assert "q3-report.pdf" in msg_event.text
    assert "summarize the attached report" in msg_event.text
