"""Inbound Slack routing of bot mentions and legacy attachments.

Covers which attachments reach the agent (automatic unfurls are filtered, forwarded
shares stay visible but cannot summon the bot) and how bot-mention routing resolves
the per-workspace bot user id.
"""

import importlib
import sys
from importlib.machinery import PathFinder
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _load_installed_package(name):
    if PathFinder.find_spec(name) is None:
        return None
    prefix = f"{name}."
    displaced = {
        m: sys.modules.pop(m)
        for m in tuple(sys.modules)
        if (m == name or m.startswith(prefix)) and not isinstance(sys.modules[m], ModuleType)
    }
    try:
        return importlib.import_module(name)
    except ImportError:
        sys.modules.update(displaced)
        return None


_load_installed_package("slack_bolt")
_load_installed_package("slack_sdk")

SlackAdapter = importlib.import_module("plugins.platforms.slack.adapter").SlackAdapter


@pytest.fixture()
def adapter():
    a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(
        return_value={
            "user": {
                "is_bot": False,
                "profile": {"display_name": "Test User"},
                "real_name": "Test User",
            }
        }
    )
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


class TestSlackAttachmentUnfurls:
    @staticmethod
    def _event(text, attachments, blocks=None):
        event = {
            "text": text,
            "user": "U_USER",
            "channel": "D123",
            "channel_type": "im",
            "ts": "1234567890.000001",
            "attachments": attachments,
        }
        if blocks is not None:
            event["blocks"] = blocks
        return event

    @pytest.mark.asyncio
    @pytest.mark.parametrize("flag", ["is_app_unfurl", "is_msg_unfurl"])
    async def test_explicit_unfurl_is_not_added_to_current_event(self, adapter, flag):
        event = self._event(
            "review https://example.com/report",
            [
                {
                    flag: True,
                    "title": "Preview title",
                    "text": "Preview body",
                    "footer": "Preview footer",
                }
            ],
        )

        await adapter._handle_slack_message(event)

        delivered = adapter.handle_message.await_args.args[0]
        assert delivered.text == "review https://example.com/report"

    @pytest.mark.asyncio
    async def test_matching_authored_url_preview_is_not_added_to_current_event(
        self, adapter
    ):
        event = self._event(
            "review <https://example.com/report|the report>",
            [
                {
                    "original_url": "https://EXAMPLE.com/report/",
                    "title": "Preview title",
                    "text": "Preview body",
                    "footer": "Preview footer",
                }
            ],
        )

        await adapter._handle_slack_message(event)

        delivered = adapter.handle_message.await_args.args[0]
        assert delivered.text == "review <https://example.com/report|the report>"

    @pytest.mark.asyncio
    async def test_mixed_current_event_keeps_only_genuine_attachment(self, adapter):
        event = self._event(
            "review https://example.com/report",
            [
                {
                    "from_url": "https://example.com/report",
                    "title": "Preview title",
                    "text": "Preview body",
                },
                {
                    "title": "Production alert",
                    "text": "Disk usage is 91%",
                    "footer": "Alertmanager",
                },
            ],
        )

        await adapter._handle_slack_message(event)

        delivered = adapter.handle_message.await_args.args[0]
        assert "Preview title" not in delivered.text
        assert "Preview body" not in delivered.text
        assert "📎 Production alert" in delivered.text
        assert "Disk usage is 91%" in delivered.text
        assert "Alertmanager" in delivered.text

    def test_render_message_text_filters_preview_and_keeps_genuine_attachment(
        self, adapter
    ):
        msg = self._event(
            "review the report",
            [
                {
                    "title_link": "https://example.com/report",
                    "title": "Preview title",
                    "text": "Preview body",
                },
                {"title": "Production alert", "text": "Disk usage is 91%"},
            ],
            blocks=[
                {
                    "type": "section",
                    "accessory": {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Open"},
                        "url": "https://example.com/report",
                    },
                }
            ],
        )

        rendered = adapter._render_message_text(msg)

        assert "Preview title" not in rendered
        assert "Preview body" not in rendered
        assert "Production alert" in rendered
        assert "Disk usage is 91%" in rendered

    @pytest.mark.asyncio
    async def test_structured_attachment_matching_authored_url_fails_open(
        self, adapter
    ):
        event = self._event(
            "alert https://example.com/incident",
            [
                {
                    "title_link": "https://example.com/incident",
                    "title": "Production alert",
                    "text": "Disk usage is 91%",
                    "fields": [{"title": "host", "value": "web-1"}],
                }
            ],
        )

        await adapter._handle_slack_message(event)

        delivered = adapter.handle_message.await_args.args[0]
        assert "📎 [Production alert](https://example.com/incident)" in delivered.text
        assert "Disk usage is 91%" in delivered.text

        rendered = adapter._render_message_text(event)
        assert "Production alert" in rendered
        assert "Disk usage is 91%" in rendered
        assert "host" in rendered
        assert "web-1" in rendered

    @pytest.mark.asyncio
    async def test_forwarded_share_is_visible_but_cannot_supply_mention(self, adapter):
        event = self._event(
            "please summarize the forwarded message",
            [
                {
                    "is_share": True,
                    "title": "Forwarded incident",
                    "text": "Earlier request for <@U_BOT>",
                }
            ],
        )

        await adapter._handle_slack_message(event)

        delivered = adapter.handle_message.await_args.args[0]
        assert "Forwarded incident" in delivered.text
        assert "Earlier request for <@U_BOT>" in delivered.text
        assert adapter._slack_event_mentions_bot(event, "U_BOT") is False
        assert "Forwarded incident" in adapter._render_message_text(event)

    @pytest.mark.asyncio
    async def test_outer_mention_routes_and_keeps_shared_content(self, adapter):
        event = self._event(
            "<@U_BOT> review this",
            [
                {
                    "is_share": True,
                    "is_app_unfurl": True,
                    "title": "Forwarded report",
                    "text": "Historic <@U_BOT> request",
                }
            ],
        )

        await adapter._handle_slack_message(event)

        delivered = adapter.handle_message.await_args.args[0]
        assert delivered.text.startswith("review this")
        assert "Forwarded report" in delivered.text
        assert "Historic  request" in delivered.text
        rendered = adapter._render_message_text(event)
        assert "Forwarded report" in rendered
        assert "Historic <@U_BOT> request" in rendered

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("is_share", "false"),
            ("is_app_unfurl", "true"),
            ("is_msg_unfurl", 1),
        ],
    )
    async def test_malformed_flags_fail_open_as_content(self, adapter, flag, value):
        event = self._event(
            "review this",
            [{flag: value, "title": "Current alert", "text": "Disk at 91%"}],
        )

        await adapter._handle_slack_message(event)

        delivered = adapter.handle_message.await_args.args[0]
        assert "Current alert" in delivered.text
        assert "Disk at 91%" in delivered.text
        assert "Current alert" in adapter._render_message_text(event)

    def test_render_message_text_preserves_attachment_only_alert(self, adapter):
        rendered = adapter._render_message_text(
            {
                "text": "",
                "attachments": [
                    {
                        "title": "Production alert",
                        "text": "Disk usage is 91%",
                        "fields": [{"title": "host", "value": "web-1"}],
                    }
                ],
            }
        )

        assert rendered == "Production alert\nDisk usage is 91%\nhost\nweb-1"


class TestSlackBotMentionRouting:
    @pytest.mark.asyncio
    async def test_allow_bots_mentions_uses_workspace_bot_user_id(self, adapter):
        adapter.config.extra["allow_bots"] = "mentions"
        adapter._team_bot_user_ids = {"T_OTHER": "U_OTHER_BOT"}
        event = {
            "text": "<@U_OTHER_BOT> investigate",
            "user": "U_PEER_BOT",
            "subtype": "bot_message",
            "channel": "C123",
            "channel_type": "channel",
            "team": "T_OTHER",
            "ts": "123.456",
        }

        await adapter._handle_slack_message(event)

        adapter.handle_message.assert_awaited_once()
        delivered = adapter.handle_message.await_args.args[0]
        assert delivered.text == "investigate"

    @pytest.mark.asyncio
    async def test_allow_bots_drops_own_workspace_bot_message(self, adapter):
        adapter.config.extra["allow_bots"] = "all"
        adapter._team_bot_user_ids = {"T_OTHER": "U_OTHER_BOT"}
        event = {
            "text": "status update",
            "user": "U_OTHER_BOT",
            "subtype": "bot_message",
            "channel": "C123",
            "channel_type": "channel",
            "team": "T_OTHER",
            "ts": "123.456",
        }

        await adapter._handle_slack_message(event)

        adapter.handle_message.assert_not_called()
