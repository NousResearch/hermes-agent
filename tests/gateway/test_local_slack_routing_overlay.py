"""Host-specific Slack routing regressions preserved across updates."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )
    monkeypatch.setattr(
        "gateway.platforms.base.VIDEO_CACHE_DIR", tmp_path / "video_cache"
    )
    config = PlatformConfig(enabled=True, token="***")
    instance = SlackAdapter(config)
    instance._app = MagicMock()
    instance._app.client = AsyncMock()
    instance._app.client.users_info = AsyncMock(
        return_value={
            "user": {
                "is_bot": False,
                "profile": {"display_name": "Test User"},
                "real_name": "Test User",
            }
        }
    )
    instance._bot_user_id = "U_BOT"
    instance._running = True
    instance.handle_message = AsyncMock()
    return instance


@pytest.mark.asyncio
async def test_admitted_bot_without_user_uses_stable_identity(adapter):
    adapter.config.extra.update(
        {
            "allow_bots": "all",
            "allowed_bots": "B_BACKEND",
            "bot_auto_response_channels": "C_BOTS",
        }
    )
    event = {
        "channel": "C_BOTS",
        "channel_type": "channel",
        "subtype": "bot_message",
        "user": "U_BACKEND_BOT",
        "bot_id": "B_BACKEND",
        "app_id": "A_BACKEND",
        "username": "backend-bot",
        "text": "Investigate this alert",
        "ts": "1700000000.000001",
    }

    await adapter._handle_slack_message(event)

    message = adapter.handle_message.await_args.args[0]
    assert message.source.user_id == "B_BACKEND"
    assert message.source.user_name == "backend-bot"
    assert message.source.is_bot is True
    assert message.source.role_authorized is False


@pytest.mark.asyncio
async def test_admitted_bot_passes_early_auth_with_exact_id(adapter):
    seen = []

    class Runner:
        def _is_user_authorized(self, source):
            seen.append(source)
            return source.is_bot and source.user_id == "B_BACKEND"

        async def handler(self, event):
            return None

    adapter._message_handler = Runner().handler
    adapter.config.extra.update(
        {
            "allow_bots": "all",
            "allowed_bots": "B_BACKEND",
            "bot_auto_response_channels": "C_BOTS",
        }
    )
    event = {
        "channel": "C_BOTS",
        "channel_type": "channel",
        "team": "T1",
        "subtype": "bot_message",
        "user": "U_BACKEND_BOT",
        "bot_id": "B_BACKEND",
        "username": "backend-bot",
        "text": "Investigate this alert",
        "ts": "1700000000.000004",
    }

    await adapter._handle_slack_message(event)

    assert len(seen) == 1
    assert seen[0].user_id == "B_BACKEND"
    assert seen[0].is_bot is True
    assert seen[0].role_authorized is False
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_early_unauthorized_rejection_marks_failure_with_x(adapter):
    class Runner:
        def _is_user_authorized(self, source):
            return False

        async def handler(self, event):
            return None

    adapter._message_handler = Runner().handler
    adapter._add_reaction = AsyncMock(return_value=True)
    event = {
        "channel": "C_SHARED",
        "channel_type": "channel",
        "team": "T1",
        "user": "U_DENIED",
        "client_msg_id": "client-1",
        "text": "hello",
        "ts": "1700000000.000005",
    }

    await adapter._handle_slack_message(event)

    adapter.handle_message.assert_not_awaited()
    adapter._add_reaction.assert_awaited_once_with(
        "C_SHARED", "1700000000.000005", "x", "T1"
    )


@pytest.mark.asyncio
async def test_unlisted_bot_identity_is_dropped(adapter):
    adapter.config.extra.update(
        {
            "allow_bots": "all",
            "allowed_bots": "B_BACKEND",
            "bot_auto_response_channels": "C_BOTS",
        }
    )
    event = {
        "channel": "C_BOTS",
        "channel_type": "channel",
        "subtype": "bot_message",
        "bot_id": "B_OTHER",
        "username": "other-bot",
        "text": "untrusted automation",
        "ts": "1700000000.000003",
    }

    await adapter._handle_slack_message(event)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_spoofed_allowed_display_name_does_not_grant_bot_access(adapter):
    adapter.config.extra.update(
        {
            "allow_bots": "all",
            "allowed_bots": "B_BACKEND",
            "bot_auto_response_channels": "C_BOTS",
        }
    )
    event = {
        "channel": "C_BOTS",
        "channel_type": "channel",
        "subtype": "bot_message",
        "bot_id": "B_OTHER",
        "username": "B_BACKEND",
        "text": "spoofed automation",
        "ts": "1700000000.000006",
    }

    await adapter._handle_slack_message(event)

    adapter.handle_message.assert_not_awaited()
