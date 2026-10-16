"""Adapter boundary checks for opt-in Slack and Teams destination permissions."""

from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack import adapter as slack_adapter
from plugins.platforms.teams import adapter as teams_adapter
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.teams.adapter import TeamsAdapter


def _grant(scope, kind, resource, *, read=False, write=False):
    return {"scope_id": scope, "kind": kind, "id": resource, "read": read, "write": write}


def _config(*grants):
    return PlatformConfig(enabled=True, extra={
        "enterprise_permissions": {"enabled": True, "grants": list(grants)}})


@pytest.mark.asyncio
async def test_slack_denies_send_and_media_before_client_use():
    adapter = SlackAdapter(_config(_grant("T1", "user", "U1", read=True),
                                   _grant("T1", "channel", "C1", read=True)))
    adapter._app = MagicMock()
    adapter._team_clients = {"T1": MagicMock()}
    adapter._remember_channel_team("C1", "T1")
    client = AsyncMock()
    adapter._get_client = MagicMock(return_value=client)

    result = await adapter.send("C1", "secret", metadata={"slack_team_id": "T1"})
    assert result.error == "enterprise_write_denied"
    result = await adapter.send_image("C1", "https://example.com/secret.png", metadata={"slack_team_id": "T1"})
    assert result.error == "enterprise_write_denied"
    client.chat_postMessage.assert_not_awaited()
    client.files_upload_v2.assert_not_awaited()


@pytest.mark.asyncio
async def test_slack_denies_inbound_before_files_are_processed():
    adapter = SlackAdapter(_config(_grant("T1", "user", "U1", read=True)))
    event = {"team": "T1", "channel": "C1", "channel_type": "channel", "user": "U1",
             "text": "hello", "ts": "123", "files": [{"id": "F1"}]}
    adapter._prefilter_inbound = AsyncMock(return_value=(event, "T1", "C1"))
    adapter._lookup_assistant_thread_metadata = MagicMock(return_value={})
    adapter._agent_view_context_for_event = MagicMock(return_value=None)
    adapter._early_reject_unauthorized = MagicMock(return_value=False)
    adapter.handle_message = AsyncMock()

    await adapter._handle_slack_message_impl(event)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_teams_denies_unknown_and_read_only_conversations():
    adapter = TeamsAdapter(_config(_grant("tenant", "user", "aad-user", read=True)))
    adapter._app = AsyncMock()
    adapter._enterprise_conversations["personal"] = ("tenant", "user", "aad-user")

    assert (await adapter.send("personal", "secret")).error == "enterprise_write_denied"
    assert (await adapter.send("unknown", "secret")).error == "enterprise_write_denied"
    adapter._app.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_teams_denies_inbound_before_attachment_download():
    adapter = TeamsAdapter(_config(_grant("tenant", "user", "aad-user", read=True)))
    adapter._app = SimpleNamespace(id="bot")
    adapter._require_mention = False
    adapter._cache_attachment = AsyncMock()
    adapter.handle_message = AsyncMock()
    activity = SimpleNamespace(
        id="message-1", recipient=SimpleNamespace(id="bot"),
        from_=SimpleNamespace(id="user", aad_object_id="aad-user", name="User"),
        conversation=SimpleNamespace(id="chat-1", tenant_id="tenant", conversation_type="groupChat", name="Group"),
        text="hello", attachments=[SimpleNamespace(name="private.txt")],
    )

    await adapter._on_message(SimpleNamespace(activity=activity, conversation_ref=object()))

    adapter._cache_attachment.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_standalone_send_fails_closed_without_verified_destination_type():
    config = _config(_grant("T1", "channel", "C1", write=True))
    assert "enterprise_write_denied" in (await slack_adapter._standalone_send(config, "C1", "secret"))["error"]
    assert "enterprise_write_denied" in (await teams_adapter._standalone_send(config, "chat", "secret"))["error"]
