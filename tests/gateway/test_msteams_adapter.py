"""Text-only Microsoft Teams adapter tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.msteams import MsTeamsAdapter, _activities_url, strip_bot_mention


def _config(**extra):
    base = {
        "app_id": "app-id",
        "app_password": "secret",
        "bot_display_name": "Hermes",
    }
    base.update(extra)
    return PlatformConfig(enabled=True, extra=base)


def _activity(**overrides):
    conversation_type = overrides.get("conversation_type", "personal")
    activity = {
        "type": "message",
        "id": overrides.get("id", "act-1"),
        "serviceUrl": overrides.get("service_url", "https://smba.trafficmanager.net/amer/"),
        "channelId": "msteams",
        "conversation": {
            "id": overrides.get("conversation_id", "a:dm-conversation"),
            "conversationType": conversation_type,
            "name": overrides.get("conversation_name", "Chat"),
        },
        "from": {
            "id": overrides.get("from_id", "29:user"),
            "aadObjectId": overrides.get("aad_id", "aad-user"),
            "name": overrides.get("from_name", "Alice"),
        },
        "recipient": {
            "id": overrides.get("recipient_id", "28:app-id"),
            "name": overrides.get("recipient_name", "Hermes"),
        },
        "text": overrides.get("text", "hello"),
        "entities": overrides.get("entities", []),
        "channelData": overrides.get("channel_data", {}),
    }
    if "reply_to" in overrides:
        activity["replyToId"] = overrides["reply_to"]
    return activity


def test_strip_bot_mention_from_at_tag():
    cleaned, mentioned = strip_bot_mention(
        "<at>Hermes</at> please summarize this",
        bot_ids={"28:app-id"},
        bot_names={"Hermes"},
    )
    assert mentioned is True
    assert cleaned == "please summarize this"


def test_build_event_dm_normalizes_source_without_mention():
    adapter = MsTeamsAdapter(_config())
    event = adapter._build_event(_activity(text="hello from a DM"))

    assert event is not None
    assert event.text == "hello from a DM"
    assert event.message_type == MessageType.TEXT
    assert event.source.platform is Platform.MSTEAMS
    assert event.source.chat_type == "dm"
    assert event.source.chat_id == "a:dm-conversation"
    assert event.source.user_id == "aad-user"
    assert event.source.user_name == "Alice"
    assert event.message_id == "act-1"


def test_channel_requires_mention_by_default():
    adapter = MsTeamsAdapter(_config())
    activity = _activity(
        conversation_type="channel",
        conversation_id="19:channel@thread.tacv2",
        text="hello without mention",
        channel_data={
            "team": {"id": "team-1", "name": "Engineering"},
            "channel": {"id": "19:channel@thread.tacv2", "name": "General"},
            "tenant": {"id": "tenant-1"},
        },
    )

    assert adapter._build_event(activity) is None


def test_channel_mention_dispatches_and_normalizes_channel_source():
    adapter = MsTeamsAdapter(_config())
    activity = _activity(
        conversation_type="channel",
        conversation_id="19:channel@thread.tacv2",
        text="<at>Hermes</at> please help",
        channel_data={
            "team": {"id": "team-1", "name": "Engineering"},
            "channel": {"id": "19:channel@thread.tacv2", "name": "General"},
            "tenant": {"id": "tenant-1"},
        },
    )

    event = adapter._build_event(activity)

    assert event is not None
    assert event.text == "please help"
    assert event.source.chat_type == "channel"
    assert event.source.chat_id == "19:channel@thread.tacv2"
    assert event.source.thread_id == "19:channel@thread.tacv2"
    assert event.source.chat_id_alt == "team-1"
    assert event.source.guild_id == "team-1"
    assert event.source.parent_chat_id == "team-1"


def test_group_chat_can_use_mention_patterns():
    adapter = MsTeamsAdapter(_config(mention_patterns=[r"^hermes[:, ]"]))
    activity = _activity(
        conversation_type="groupChat",
        conversation_id="19:groupchat@unq.gbl.spaces",
        text="Hermes, status please",
    )

    event = adapter._build_event(activity)

    assert event is not None
    assert event.source.chat_type == "group"
    assert event.text == "Hermes, status please"


def test_free_response_conversation_bypasses_mention_gate():
    adapter = MsTeamsAdapter(
        _config(free_response_conversations=["19:channel@thread.tacv2"])
    )
    activity = _activity(
        conversation_type="channel",
        conversation_id="19:channel@thread.tacv2",
        text="no mention needed here",
        channel_data={
            "team": {"id": "team-1"},
            "channel": {"id": "19:channel@thread.tacv2"},
        },
    )

    event = adapter._build_event(activity)

    assert event is not None
    assert event.text == "no mention needed here"


def test_activities_url_includes_v3_and_encodes_conversation_id():
    assert _activities_url(
        "https://smba.trafficmanager.net/amer/",
        "19:abc@thread.tacv2",
    ) == (
        "https://smba.trafficmanager.net/amer/v3/conversations/"
        "19%3Aabc%40thread.tacv2/activities"
    )


class _FakeResponse:
    def __init__(self, status_code=201, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"id": "reply-1"}
        self.text = text

    def json(self):
        return self._payload


class _FakeHTTPClient:
    def __init__(self):
        self.posts = []

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return _FakeResponse()


class _FakeTokenProvider:
    async def get_token(self, scope):
        return "bf-token"


@pytest.mark.asyncio
async def test_send_posts_text_payload_to_bot_framework_endpoint():
    adapter = MsTeamsAdapter(_config())
    adapter._service_urls["19:abc@thread.tacv2"] = "https://smba.trafficmanager.net/amer/"
    adapter._token_provider = _FakeTokenProvider()
    adapter._http_client = _FakeHTTPClient()

    result = await adapter.send(
        "19:abc@thread.tacv2",
        "hello teams",
        reply_to="activity-1",
    )

    assert result.success is True
    assert result.message_id == "reply-1"
    url, kwargs = adapter._http_client.posts[0]
    assert url.endswith("/v3/conversations/19%3Aabc%40thread.tacv2/activities")
    assert kwargs["json"] == {
        "type": "message",
        "text": "hello teams",
        "replyToId": "activity-1",
    }
    assert kwargs["headers"]["Authorization"] == "Bearer bf-token"


class _FakeRequest:
    def __init__(self, body, *, authorization="Bearer token"):
        self._body = json.dumps(body).encode("utf-8")
        self.headers = {"Authorization": authorization}
        self.content_length = len(self._body)

    async def read(self):
        return self._body


@pytest.mark.asyncio
async def test_webhook_rejects_invalid_auth_before_dispatch():
    adapter = MsTeamsAdapter(_config())
    adapter._jwt_validator = type(
        "Validator",
        (),
        {"validate_authorization_header": AsyncMock(return_value=False)},
    )()
    adapter.handle_message = AsyncMock()

    response = await adapter._handle_activity(_FakeRequest(_activity()))

    assert response.status == 401
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_webhook_accepts_valid_text_activity_and_dispatches():
    adapter = MsTeamsAdapter(_config())
    adapter._jwt_validator = type(
        "Validator",
        (),
        {"validate_authorization_header": AsyncMock(return_value=True)},
    )()
    adapter.handle_message = AsyncMock()

    response = await adapter._handle_activity(_FakeRequest(_activity(text="hello")))

    assert response.status == 200
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"
    assert event.source.chat_type == "dm"
