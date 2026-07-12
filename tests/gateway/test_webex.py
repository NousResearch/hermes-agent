"""Tests for the Webex gateway adapter."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _make_adapter(**extra):
    from plugins.platforms.webex.adapter import WebexAdapter

    config = PlatformConfig(
        enabled=True,
        token="webex-token",
        extra=extra,
    )
    adapter = WebexAdapter(config)
    adapter._bot_id = "bot-person-id"
    adapter._bot_email = "hermes@example.com"
    adapter._bot_display_name = "Hermes"
    return adapter


class TestWebexConfigLoading:
    def test_plugin_env_enablement(self, monkeypatch):
        from plugins.platforms.webex.adapter import _env_enablement

        monkeypatch.setenv("WEBEX_BOT_TOKEN", "webex-token")
        monkeypatch.setenv("WEBEX_CONNECTION_MODE", "websocket")
        monkeypatch.setenv("WEBEX_WEBHOOK_PUBLIC_URL", "https://bot.example.com")
        monkeypatch.setenv("WEBEX_WEBHOOK_SECRET", "super-secret")
        monkeypatch.setenv("WEBEX_HOME_CHANNEL", "Y2lzY29zcGFyazovL3VzL1JPT00vabc")
        monkeypatch.setenv("WEBEX_HOME_CHANNEL_NAME", "Ops")

        seed = _env_enablement()
        assert seed["connection_mode"] == "websocket"
        assert seed["public_url"] == "https://bot.example.com"
        assert seed["secret"] == "super-secret"
        assert seed["home_channel"]["chat_id"] == "Y2lzY29zcGFyazovL3VzL1JPT00vabc"


class TestWebexSignatures:
    def test_verify_legacy_sha1_signature(self):
        adapter = _make_adapter(secret="super-secret")
        body = b'{"id":"evt-1"}'
        digest = hmac.new(b"super-secret", body, hashlib.sha1).hexdigest()

        assert adapter._verify_signature({"X-Spark-Signature": digest}, body) is True

    def test_verify_modern_sha256_signature(self):
        adapter = _make_adapter(secret="super-secret")
        body = b'{"id":"evt-2"}'
        digest = hmac.new(b"super-secret", body, hashlib.sha256).hexdigest()

        assert adapter._verify_signature({"X-Webex-Signature": f"sha256={digest}"}, body) is True


class TestWebexStreamingSupport:
    def test_supports_message_editing_is_true(self):
        adapter = _make_adapter()
        assert adapter.SUPPORTS_MESSAGE_EDITING is True

    @pytest.mark.asyncio
    async def test_edit_message_uses_room_target(self):
        adapter = _make_adapter()
        adapter._api_put_json = AsyncMock(
            return_value={
                "id": "msg-1",
                "roomId": "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
                "markdown": "Updated text",
            }
        )

        result = await adapter.edit_message(
            chat_id="Y2lzY29zcGFyazovL3VzL1JPT00vroom",
            message_id="msg-1",
            content="Updated text",
        )

        assert result.success is True
        assert result.message_id == "msg-1"
        adapter._api_put_json.assert_awaited_once_with(
            "messages/msg-1",
            {
                "roomId": "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
                "markdown": "Updated text",
            },
        )

    @pytest.mark.asyncio
    async def test_edit_message_resolves_room_for_direct_target(self):
        adapter = _make_adapter()
        adapter._api_get_json = AsyncMock(
            return_value={"id": "msg-2", "roomId": "Y2lzY29zcGFyazovL3VzL1JPT00vdm-room"}
        )
        adapter._api_put_json = AsyncMock(
            return_value={
                "id": "msg-2",
                "roomId": "Y2lzY29zcGFyazovL3VzL1JPT00vdm-room",
                "markdown": "Edited DM text",
            }
        )

        result = await adapter.edit_message(
            chat_id="user@example.com",
            message_id="msg-2",
            content="Edited DM text",
        )

        assert result.success is True
        adapter._api_get_json.assert_awaited_once_with("messages/msg-2")
        adapter._api_put_json.assert_awaited_once_with(
            "messages/msg-2",
            {
                "roomId": "Y2lzY29zcGFyazovL3VzL1JPT00vdm-room",
                "markdown": "Edited DM text",
            },
        )


class TestWebexEventBuilding:
    @pytest.mark.asyncio
    async def test_build_event_shapes_group_mention(self):
        adapter = _make_adapter()

        async def _fake_api_get(path):
            if path == "rooms/Y2lzY29zcGFyazovL3VzL1JPT00vroom":
                return {
                    "id": "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
                    "title": "Incident Room",
                    "type": "group",
                }
            if path == "people/person-1":
                return {"displayName": "Alice"}
            raise AssertionError(f"Unexpected path: {path}")

        adapter._api_get_json = AsyncMock(side_effect=_fake_api_get)
        adapter._download_attachments = AsyncMock(return_value=([], []))

        event = await adapter._build_event(
            {
                "resource": "messages",
                "event": "created",
                "data": {
                    "id": "msg-1",
                    "roomId": "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
                    "roomType": "group",
                    "personId": "person-1",
                    "personEmail": "user@example.com",
                    "text": "@Hermes: investigate this",
                    "files": [],
                },
            }
        )

        assert event is not None
        assert event.text == "investigate this"
        assert event.message_type.value == "text"
        assert event.source.chat_id == "Y2lzY29zcGFyazovL3VzL1JPT00vroom"
        assert event.source.chat_name == "Incident Room"
        assert event.source.chat_type == "group"
        assert event.source.user_id == "user@example.com"
        assert event.source.user_name == "Alice"
        assert event.source.user_id_alt == "person-1"

    @pytest.mark.asyncio
    async def test_build_event_fetches_full_message_when_event_has_metadata_only(self):
        adapter = _make_adapter()
        room_id = "Y2lzY29zcGFyazovL3VzL1JPT00vroom"

        async def _fake_api_get(path, params=None):
            if path == "messages/msg-6":
                return {
                    "id": "msg-6",
                    "roomId": room_id,
                    "roomType": "group",
                    "personId": "person-1",
                    "personEmail": "user@example.com",
                    "mentionedPeople": ["bot-person-id"],
                    "text": "@Hermes: fetched text",
                    "files": [],
                }
            if path == f"rooms/{room_id}":
                return {
                    "id": room_id,
                    "title": "Incident Room",
                    "type": "group",
                }
            if path == "people/person-1":
                return {"displayName": "Alice"}
            raise AssertionError(f"Unexpected path: {path} params={params!r}")

        adapter._api_get_json = AsyncMock(side_effect=_fake_api_get)
        adapter._download_attachments = AsyncMock(return_value=([], []))

        event = await adapter._build_event(
            {
                "resource": "messages",
                "event": "created",
                "data": {
                    "id": "msg-6",
                    "roomId": room_id,
                    "roomType": "group",
                    "personId": "person-1",
                    "personEmail": "user@example.com",
                },
            }
        )

        assert event is not None
        assert event.text == "fetched text"
        adapter._api_get_json.assert_any_await("messages/msg-6")

    @pytest.mark.asyncio
    async def test_build_event_ignores_bot_messages(self):
        adapter = _make_adapter()
        adapter._api_get_json = AsyncMock(
            return_value={
                "id": "msg-2",
                "roomId": "room-1",
                "personId": "bot-person-id",
                "personEmail": "hermes@example.com",
                "text": "hello",
                "files": [],
            }
        )

        event = await adapter._build_event({"data": {"id": "msg-2"}})

        assert event is None

    @pytest.mark.asyncio
    async def test_build_event_ignores_unmentioned_group_message(self):
        adapter = _make_adapter()

        async def _fake_api_get(path):
            if path == "rooms/Y2lzY29zcGFyazovL3VzL1JPT00vroom":
                return {
                    "id": "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
                    "title": "Incident Room",
                    "type": "group",
                }
            if path == "people/person-1":
                return {"displayName": "Alice"}
            raise AssertionError(f"Unexpected path: {path}")

        adapter._api_get_json = AsyncMock(side_effect=_fake_api_get)
        adapter._download_attachments = AsyncMock(return_value=([], []))

        event = await adapter._build_event(
            {
                "resource": "messages",
                "event": "created",
                "data": {
                    "id": "msg-3",
                    "roomId": "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
                    "roomType": "group",
                    "personId": "person-1",
                    "personEmail": "user@example.com",
                    "text": "please investigate this",
                    "files": [],
                },
            }
        )

        assert event is None

    @pytest.mark.asyncio
    async def test_build_event_allows_bare_group_slash_command(self):
        adapter = _make_adapter()

        async def _fake_api_get(path):
            if path == "rooms/Y2lzY29zcGFyazovL3VzL1JPT00vroom":
                return {
                    "id": "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
                    "title": "Incident Room",
                    "type": "group",
                }
            if path == "people/person-1":
                return {"displayName": "Alice"}
            raise AssertionError(f"Unexpected path: {path}")

        adapter._api_get_json = AsyncMock(side_effect=_fake_api_get)
        adapter._download_attachments = AsyncMock(return_value=([], []))

        event = await adapter._build_event(
            {
                "resource": "messages",
                "event": "created",
                "data": {
                    "id": "msg-4",
                    "roomId": "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
                    "roomType": "group",
                    "personId": "person-1",
                    "personEmail": "user@example.com",
                    "text": "/sethome",
                    "files": [],
                },
            }
        )

        assert event is not None
        assert event.text == "/sethome"
        assert event.message_type.value == "command"

    @pytest.mark.asyncio
    async def test_build_event_normalizes_mention_space_command(self):
        adapter = _make_adapter()

        async def _fake_api_get(path):
            if path == "rooms/Y2lzY29zcGFyazovL3VzL1JPT00vroom":
                return {
                    "id": "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
                    "title": "Incident Room",
                    "type": "group",
                }
            if path == "people/person-1":
                return {"displayName": "Alice"}
            raise AssertionError(f"Unexpected path: {path}")

        adapter._api_get_json = AsyncMock(side_effect=_fake_api_get)
        adapter._download_attachments = AsyncMock(return_value=([], []))

        event = await adapter._build_event(
            {
                "resource": "messages",
                "event": "created",
                "data": {
                    "id": "msg-5",
                    "roomId": "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
                    "roomType": "group",
                    "personId": "person-1",
                    "personEmail": "user@example.com",
                    "text": "Hermes /sethome",
                    "files": [],
                },
            }
        )

        assert event is not None
        assert event.text == "/sethome"
        assert event.message_type.value == "command"


class TestWebexThreadContext:
    @pytest.mark.asyncio
    async def test_build_event_prepends_thread_context_for_first_thread_turn(self):
        adapter = _make_adapter()
        room_id = "Y2lzY29zcGFyazovL3VzL1JPT00vroom"

        async def _fake_api_get(path, params=None):
            if path == f"rooms/{room_id}":
                return {"id": room_id, "title": "Incident Room", "type": "group"}
            if path == "messages/parent-1":
                return {
                    "id": "parent-1",
                    "roomId": room_id,
                    "personId": "person-0",
                    "personEmail": "carol@example.com",
                    "text": "Original incident summary",
                }
            if path == "messages":
                assert params == {
                    "roomId": room_id,
                    "parentId": "parent-1",
                    "max": 31,
                }
                return {
                    "items": [
                        {
                            "id": "msg-current",
                            "personId": "person-1",
                            "personEmail": "alice@example.com",
                            "text": "@Hermes can you summarize this?",
                        },
                        {
                            "id": "msg-self",
                            "personId": "bot-person-id",
                            "personEmail": "hermes@example.com",
                            "text": "Prior Hermes reply",
                        },
                        {
                            "id": "msg-prior",
                            "personId": "person-2",
                            "personEmail": "bob@example.com",
                            "text": "The database is noisy",
                        },
                    ]
                }
            raise AssertionError(f"Unexpected path: {path} params={params!r}")

        async def _fake_lookup(person_id):
            return {
                "person-0": "Carol",
                "person-1": "Alice",
                "person-2": "Bob",
            }.get(person_id, "")

        adapter._api_get_json = AsyncMock(side_effect=_fake_api_get)
        adapter._lookup_display_name = AsyncMock(side_effect=_fake_lookup)
        adapter._download_attachments = AsyncMock(return_value=([], []))

        event = await adapter._build_event(
            {
                "resource": "messages",
                "event": "created",
                "data": {
                    "id": "msg-current",
                    "roomId": room_id,
                    "roomType": "group",
                    "parentId": "parent-1",
                    "personId": "person-1",
                    "personEmail": "alice@example.com",
                    "mentionedPeople": ["bot-person-id"],
                    "text": "@Hermes can you summarize this?",
                    "files": [],
                },
            }
        )

        assert event is not None
        assert event.source.thread_id == "parent-1"
        assert event.reply_to_message_id == "parent-1"
        assert event.reply_to_text == "Original incident summary"
        assert "[Webex thread context" in event.text
        assert "[thread parent] Carol: Original incident summary" in event.text
        assert "Bob: The database is noisy" in event.text
        assert "Prior Hermes reply" not in event.text
        assert event.text.endswith("can you summarize this?")

    @pytest.mark.asyncio
    async def test_build_event_skips_thread_context_when_session_exists(self):
        adapter = _make_adapter()
        room_id = "Y2lzY29zcGFyazovL3VzL1JPT00vroom"
        adapter._has_active_session_for_thread = lambda **_: True

        async def _fake_api_get(path, params=None):
            if path == f"rooms/{room_id}":
                return {"id": room_id, "title": "Incident Room", "type": "group"}
            if path == "messages/parent-1":
                return {
                    "id": "parent-1",
                    "roomId": room_id,
                    "personId": "person-0",
                    "personEmail": "carol@example.com",
                    "text": "Original incident summary",
                }
            if path == "messages":
                raise AssertionError("thread replies should not be fetched for active sessions")
            raise AssertionError(f"Unexpected path: {path} params={params!r}")

        adapter._api_get_json = AsyncMock(side_effect=_fake_api_get)
        adapter._lookup_display_name = AsyncMock(return_value="Alice")
        adapter._download_attachments = AsyncMock(return_value=([], []))

        event = await adapter._build_event(
            {
                "resource": "messages",
                "event": "created",
                "data": {
                    "id": "msg-current",
                    "roomId": room_id,
                    "roomType": "group",
                    "parentId": "parent-1",
                    "personId": "person-1",
                    "personEmail": "alice@example.com",
                    "mentionedPeople": ["bot-person-id"],
                    "text": "@Hermes can you summarize this?",
                    "files": [],
                },
            }
        )

        assert event is not None
        assert event.text == "can you summarize this?"
        assert event.reply_to_text == "Original incident summary"

    @pytest.mark.asyncio
    async def test_build_event_does_not_prepend_context_to_thread_command(self):
        adapter = _make_adapter()
        room_id = "Y2lzY29zcGFyazovL3VzL1JPT00vroom"

        async def _fake_api_get(path, params=None):
            if path == f"rooms/{room_id}":
                return {"id": room_id, "title": "Incident Room", "type": "group"}
            if path == "messages/parent-1":
                return {
                    "id": "parent-1",
                    "roomId": room_id,
                    "personId": "person-0",
                    "personEmail": "carol@example.com",
                    "text": "Original incident summary",
                }
            if path == "messages":
                raise AssertionError("thread context should not be prepended to commands")
            raise AssertionError(f"Unexpected path: {path} params={params!r}")

        adapter._api_get_json = AsyncMock(side_effect=_fake_api_get)
        adapter._lookup_display_name = AsyncMock(return_value="Alice")
        adapter._download_attachments = AsyncMock(return_value=([], []))

        event = await adapter._build_event(
            {
                "resource": "messages",
                "event": "created",
                "data": {
                    "id": "msg-current",
                    "roomId": room_id,
                    "roomType": "group",
                    "parentId": "parent-1",
                    "personId": "person-1",
                    "personEmail": "alice@example.com",
                    "mentionedPeople": ["bot-person-id"],
                    "text": "@Hermes /status",
                    "files": [],
                },
            }
        )

        assert event is not None
        assert event.text == "/status"
        assert event.message_type.value == "command"
        assert event.reply_to_text == "Original incident summary"


class TestWebexSend:
    @pytest.mark.asyncio
    async def test_send_uses_markdown_and_parent_id(self):
        adapter = _make_adapter()
        adapter._api_post_json = AsyncMock(
            return_value={
                "id": "sent-1",
                "roomId": "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
                "parentId": "thread-1",
            }
        )

        result = await adapter.send(
            "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
            "Hello from Hermes",
            metadata={"thread_id": "thread-1"},
        )

        assert result.success is True
        adapter._api_post_json.assert_awaited_once_with(
            "messages",
            {
                "roomId": "Y2lzY29zcGFyazovL3VzL1JPT00vroom",
                "parentId": "thread-1",
                "markdown": "Hello from Hermes",
            },
        )
