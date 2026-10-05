"""BlueBubbles reply ux behavior."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from tests.gateway.bluebubbles_test_support import (
    _make_adapter,
    _FakeBlueBubblesRequest,
)

pytestmark = pytest.mark.usefixtures("_isolate_bluebubbles_environment")


class TestBlueBubblesReplyUX:
    def test_hardened_defaults_are_active(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        assert adapter.auto_react is True
        assert adapter.auto_react_type == "like"
        assert adapter.typing_indicators is True
        assert adapter.config.typing_indicator is True
        assert adapter.typing_refresh_interval == 4.0
        assert adapter.split_paragraph_replies is False
        assert not hasattr(adapter, "_typing_refresh_tasks")
        assert not hasattr(adapter, "delayed_ack")

    def test_string_false_values_are_false(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            auto_react="false",
            typing_indicators="false",
            send_read_receipts="false",
            split_paragraph_replies="false",
        )

        assert adapter.auto_react is False
        assert adapter.typing_indicators is False
        assert adapter.config.typing_indicator is False
        assert adapter.send_read_receipts is False
        assert adapter.split_paragraph_replies is False

    @pytest.mark.asyncio
    async def test_send_preserves_paragraphs_in_one_bubble_by_default(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        sent = []

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        async def fake_api_post(path, payload):
            sent.append((path, payload.copy()))
            return {"data": {"guid": "msg-1"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send(
            "user@example.com", "first thought\n\nsecond thought"
        )

        assert result.success is True
        assert [payload["message"] for _, payload in sent] == [
            "first thought\n\nsecond thought"
        ]

    @pytest.mark.asyncio
    async def test_send_can_split_paragraphs_when_configured(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, split_paragraph_replies=True)
        sent = []

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        async def fake_api_post(path, payload):
            sent.append(payload["message"])
            return {"data": {"guid": f"msg-{len(sent)}"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send(
            "user@example.com", "first thought\n\nsecond thought"
        )

        assert result.success is True
        assert sent == ["first thought", "second thought"]

    @pytest.mark.asyncio
    async def test_over_limit_reply_still_splits_when_paragraph_splitting_is_off(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, split_paragraph_replies=False)
        sent = []

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        async def fake_api_post(path, payload):
            sent.append(payload["message"])
            return {"data": {"guid": f"msg-{len(sent)}"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        content = "a" * (adapter.MAX_MESSAGE_LENGTH + 1)

        result = await adapter.send("user@example.com", content)

        assert result.success is True
        assert len(sent) == 2
        assert all(len(chunk) <= adapter.MAX_MESSAGE_LENGTH for chunk in sent)
        assert "".join(sent) == content

    @pytest.mark.asyncio
    async def test_over_limit_new_chat_sends_all_chunks(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, split_paragraph_replies=False)
        adapter._private_api_enabled = True
        resolutions = [None, "iMessage;-;new@example.com"]
        created = []
        sent = []

        async def fake_resolve_chat_guid(chat_id):
            return resolutions.pop(0)

        async def fake_create_chat(address, message):
            created.append((address, message))
            return SendResult(
                success=True,
                message_id="created-message",
                raw_response={"data": {"messageGuid": "created-message"}},
            )

        async def fake_api_post(path, payload):
            sent.append((path, payload.copy()))
            return {"data": {"guid": "remainder-message"}}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_create_chat_for_handle", fake_create_chat)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        content = "a" * (adapter.MAX_MESSAGE_LENGTH + 1)

        result = await adapter.send("new@example.com", content)

        assert result.success is True
        assert len(created) == 1
        delivered = created[0][1] + "".join(payload["message"] for _, payload in sent)
        assert delivered == content
        assert len(sent) == 1

    @pytest.mark.asyncio
    async def test_over_limit_new_chat_fails_if_remainder_cannot_resolve(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, split_paragraph_replies=False)
        adapter._private_api_enabled = True
        created = []

        async def fake_resolve_chat_guid(chat_id):
            return None

        async def fake_create_chat(address, message):
            created.append((address, message))
            return SendResult(
                success=True,
                message_id="created-message",
                raw_response={"data": {"messageGuid": "created-message"}},
            )

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_create_chat_for_handle", fake_create_chat)

        result = await adapter.send(
            "new@example.com", "a" * (adapter.MAX_MESSAGE_LENGTH + 1)
        )

        assert result.success is False
        assert "remaining message chunks" in (result.error or "")
        assert len(created) == 1

    @pytest.mark.asyncio
    async def test_reaction_uses_bluebubbles_endpoint_and_payload(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        adapter.client = object()
        posts = []

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        async def fake_api_post(path, payload):
            posts.append((path, payload.copy()))
            return {"status": 200}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        assert (
            await adapter._send_reaction("user@example.com", "inbound-guid", "like")
            is True
        )
        assert posts == [
            (
                "/api/v1/message/react",
                {
                    "chatGuid": "iMessage;-;user@example.com",
                    "selectedMessageGuid": "inbound-guid",
                    "reaction": "like",
                    "partIndex": 0,
                },
            )
        ]

    @pytest.mark.asyncio
    async def test_reaction_noops_without_private_api_helper(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = False
        adapter._helper_connected = False
        adapter.client = object()
        posts = []

        async def fake_api_post(path, payload):
            posts.append((path, payload))

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        assert (
            await adapter._send_reaction("user@example.com", "inbound-guid", "like")
            is False
        )
        assert posts == []

    def test_reaction_type_is_normalized_and_invalid_values_fall_back(
        self, monkeypatch
    ):
        assert (
            _make_adapter(monkeypatch, auto_react_type="loved").auto_react_type
            == "love"
        )
        assert (
            _make_adapter(monkeypatch, auto_react_type="not-a-tapback").auto_react_type
            == "like"
        )

    @pytest.mark.asyncio
    async def test_reaction_rejects_unsuccessful_response_envelope(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        adapter.client = AsyncMock()

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        async def fake_api_post(path, payload):
            return {"status": 500, "message": "reaction rejected"}

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        assert (
            await adapter._send_reaction("user@example.com", "inbound-guid", "like")
            is False
        )

        async def malformed_api_post(path, payload):
            return {}

        monkeypatch.setattr(adapter, "_api_post", malformed_api_post)
        assert (
            await adapter._send_reaction("user@example.com", "inbound-guid", "like")
            is False
        )

    @pytest.mark.asyncio
    async def test_processing_start_reacts_without_text_ack(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        source = adapter.build_source(
            chat_id="iMessage;-;user@example.com", user_id="user@example.com"
        )
        event = MessageEvent(text="hello", source=source, message_id="msg-1")
        reactions = []
        sent = []

        async def fake_reaction(chat_id, message_id, reaction):
            reactions.append((chat_id, message_id, reaction))
            return True

        async def fake_send(chat_id, content, reply_to=None, metadata=None):
            sent.append((chat_id, content, reply_to))

        monkeypatch.setattr(adapter, "_send_reaction", fake_reaction)
        monkeypatch.setattr(adapter, "send", fake_send)

        await adapter.on_processing_start(event)

        assert reactions == [("iMessage;-;user@example.com", "msg-1", "like")]
        assert sent == []

    @pytest.mark.asyncio
    async def test_background_processing_runs_reaction_hook_and_shared_typing_cadence(
        self, monkeypatch
    ):
        adapter = _make_adapter(
            monkeypatch,
            typing_refresh_interval=7,
            send_read_receipts=False,
        )
        intervals = []
        reactions = []

        async def fake_keep_typing(
            chat_id, *, interval, metadata=None, stop_event=None
        ):
            intervals.append((chat_id, interval))

        async def fake_reaction(chat_id, message_id, reaction):
            reactions.append((chat_id, message_id, reaction))
            return True

        async def fake_handler(event):
            await asyncio.sleep(0)
            return None

        monkeypatch.setattr(adapter, "_keep_typing", fake_keep_typing)
        monkeypatch.setattr(adapter, "_send_reaction", fake_reaction)
        adapter.set_message_handler(fake_handler)
        source = adapter.build_source(
            chat_id="iMessage;-;user@example.com", user_id="user@example.com"
        )
        event = MessageEvent(text="hello", source=source, message_id="msg-hook")

        await adapter._process_message_background(event, "bluebubbles:ux-hook")

        assert intervals == [("iMessage;-;user@example.com", 7.0)]
        assert reactions == [("iMessage;-;user@example.com", "msg-hook", "like")]
        assert not hasattr(adapter, "_typing_refresh_tasks")

    @pytest.mark.asyncio
    async def test_typing_opt_out_blocks_post_and_shared_scheduler(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            typing_indicators=False,
            auto_react=False,
            send_read_receipts=False,
        )
        adapter._private_api_enabled = True
        adapter._helper_connected = True
        posts = []
        keep_typing_calls = []

        class Client:
            async def post(self, *args, **kwargs):
                posts.append((args, kwargs))

        async def fake_keep_typing(*args, **kwargs):
            keep_typing_calls.append((args, kwargs))

        async def fake_handler(event):
            return None

        adapter.client = Client()
        monkeypatch.setattr(adapter, "_keep_typing", fake_keep_typing)
        adapter.set_message_handler(fake_handler)
        source = adapter.build_source(
            chat_id="iMessage;-;user@example.com", user_id="user@example.com"
        )
        event = MessageEvent(text="hello", source=source, message_id="msg-off")

        await adapter.send_typing("user@example.com")
        await adapter._process_message_background(event, "bluebubbles:typing-off")

        assert posts == []
        assert keep_typing_calls == []

    @pytest.mark.asyncio
    async def test_webhook_handoff_schedules_read_receipt(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=True)
        handled = []
        marked = []

        async def fake_handle_message(event):
            handled.append(event)
            event._gateway_accepted = True

        async def fake_mark_read(chat_id):
            marked.append(chat_id)
            return True

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "mark_read", fake_mark_read)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "receipt-guid",
                "text": "hello",
                "chatGuid": "iMessage;-;user@example.com",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
            },
        }

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        await asyncio.sleep(0)

        assert response.status == 200
        assert [event.message_id for event in handled] == ["receipt-guid"]
        assert marked == ["iMessage;-;user@example.com"]

    @pytest.mark.asyncio
    async def test_webhook_to_reply_lifecycle_is_single_turn_single_bubble(
        self, monkeypatch
    ):
        adapter = _make_adapter(
            monkeypatch,
            auto_react=True,
            send_read_receipts=True,
            split_paragraph_replies=False,
            typing_indicators=True,
            typing_refresh_interval=7,
        )
        adapter._private_api_enabled = True
        adapter._helper_connected = True

        class RecordingClient:
            def __init__(self):
                self.posts = []
                self.deletes = []

            async def post(self, url, **kwargs):
                self.posts.append(url)

            async def delete(self, url, **kwargs):
                self.deletes.append(url)

        client = RecordingClient()
        monkeypatch.setattr(adapter, "client", client)
        handled = []
        api_posts = []
        read_receipts = []
        handler_started = asyncio.Event()
        release_handler = asyncio.Event()

        async def handler(event):
            handled.append(event)
            event._gateway_accepted = True
            handler_started.set()
            await release_handler.wait()
            return "first paragraph\n\nsecond paragraph"

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;-;user@example.com"

        async def fake_api_post(path, payload):
            api_posts.append((path, payload.copy()))
            if path == "/api/v1/message/react":
                return {"status": 200}
            return {"status": 200, "data": {"guid": "reply-guid"}}

        original_mark_read = adapter.mark_read
        original_send_typing = adapter.send_typing
        typing_calls = []
        typing_completed = asyncio.Event()

        async def tracked_mark_read(chat_id):
            result = await original_mark_read(chat_id)
            read_receipts.append((chat_id, result))
            return result

        async def tracked_send_typing(chat_id, metadata=None):
            typing_calls.append(chat_id)
            await original_send_typing(chat_id, metadata=metadata)
            typing_completed.set()

        adapter.set_message_handler(handler)
        monkeypatch.setattr(adapter, "mark_read", tracked_mark_read)
        monkeypatch.setattr(adapter, "send_typing", tracked_send_typing)
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        monkeypatch.setenv("HERMES_HUMAN_DELAY_MODE", "off")
        new_payload = {
            "type": "new-message",
            "data": {
                "guid": "lifecycle-guid",
                "text": "hello",
                "chatGuid": "iMessage;-;user@example.com",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
            },
        }

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(new_payload))
        assert response.status == 200
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        await asyncio.wait_for(typing_completed.wait(), timeout=1)
        tasks = list(adapter._session_tasks.values())
        release_handler.set()
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.01)

        updated_payload = json.loads(json.dumps(new_payload))
        updated_payload["type"] = "updated-message"
        updated_response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(updated_payload)
        )

        reaction_posts = [item for item in api_posts if item[0].endswith("/react")]
        text_posts = [item for item in api_posts if item[0].endswith("/text")]
        client_post_urls = client.posts
        client_delete_urls = client.deletes

        assert updated_response.status == 200
        assert [event.message_id for event in handled] == ["lifecycle-guid"]
        assert len(reaction_posts) == 1
        assert reaction_posts[0][1]["reaction"] == "like"
        assert len(text_posts) == 1
        assert text_posts[0][1]["message"] == ("first paragraph\n\nsecond paragraph")
        assert read_receipts == [("iMessage;-;user@example.com", True)]
        assert typing_calls == ["iMessage;-;user@example.com"]
        assert any("/typing?" in url for url in client_post_urls)
        assert any("/typing?" in url for url in client_delete_urls)
