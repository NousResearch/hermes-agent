"""BlueBubbles inbound behavior."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from tests.gateway.bluebubbles_test_support import (
    _make_adapter,
    _FakeBlueBubblesRequest,
)

pytestmark = pytest.mark.usefixtures("_isolate_bluebubbles_environment")


class TestBlueBubblesDuplicateDelivery:
    @pytest.mark.asyncio
    async def test_v019_new_and_updated_chat_variants_dispatch_once(self, monkeypatch):
        """Regression for #30708/#34372 as reproduced on Hermes v0.19.0."""
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)
            event._gateway_accepted = True

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        app = web.Application()
        app.router.add_post("/bluebubbles-webhook", adapter._handle_webhook)

        async with TestClient(TestServer(app)) as client:
            first = await client.post(
                "/bluebubbles-webhook?password=secret",
                json={
                    "type": "new-message",
                    "data": {
                        "guid": "v019-msg-1",
                        "text": "approve",
                        "chatGuid": "any;-;+15555550100",
                        "chatIdentifier": "+15555550100",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                    },
                },
            )
            second = await client.post(
                "/bluebubbles-webhook?password=secret",
                json={
                    "type": "updated-message",
                    "data": {
                        "guid": "v019-msg-1",
                        "text": "approve",
                        "chatIdentifier": "+15555550100",
                        "handle": {"address": "+15555550100"},
                        "isFromMe": False,
                    },
                },
            )
            await asyncio.sleep(0)

        assert first.status == 200
        assert second.status == 200
        assert [(event.message_id, event.text) for event in handled] == [
            ("v019-msg-1", "approve")
        ]

    @pytest.mark.asyncio
    async def test_duplicate_guid_is_dropped_but_same_text_new_guid_is_kept(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            handled.append(event)
            event._gateway_accepted = True

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "duplicate-guid-1",
                "text": "hello",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
            },
        }

        first = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        second = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        distinct = {**payload, "data": {**payload["data"], "guid": "duplicate-guid-2"}}
        third = await adapter._handle_webhook(_FakeBlueBubblesRequest(distinct))
        await asyncio.sleep(0)

        assert first.status == 200
        assert second.status == 200
        assert third.status == 200
        assert [event.message_id for event in handled] == [
            "duplicate-guid-1",
            "duplicate-guid-2",
        ]

    @pytest.mark.asyncio
    async def test_failed_handoff_releases_guid_for_redelivery(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = 0
        handled = []

        async def flaky_handle_message(event):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient handoff failure")
            handled.append(event)
            event._gateway_accepted = True

        monkeypatch.setattr(adapter, "handle_message", flaky_handle_message)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "retry-guid-1",
                "text": "retry me",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
            },
        }

        first = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        second = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert first.status == 503
        assert second.status == 200
        assert attempts == 2
        assert [event.message_id for event in handled] == ["retry-guid-1"]

    @pytest.mark.asyncio
    async def test_cancelled_handoff_releases_guid_for_redelivery(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = 0

        async def cancelled_once(event):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise asyncio.CancelledError
            event._gateway_accepted = True

        monkeypatch.setattr(adapter, "handle_message", cancelled_once)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "cancelled-guid-1",
                "text": "retry after cancellation",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
            },
        }

        with pytest.raises(asyncio.CancelledError):
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        retry = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert retry.status == 200
        assert attempts == 2

    @pytest.mark.asyncio
    async def test_single_delivery_retries_attachment_and_preserves_caption(
        self, monkeypatch
    ):
        """BlueBubbles does not redeliver a webhook after a non-2xx response."""
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = 0
        handled = []

        async def failed_download(attachment_guid, metadata):
            nonlocal attempts
            attempts += 1
            return None

        async def fake_handle_message(event):
            handled.append(event)
            event._gateway_accepted = True

        monkeypatch.setattr(adapter, "_download_attachment", failed_download)
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles._ATTACHMENT_RETRY_DELAYS", (0, 0)
        )
        payload = {
            "type": "new-message",
            "data": {
                "guid": "attachment-guid-1",
                "text": "image caption",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "attachments": [
                    {
                        "guid": "file-guid-1",
                        "mimeType": "image/png",
                        "transferName": "image.png",
                    }
                ],
            },
        }

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert response.status == 200
        assert attempts == 3
        assert [(event.text, event.media_urls) for event in handled] == [
            ("image caption", [])
        ]

    @pytest.mark.asyncio
    async def test_single_delivery_recovers_attachment_on_internal_retry(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = 0
        handled = []

        async def transient_download(attachment_guid, metadata):
            nonlocal attempts
            attempts += 1
            return None if attempts == 1 else "/tmp/recovered-image.png"

        async def fake_handle_message(event):
            handled.append(event)
            event._gateway_accepted = True

        monkeypatch.setattr(adapter, "_download_attachment", transient_download)
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles._ATTACHMENT_RETRY_DELAYS", (0, 0)
        )
        payload = {
            "type": "new-message",
            "data": {
                "guid": "attachment-guid-recovered",
                "text": "recovered caption",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "attachments": [
                    {"guid": "transient-file", "mimeType": "image/png"},
                ],
            },
        }

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert response.status == 200
        assert attempts == 2
        assert [(event.text, event.media_urls) for event in handled] == [
            ("recovered caption", ["/tmp/recovered-image.png"])
        ]

    @pytest.mark.asyncio
    async def test_single_delivery_preserves_successful_attachment_siblings(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = {"good-file": 0, "bad-file": 0}
        handled = []

        async def partial_download(attachment_guid, metadata):
            attempts[attachment_guid] += 1
            if attachment_guid == "good-file":
                return "/tmp/good-image.png"
            return None

        async def fake_handle_message(event):
            handled.append(event)
            event._gateway_accepted = True

        monkeypatch.setattr(adapter, "_download_attachment", partial_download)
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles._ATTACHMENT_RETRY_DELAYS", (0, 0)
        )
        payload = {
            "type": "new-message",
            "data": {
                "guid": "attachment-guid-2",
                "text": "two files",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "attachments": [
                    {"guid": "good-file", "mimeType": "image/png"},
                    {"guid": "bad-file", "mimeType": "application/pdf"},
                ],
            },
        }

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert response.status == 200
        assert attempts == {"good-file": 1, "bad-file": 3}
        assert [(event.text, event.media_urls) for event in handled] == [
            ("two files", ["/tmp/good-image.png"])
        ]

    @pytest.mark.asyncio
    async def test_unexpected_attachment_error_preserves_caption_and_siblings(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        handled = []

        async def successful_download(attachment_guid, metadata):
            return f"/tmp/{attachment_guid}"

        async def fake_handle_message(event):
            handled.append(event)
            event._gateway_accepted = True

        monkeypatch.setattr(adapter, "_download_attachment", successful_download)
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        payload = {
            "type": "new-message",
            "data": {
                "guid": "attachment-guid-unexpected",
                "text": "keep this caption",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "attachments": [
                    {"guid": "good-file", "mimeType": "image/png"},
                    {"guid": "malformed-file", "mimeType": 42},
                ],
            },
        }

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert response.status == 200
        assert [(event.text, event.media_urls) for event in handled] == [
            ("keep this caption", ["/tmp/good-file"])
        ]

    @pytest.mark.asyncio
    async def test_single_delivery_acknowledges_unrecoverable_attachment_only_message(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        attempts = 0
        handled = []

        async def failed_download(attachment_guid, metadata):
            nonlocal attempts
            attempts += 1
            return None

        async def fake_handle_message(event):
            handled.append(event)
            event._gateway_accepted = True

        monkeypatch.setattr(adapter, "_download_attachment", failed_download)
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles._ATTACHMENT_RETRY_DELAYS", (0, 0)
        )
        payload = {
            "type": "new-message",
            "data": {
                "guid": "attachment-guid-3",
                "text": "",
                "chatIdentifier": "user@example.com",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
                "attachments": [
                    {"guid": "bad-file", "mimeType": "image/png"},
                ],
            },
        }

        response = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))

        assert response.status == 200
        assert attempts == 3
        assert [(event.text, event.media_urls) for event in handled] == [
            ("(attachment unavailable)", [])
        ]


class TestBlueBubblesMentionGating:
    @pytest.mark.asyncio
    async def test_group_message_without_mention_is_acknowledged_and_skipped(
        self, monkeypatch
    ):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            send_read_receipts=False,
        )
        handled = []

        async def fake_handle_message(event):
            handled.append(event)
            event._gateway_accepted = True

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest({
                "type": "new-message",
                "data": {
                    "guid": "msg-1",
                    "text": "casual family chatter",
                    "handle": {"address": "+15555550100"},
                    "isFromMe": False,
                    "isGroup": True,
                    "chats": [{"guid": "iMessage;+;group-chat"}],
                },
            })
        )
        await asyncio.sleep(0)

        assert response.status == 200
        assert handled == []


class TestBlueBubblesWebhookParsing:
    def test_webhook_can_fall_back_to_sender_when_chat_fields_missing(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        payload = {
            "data": {
                "guid": "MESSAGE-GUID",
                "text": "hello",
                "handle": {"address": "user@example.com"},
                "isFromMe": False,
            }
        }
        record = adapter._extract_payload_record(payload) or {}
        chat_guid = adapter._value(
            record.get("chatGuid"),
            payload.get("chatGuid"),
            record.get("chat_guid"),
            payload.get("chat_guid"),
            payload.get("guid"),
        )
        chat_identifier = adapter._value(
            record.get("chatIdentifier"),
            record.get("identifier"),
            payload.get("chatIdentifier"),
            payload.get("identifier"),
        )
        sender = (
            adapter._value(
                record.get("handle", {}).get("address")
                if isinstance(record.get("handle"), dict)
                else None,
                record.get("sender"),
                record.get("from"),
                record.get("address"),
            )
            or chat_identifier
            or chat_guid
        )
        if not (chat_guid or chat_identifier) and sender:
            chat_identifier = sender
        assert chat_identifier == "user@example.com"

    def test_extract_payload_record_accepts_list_data(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        payload = {
            "type": "new-message",
            "data": [
                {
                    "text": "hello",
                    "chatGuid": "iMessage;-;user@example.com",
                    "chatIdentifier": "user@example.com",
                }
            ],
        }
        record = adapter._extract_payload_record(payload)
        assert record == payload["data"][0]


class TestBlueBubblesGateBeforeDownload:
    """The require_mention gate must run BEFORE attachments are downloaded (review follow-up)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text, downloads, handled_count",
        [
            ("look at this", 0, 0),  # unmentioned group attachment: never fetched
            ("hermes look at this", 1, 1),  # mentioned: fetched and dispatched
        ],
    )
    async def test_unmentioned_group_attachment_is_not_downloaded(
        self, monkeypatch, text, downloads, handled_count
    ):
        adapter = _make_adapter(
            monkeypatch, require_mention=True, send_read_receipts=False
        )
        handled = []

        async def fake_handle_message(event):
            handled.append(event)
            event._gateway_accepted = True

        download = AsyncMock(return_value="/tmp/cached.jpg")
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", download)
        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest({
                "type": "new-message",
                "data": {
                    "guid": "msg-att-1",
                    "text": text,
                    "handle": {"address": "+15555550100"},
                    "isFromMe": False,
                    "isGroup": True,
                    "chats": [{"guid": "iMessage;+;group-chat"}],
                    "attachments": [{"guid": "att-1", "mimeType": "image/jpeg"}],
                },
            })
        )
        await asyncio.sleep(0)

        assert response.status == 200
        assert download.await_count == downloads
        assert len(handled) == handled_count


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["missing_handler", "busy_refusal", "inline", "debounce"]
)
async def test_webhook_uses_actual_gateway_admission_receipt(monkeypatch, path):
    adapter = _make_adapter(monkeypatch, send_read_receipts=True)
    adapter.mark_read = AsyncMock(return_value=True)
    adapter.send_typing = AsyncMock()
    handler = AsyncMock(return_value=None)
    owner = None
    if path != "missing_handler":
        adapter.set_message_handler(handler)
        source = adapter.build_source(
            chat_id="user@example.com", user_id="user@example.com"
        )
        key = adapter._event_session_key(MessageEvent(text="hello", source=source))
        owner = asyncio.create_task(asyncio.Event().wait())
        adapter._active_sessions[key] = asyncio.Event()
        adapter._session_tasks[key] = owner
    if path == "busy_refusal":

        async def refuse(event, session_key):
            return True  # rejected by the runner, no admission receipt

        adapter.set_busy_session_handler(refuse)
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 60
    payload = {
        "type": "new-message",
        "data": {
            "guid": "admission-guid",
            "text": "/status" if path == "inline" else "hello",
            "chatIdentifier": "user@example.com",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
        },
    }
    try:
        first = await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        if path in {"missing_handler", "busy_refusal"}:
            assert first.status == 503
            adapter.mark_read.assert_not_awaited()
            handler.assert_not_awaited()
            if owner:
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
            adapter._session_tasks.clear()
            adapter._active_sessions.clear()
            adapter.set_busy_session_handler(None)
            adapter.set_message_handler(handler)
            assert (
                await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
            ).status == 200
            for _ in range(20):
                if handler.await_count:
                    break
                await asyncio.sleep(0)
            handler.assert_awaited_once()
        else:
            assert first.status == 200
        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        if path == "debounce":
            assert adapter._text_debounce[key].event.message_id == "admission-guid"
            handler.assert_not_awaited()
        else:
            handler.assert_awaited_once()
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "after_acceptance", [False, True], ids=["before_admission", "after_admission"]
)
async def test_cancelled_gateway_admission_settles_guid_against_receipt(
    monkeypatch,
    after_acceptance,
):
    """Cancellation must not replay a command the real gateway already handled."""
    adapter = _make_adapter(monkeypatch, send_read_receipts=False)
    entered = asyncio.Event()
    release = asyncio.Event()
    handled = []
    events = []

    async def handler(event):
        events.append(event)
        if not after_acceptance:
            entered.set()
            await release.wait()
        handled.append(event.message_id)
        return "status reply"

    async def send(chat_id, content, **kwargs):
        if after_acceptance:
            entered.set()
            await release.wait()
        return SendResult(success=True, message_id="reply")

    adapter.set_message_handler(handler)
    monkeypatch.setattr(adapter, "send", send)
    source = adapter.build_source(
        chat_id="user@example.com", user_id="user@example.com"
    )
    key = adapter._event_session_key(MessageEvent(text="/status", source=source))
    owner = asyncio.create_task(asyncio.Event().wait())
    adapter._active_sessions[key] = asyncio.Event()
    adapter._session_tasks[key] = owner
    payload = {
        "type": "new-message",
        "data": {
            "guid": "cancel-admission-guid",
            "text": "/status",
            "chatIdentifier": "user@example.com",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
        },
    }
    first = asyncio.create_task(
        adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
    )
    duplicate = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert events[0]._gateway_accepted is after_acceptance
        duplicate = asyncio.create_task(
            adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        )
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        duplicate_response = await asyncio.wait_for(duplicate, timeout=5)
        assert handled == (["cancel-admission-guid"] if after_acceptance else [])
        release.set()
        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        assert handled == ["cancel-admission-guid"]
        assert len(events) == (1 if after_acceptance else 2)
        assert duplicate_response.status == (200 if after_acceptance else 503)
        assert not adapter._inflight_message_ids
    finally:
        for task in (first, duplicate):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, duplicate) if task is not None),
            return_exceptions=True,
        )
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [False, True], ids=["dm", "group"])
async def test_unseen_updated_attachment_is_acknowledged_without_dispatch(
    monkeypatch, group
):
    adapter = _make_adapter(monkeypatch, send_read_receipts=False)
    handler = AsyncMock()
    download = AsyncMock()
    monkeypatch.setattr(adapter, "handle_message", handler)
    monkeypatch.setattr(adapter, "_download_attachment", download)
    payload = {
        "type": "updated-message",
        "data": {
            "guid": "unseen-update-guid",
            "text": "newly hydrated caption",
            "isGroup": group,
            "chatGuid": "iMessage;+;group" if group else "iMessage;-;user@example.com",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
            "attachments": [{"guid": "new-attachment", "mimeType": "image/jpeg"}],
        },
    }
    assert (
        await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
    ).status == 200
    handler.assert_not_awaited()
    download.assert_not_awaited()
    assert not adapter._inflight_message_ids
    assert not adapter._message_dedup.contains("unseen-update-guid")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "associated_type, filtered",
    [
        *[(code, True) for code in range(2000, 2006)],
        *[(code, True) for code in range(3000, 3006)],
        (0, False),
        (1999, False),
        (2006, False),
        (3006, False),
        ("2001", False),
        (None, False),
    ],
)
async def test_integer_tapback_policy_acknowledges_without_starting_turn(
    monkeypatch,
    associated_type,
    filtered,
):
    """Known additions/removals are filtered; other message types still dispatch."""
    adapter = _make_adapter(
        monkeypatch,
        send_read_receipts=False,
        typing_indicators=False,
        auto_react=False,
    )
    handled = []
    dispatched = asyncio.Event()

    async def handler(event):
        handled.append(event.message_id)
        dispatched.set()
        return None

    adapter.set_message_handler(handler)
    payload = {
        "type": "new-message",
        "data": {
            "guid": "associated-type-guid",
            "text": "message content",
            "associatedMessageType": associated_type,
            "chatIdentifier": "user@example.com",
            "handle": {"address": "user@example.com"},
            "isFromMe": False,
        },
    }
    try:
        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        if not filtered:
            await asyncio.wait_for(dispatched.wait(), timeout=5)
        assert (
            await adapter._handle_webhook(_FakeBlueBubblesRequest(payload))
        ).status == 200
        assert handled == ([] if filtered else ["associated-type-guid"])
        assert not adapter._inflight_message_ids
        assert adapter._message_dedup.contains("associated-type-guid") is (not filtered)
    finally:
        await adapter.disconnect()
