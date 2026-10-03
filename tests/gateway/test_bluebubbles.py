"""Tests for the BlueBubbles iMessage gateway adapter."""
import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageType


def _make_adapter(monkeypatch, **extra):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    from gateway.platforms.bluebubbles import BlueBubblesAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "server_url": "http://localhost:1234",
            "password": "secret",
            **extra,
        },
    )
    adapter = BlueBubblesAdapter(cfg)
    return adapter


class TestBlueBubblesConfigLoading:
    def test_apply_env_overrides_bluebubbles(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_PORT", "9999")
        monkeypatch.setenv("BLUEBUBBLES_REQUIRE_MENTION", "true")
        monkeypatch.setenv("BLUEBUBBLES_MENTION_PATTERNS", r'["(?i)^amos\\b"]')
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.BLUEBUBBLES in config.platforms
        bc = config.platforms[Platform.BLUEBUBBLES]
        assert bc.enabled is True
        assert bc.extra["server_url"] == "http://localhost:1234"
        assert bc.extra["password"] == "secret"
        assert bc.extra["webhook_port"] == 9999
        assert bc.extra["require_mention"] is True
        assert bc.extra["mention_patterns"] == ["(?i)^amos\\b"]


class TestBlueBubblesHelpers:


    def test_format_message_preserves_underscores_in_identifiers(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        text = "Use /api_v2 with FEATURE_FLAG_NAME and config_file.json"
        assert adapter.format_message(text) == text

    def test_strip_markdown_headers(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.format_message("## Heading\ntext") == "Heading\ntext"


    def test_init_normalizes_webhook_path(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_path="bluebubbles-webhook")
        assert adapter.webhook_path == "/bluebubbles-webhook"


    def test_server_url_normalized(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, server_url="http://localhost:1234/")
        assert adapter.server_url == "http://localhost:1234"


class _FakeBlueBubblesRequest:
    def __init__(self, payload, password="secret"):
        self.query = {"password": password}
        self.headers = {}
        self._body = json.dumps(payload).encode("utf-8")

    async def read(self):
        return self._body


class TestBlueBubblesMentionGating:
    @pytest.mark.asyncio
    async def test_group_message_without_mention_is_acknowledged_and_skipped(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            send_read_receipts=False,
        )
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
            "type": "new-message",
            "data": {
                "guid": "msg-1",
                "text": "casual family chatter",
                "handle": {"address": "+15555550100"},
                "isFromMe": False,
                "isGroup": True,
                "chats": [{"guid": "iMessage;+;group-chat"}],
            },
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert handled == []


class TestBlueBubblesWebhookParsing:

    def test_webhook_can_fall_back_to_sender_when_chat_fields_missing(self, monkeypatch):
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


class TestBlueBubblesGuidResolution:


    @pytest.mark.asyncio
    async def test_participant_only_match_does_not_resolve_to_group(self, monkeypatch):
        """Regression for #24157: contact appearing as a participant in a group
        chat must NOT be selected when no DM with that exact chatIdentifier exists.

        Otherwise an outbound DM reply leaks into the group thread.
        """
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            return {
                "data": [
                    {
                        "guid": "iMessage;+;chat0000000000-family-group",
                        "chatIdentifier": "chat0000000000",
                        "participants": [
                            {"address": "user@example.com"},
                            {"address": "+15555550100"},
                        ],
                    }
                ]
            }

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        result = await adapter._resolve_chat_guid("user@example.com")
        assert result is None, (
            "participant-only match must not resolve to a group GUID — DM "
            "replies would leak into the group thread"
        )


    @pytest.mark.asyncio
    async def test_unresolved_target_is_not_cached(self, monkeypatch):
        """When no exact match is found, the resolver must NOT cache anything.

        Otherwise a later attempt — after the DM has been created — would
        keep returning the stale ``None`` from cache. Also guards against a
        latent variant of #24157 where a group GUID could be cached under a
        bare address key and persist across calls.
        """
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            return {
                "data": [
                    {
                        "guid": "iMessage;+;chat0000000000-family-group",
                        "chatIdentifier": "chat0000000000",
                        "participants": [{"address": "user@example.com"}],
                    }
                ]
            }

        monkeypatch.setattr(adapter, "_api_post", fake_api_post)
        await adapter._resolve_chat_guid("user@example.com")
        assert "user@example.com" not in adapter._guid_cache


class TestBlueBubblesAttachmentDownload:
    """Verify _download_attachment routes to the correct cache helper."""

    def test_download_image_uses_image_cache(self, monkeypatch):
        """Image MIME routes to cache_image_from_bytes."""
        adapter = _make_adapter(monkeypatch)
        import asyncio

        # Mock the HTTP client response
        class MockResponse:
            status_code = 200
            content = b"\x89PNG\r\n\x1a\n"

            def raise_for_status(self):
                pass

        async def mock_get(*args, **kwargs):
            return MockResponse()

        adapter.client = type("MockClient", (), {"get": mock_get})()

        cached_path = None

        async def mock_cache_image(data, ext):
            nonlocal cached_path
            cached_path = f"/tmp/test_image{ext}"
            return cached_path

        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.cache_image_from_bytes_async",
            mock_cache_image,
        )

        att_meta = {"mimeType": "image/png", "transferName": "photo.png"}
        result = asyncio.get_event_loop().run_until_complete(
            adapter._download_attachment("att-guid-123", att_meta)
        )
        assert result == "/tmp/test_image.png"


class TestBlueBubblesAttachmentSend:
    @pytest.mark.asyncio
    async def test_attachment_payload_is_read_before_async_upload(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        file_path = tmp_path / "payload.bin"
        payload = b"attachment-payload"
        file_path.write_bytes(payload)

        captured = {}

        async def fake_resolve_chat_guid(chat_id):
            return "iMessage;+;chat-guid"

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"status": 200, "data": {"guid": "message-guid"}}

        class MockClient:
            async def post(self, url, *, files, data, timeout):
                captured.update(url=url, files=files, data=data, timeout=timeout)
                return MockResponse()

        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve_chat_guid)
        adapter.client = MockClient()

        result = await adapter._send_attachment(
            "target", str(file_path), filename="payload.bin"
        )

        assert result.success is True
        assert captured["files"]["attachment"] == (
            "payload.bin",
            payload,
            "application/octet-stream",
        )
        assert captured["data"]["chatGuid"] == "iMessage;+;chat-guid"


# ---------------------------------------------------------------------------
# Webhook registration
# ---------------------------------------------------------------------------


class TestBlueBubblesWebhookUrl:
    """_webhook_url property normalises local hosts to 'localhost'."""

    def test_default_host(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # Default webhook_host is 0.0.0.0 → normalized to localhost
        assert "localhost" in adapter._webhook_url
        assert str(adapter.webhook_port) in adapter._webhook_url
        assert adapter.webhook_path in adapter._webhook_url


    def test_register_url_omits_query_when_no_password(self, monkeypatch):
        """If no password is configured, the register URL should be the bare URL."""
        monkeypatch.delenv("BLUEBUBBLES_PASSWORD", raising=False)
        from gateway.platforms.bluebubbles import BlueBubblesAdapter
        cfg = PlatformConfig(
            enabled=True,
            extra={"server_url": "http://localhost:1234", "password": ""},
        )
        adapter = BlueBubblesAdapter(cfg)
        assert adapter._webhook_register_url == adapter._webhook_url


class TestBlueBubblesWebhookRegistration:
    """Tests for _register_webhook, _unregister_webhook, _find_registered_webhooks."""

    @staticmethod
    def _mock_client(get_response=None, post_response=None, delete_ok=True):
        """Build a tiny mock httpx.AsyncClient."""

        async def mock_get(*args, **kwargs):
            class R:
                status_code = 200
                def raise_for_status(self):
                    pass
                def json(self):
                    return get_response or {"status": 200, "data": []}
            return R()

        async def mock_post(*args, **kwargs):
            class R:
                status_code = 200
                def raise_for_status(self):
                    pass
                def json(self):
                    return post_response or {"status": 200, "data": {}}
            return R()

        async def mock_delete(*args, **kwargs):
            class R:
                status_code = 200 if delete_ok else 500
                def raise_for_status(self_inner):
                    if not delete_ok:
                        raise Exception("delete failed")
            return R()

        return type(
            "MockClient", (),
            {"get": mock_get, "post": mock_post, "delete": mock_delete},
        )()

    # -- _find_registered_webhooks --

    def test_find_registered_webhooks_returns_matches(self, monkeypatch):
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_url
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 1, "url": url, "events": ["new-message"]},
                {"id": 2, "url": "http://other:9999/hook", "events": ["message"]},
            ]}
        )
        result = asyncio.get_event_loop().run_until_complete(
            adapter._find_registered_webhooks(url)
        )
        assert len(result) == 1
        assert result[0]["id"] == 1


    # -- _register_webhook --

    def test_register_fresh(self, monkeypatch):
        """No existing webhook → POST creates and verifies one."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        adapter.client = AsyncMock()
        url = adapter._webhook_register_url
        state = []

        async def find(_url):
            return list(state)

        async def post(_path, payload):
            created = {"id": 42, "url": url, "events": payload["events"]}
            state.append(created)
            return {"status": 200, "data": created}

        monkeypatch.setattr(adapter, "_find_registered_webhooks", find)
        monkeypatch.setattr(adapter, "_api_post", post)
        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is True
        assert state == [{"id": 42, "url": url, "events": ["new-message", "updated-message"]}]


    def test_register_reuses_existing(self, monkeypatch):
        """Crash resilience — existing registration is reused, no POST needed."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 7, "url": url, "events": ["new-message", "updated-message"]},
            ]},
        )

        # Track whether POST was called
        post_called = False
        orig_api_post = adapter._api_post
        async def tracking_post(path, payload):
            nonlocal post_called
            post_called = True
            return await orig_api_post(path, payload)
        adapter._api_post = tracking_post

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is True
        assert not post_called, "Should reuse existing, not POST again"


    # -- _unregister_webhook --


    def test_unregister_removes_all_duplicates(self, monkeypatch):
        """Multiple orphaned registrations for same URL — all get removed."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        deleted_ids = []

        async def mock_delete(*args, **kwargs):
            # Extract ID from URL
            url_str = args[0] if args else ""
            deleted_ids.append(url_str)
            class R:
                status_code = 200
                def raise_for_status(self):
                    pass
            return R()

        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 1, "url": url},
                {"id": 2, "url": url},
                {"id": 3, "url": "http://other/hook"},
            ]},
        )
        adapter.client.delete = mock_delete

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._unregister_webhook()
        )
        assert ok is True
        assert len(deleted_ids) == 2


# ---------------------------------------------------------------------------
# Regression for #78183: httpx timeout exceptions stringify to "" which
# defeats _is_timeout_error, causing the plain-text fallback to re-send an
# already-delivered message (duplicate delivery).
# ---------------------------------------------------------------------------

class TestBlueBubblesTimeoutErrorNormalization:
    """When an httpx timeout has an empty string representation, the adapter
    must fall back to the exception type name so the base-layer timeout guard
    can still recognise it."""

    @pytest.mark.asyncio
    async def test_send_read_timeout_produces_matchable_error(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        async def fake_resolve(chat_id):
            return "iMessage;+;chat-123"
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve)

        async def fake_api_post(path, payload):
            raise httpx.ReadTimeout("")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send("chat-1", "hello world")

        assert not result.success
        assert result.error, "error must not be empty"
        assert BasePlatformAdapter._is_timeout_error(result.error), (
            f"_is_timeout_error must recognise {result.error!r}"
        )


    @pytest.mark.asyncio
    async def test_create_chat_for_handle_timeout_produces_matchable_error(
        self, monkeypatch,
    ):
        """Sibling call path — _create_chat_for_handle has the same
        error=str(exc) pattern and must also preserve the exception type."""
        adapter = _make_adapter(monkeypatch)

        async def fake_api_post(path, payload):
            raise httpx.ReadTimeout("")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter._create_chat_for_handle("test@example.com", "hi")

        assert not result.success
        assert result.error
        assert BasePlatformAdapter._is_timeout_error(result.error)

    @pytest.mark.asyncio
    async def test_non_empty_error_string_is_unchanged(self, monkeypatch):
        """A normal exception with a message must keep its original text."""
        adapter = _make_adapter(monkeypatch)

        async def fake_resolve(chat_id):
            return "iMessage;+;chat-123"
        monkeypatch.setattr(adapter, "_resolve_chat_guid", fake_resolve)

        async def fake_api_post(path, payload):
            raise RuntimeError("Server error '500 Internal Server Error'")
        monkeypatch.setattr(adapter, "_api_post", fake_api_post)

        result = await adapter.send("chat-1", "hello world")

        assert not result.success
        assert "500 Internal Server Error" in (result.error or "")




class TestBlueBubblesGateBeforeDownload:
    """The require_mention gate must run BEFORE attachments are downloaded (review follow-up)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text, downloads, handled_count", [
        ("look at this", 0, 0),          # unmentioned group attachment: never fetched
        ("hermes look at this", 1, 1),   # mentioned: fetched and dispatched
    ])
    async def test_unmentioned_group_attachment_is_not_downloaded(
            self, monkeypatch, text, downloads, handled_count):
        adapter = _make_adapter(monkeypatch, require_mention=True, send_read_receipts=False)
        handled = []

        async def fake_handle_message(event):
            event._gateway_accepted = True
            handled.append(event)

        download = AsyncMock(return_value="/tmp/cached.jpg")
        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachment", download)
        response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
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
        }))
        await asyncio.sleep(0)

        assert response.status == 200
        assert download.await_count == downloads
        assert len(handled) == handled_count


class TestBlueBubblesInboundRegression:
    @staticmethod
    def _payload(event_type="new-message", guid="msg-1", *, chat=None, **fields):
        data = {
            "guid": guid,
            "text": "hello",
            "handle": {"address": "+155****0100"},
            "isFromMe": False,
            **fields,
        }
        if chat is None:
            data["chatIdentifier"] = "+155****0100"
        else:
            data["chats"] = [chat]
        return {"type": event_type, "data": data}

    @staticmethod
    def _capture(monkeypatch, **extra):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False, require_mention=False, **extra)
        handled = []

        async def capture(event):
            event._gateway_accepted = True
            handled.append(event)

        monkeypatch.setattr(adapter, "handle_message", capture)
        return adapter, handled

    @pytest.mark.asyncio
    async def test_inbound_lifecycle_routes_once_without_losing_rich_metadata(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def download(guid, metadata):
            calls.append(guid)
            started.set()
            await release.wait()
            return "/tmp/photo.jpg"

        monkeypatch.setattr(adapter, "_download_attachment", download)
        dm = {"guid": "any;-;+155****0100", "style": 45}
        first = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload("new-message", guid="enriched", chat=dm,
                          attachments=[{"guid": "att-1", "mimeType": ""}]))))
        await started.wait()
        second = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
            "updated-message", guid="enriched", chat=dm,
            attachments=[{"guid": "att-1", "mimeType": "image/jpeg", "transferName": "photo.jpg"}]))))
        await asyncio.sleep(0)
        release.set()
        responses = await asyncio.gather(first, second)
        assert [response.status for response in responses] == [200, 200]
        assert calls == ["att-1"]
        assert [(event.source.chat_id, event.source.chat_type, event.message_type, event.media_urls)
                for event in handled] == [
            ("+155****0100", "dm", MessageType.PHOTO, ["/tmp/photo.jpg"])]

        # Receipt-only updates do not claim the GUID needed by a subsequent real event.
        receipt_adapter, receipt_handled = self._capture(monkeypatch)
        receipt = await receipt_adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload("updated-message", guid="receipt", chat=dm, text="", dateRead=1789859535544)))
        message = await receipt_adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload("new-message", guid="receipt", chat=dm)))
        await asyncio.sleep(0)
        assert receipt.status == message.status == 200
        assert [(event.source.chat_id, event.source.chat_type) for event in receipt_handled] == [
            ("+155****0100", "dm")]

        # Sparse private-API group records retry hydration inside this webhook and never become DMs.
        group_adapter, group_handled = self._capture(monkeypatch)
        group_adapter.client = AsyncMock()
        get = AsyncMock(side_effect=[
            httpx.ReadTimeout("temporary"),
            {"data": {"chats": []}},
            {"data": {"chats": [{
                "[auth-key]": "any;+;family-group",
                "style": 43,
                "chatIdentifier": "family-group",
            }]}},
        ])
        monkeypatch.setattr(group_adapter, "_api_get", get)
        response = await group_adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload("updated-message", guid="group")))
        await asyncio.sleep(0)
        assert response.status == 200
        assert get.await_count == 3
        assert [(event.source.chat_id, event.source.chat_type) for event in group_handled] == [
            ("any;+;family-group", "group")]

        # A rich group event followed by a sparse echo remains one group dispatch.
        echo_adapter, echo_handled = self._capture(monkeypatch)
        echo_adapter.client = AsyncMock()
        monkeypatch.setattr(echo_adapter, "_api_get", AsyncMock(return_value={"data": {
            "chats": [{"guid": "any;+;same-group", "style": 43}],
        }}))
        await echo_adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
            "new-message", guid="echo", chat={"guid": "any;+;same-group", "style": 43})))
        await echo_adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
            "updated-message", guid="echo")))
        await asyncio.sleep(0)
        assert [(event.source.chat_id, event.source.chat_type) for event in echo_handled] == [
            ("any;+;same-group", "group")]

    @pytest.mark.asyncio
    async def test_webhook_reconciliation_preserves_delivery_registration(self, monkeypatch):
        # BlueBubbles POST is idempotent by URL, so migration must delete stale state before creation.
        adapter = _make_adapter(monkeypatch)
        adapter.client = AsyncMock()
        url = adapter._webhook_register_url
        state = [{"id": 1, "url": url, "events": ["new-message"]}]
        order = []

        async def find(_url):
            return list(state)

        async def post(_path, payload):
            order.append(("post", payload["events"]))
            if state:
                return {"status": 200, "data": state[0]}
            created = {"id": 2, "url": url, "events": payload["events"]}
            state.append(created)
            return {"status": 200, "data": created}

        async def delete(webhook_id):
            order.append(("delete", webhook_id))
            state[:] = [item for item in state if item["id"] != webhook_id]

        monkeypatch.setattr(adapter, "_find_registered_webhooks", find)
        monkeypatch.setattr(adapter, "_api_post", post)
        monkeypatch.setattr(adapter, "_delete_webhook_id", delete)
        assert await adapter._register_webhook() is True
        assert order == [("delete", 1), ("post", ["new-message", "updated-message"])]
        assert state == [{"id": 2, "url": url, "events": ["new-message", "updated-message"]}]

        # If replacement fails after deletion, restore the prior event set best-effort.
        rollback = _make_adapter(monkeypatch)
        rollback.client = AsyncMock()
        rollback_state = [{"id": 3, "url": url, "events": ["new-message"]}]
        posts = []

        async def rollback_delete(webhook_id):
            rollback_state[:] = [item for item in rollback_state if item["id"] != webhook_id]

        async def rollback_post(_path, payload):
            posts.append(payload["events"])
            if len(posts) == 1:
                raise httpx.ReadTimeout("replacement failed")
            restored = {"id": 4, "url": url, "events": payload["events"]}
            rollback_state.append(restored)
            return {"status": 200, "data": restored}

        monkeypatch.setattr(rollback, "_find_registered_webhooks", AsyncMock(return_value=list(rollback_state)))
        monkeypatch.setattr(rollback, "_api_post", rollback_post)
        monkeypatch.setattr(rollback, "_delete_webhook_id", rollback_delete)
        assert await rollback._register_webhook() is False
        assert posts == [["new-message", "updated-message"], ["new-message"]]
        assert rollback_state == [{"id": 4, "url": url, "events": ["new-message"]}]


class TestBlueBubblesAdmissionBoundaries:
    _payload = staticmethod(TestBlueBubblesInboundRegression._payload)
    _capture = staticmethod(TestBlueBubblesInboundRegression._capture)
    @pytest.mark.asyncio
    @pytest.mark.parametrize("caption", ["caption", ""])
    async def test_failed_download_can_complete_without_losing_caption(self, monkeypatch, caption):
        adapter, handled = self._capture(monkeypatch)
        ready = False
        calls = []

        async def download(guid, metadata):
            calls.append(guid)
            return "/tmp/photo.jpg" if ready else None

        monkeypatch.setattr(adapter, "_download_attachment", download)
        dm = {"guid": "any;-;+155****0100", "style": 45}
        data = self._payload(chat=dm, text=caption,
                             attachments=[{"guid": "photo", "mimeType": "image/jpeg"}])
        refused = await adapter._handle_webhook(_FakeBlueBubblesRequest(data))
        assert refused.status == 503
        assert handled == []
        ready = True
        data["type"] = "updated-message"
        data["data"]["isDelivered"] = True
        accepted = await adapter._handle_webhook(_FakeBlueBubblesRequest(data))
        assert accepted.status == 200
        assert [(e.text, e.media_urls) for e in handled] == [(caption or "(attachment)", ["/tmp/photo.jpg"])]

    @pytest.mark.asyncio
    async def test_partial_downloads_survive_refused_admission(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        ready = False
        calls = []

        async def download(guid, metadata):
            calls.append(guid)
            return "/tmp/" + guid + ".jpg" if guid == "first" or ready else None

        monkeypatch.setattr(adapter, "_download_attachment", download)
        data = self._payload(chat={"guid": "any;-;+155****0100"}, attachments=[
            {"guid": "first", "mimeType": "image/jpeg"}, {"guid": "second", "mimeType": "image/jpeg"}])
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 503
        ready = True
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert calls.count("first") == 1
        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/first.jpg", "/tmp/second.jpg"]

    @pytest.mark.asyncio
    async def test_actual_gateway_refusal_does_not_consume_message(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        data = self._payload(chat={"guid": "any;-;+155****0100"})
        refused = await adapter._handle_webhook(_FakeBlueBubblesRequest(data))
        assert refused.status == 503  # Real BasePlatformAdapter: no gateway handler installed.
        accepted = []

        async def admit(event):
            event._gateway_accepted = True
            accepted.append(event)

        monkeypatch.setattr(adapter, "handle_message", admit)
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert len(accepted) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("refresh", [{"transferState": 2}, {"mimeType": "image/png"}])
    async def test_stale_cached_attachment_is_not_claimed_without_delivery(self, monkeypatch, refresh):
        adapter, handled = self._capture(monkeypatch)
        ready = False

        async def download(guid, metadata):
            if ready or (guid == "first" and metadata.get("mimeType") == "image/jpeg"):
                return "/tmp/" + guid + ".jpg"
            return None

        monkeypatch.setattr(adapter, "_download_attachment", download)
        dm = {"guid": "any;-;+155****0100"}
        data = self._payload(chat=dm, attachments=[
            {"guid": "first", "mimeType": "image/jpeg"}, {"guid": "second", "mimeType": "image/jpeg"}])
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 503
        data["data"]["attachments"][0].update(refresh)
        ready = True
        if "mimeType" in refresh:
            ready = False
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 503
        assert handled == []
        ready = True
        data["data"]["attachments"][0]["transferState"] = 5
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/first.jpg", "/tmp/second.jpg"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("accepted_before_cancel", [False, True])
    async def test_cancellation_preserves_admission_boundary(self, monkeypatch, accepted_before_cancel):
        adapter, handled = self._capture(monkeypatch)
        data = self._payload(chat={"guid": "any;-;+155****0100"})
        original = adapter.handle_message

        async def cancel(event):
            event._gateway_accepted = accepted_before_cancel
            raise asyncio.CancelledError

        monkeypatch.setattr(adapter, "handle_message", cancel)
        with pytest.raises(asyncio.CancelledError):
            await adapter._handle_webhook(_FakeBlueBubblesRequest(data))
        monkeypatch.setattr(adapter, "handle_message", original)
        assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert len(handled) == (0 if accepted_before_cancel else 1)
        assert adapter._inbound_chat_tails == {}

    @pytest.mark.asyncio
    async def test_late_attachment_is_delivered_without_replaying_caption(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        monkeypatch.setattr(adapter, "_download_attachment", AsyncMock(return_value="/tmp/photo.jpg"))
        dm = {"guid": "any;-;+155****0100"}
        initial = self._payload(chat=dm)
        late = self._payload("updated-message", chat=dm, isDelivered=True, attachments=[
            {"guid": "photo", "mimeType": "image/jpeg"}])
        for data in (initial, late, late):
            assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert [(e.text, e.media_urls) for e in handled] == [
            ("hello", []), ("(attachment)", ["/tmp/photo.jpg"])]
        assert adapter._download_attachment.await_count == 1

    @pytest.mark.asyncio
    async def test_same_guid_completion_across_download_has_one_owner(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def download(guid, metadata):
            calls.append(guid)
            if guid == "first":
                started.set()
                await release.wait()
            return "/tmp/" + guid + ".jpg"

        monkeypatch.setattr(adapter, "_download_attachment", download)
        dm = {"guid": "any;-;+155****0100"}
        first_att = {"guid": "first", "mimeType": "image/jpeg"}
        first = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload(chat=dm, attachments=[first_att]))))
        await started.wait()
        second = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload("updated-message", chat=dm, isDelivered=True, attachments=[
                first_att, {"guid": "second", "mimeType": "image/jpeg"}]))))
        await asyncio.sleep(0)
        release.set()
        assert [r.status for r in await asyncio.gather(first, second)] == [200, 200]
        assert calls == ["first", "second"]
        assert len(handled) == 1
        assert handled[0].media_urls == ["/tmp/first.jpg", "/tmp/second.jpg"]

    @pytest.mark.asyncio
    async def test_distinct_messages_keep_order_without_blocking_other_chats(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        started, release = asyncio.Event(), asyncio.Event()

        async def download(guid, metadata):
            started.set()
            await release.wait()
            return "/tmp/photo.jpg"

        monkeypatch.setattr(adapter, "_download_attachment", download)
        group = {"guid": "any;+;group", "style": 43}
        first = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload(guid="first", chat=group, attachments=[{"guid": "photo", "mimeType": "image/jpeg"}]))))
        await started.wait()
        second = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload(guid="second", chat=group))))
        await asyncio.sleep(0)
        unrelated = await adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
            guid="unrelated", chat={"guid": "any;-;+155****0100"})))
        assert unrelated.status == 200
        assert [e.message_id for e in handled] == ["unrelated"]
        release.set()
        await asyncio.gather(first, second)
        assert [e.message_id for e in handled] == ["unrelated", "first", "second"]
        assert [e.source.chat_type for e in handled] == ["dm", "group", "group"]

    @pytest.mark.asyncio
    async def test_cancelled_fifo_waiter_cannot_release_its_successor_early(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        started, release = asyncio.Event(), asyncio.Event()

        async def download(guid, metadata):
            started.set()
            await release.wait()
            return "/tmp/photo.jpg"

        monkeypatch.setattr(adapter, "_download_attachment", download)
        dm = {"guid": "any;-;+155****0100"}
        first = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
            guid="first", chat=dm, attachments=[{"guid": "photo", "mimeType": "image/jpeg"}]))))
        await started.wait()
        cancelled = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload(guid="cancelled", chat=dm))))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        third = asyncio.create_task(adapter._handle_webhook(_FakeBlueBubblesRequest(
            self._payload(guid="third", chat=dm))))
        await asyncio.sleep(0)
        assert handled == []
        release.set()
        await asyncio.gather(first, third)
        assert [e.message_id for e in handled] == ["first", "third"]
        assert adapter._inbound_chat_tails == {}

    @pytest.mark.asyncio
    async def test_known_replays_do_not_wait_or_download_again(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        download = AsyncMock(return_value="/tmp/photo.jpg")
        monkeypatch.setattr(adapter, "_download_attachment", download)
        sleep = AsyncMock(side_effect=AssertionError("unexpected fixed delay"))
        monkeypatch.setattr(asyncio, "sleep", sleep)
        data = self._payload(chat={"guid": "any;-;+155****0100"}, attachments=[
            {"guid": "photo", "mimeType": "image/jpeg"}])
        for _ in range(3):
            assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert len(handled) == 1
        assert download.await_count == 1
        assert sleep.await_count == 0

    @pytest.mark.asyncio
    async def test_rest_hydration_requests_relationships_and_preserves_group_identity(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        requests = []

        def transport(request):
            requests.append(request)
            assert request.url.params['password'] == 'secret'
            with_fields = request.url.params.get('with', '').split(',')
            chats = [{'guid': 'any;+;group', 'style': 43, 'chatIdentifier': 'group'}] if 'chats' in with_fields else []
            return httpx.Response(200, json={'status': 200, 'data': {'chats': chats}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            adapter.client = client
            data = self._payload('updated-message', isDelivered=True)
            assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        assert requests and all('chats' in r.url.params['with'].split(',') for r in requests)
        assert [(e.source.chat_id, e.source.chat_type, e.source.user_id) for e in handled] == [
            ('any;+;group', 'group', '+155****0100')]

    @pytest.mark.asyncio
    async def test_attachment_readiness_uses_rest_metadata_before_download(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        snapshots = iter([2, 2, 5])
        downloads = []

        def transport(request):
            if request.url.path.startswith('/api/v1/message/'):
                assert set(request.url.params['with'].split(',')) >= {'chats', 'attachments'}
                return httpx.Response(200, json={'status': 200, 'data': {
                    'chats': [{'guid': 'any;-;+155****0100', 'style': 45}],
                    'attachments': [{'guid': 'photo', 'mimeType': 'image/png', 'transferState': next(snapshots)}]}})
            assert request.url.path == '/api/v1/attachment/photo/download'
            downloads.append(request.url.path)
            return httpx.Response(200, content=b'probe-photo')

        from gateway.platforms import bluebubbles as module
        monkeypatch.setattr(module, 'cache_image_from_bytes_async', AsyncMock(return_value='/tmp/photo.png'))
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            adapter.client = client
            response = await adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
                chat={'guid': 'any;-;+155****0100'}, attachments=[{'guid': 'photo', 'mimeType': ''}])))
        assert response.status == 200
        assert downloads == ['/api/v1/attachment/photo/download']
        assert [(e.message_type, e.media_types) for e in handled] == [(MessageType.PHOTO, ['image/png'])]
        module.cache_image_from_bytes_async.assert_awaited_once_with(b'probe-photo', '.png')

    @pytest.mark.asyncio
    async def test_dm_reply_retains_inbound_route_and_recovers_after_cache_eviction(self, monkeypatch):
        adapter, handled = self._capture(monkeypatch)
        address = '+15555550100'
        guid = 'any;-;' + address
        chats = [{'guid': 'any;-;+1555555' + str(i).zfill(4),
                  'chatIdentifier': '+1555555' + str(i).zfill(4)} for i in range(100)]
        chats.append({'guid': guid, 'chatIdentifier': address})
        query_offsets, sent = [], []

        def transport(request):
            body = json.loads(request.content)
            if request.url.path == '/api/v1/chat/query':
                query_offsets.append(body['offset'])
                return httpx.Response(200, json={'status': 200,
                    'data': chats[body['offset']:body['offset'] + body['limit']]})
            assert request.url.path == '/api/v1/message/text'
            sent.append(body['chatGuid'])
            return httpx.Response(200, json={'status': 200, 'data': {'guid': 'reply'}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            adapter.client = client
            for message_id, chat in [('sparse', {'guid': guid, 'style': 45, 'displayName': 'Alice'}),
                                     ('rich', {'guid': guid, 'style': 45, 'chatIdentifier': address})]:
                await adapter._handle_webhook(_FakeBlueBubblesRequest(self._payload(
                    guid=message_id, chat=chat, handle={'address': address})))
            assert [e.source.chat_id for e in handled] == [address, address]
            assert handled[0].source.chat_name == 'Alice'
            assert (await adapter.send(handled[0].source.chat_id, 'reply')).success
            assert query_offsets == []
            adapter._guid_cache.clear()
            assert (await adapter.send(handled[1].source.chat_id, 'reply')).success
        assert query_offsets == [0, 100]
        assert sent == [guid, guid]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('busy_mode', ['inline_command', 'fifo', 'debounce'])
    async def test_actual_base_busy_admission_prevents_replay(self, monkeypatch, busy_mode):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False, require_mention=False)
        data = self._payload(chat={'guid': 'any;-;+155****0100'},
                             text='/status' if busy_mode == 'inline_command' else 'hello')
        seen = []

        async def handler(event):
            seen.append(event.message_id)
            return None

        adapter.set_message_handler(handler)
        source = adapter.build_source(chat_id='+155****0100', chat_type='dm', user_id='+155****0100')
        session_key = adapter._source_session_key(source)
        adapter._active_sessions[session_key] = asyncio.Event()
        if busy_mode == 'fifo':
            async def busy(event, key):
                event._gateway_accepted = True
                seen.append(event.message_id)
                return True
            adapter._busy_session_handler = busy
        elif busy_mode == 'debounce':
            adapter._busy_text_mode = 'queue'
            adapter._busy_text_debounce_seconds = 60
        try:
            for _ in range(2):
                assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
            if busy_mode == 'debounce':
                assert adapter._text_debounce_store()[session_key].event.text == 'hello'
            else:
                assert seen == ['msg-1']
        finally:
            adapter._discard_text_debounce(session_key)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mode', ['queue', 'steer', 'interrupt'])
    async def test_real_busy_runner_claims_only_accepted_input(self, monkeypatch, mode):
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        monkeypatch.setenv('GATEWAY_ALLOW_ALL_USERS', 'true')
        monkeypatch.setenv('HERMES_GATEWAY_BUSY_ACK_ENABLED', 'false')
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter.set_message_handler(AsyncMock(return_value=None))
        runner = GatewayRunner(GatewayConfig())
        runner.adapters = {Platform.BLUEBUBBLES: adapter}
        runner._busy_input_mode = mode
        runner._busy_text_mode = 'interrupt'
        adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
        data = self._payload(chat={'guid': 'any;-;+155****0100'})
        source = adapter.build_source(chat_id='+155****0100', chat_type='dm', user_id='+155****0100')
        key = adapter._source_session_key(source)
        adapter._active_sessions[key] = asyncio.Event()
        calls = []

        class Receiver:
            _supports_active_turn_redirect = True

            def steer(self, text):
                calls.append(text)
                return True

            redirect = steer

        runner._session_state(key).turn.agent = Receiver()
        if mode == 'queue':
            runner._BUSY_QUEUE_MAX_PENDING = 0
            assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 503
            assert key not in adapter._pending_messages
            runner._BUSY_QUEUE_MAX_PENDING = 1
        for _ in range(2):
            assert (await adapter._handle_webhook(_FakeBlueBubblesRequest(data))).status == 200
        if mode == 'queue':
            assert adapter._pending_messages[key].text == 'hello'
            assert runner._queue_depth(key, adapter=adapter) == 1
        else:
            assert len(calls) == 1 and calls[0].endswith('\n\nhello')
