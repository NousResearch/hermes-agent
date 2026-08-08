"""Tests for the BlueBubbles iMessage gateway adapter."""
import asyncio
import json
import os
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult


@pytest.fixture(autouse=True)
def _isolate_bluebubbles_environment(monkeypatch):
    """Keep host BlueBubbles settings from changing adapter test behavior."""
    for key in tuple(os.environ):
        if key.startswith("BLUEBUBBLES_"):
            monkeypatch.delenv(key, raising=False)


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
    return BlueBubblesAdapter(cfg)


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

    def test_yaml_bridges_reply_ux_and_env_does_not_stomp_it(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        for key in (
            "BLUEBUBBLES_WEBHOOK_HOST",
            "BLUEBUBBLES_WEBHOOK_PORT",
            "BLUEBUBBLES_WEBHOOK_PATH",
            "BLUEBUBBLES_SEND_READ_RECEIPTS",
        ):
            monkeypatch.delenv(key, raising=False)
        (tmp_path / "config.yaml").write_text(
            """
platforms:
  bluebubbles:
    enabled: true
    auto_react: false
    auto_react_type: loved
    send_read_receipts: false
    split_paragraph_replies: true
    typing_indicators: false
    typing_refresh_interval: 7
    webhook_host: 0.0.0.0
    webhook_port: 9876
    webhook_path: /custom-hook
""".strip()
        )
        from gateway.config import load_gateway_config

        config = load_gateway_config()
        platform_config = config.platforms[Platform.BLUEBUBBLES]
        extra = platform_config.extra

        assert platform_config.typing_indicator is False
        assert extra["auto_react"] is False
        assert extra["auto_react_type"] == "loved"
        assert extra["send_read_receipts"] is False
        assert extra["split_paragraph_replies"] is True
        assert extra["typing_indicators"] is False
        assert extra["typing_refresh_interval"] == 7
        assert extra["webhook_host"] == "0.0.0.0"
        assert extra["webhook_port"] == 9876
        assert extra["webhook_path"] == "/custom-hook"

    def test_explicit_env_values_override_yaml_backed_defaults(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_HOST", "127.0.0.2")
        monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_PORT", "9999")
        monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_PATH", "/env-hook")
        monkeypatch.setenv("BLUEBUBBLES_SEND_READ_RECEIPTS", "true")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig(
            platforms={
                Platform.BLUEBUBBLES: PlatformConfig(
                    enabled=True,
                    extra={
                        "server_url": "http://configured.example",
                        "password": "configured-secret",
                        "webhook_host": "0.0.0.0",
                        "webhook_port": 9876,
                        "webhook_path": "/yaml-hook",
                        "send_read_receipts": False,
                    },
                )
            }
        )

        _apply_env_overrides(config)
        extra = config.platforms[Platform.BLUEBUBBLES].extra

        assert extra["server_url"] == "http://configured.example"
        assert extra["password"] == "configured-secret"
        assert extra["webhook_host"] == "127.0.0.2"
        assert extra["webhook_port"] == 9999
        assert extra["webhook_path"] == "/env-hook"
        assert extra["send_read_receipts"] is True


class TestBlueBubblesConnectionLifecycle:
    @pytest.mark.asyncio
    async def test_registration_failure_cleans_up_and_reports_disconnected(
        self, monkeypatch
    ):
        from aiohttp import web

        adapter = _make_adapter(monkeypatch)
        client = AsyncMock()
        runner = AsyncMock()
        site = AsyncMock()
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.httpx.AsyncClient",
            lambda **kwargs: client,
        )
        monkeypatch.setattr(web, "AppRunner", lambda *args, **kwargs: runner)
        monkeypatch.setattr(web, "TCPSite", lambda *args, **kwargs: site)
        monkeypatch.setattr(
            adapter,
            "_api_get",
            AsyncMock(
                side_effect=[
                    {"status": 200},
                    {"data": {"private_api": True, "helper_connected": True}},
                ]
            ),
        )
        monkeypatch.setattr(
            adapter, "_register_webhook", AsyncMock(return_value=False)
        )

        connected = await adapter.connect()

        assert connected is False
        assert adapter.is_connected is False
        assert adapter.client is None
        assert adapter._runner is None
        runner.cleanup.assert_awaited_once()
        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_listener_bind_failure_cleans_client_and_runner(self, monkeypatch):
        from aiohttp import web

        adapter = _make_adapter(monkeypatch)
        client = AsyncMock()
        runner = AsyncMock()
        site = AsyncMock()
        site.start.side_effect = OSError("address already in use")
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.httpx.AsyncClient",
            lambda **kwargs: client,
        )
        monkeypatch.setattr(web, "AppRunner", lambda *args, **kwargs: runner)
        monkeypatch.setattr(web, "TCPSite", lambda *args, **kwargs: site)
        monkeypatch.setattr(
            adapter,
            "_api_get",
            AsyncMock(side_effect=[{"status": 200}, {"data": {}}]),
        )

        assert await adapter.connect() is False
        assert adapter.client is None
        assert adapter._runner is None
        runner.cleanup.assert_awaited_once()
        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_listener_start_cancellation_cleans_client_and_runner(
        self, monkeypatch
    ):
        from aiohttp import web

        adapter = _make_adapter(monkeypatch)
        client = AsyncMock()
        runner = AsyncMock()
        site = AsyncMock()
        site.start.side_effect = asyncio.CancelledError
        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.httpx.AsyncClient",
            lambda **kwargs: client,
        )
        monkeypatch.setattr(web, "AppRunner", lambda *args, **kwargs: runner)
        monkeypatch.setattr(web, "TCPSite", lambda *args, **kwargs: site)
        monkeypatch.setattr(
            adapter,
            "_api_get",
            AsyncMock(side_effect=[{"status": 200}, {"data": {}}]),
        )

        with pytest.raises(asyncio.CancelledError):
            await adapter.connect()

        assert adapter.client is None
        assert adapter._runner is None
        runner.cleanup.assert_awaited_once()
        client.aclose.assert_awaited_once()


class TestBlueBubblesHelpers:
    def test_check_requirements(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        from gateway.platforms.bluebubbles import check_bluebubbles_requirements

        assert check_bluebubbles_requirements() is True


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
    async def test_send_preserves_paragraphs_in_one_bubble_by_default(self, monkeypatch):
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
        delivered = created[0][1] + "".join(
            payload["message"] for _, payload in sent
        )
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

        assert await adapter._send_reaction(
            "user@example.com", "inbound-guid", "like"
        ) is True
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

        assert await adapter._send_reaction(
            "user@example.com", "inbound-guid", "like"
        ) is False
        assert posts == []

    def test_reaction_type_is_normalized_and_invalid_values_fall_back(
        self, monkeypatch
    ):
        assert _make_adapter(monkeypatch, auto_react_type="loved").auto_react_type == "love"
        assert _make_adapter(monkeypatch, auto_react_type="not-a-tapback").auto_react_type == "like"

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

        assert await adapter._send_reaction(
            "user@example.com", "inbound-guid", "like"
        ) is False

        async def malformed_api_post(path, payload):
            return {}

        monkeypatch.setattr(adapter, "_api_post", malformed_api_post)
        assert await adapter._send_reaction(
            "user@example.com", "inbound-guid", "like"
        ) is False

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
        assert reactions == [
            ("iMessage;-;user@example.com", "msg-hook", "like")
        ]
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

        response = await adapter._handle_webhook(
            _FakeBlueBubblesRequest(new_payload)
        )
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
        assert text_posts[0][1]["message"] == (
            "first paragraph\n\nsecond paragraph"
        )
        assert read_receipts == [("iMessage;-;user@example.com", True)]
        assert typing_calls == ["iMessage;-;user@example.com"]
        assert any("/typing?" in url for url in client_post_urls)
        assert any("/typing?" in url for url in client_delete_urls)


class _FakeBlueBubblesRequest:
    def __init__(self, payload, password="secret"):
        self.query = {"password": password}
        self.headers = {}
        self._body = json.dumps(payload).encode("utf-8")

    async def read(self):
        return self._body


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
    async def test_group_message_without_mention_is_acknowledged_and_skipped(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            require_mention=True,
            send_read_receipts=False,
        )
        handled = []

        async def fake_handle_message(event):
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

        def mock_cache_image(data, ext):
            nonlocal cached_path
            cached_path = f"/tmp/test_image{ext}"
            return cached_path

        monkeypatch.setattr(
            "gateway.platforms.bluebubbles.cache_image_from_bytes",
            mock_cache_image,
        )

        att_meta = {"mimeType": "image/png", "transferName": "photo.png"}
        result = asyncio.get_event_loop().run_until_complete(
            adapter._download_attachment("att-guid-123", att_meta)
        )
        assert result == "/tmp/test_image.png"


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
        """No existing webhook → POST creates one."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": []},
            post_response={"status": 200, "data": {"id": 42}},
        )
        ok = asyncio.get_event_loop().run_until_complete(
            adapter._register_webhook()
        )
        assert ok is True

    @pytest.mark.asyncio
    async def test_register_fails_closed_when_lookup_fails(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        monkeypatch.setattr(adapter, "client", self._mock_client())
        monkeypatch.setattr(
            adapter, "_api_get", AsyncMock(side_effect=RuntimeError("lookup failed"))
        )
        post = AsyncMock(return_value={"status": 200})
        monkeypatch.setattr(adapter, "_api_post", post)

        ok = await adapter._register_webhook()

        assert ok is False
        post.assert_not_awaited()


    def test_register_reuses_existing(self, monkeypatch):
        """Crash resilience — existing registration is reused, no POST needed."""
        import asyncio
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        adapter.client = self._mock_client(
            get_response={"status": 200, "data": [
                {"id": 7, "url": url, "events": ["new-message"]},
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

    @pytest.mark.asyncio
    async def test_register_migrates_existing_hook_without_inbound_event(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [
            {"id": 7, "url": url, "events": ["updated-message"]}
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete
        calls = []

        async def fake_find(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            calls.append(("delete", args[0]))
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def tracking_post(path, payload):
            calls.append(("post", payload))
            created = {"id": 8, **payload}
            registrations.append(created)
            return {"status": 200, "data": created}

        monkeypatch.setattr(adapter, "_find_registered_webhooks", fake_find)
        monkeypatch.setattr(adapter, "_api_post", tracking_post)
        adapter.client.delete = tracking_delete

        assert await adapter._register_webhook() is True
        assert calls == [
            ("delete", adapter._api_url("/api/v1/webhook/7")),
            ("post", {"url": url, "events": ["new-message"]}),
        ]
        assert registrations == [
            {"id": 8, "url": url, "events": ["new-message"]}
        ]

    @pytest.mark.asyncio
    async def test_register_migrates_realistic_idempotent_same_url_webhook(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        calls = []
        registrations = [
            {
                "id": 7,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_post(path, payload):
            calls.append(("post", payload))
            # BlueBubbles addWebhook() returns an existing same-URL row
            # unchanged instead of updating its event list.
            existing = next(
                (item for item in registrations if item["url"] == payload["url"]),
                None,
            )
            if existing:
                return {"status": 200, "data": dict(existing)}
            created = {"id": 8, **payload}
            registrations.append(created)
            return {"status": 200, "data": dict(created)}

        async def tracking_delete(*args, **kwargs):
            calls.append(("delete", args[0]))
            registrations[:] = [item for item in registrations if item["id"] != 7]
            return await original_delete(*args, **kwargs)

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        monkeypatch.setattr(adapter, "_api_post", tracking_post)
        adapter.client.delete = tracking_delete

        assert await adapter._register_webhook() is True
        assert calls == [
            ("delete", adapter._api_url("/api/v1/webhook/7")),
            ("post", {"url": url, "events": ["new-message"]}),
        ]
        assert registrations == [
            {"id": 8, "url": url, "events": ["new-message"]}
        ]

    @pytest.mark.asyncio
    async def test_reused_registration_is_not_owned_or_removed_on_disconnect(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        deleted = []
        adapter.client = self._mock_client(
            get_response={
                "status": 200,
                "data": [{"id": 7, "url": url, "events": ["new-message"]}],
            }
        )
        original_delete = adapter.client.delete

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            return await original_delete(*args, **kwargs)

        adapter.client.delete = tracking_delete

        assert await adapter._register_webhook() is True
        assert await adapter._unregister_webhook() is False
        assert deleted == []

    @pytest.mark.asyncio
    async def test_cancelled_fresh_registration_remains_durable(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        started = asyncio.Event()
        release = asyncio.Event()
        registrations = []
        deleted = []
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def delayed_post(path, payload):
            started.set()
            await release.wait()
            created = {"id": 8, **payload}
            registrations.append(created)
            return {"status": 200, "data": created}

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            return await original_delete(*args, **kwargs)

        monkeypatch.setattr(adapter, "_find_registered_webhooks", fake_find)
        monkeypatch.setattr(adapter, "_api_post", delayed_post)
        adapter.client.delete = tracking_delete

        registration = asyncio.create_task(adapter._register_webhook())
        await started.wait()
        registration.cancel()
        await asyncio.sleep(0)
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await registration

        assert await adapter._unregister_webhook() is False
        assert deleted == []

    @pytest.mark.asyncio
    async def test_cancelled_registration_post_leaves_durable_replacement(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        post_started = asyncio.Event()
        release_post = asyncio.Event()
        deleted = []
        registrations = [
            {"id": 7, "url": url, "events": ["new-message", "updated-message"]}
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return list(registrations)

        async def delayed_post(path, payload):
            post_started.set()
            await release_post.wait()
            registrations.append(
                {"id": 8, "url": url, "events": list(payload["events"])}
            )
            return {"status": 200, "data": {"id": 8}}

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            return await original_delete(*args, **kwargs)

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        monkeypatch.setattr(adapter, "_api_post", delayed_post)
        adapter.client.delete = tracking_delete

        registration = asyncio.create_task(adapter._register_webhook())
        await post_started.wait()
        registration.cancel()
        release_post.set()

        with pytest.raises(asyncio.CancelledError):
            await registration

        assert await adapter._unregister_webhook() is False
        assert deleted == [adapter._api_url("/api/v1/webhook/7")]

    @pytest.mark.asyncio
    async def test_partial_stale_delete_failure_restores_when_url_is_empty(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [
            {"id": 7, "url": url, "events": ["updated-message"]},
            {"id": 8, "url": url, "events": ["updated-message"]},
        ]
        posted = []
        adapter.client = self._mock_client()

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def uncertain_delete(endpoint, **kwargs):
            webhook_id = int(endpoint.split("/api/v1/webhook/")[1].split("?")[0])
            registrations[:] = [item for item in registrations if item["id"] != webhook_id]
            if webhook_id == 8:
                raise TimeoutError("delete response lost after commit")

            class Response:
                def raise_for_status(self):
                    return None

            return Response()

        async def restore_post(path, payload):
            posted.append(payload)
            restored = {"id": 9, **payload}
            registrations.append(restored)
            return {"status": 200, "data": restored}

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        adapter.client.delete = uncertain_delete
        adapter._api_post = restore_post

        assert await adapter._register_webhook() is False
        assert posted == [{"url": url, "events": ["updated-message"]}]
        assert registrations == [
            {"id": 9, "url": url, "events": ["updated-message"]}
        ]

    @pytest.mark.asyncio
    async def test_register_post_failure_restores_stale_hook(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        deleted = []
        posted = []
        registrations = [
            {
                "id": 7,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def replacement_then_rollback(path, payload):
            posted.append(payload)
            if len(posted) == 1:
                return {"status": 500, "message": "internal error"}
            restored = {"id": 9, **payload}
            registrations.append(restored)
            return {"status": 200, "data": restored}

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        adapter.client.delete = tracking_delete
        adapter._api_post = replacement_then_rollback

        assert await adapter._register_webhook() is False
        assert deleted == [adapter._api_url("/api/v1/webhook/7")]
        assert posted == [
            {"url": url, "events": ["new-message"]},
            {"url": url, "events": ["new-message", "updated-message"]},
        ]
        assert registrations == [
            {
                "id": 9,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]

    @pytest.mark.asyncio
    async def test_register_reconciles_replacement_committed_before_timeout(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [
            {
                "id": 7,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        posted = []
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def committed_then_timed_out(path, payload):
            posted.append(payload)
            committed = {"id": 8, **payload}
            registrations.append(committed)
            raise TimeoutError("response lost after commit")

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        adapter.client.delete = tracking_delete
        adapter._api_post = committed_then_timed_out

        assert await adapter._register_webhook() is True
        assert posted == [{"url": url, "events": ["new-message"]}]
        assert registrations == [
            {"id": 8, "url": url, "events": ["new-message"]}
        ]

    @pytest.mark.asyncio
    async def test_register_does_not_delete_unexpected_post_failure_owner(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [
            {
                "id": 7,
                "url": url,
                "events": ["new-message", "updated-message"],
            }
        ]
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find_registered_webhooks(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def unexpected_owner(path, payload):
            occupied = {
                "id": 8,
                "url": url,
                "events": ["updated-message"],
            }
            registrations.append(occupied)
            return {"status": 200, "data": occupied}

        monkeypatch.setattr(
            adapter, "_find_registered_webhooks", fake_find_registered_webhooks
        )
        adapter.client.delete = tracking_delete
        adapter._api_post = unexpected_owner

        assert await adapter._register_webhook() is False
        assert registrations == [
            {"id": 8, "url": url, "events": ["updated-message"]}
        ]

    @pytest.mark.asyncio
    async def test_connect_cancellation_cleans_listener_without_deleting_hook(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch, webhook_port="0")
        cleanup_calls = []

        async def fake_api_get(path):
            if path == "/api/v1/server/info":
                return {"data": {}}
            return {"status": 200}

        async def cancelled_registration():
            raise asyncio.CancelledError

        async def tracking_unregister():
            cleanup_calls.append("unregister")
            return False

        monkeypatch.setattr(adapter, "_api_get", fake_api_get)
        monkeypatch.setattr(adapter, "_register_webhook", cancelled_registration)
        monkeypatch.setattr(adapter, "_unregister_webhook", tracking_unregister)

        with pytest.raises(asyncio.CancelledError):
            await adapter.connect()

        assert cleanup_calls == ["unregister"]
        assert adapter.client is None
        assert adapter._runner is None

    @pytest.mark.asyncio
    async def test_fresh_concurrent_winner_is_never_deleted(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = []
        deleted = []
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def concurrent_winner(path, payload):
            winner = {"id": 99, **payload}
            registrations.append(winner)
            return {"status": 200, "data": winner}

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            return await original_delete(*args, **kwargs)

        monkeypatch.setattr(adapter, "_find_registered_webhooks", fake_find)
        monkeypatch.setattr(adapter, "_api_post", concurrent_winner)
        adapter.client.delete = tracking_delete

        assert await adapter._register_webhook() is True
        assert await adapter._unregister_webhook() is False
        assert deleted == []
        assert registrations == [
            {"id": 99, "url": url, "events": ["new-message"]}
        ]

    @pytest.mark.asyncio
    async def test_migration_concurrent_winner_is_never_deleted(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        url = adapter._webhook_register_url
        registrations = [
            {"id": 7, "url": url, "events": ["updated-message"]}
        ]
        deleted = []
        adapter.client = self._mock_client()
        original_delete = adapter.client.delete

        async def fake_find(candidate_url):
            return [item for item in registrations if item["url"] == candidate_url]

        async def tracking_delete(*args, **kwargs):
            deleted.append(args[0])
            registrations.clear()
            return await original_delete(*args, **kwargs)

        async def concurrent_winner(path, payload):
            winner = {"id": 99, **payload}
            registrations.append(winner)
            return {"status": 200, "data": winner}

        monkeypatch.setattr(adapter, "_find_registered_webhooks", fake_find)
        monkeypatch.setattr(adapter, "_api_post", concurrent_winner)
        adapter.client.delete = tracking_delete

        assert await adapter._register_webhook() is True
        assert await adapter._unregister_webhook() is False
        assert deleted == [adapter._api_url("/api/v1/webhook/7")]
        assert registrations == [
            {"id": 99, "url": url, "events": ["new-message"]}
        ]

    # -- _unregister_webhook --


    def test_unregister_preserves_durable_registration(self, monkeypatch):
        """Disconnect never deletes an ambiguous fixed-URL registration."""
        import asyncio

        adapter = _make_adapter(monkeypatch)
        deleted_urls = []

        async def mock_delete(*args, **kwargs):
            deleted_urls.append(args[0] if args else "")

        adapter.client = self._mock_client()
        adapter.client.delete = mock_delete

        ok = asyncio.get_event_loop().run_until_complete(
            adapter._unregister_webhook()
        )

        assert ok is False
        assert deleted_urls == []


