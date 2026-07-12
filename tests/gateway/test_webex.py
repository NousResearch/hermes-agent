"""Tests for the Webex gateway adapter."""

import hashlib
import hmac
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter(*, token="webex-token", **extra):
    from plugins.platforms.webex.adapter import WebexAdapter

    config = PlatformConfig(
        enabled=True,
        token=token,
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

    def test_person_id_and_email_are_equivalent_allowlist_principals(self):
        from gateway.authz_mixin import _principal_matches_allowlist

        source = SimpleNamespace(
            platform=Platform("webex"),
            user_id_alt="Y2lzY29zcGFyazovL3VzL1BFT1BMRS9hbGljZQ",
        )

        assert _principal_matches_allowlist(
            source,
            "alice@example.com",
            {"Y2lzY29zcGFyazovL3VzL1BFT1BMRS9hbGljZQ"},
        )
        assert _principal_matches_allowlist(
            source, "alice@example.com", {"alice@example.com"}
        )

    def test_shared_dependency_probe_does_not_require_websocket_packages(
        self, monkeypatch
    ):
        from plugins.platforms.webex import adapter as webex

        monkeypatch.setattr(webex, "AIOHTTP_AVAILABLE", True)
        monkeypatch.setattr(
            webex,
            "_listener_dependencies_available",
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("universal probe must not inspect WebSocket packages")
            ),
        )

        assert webex.check_webex_requirements() is True

    def test_listener_inputs_mirror_to_profile_storage(self, tmp_path, monkeypatch):
        from plugins.platforms.webex import adapter as webex

        source = tmp_path / "source"
        source.mkdir()
        for name in webex._LISTENER_FILES:
            (source / name).write_text(f"fixture:{name}", encoding="utf-8")

        monkeypatch.setattr(webex, "_PLUGIN_DIR", source)
        monkeypatch.setattr(webex, "get_hermes_home", lambda: tmp_path / "home")
        monkeypatch.setattr(
            webex, "_listener_dependencies_available", lambda _path: False
        )

        runtime_dir = webex._listener_runtime_dir()

        assert runtime_dir == tmp_path / "home" / "platforms" / "webex" / "listener"
        for name in webex._LISTENER_FILES:
            assert (runtime_dir / name).read_text(encoding="utf-8") == f"fixture:{name}"


class TestWebexSignatures:
    def test_missing_secret_fails_closed(self):
        adapter = _make_adapter()

        assert adapter._verify_signature({}, b"{}") is False

    def test_verify_legacy_sha1_signature(self):
        adapter = _make_adapter(secret="super-secret")
        body = b'{"id":"evt-1"}'
        digest = hmac.new(b"super-secret", body, hashlib.sha1).hexdigest()

        assert adapter._verify_signature({"X-Spark-Signature": digest}, body) is True

    def test_verify_modern_sha256_signature(self):
        adapter = _make_adapter(secret="super-secret")
        body = b'{"id":"evt-2"}'
        digest = hmac.new(b"super-secret", body, hashlib.sha256).hexdigest()

        assert (
            adapter._verify_signature({"X-Webex-Signature": f"sha256={digest}"}, body)
            is True
        )


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
            finalize=True,
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
            return_value={
                "id": "msg-2",
                "roomId": "Y2lzY29zcGFyazovL3VzL1JPT00vdm-room",
            }
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

        event = await adapter._build_event({
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
        })

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

        event = await adapter._build_event({
            "resource": "messages",
            "event": "created",
            "data": {
                "id": "msg-6",
                "roomId": room_id,
                "roomType": "group",
                "personId": "person-1",
                "personEmail": "user@example.com",
            },
        })

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

        event = await adapter._build_event({
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
        })

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

        event = await adapter._build_event({
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
        })

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

        event = await adapter._build_event({
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
        })

        assert event is not None
        assert event.text == "/sethome"
        assert event.message_type.value == "command"


class TestWebexThreadContext:
    def test_active_thread_lookup_uses_public_profile_scoped_store_api(self):
        from gateway.config import Platform
        from gateway.session import SessionSource, build_session_key

        adapter = _make_adapter()
        adapter.set_owner_profile("secondary")
        store = SimpleNamespace(
            config=SimpleNamespace(
                group_sessions_per_user=True,
                thread_sessions_per_user=False,
            ),
            lookup_by_session_key=MagicMock(return_value=object()),
        )
        adapter.set_session_store(store)

        assert adapter._has_active_session_for_thread(
            room_id="room-1",
            chat_type="group",
            parent_id="parent-1",
            user_id="alice@example.com",
            user_id_alt="person-1",
        )

        source = SessionSource(
            platform=Platform("webex"),
            chat_id="room-1",
            chat_type="group",
            user_id="alice@example.com",
            thread_id="parent-1",
            user_id_alt="person-1",
        )
        expected_key = build_session_key(
            source,
            group_sessions_per_user=True,
            thread_sessions_per_user=False,
            profile="secondary",
        )
        store.lookup_by_session_key.assert_called_once_with(expected_key)

    @pytest.mark.asyncio
    async def test_build_event_keeps_thread_context_separate_from_trigger_text(self):
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

        event = await adapter._build_event({
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
        })

        assert event is not None
        assert event.source.thread_id == "parent-1"
        assert event.reply_to_message_id == "parent-1"
        assert event.reply_to_text == "Original incident summary"
        assert event.text == "can you summarize this?"
        assert "[Webex thread context" in event.channel_context
        assert (
            "[thread parent] Carol: Original incident summary" in event.channel_context
        )
        assert "Bob: The database is noisy" in event.channel_context
        assert "Prior Hermes reply" not in event.channel_context

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
                raise AssertionError(
                    "thread replies should not be fetched for active sessions"
                )
            raise AssertionError(f"Unexpected path: {path} params={params!r}")

        adapter._api_get_json = AsyncMock(side_effect=_fake_api_get)
        adapter._lookup_display_name = AsyncMock(return_value="Alice")
        adapter._download_attachments = AsyncMock(return_value=([], []))

        event = await adapter._build_event({
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
        })

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
                raise AssertionError(
                    "thread context should not be prepended to commands"
                )
            raise AssertionError(f"Unexpected path: {path} params={params!r}")

        adapter._api_get_json = AsyncMock(side_effect=_fake_api_get)
        adapter._lookup_display_name = AsyncMock(return_value="Alice")
        adapter._download_attachments = AsyncMock(return_value=([], []))

        event = await adapter._build_event({
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
        })

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

    @pytest.mark.asyncio
    async def test_standalone_send_preserves_email_target_and_thread(self, monkeypatch):
        from plugins.platforms.webex import adapter as webex

        class _StandaloneResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def json(self, **_kwargs):
                return {"id": "sent-standalone"}

        class _StandaloneSession:
            def __init__(self, **_kwargs):
                self.post = AsyncMock(return_value=_StandaloneResponse())

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

        session = _StandaloneSession()
        monkeypatch.setattr(webex.aiohttp, "ClientSession", lambda **_kwargs: session)

        result = await webex._standalone_send(
            PlatformConfig(token="webex-token"),
            "alice@example.com",
            "Thread update",
            thread_id="parent-message-id",
        )

        assert result == {"success": True, "message_id": "sent-standalone"}
        session.post.assert_awaited_once_with(
            f"{webex.API_BASE}/messages",
            headers={"Authorization": "Bearer webex-token"},
            json={
                "toPersonEmail": "alice@example.com",
                "markdown": "Thread update",
                "parentId": "parent-message-id",
            },
        )


class _ChunkStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.yield_count = 0

    async def iter_chunked(self, chunk_size):
        assert chunk_size > 0
        for chunk in self._chunks:
            self.yield_count += 1
            yield chunk


class _Response:
    def __init__(self, *, headers, chunks, status=200, text=""):
        self.headers = headers
        self.status = status
        self.content = _ChunkStream(chunks)
        self.read_called = False
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def read(self):
        self.read_called = True
        raise AssertionError("Webex attachment downloads must stream bounded chunks")

    async def text(self):
        return self._text


class _Session:
    def __init__(self, response):
        self.response = response
        self.closed = False

    def get(self, *_args, **_kwargs):
        return self.response

    async def close(self):
        self.closed = True


class TestWebexAttachmentBounds:
    @pytest.mark.asyncio
    async def test_rejects_oversized_content_length_before_reading(self, monkeypatch):
        from plugins.platforms.webex import adapter as webex

        response = _Response(
            headers={
                "Content-Type": "application/pdf",
                "Content-Length": "5",
                "Content-Disposition": 'attachment; filename="report.pdf"',
            },
            chunks=[b"abcde"],
        )
        adapter = _make_adapter()
        adapter._session = _Session(response)
        monkeypatch.setattr(webex, "get_inbound_media_max_bytes", lambda: 4)
        cache_document = MagicMock()
        monkeypatch.setattr(webex, "cache_document_from_bytes", cache_document)

        with pytest.raises(ValueError, match="Inbound document payload is too large"):
            await adapter._download_attachment("https://webex.example/report.pdf")

        assert response.content.yield_count == 0
        assert response.read_called is False
        cache_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_stops_when_stream_exceeds_actual_limit(self, monkeypatch):
        from plugins.platforms.webex import adapter as webex

        response = _Response(
            headers={
                "Content-Type": "application/pdf",
                "Content-Length": "2",
                "Content-Disposition": 'attachment; filename="report.pdf"',
            },
            chunks=[b"abc", b"de", b"unreachable"],
        )
        adapter = _make_adapter()
        adapter._session = _Session(response)
        monkeypatch.setattr(webex, "get_inbound_media_max_bytes", lambda: 4)

        with pytest.raises(ValueError, match="Inbound document payload is too large"):
            await adapter._download_attachment("https://webex.example/report.pdf")

        assert response.content.yield_count == 2
        assert response.read_called is False

    @pytest.mark.asyncio
    async def test_stream_within_limit_reaches_document_cache(self, monkeypatch):
        from plugins.platforms.webex import adapter as webex

        response = _Response(
            headers={
                "Content-Type": "application/pdf",
                "Content-Length": "5",
                "Content-Disposition": 'attachment; filename="report.pdf"',
            },
            chunks=[b"ab", b"cde"],
        )
        adapter = _make_adapter()
        adapter._session = _Session(response)
        monkeypatch.setattr(webex, "get_inbound_media_max_bytes", lambda: 5)
        cache_document = MagicMock(return_value="/cache/report.pdf")
        monkeypatch.setattr(webex, "cache_document_from_bytes", cache_document)

        path, mime = await adapter._download_attachment(
            "https://webex.example/report.pdf"
        )

        assert (path, mime) == ("/cache/report.pdf", "application/pdf")
        assert response.read_called is False
        cache_document.assert_called_once_with(b"abcde", "report.pdf")


class TestWebexStartupClassification:
    @pytest.mark.asyncio
    async def test_missing_token_is_nonretryable(self, monkeypatch):
        monkeypatch.delenv("WEBEX_BOT_TOKEN", raising=False)
        adapter = _make_adapter(token="")

        assert await adapter.connect() is False
        assert adapter.fatal_error_code == "webex_missing_token"
        assert adapter.fatal_error_retryable is False

    @pytest.mark.asyncio
    async def test_explicit_auth_status_is_nonretryable(self, monkeypatch):
        from plugins.platforms.webex import adapter as webex

        session = _Session(
            _Response(headers={}, chunks=[], status=401, text="unauthorized")
        )
        adapter = _make_adapter()
        adapter._acquire_platform_lock = MagicMock(return_value=True)
        adapter._release_platform_lock = MagicMock()
        monkeypatch.setattr(
            webex.aiohttp,
            "ClientSession",
            MagicMock(return_value=session),
        )

        assert await adapter.connect() is False
        assert adapter.fatal_error_code == "webex_auth_error"
        assert adapter.fatal_error_retryable is False
        assert session.closed is True

    @pytest.mark.asyncio
    async def test_transport_failure_remains_retryable(self, monkeypatch):
        from plugins.platforms.webex import adapter as webex

        session = _Session(
            _Response(headers={}, chunks=[], status=503, text="unavailable")
        )
        adapter = _make_adapter()
        adapter._acquire_platform_lock = MagicMock(return_value=True)
        adapter._release_platform_lock = MagicMock()
        monkeypatch.setattr(
            webex.aiohttp,
            "ClientSession",
            MagicMock(return_value=session),
        )

        assert await adapter.connect() is False
        assert adapter.fatal_error_code == "webex_api_unavailable"
        assert adapter.fatal_error_retryable is True
        assert session.closed is True

    def test_listener_classification_is_narrow(self):
        adapter = _make_adapter()

        missing_sdk = adapter._classify_listener_failure(
            "The Webex JS SDK packages are not installed. Run npm ci."
        )
        ambiguous_crash = adapter._classify_listener_failure(
            "Webex listener exited before startup completed (code 1)"
        )

        assert missing_sdk.code == "webex_missing_dependency"
        assert missing_sdk.retryable is False
        assert ambiguous_crash.code == "webex_listener_fatal"
        assert ambiguous_crash.retryable is True
