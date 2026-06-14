"""Tests for the Rocket.Chat platform plugin."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.base import MessageType, SendResult
from gateway.platform_registry import platform_registry
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


class _MockPluginContext:
    def register_platform(self, **kwargs):
        from gateway.platform_registry import PlatformEntry

        platform_registry.register(PlatformEntry(**kwargs))


class FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        payload=None,
        text: str | None = None,
        content_type: str = "application/json",
        data: bytes | None = None,
    ) -> None:
        self.status = status
        self._payload = payload if payload is not None else {}
        self._text = text
        self._data = data if data is not None else b""
        self.headers = {"Content-Type": content_type}
        self.content_type = content_type.split(";", 1)[0]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        if self._text is not None:
            return self._text
        return str(self._payload)

    async def read(self):
        return self._data


class FakeSession:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.requests = []
        self.closed = False

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError(f"No fake response queued for {method} {url}")
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    async def close(self):
        self.closed = True


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


@pytest.fixture
def adapter_mod():
    return load_plugin_adapter("rocketchat")


@pytest.fixture
def auth_mod():
    return importlib.import_module("plugins.platforms.rocketchat.auth")


@pytest.fixture(autouse=True)
def fake_aiohttp(monkeypatch):
    class ClientTimeout:
        def __init__(self, total=None):
            self.total = total

    class FormData:
        def __init__(self):
            self.fields = []

        def add_field(self, name, value, **kwargs):
            self.fields.append((name, value, kwargs))

    class ClientError(Exception):
        pass

    class ClientSession:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def close(self):
            self.closed = True

    class WSMsgType:
        TEXT = "TEXT"
        CLOSED = "CLOSED"
        CLOSE = "CLOSE"
        CLOSING = "CLOSING"
        ERROR = "ERROR"

    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(
            ClientTimeout=ClientTimeout,
            FormData=FormData,
            ClientError=ClientError,
            ClientSession=ClientSession,
            WSMsgType=WSMsgType,
        ),
    )


@pytest.fixture(autouse=True)
def clear_rocketchat_env(monkeypatch):
    for name in (
        "ROCKETCHAT_URL",
        "ROCKETCHAT_USER_ID",
        "ROCKETCHAT_AUTH_TOKEN",
        "ROCKETCHAT_ALLOWED_USERS",
        "ROCKETCHAT_ALLOW_ALL_USERS",
        "ROCKETCHAT_HOME_CHANNEL",
        "ROCKETCHAT_HOME_CHANNEL_NAME",
        "ROCKETCHAT_HOME_CHANNEL_THREAD_ID",
        "ROCKETCHAT_REQUIRE_MENTION",
        "ROCKETCHAT_FREE_RESPONSE_ROOMS",
        "ROCKETCHAT_ALLOWED_ROOMS",
        "ROCKETCHAT_COMMAND_PREFIX",
        "ROCKETCHAT_BOOTSTRAP_ENABLED",
        "ROCKETCHAT_BOOTSTRAP_ARTIFACT",
        "ROCKETCHAT_BOOTSTRAP_PAT_NAME",
        "ROCKETCHAT_OAUTH_SERVICE_NAME",
        "ROCKETCHAT_OAUTH_AUTHORIZE_URL",
        "ROCKETCHAT_OAUTH_TOKEN_URL",
        "ROCKETCHAT_OAUTH_CLIENT_ID",
        "ROCKETCHAT_OAUTH_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def clean_registry():
    original = dict(platform_registry._entries)
    platform_registry._entries.clear()
    yield
    platform_registry._entries.clear()
    platform_registry._entries.update(original)


def _register_plugin(adapter_mod):
    adapter_mod.register(_MockPluginContext())


def _make_adapter(adapter_mod, *, extra=None):
    adapter = adapter_mod.RocketChatAdapter(
        PlatformConfig(enabled=True, extra={"url": "https://chat.example.com", **(extra or {})})
    )
    adapter._runtime_creds = SimpleNamespace(
        auth_token="runtime-token",
        user_id="bot-user",
        auth_type="auth_token",
    )
    adapter._bot_user_id = "bot-user"
    adapter._bot_username = "hermes"
    return adapter


class TestRocketChatConfig:
    def test_registers_platform_entry(self, adapter_mod, clean_registry):
        _register_plugin(adapter_mod)
        entry = platform_registry.get("rocketchat")
        assert entry is not None
        assert entry.name == "rocketchat"
        assert entry.allowed_users_env == "ROCKETCHAT_ALLOWED_USERS"
        assert entry.allow_all_env == "ROCKETCHAT_ALLOW_ALL_USERS"
        assert entry.cron_deliver_env_var == "ROCKETCHAT_HOME_CHANNEL"

    def test_env_only_config_loads(self, adapter_mod, clean_registry, monkeypatch):
        _register_plugin(adapter_mod)
        monkeypatch.setenv("ROCKETCHAT_URL", "https://chat.example.com")
        monkeypatch.setenv("ROCKETCHAT_USER_ID", "user-1")
        monkeypatch.setenv("ROCKETCHAT_AUTH_TOKEN", "token-1")
        monkeypatch.setenv("ROCKETCHAT_HOME_CHANNEL", "ROOM123")
        monkeypatch.setenv("ROCKETCHAT_HOME_CHANNEL_NAME", "Ops")
        monkeypatch.setenv("ROCKETCHAT_HOME_CHANNEL_THREAD_ID", "root-thread")

        config = GatewayConfig()
        _apply_env_overrides(config)

        platform = Platform("rocketchat")
        assert platform in config.platforms
        rc = config.platforms[platform]
        assert rc.enabled is True
        assert rc.extra["url"] == "https://chat.example.com"
        assert rc.home_channel is not None
        assert rc.home_channel.chat_id == "ROOM123"
        assert rc.home_channel.name == "Ops"
        assert rc.home_channel.thread_id == "root-thread"

    def test_yaml_bridge_preserves_env_precedence(self, adapter_mod, monkeypatch):
        monkeypatch.setenv("ROCKETCHAT_COMMAND_PREFIX", "?")
        adapter_mod._apply_yaml_config(
            {},
            {
                "command_prefix": "!",
                "allowed_rooms": ["room-a", "room-b"],
                "free_response_rooms": ["room-c"],
            },
        )
        assert os.getenv("ROCKETCHAT_COMMAND_PREFIX") == "?"
        assert os.getenv("ROCKETCHAT_ALLOWED_ROOMS") == "room-a,room-b"
        assert os.getenv("ROCKETCHAT_FREE_RESPONSE_ROOMS") == "room-c"

    def test_bootstrap_artifact_only_overrides_env_when_enabled(self, auth_mod, tmp_path, monkeypatch):
        artifact = tmp_path / "rocketchat-auth.json"
        auth_mod.save_bootstrap_artifact(
            artifact,
            auth_mod.RocketChatRuntimeCredentials(
                url="https://chat.example.com",
                user_id="artifact-user",
                auth_token="artifact-token",
                auth_type="pat",
            ),
        )
        monkeypatch.setenv("ROCKETCHAT_URL", "https://chat.example.com")
        monkeypatch.setenv("ROCKETCHAT_USER_ID", "env-user")
        monkeypatch.setenv("ROCKETCHAT_AUTH_TOKEN", "env-token")

        creds = auth_mod.resolve_runtime_credentials(
            "https://chat.example.com",
            extra={"bootstrap_enabled": False, "bootstrap_artifact": str(artifact)},
        )
        assert creds.user_id == "env-user"
        assert creds.auth_token == "env-token"

        creds = auth_mod.resolve_runtime_credentials(
            "https://chat.example.com",
            extra={"bootstrap_enabled": True, "bootstrap_artifact": str(artifact)},
        )
        assert creds.user_id == "artifact-user"
        assert creds.auth_token == "artifact-token"


class TestRocketChatAuth:
    def test_validate_bootstrap_artifact_rejects_malformed(self, auth_mod):
        with pytest.raises(auth_mod.RocketChatBootstrapError, match="missing one of"):
            auth_mod.validate_bootstrap_artifact({"url": "https://chat.example.com"})

    def test_save_and_load_bootstrap_artifact(self, auth_mod, tmp_path):
        artifact = tmp_path / "rocketchat-auth.json"
        runtime = auth_mod.RocketChatRuntimeCredentials(
            url="https://chat.example.com",
            user_id="user-1",
            auth_token="pat-token",
            auth_type="pat",
            username="alice",
        )
        auth_mod.save_bootstrap_artifact(artifact, runtime)
        loaded = auth_mod.load_bootstrap_artifact(artifact, expected_url="https://chat.example.com")
        assert loaded.auth_type == "pat"
        assert loaded.username == "alice"
        assert loaded.auth_token == "pat-token"

    @pytest.mark.asyncio
    async def test_bootstrap_persists_pat_when_promotion_succeeds(self, auth_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(auth_mod.secrets, "token_urlsafe", lambda *_: "state123")
        monkeypatch.setattr(auth_mod, "_make_pkce_verifier", lambda: "verifier")
        monkeypatch.setattr(auth_mod._CallbackCapture, "wait", lambda self, timeout: {"state": ["state123"], "code": ["auth-code"]})
        monkeypatch.setattr(auth_mod.webbrowser, "open", lambda *_: True)

        async def fake_exchange_provider_code(session, *, config, code, code_verifier):
            assert code == "auth-code"
            assert code_verifier == "verifier"
            return {"access_token": "provider-token"}

        async def fake_exchange_login(session, *, config, provider_tokens):
            assert provider_tokens["access_token"] == "provider-token"
            return auth_mod.RocketChatRuntimeCredentials(
                url=config.rocketchat_url,
                user_id="user-1",
                auth_token="short-lived",
                auth_type="auth_token",
                username="alice",
            )

        async def fake_validate(session, creds):
            return creds

        async def fake_promote(session, creds, **kwargs):
            assert kwargs["pat_name"] == "hermes-bootstrap"
            return auth_mod.RocketChatRuntimeCredentials(
                url=creds.url,
                user_id=creds.user_id,
                auth_token="pat-token",
                auth_type="pat",
                username=creds.username,
                pat_name=kwargs["pat_name"],
            )

        monkeypatch.setattr(auth_mod, "_exchange_provider_code", fake_exchange_provider_code)
        monkeypatch.setattr(auth_mod, "_exchange_rocketchat_login", fake_exchange_login)
        monkeypatch.setattr(auth_mod, "_validate_runtime_credentials", fake_validate)
        monkeypatch.setattr(auth_mod, "_maybe_promote_to_pat", fake_promote)

        artifact = tmp_path / "bootstrap.json"
        cfg = auth_mod.OAuthBootstrapConfig(
            rocketchat_url="https://chat.example.com",
            service_name="oidc",
            authorize_url="https://id.example.com/authorize",
            token_url="https://id.example.com/token",
            client_id="client-id",
            redirect_uri="http://127.0.0.1:0/callback",
            pat_name="hermes-bootstrap",
            artifact_path=artifact,
            open_browser=True,
        )

        runtime = await auth_mod.bootstrap_via_oauth(cfg)
        assert runtime.auth_type == "pat"
        assert runtime.auth_token == "pat-token"
        saved = auth_mod.load_bootstrap_artifact(artifact)
        assert saved.auth_type == "pat"
        assert saved.pat_name == "hermes-bootstrap"

    @pytest.mark.asyncio
    async def test_bootstrap_falls_back_to_auth_token_when_pat_unavailable(self, auth_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(auth_mod.secrets, "token_urlsafe", lambda *_: "state123")
        monkeypatch.setattr(auth_mod, "_make_pkce_verifier", lambda: "verifier")
        monkeypatch.setattr(auth_mod._CallbackCapture, "wait", lambda self, timeout: {"state": ["state123"], "code": ["auth-code"]})
        monkeypatch.setattr(auth_mod.webbrowser, "open", lambda *_: True)

        async def fake_exchange_provider_code(session, *, config, code, code_verifier):
            return {"access_token": "provider-token"}

        async def fake_exchange_login(session, *, config, provider_tokens):
            return auth_mod.RocketChatRuntimeCredentials(
                url=config.rocketchat_url,
                user_id="user-1",
                auth_token="short-lived",
                auth_type="auth_token",
            )

        async def fake_validate(session, creds):
            return creds

        async def fake_promote(session, creds, **kwargs):
            return creds

        monkeypatch.setattr(auth_mod, "_exchange_provider_code", fake_exchange_provider_code)
        monkeypatch.setattr(auth_mod, "_exchange_rocketchat_login", fake_exchange_login)
        monkeypatch.setattr(auth_mod, "_validate_runtime_credentials", fake_validate)
        monkeypatch.setattr(auth_mod, "_maybe_promote_to_pat", fake_promote)

        artifact = tmp_path / "bootstrap.json"
        cfg = auth_mod.OAuthBootstrapConfig(
            rocketchat_url="https://chat.example.com",
            service_name="oidc",
            authorize_url="https://id.example.com/authorize",
            token_url="https://id.example.com/token",
            client_id="client-id",
            redirect_uri="http://127.0.0.1:0/callback",
            artifact_path=artifact,
            open_browser=True,
        )

        runtime = await auth_mod.bootstrap_via_oauth(cfg)
        assert runtime.auth_type == "auth_token"
        saved = auth_mod.load_bootstrap_artifact(artifact)
        assert saved.auth_type == "auth_token"


class TestRocketChatAdapter:
    def test_truncate_message_splits_long_text(self, adapter_mod):
        adapter = _make_adapter(adapter_mod)
        chunks = adapter.truncate_message("a" * 9000, 4096)
        assert len(chunks) == 3
        assert all(len(chunk) <= 4096 for chunk in chunks)

    def test_command_prefix_rewrite(self, adapter_mod):
        adapter = _make_adapter(adapter_mod)
        text, is_command = adapter._normalize_command_text("!new")
        assert text == "/new"
        assert is_command is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("room_type", "expected_chat_type"),
        [("d", "dm"), ("c", "channel"), ("p", "group")],
    )
    async def test_room_type_and_thread_mapping(self, adapter_mod, room_type, expected_chat_type):
        adapter = _make_adapter(adapter_mod, extra={"allow_all_users": True})
        adapter._allow_all_users = True
        adapter._room_cache["ROOM1"] = {"_id": "ROOM1", "t": room_type, "fname": "Room One"}
        adapter._download_incoming_attachments = AsyncMock(return_value=([], [], MessageType.TEXT))
        captured = []
        message = "!new"
        payload: dict[str, object] = {}
        if room_type != "d":
            adapter._free_response_rooms.add("ROOM1")
            message = "@hermes !new"
            payload["mentions"] = [{"_id": "bot-user", "username": "hermes"}]

        async def capture(event):
            captured.append(event)

        adapter.handle_message = capture
        await adapter._dispatch_incoming_message(
            "ROOM1",
            {
                "_id": "m1",
                "rid": "ROOM1",
                "msg": message,
                "u": {"_id": "user-1", "username": "alice"},
                "tmid": "thread-root",
                **payload,
            },
        )
        assert len(captured) == 1
        event = captured[0]
        assert event.source.chat_type == expected_chat_type
        assert event.source.thread_id == "thread-root"
        assert event.text == "/new"
        assert event.message_type == MessageType.COMMAND

    @pytest.mark.asyncio
    async def test_mention_requirement_and_free_response(self, adapter_mod):
        adapter = _make_adapter(adapter_mod, extra={"allowed_users": "user-1"})
        adapter._allowed_users = {"user-1"}
        adapter._room_cache["ROOM1"] = {"_id": "ROOM1", "t": "c", "fname": "Ops"}
        adapter._download_incoming_attachments = AsyncMock(return_value=([], [], MessageType.TEXT))
        adapter.handle_message = AsyncMock()

        await adapter._dispatch_incoming_message(
            "ROOM1",
            {"_id": "m1", "rid": "ROOM1", "msg": "hello", "u": {"_id": "user-1", "username": "alice"}},
        )
        adapter.handle_message.assert_not_called()

        await adapter._dispatch_incoming_message(
            "ROOM1",
            {
                "_id": "m2",
                "rid": "ROOM1",
                "msg": "@hermes hello there",
                "mentions": [{"_id": "bot-user", "username": "hermes"}],
                "u": {"_id": "user-1", "username": "alice"},
            },
        )
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "hello there"

        adapter.handle_message.reset_mock()
        adapter._free_response_rooms.add("ROOM1")
        await adapter._dispatch_incoming_message(
            "ROOM1",
            {"_id": "m3", "rid": "ROOM1", "msg": "free response", "u": {"_id": "user-1", "username": "alice"}},
        )
        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_allowed_room_and_user_gating(self, adapter_mod):
        adapter = _make_adapter(adapter_mod, extra={"allowed_users": "user-1", "allowed_rooms": "ROOM1"})
        adapter._room_cache["ROOM1"] = {"_id": "ROOM1", "t": "c", "fname": "Ops"}
        adapter._room_cache["ROOM2"] = {"_id": "ROOM2", "t": "c", "fname": "Other"}
        adapter._download_incoming_attachments = AsyncMock(return_value=([], [], MessageType.TEXT))
        adapter.handle_message = AsyncMock()

        await adapter._dispatch_incoming_message(
            "ROOM2",
            {"_id": "m1", "rid": "ROOM2", "msg": "@hermes hi", "mentions": [{"_id": "bot-user"}], "u": {"_id": "user-1"}},
        )
        adapter.handle_message.assert_not_called()

        await adapter._dispatch_incoming_message(
            "ROOM1",
            {"_id": "m2", "rid": "ROOM1", "msg": "@hermes hi", "mentions": [{"_id": "bot-user"}], "u": {"_id": "user-2"}},
        )
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_self_message_and_duplicate_suppression(self, adapter_mod):
        adapter = _make_adapter(adapter_mod, extra={"allow_all_users": True})
        adapter._room_cache["DM1"] = {"_id": "DM1", "t": "d", "fname": "Alice"}
        adapter._download_incoming_attachments = AsyncMock(return_value=([], [], MessageType.TEXT))
        adapter.handle_message = AsyncMock()

        await adapter._dispatch_incoming_message(
            "DM1",
            {"_id": "m1", "rid": "DM1", "msg": "first", "u": {"_id": "user-1", "username": "alice"}},
        )
        await adapter._dispatch_incoming_message(
            "DM1",
            {"_id": "m1", "rid": "DM1", "msg": "first", "u": {"_id": "user-1", "username": "alice"}},
        )
        await adapter._dispatch_incoming_message(
            "DM1",
            {"_id": "m2", "rid": "DM1", "msg": "ignored", "u": {"_id": "bot-user", "username": "hermes"}},
        )
        await adapter._dispatch_incoming_message(
            "DM1",
            {"_id": "m3", "rid": "DM1", "msg": "system", "u": {"_id": "user-1"}, "t": "uj"},
        )

        assert adapter.handle_message.await_count == 1

    @pytest.mark.asyncio
    async def test_inbound_attachment_download_failure_keeps_text_only(self, adapter_mod):
        adapter = _make_adapter(adapter_mod, extra={"allow_all_users": True})
        adapter._room_cache["DM1"] = {"_id": "DM1", "t": "d", "fname": "Alice"}
        adapter._download_one_attachment = AsyncMock(side_effect=RuntimeError("boom"))
        captured = []

        async def capture(event):
            captured.append(event)

        adapter.handle_message = capture
        await adapter._dispatch_incoming_message(
            "DM1",
            {
                "_id": "m1",
                "rid": "DM1",
                "msg": "here is a file",
                "u": {"_id": "user-1", "username": "alice"},
                "file": {"_id": "f1", "name": "report.pdf", "type": "application/pdf", "url": "/file-upload/f1/report.pdf"},
            },
        )
        assert len(captured) == 1
        assert captured[0].text == "here is a file"
        assert captured[0].message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_send_posts_chat_send_message(self, adapter_mod):
        adapter = _make_adapter(adapter_mod)
        adapter._session = FakeSession(
            [
                FakeResponse(
                    status=200,
                    payload={"success": True, "message": {"_id": "msg-1"}},
                )
            ]
        )
        result = await adapter.send("ROOM1", "hello world", metadata={"thread_id": "root-1"})
        assert result.success is True
        method, url, kwargs = adapter._session.requests[0]
        assert method == "POST"
        assert url.endswith("/api/v1/chat.sendMessage")
        payload = kwargs["json"]["message"]
        assert payload["rid"] == "ROOM1"
        assert payload["msg"] == "hello world"
        assert payload["tmid"] == "root-1"
        assert payload["tshow"] is False

    @pytest.mark.asyncio
    async def test_edit_failure_is_non_success_result(self, adapter_mod):
        adapter = _make_adapter(adapter_mod)
        adapter._session = FakeSession(
            [FakeResponse(status=403, payload={"error": "Editing disabled"})]
        )
        result = await adapter.edit_message("ROOM1", "msg-1", "updated")
        assert result.success is False
        assert "Editing disabled" in (result.error or "")

    @pytest.mark.asyncio
    async def test_send_document_uploads_and_confirms_in_thread(self, adapter_mod, tmp_path):
        adapter = _make_adapter(adapter_mod)
        file_path = tmp_path / "report.txt"
        file_path.write_text("hello", encoding="utf-8")
        adapter._session = FakeSession(
            [
                FakeResponse(status=200, payload={"success": True, "file": {"_id": "file-1"}}),
                FakeResponse(status=200, payload={"success": True, "message": {"_id": "msg-2"}}),
            ]
        )
        result = await adapter.send_document(
            "ROOM1",
            str(file_path),
            caption="see attached",
            metadata={"thread_id": "root-9"},
        )
        assert result.success is True
        assert len(adapter._session.requests) == 2
        _, upload_url, _ = adapter._session.requests[0]
        _, confirm_url, confirm_kwargs = adapter._session.requests[1]
        assert upload_url.endswith("/api/v1/rooms.media/ROOM1")
        assert confirm_url.endswith("/api/v1/rooms.mediaConfirm/ROOM1/file-1")
        assert confirm_kwargs["json"]["msg"] == "see attached"
        assert confirm_kwargs["json"]["tmid"] == "root-9"

    @pytest.mark.asyncio
    async def test_retryable_http_errors_surface_retryable_send_result(self, adapter_mod):
        adapter = _make_adapter(adapter_mod)
        adapter._session = FakeSession(
            [FakeResponse(status=429, payload={"error": "rate limited"})]
        )
        result = await adapter.send("ROOM1", "hello")
        assert result.success is False
        assert result.retryable is True
        assert "rate limited" in (result.error or "")

    @pytest.mark.asyncio
    async def test_standalone_sender_uses_adapter_send(self, adapter_mod, monkeypatch):
        send_mock = AsyncMock(return_value=SendResult(success=True, message_id="msg-1"))
        connect_mock = AsyncMock(return_value=True)
        disconnect_mock = AsyncMock()
        monkeypatch.setattr(adapter_mod.RocketChatAdapter, "connect", connect_mock)
        monkeypatch.setattr(adapter_mod.RocketChatAdapter, "send", send_mock)
        monkeypatch.setattr(adapter_mod.RocketChatAdapter, "disconnect", disconnect_mock)

        result = await adapter_mod._standalone_send(
            SimpleNamespace(extra={"url": "https://chat.example.com"}),
            "ROOM1",
            "hello",
            thread_id="root-1",
        )
        assert result == {"success": True, "message_id": "msg-1"}
        send_mock.assert_awaited_once()
        assert send_mock.await_args.kwargs["metadata"] == {"thread_id": "root-1"}

    @pytest.mark.asyncio
    async def test_ddp_login_uses_resume_token(self, adapter_mod):
        adapter = _make_adapter(adapter_mod)
        adapter._ws = FakeWS()
        future = asyncio.get_running_loop().create_future()
        future.set_result({"result": {"id": "bot-user"}})
        adapter._create_waiter = MagicMock(return_value=future)
        await adapter._ddp_login()
        assert adapter._ws.sent[0]["method"] == "login"
        assert adapter._ws.sent[0]["params"] == [{"resume": "runtime-token"}]

    @pytest.mark.asyncio
    async def test_room_subscription_replay(self, adapter_mod):
        adapter = _make_adapter(adapter_mod)
        adapter._ws = FakeWS()
        adapter._room_cache = {
            "ROOM1": {"_id": "ROOM1", "t": "c"},
            "ROOM2": {"_id": "ROOM2", "t": "d"},
        }
        await adapter._subscribe_all_rooms()
        assert {payload["params"][0] for payload in adapter._ws.sent} == {"ROOM1", "ROOM2"}

    @pytest.mark.asyncio
    async def test_subscription_change_updates_cache_and_subscribes(self, adapter_mod):
        adapter = _make_adapter(adapter_mod)
        adapter._ws = FakeWS()
        await adapter._handle_ddp_frame(
            {
                "msg": "changed",
                "collection": "stream-notify-user",
                "fields": {
                    "eventName": "bot-user/subscriptions-changed",
                    "args": ["inserted", {"rid": "ROOM1", "t": "c", "fname": "Ops"}],
                },
            }
        )
        assert adapter._room_cache["ROOM1"]["t"] == "c"
        assert adapter._ws.sent[0]["name"] == "stream-room-messages"
        assert adapter._ws.sent[0]["params"][0] == "ROOM1"

    @pytest.mark.asyncio
    async def test_non_message_frames_are_ignored(self, adapter_mod):
        adapter = _make_adapter(adapter_mod, extra={"allow_all_users": True})
        adapter._room_cache["ROOM1"] = {"_id": "ROOM1", "t": "c", "fname": "Ops"}
        adapter._download_incoming_attachments = AsyncMock(return_value=([], [], MessageType.TEXT))
        adapter.handle_message = AsyncMock()

        await adapter._handle_ddp_frame(
            {
                "msg": "changed",
                "collection": "stream-room-messages",
                "fields": {"eventName": "ROOM1", "args": ["not-a-message"]},
            }
        )
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_ws_loop_reconnect_backoff(self, adapter_mod, monkeypatch):
        adapter = _make_adapter(adapter_mod)
        attempts = []

        async def fake_connect():
            attempts.append("x")
            if len(attempts) == 1:
                raise RuntimeError("temporary")
            adapter._closing = True

        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)
            adapter._closing = True

        monkeypatch.setattr(adapter, "_ws_connect_and_listen", fake_connect)
        monkeypatch.setattr(adapter_mod.random, "random", lambda: 0.0)
        monkeypatch.setattr(adapter_mod.asyncio, "sleep", fake_sleep)

        await adapter._ws_loop()
        assert len(attempts) == 1
        assert sleeps and sleeps[0] == pytest.approx(2.0)

    def test_expired_auth_token_returns_rebootstrap_hint(self, adapter_mod, tmp_path):
        adapter = _make_adapter(
            adapter_mod,
            extra={"bootstrap_enabled": True, "bootstrap_artifact": str(tmp_path / "artifact.json")},
        )
        adapter._runtime_creds = SimpleNamespace(
            auth_token="expired",
            user_id="bot-user",
            auth_type="auth_token",
        )
        error = adapter._api_error_message(401, {"error": "You must be logged in"})
        assert "Re-run the Rocket.Chat bootstrap flow" in error
