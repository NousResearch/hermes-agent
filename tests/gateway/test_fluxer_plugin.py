"""Tests for the Fluxer platform-plugin adapter.

The Fluxer plugin is intentionally text-first here: REST send + Gateway
MESSAGE_CREATE normalization. Rich media can layer on once Fluxer's self-hosting
surface stabilizes.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_fluxer = load_plugin_adapter("fluxer")

FluxerAdapter = _fluxer.FluxerAdapter
check_requirements = _fluxer.check_requirements
validate_config = _fluxer.validate_config
is_connected = _fluxer.is_connected
register = _fluxer.register
_env_enablement = _fluxer._env_enablement
_build_identify_payload = _fluxer._build_identify_payload


@pytest.fixture(autouse=True)
def _isolate_fluxer_env(monkeypatch):
    """Keep Fluxer unit tests deterministic on dogfood machines with live env."""
    for key in (
        "FLUXER_BASE_URL",
        "FLUXER_BOT_TOKEN",
        "FLUXER_GATEWAY_URL",
        "FLUXER_HOME_CHANNEL",
        "FLUXER_ALLOWED_USERS",
        "FLUXER_ALLOW_ALL_USERS",
        "FLUXER_BACKLOG_RECOVERY",
        "FLUXER_BACKLOG_LIMIT",
        "FLUXER_BACKLOG_BOOTSTRAP_SECONDS",
        "FLUXER_DELIVERY_VERIFICATION",
        "FLUXER_ALLOW_MENTION_EVERYONE",
        "FLUXER_ALLOW_MENTION_ROLES",
        "FLUXER_ALLOW_MENTION_USERS",
        "FLUXER_ALLOW_MENTION_REPLIED_USER",
        "FLUXER_REQUIRE_MENTION",
        "FLUXER_STRICT_MENTION",
        "FLUXER_FREE_RESPONSE_CHANNELS",
        "FLUXER_ALLOWED_CHANNELS",
        "FLUXER_MENTION_PATTERNS",
        "FLUXER_REGISTER_NATIVE_COMMANDS",
        "FLUXER_APPLICATION_ID",
        "FLUXER_NATIVE_COMMAND_GUILDS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_platform_enum_resolves_via_plugin_scan():
    from gateway.config import Platform

    p = Platform("fluxer")
    assert p.value == "fluxer"
    assert Platform("fluxer") is p


def test_check_requirements_needs_token_only(monkeypatch):
    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    assert check_requirements() is False


def test_check_requirements_true_when_token_configured(monkeypatch):
    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.setenv("FLUXER_BOT_TOKEN", "app.secret")

    assert check_requirements() is True


def test_validate_config_uses_env_or_extra(monkeypatch):
    from gateway.config import PlatformConfig

    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    assert validate_config(PlatformConfig(enabled=True)) is False
    assert validate_config(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret"},
        )
    ) is True
    assert validate_config(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    ) is True


def test_is_connected_mirrors_validate(monkeypatch):
    from gateway.config import PlatformConfig

    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    assert is_connected(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret"},
        )
    ) is True
    assert is_connected(PlatformConfig(enabled=True)) is False


def test_env_enablement_none_when_unset(monkeypatch):
    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    assert _env_enablement() is None


def test_env_enablement_seeds_config(monkeypatch):
    monkeypatch.setenv("FLUXER_BASE_URL", "https://fluxer.example/")
    monkeypatch.setenv("FLUXER_BOT_TOKEN", "app.secret")
    monkeypatch.setenv("FLUXER_HOME_CHANNEL", "chan-1")
    monkeypatch.setenv("FLUXER_HOME_CHANNEL_NAME", "Fluxer Home")

    seed = _env_enablement()

    assert seed["base_url"] == "https://fluxer.example/"
    assert seed["bot_token"] == "app.secret"
    assert seed["home_channel"] == {"chat_id": "chan-1", "name": "Fluxer Home"}


def test_adapter_init_and_platform_identity(monkeypatch):
    from gateway.config import Platform, PlatformConfig

    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example/", "bot_token": "app.secret"},
        )
    )

    assert adapter.base_url == "https://fluxer.example"
    assert adapter.api_base_url == "https://fluxer.example/api"
    assert adapter.bot_token == "app.secret"
    assert adapter.platform is Platform("fluxer")
    assert adapter._running is False


def test_adapter_defaults_to_official_hosted_api(monkeypatch):
    from gateway.config import PlatformConfig

    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret"},
        )
    )

    assert adapter.base_url == "https://api.fluxer.app/v1"
    assert adapter.api_base_url == "https://api.fluxer.app/v1"
    assert adapter.bot_token == "app.secret"


def test_build_identify_payload_contains_bot_token_and_client_properties():
    payload = _build_identify_payload("app.secret")

    assert payload["op"] == 2
    assert payload["d"]["token"] == "app.secret"
    assert payload["d"]["properties"]["browser"] == "hermes"
    assert payload["d"]["properties"]["device"] == "hermes"


@pytest.mark.asyncio
async def test_send_posts_channel_message_and_reply_reference(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._request = AsyncMock(side_effect=[{"id": "msg-1"}, {"id": "msg-1", "content": "Hello, Fluxer!"}])

    result = await adapter.send("chan-1", "Hello, Fluxer!", reply_to="msg-0")

    first_call = adapter._request.await_args_list[0]
    assert first_call.args == ("POST", "/channels/chan-1/messages")
    assert first_call.kwargs == {
        "json": {
            "content": "Hello, Fluxer!",
            "allowed_mentions": {"parse": ["users"], "replied_user": True},
            "message_reference": {"message_id": "msg-0"},
        }
    }
    assert adapter._request.await_args_list[1].args == ("GET", "/channels/chan-1/messages/msg-1")
    assert result.success is True
    assert result.message_id == "msg-1"
    assert result.raw_response["responses"][0]["delivery_verified"] is True


@pytest.mark.asyncio
async def test_send_splits_long_channel_messages(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    first_chunk = "a" * 4000
    second_chunk = "a" * 500
    adapter._request = AsyncMock(
        side_effect=[
            {"id": "msg-1"},
            {"id": "msg-1", "content": first_chunk},
            {"id": "msg-2"},
            {"id": "msg-2", "content": second_chunk},
        ]
    )

    result = await adapter.send("chan-1", "a" * 4500, reply_to="msg-0")

    assert result.success is True
    assert result.message_id == "msg-1"
    assert result.raw_response["message_ids"] == ["msg-1", "msg-2"]
    assert adapter._request.await_count == 4
    first = adapter._request.await_args_list[0]
    second = adapter._request.await_args_list[2]
    assert first.kwargs["json"]["message_reference"] == {"message_id": "msg-0"}
    assert len(first.kwargs["json"]["content"]) <= 4000
    assert "message_reference" not in second.kwargs["json"]
    assert len(second.kwargs["json"]["content"]) <= 4000


@pytest.mark.asyncio
async def test_send_neutralizes_global_and_role_mentions_by_default(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._request = AsyncMock(return_value={"id": "msg-1"})

    result = await adapter.send("chan-1", "Careful @everyone <@&123> <@456>")

    assert result.success is True
    payload = adapter._request.await_args_list[0].kwargs["json"]
    assert payload["content"] == "Careful @\u200beveryone <@\u200b&123> <@456>"
    assert payload["allowed_mentions"] == {"parse": ["users"], "replied_user": True}


@pytest.mark.asyncio
async def test_send_can_disable_user_mentions_too(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "base_url": "https://fluxer.example",
                "bot_token": "app.secret",
                "allow_mention_users": False,
            },
        )
    )
    adapter._request = AsyncMock(return_value={"id": "msg-1"})

    result = await adapter.send("chan-1", "Hi <@456> and <@!789>")

    assert result.success is True
    payload = adapter._request.await_args_list[0].kwargs["json"]
    assert payload["content"] == "Hi <@\u200b456> and <@\u200b!789>"
    assert payload["allowed_mentions"] == {"parse": [], "replied_user": True}


@pytest.mark.asyncio
async def test_send_image_file_posts_multipart_payload(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig

    image = tmp_path / "zofka.png"
    image.write_bytes(b"png-data")
    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._multipart_request = AsyncMock(return_value={"id": "msg-img"})

    result = await adapter.send_image_file("chan-1", str(image), caption="look")

    assert result.success is True
    assert result.message_id == "msg-img"
    adapter._multipart_request.assert_awaited_once()
    call = adapter._multipart_request.await_args
    assert call is not None
    _, path = call.args
    kwargs = call.kwargs
    assert path == "/channels/chan-1/messages"
    assert kwargs["payload"]["content"] == "look"
    assert kwargs["payload"]["allowed_mentions"] == {"parse": ["users"], "replied_user": True}
    assert kwargs["payload"]["attachments"] == [{"id": 0, "filename": "zofka.png", "title": "zofka.png"}]
    assert kwargs["files"] == [("files[0]", image.resolve(), "zofka.png")]


@pytest.mark.asyncio
async def test_send_voice_marks_fluxer_voice_message(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig

    audio = tmp_path / "reply.mp3"
    audio.write_bytes(b"mp3-data")
    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._multipart_request = AsyncMock(return_value={"id": "msg-voice"})

    result = await adapter.send_voice("chan-1", str(audio), caption="spoken", duration=3, waveform="AAAA")

    assert result.success is True
    call = adapter._multipart_request.await_args
    assert call is not None
    kwargs = call.kwargs
    assert "content" not in kwargs["payload"]
    assert kwargs["payload"]["allowed_mentions"] == {"parse": ["users"], "replied_user": True}
    assert kwargs["payload"]["flags"] == 1 << 13
    assert kwargs["payload"]["attachments"] == [
        {"id": 0, "filename": "reply.mp3", "title": "reply.mp3", "duration": 3, "waveform": "AAAA"}
    ]


@pytest.mark.asyncio
async def test_standalone_send_routes_audio_as_voice(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig

    audio = tmp_path / "reply.mp3"
    audio.write_bytes(b"mp3-data")
    pconfig = PlatformConfig(
        enabled=True,
        extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
    )

    with patch.object(FluxerAdapter, "send_voice", new=AsyncMock(return_value=_fluxer.SendResult(success=True, message_id="msg-voice"))) as send_voice:
        result = await _fluxer._standalone_send(pconfig, "chan-1", "", media_files=[str(audio)])

    assert result == {"success": True, "platform": "fluxer", "chat_id": "chan-1", "message_id": "msg-voice"}
    send_voice.assert_awaited_once()
    call = send_voice.await_args
    assert call is not None
    assert call.args[:2] == ("chan-1", str(audio))


@pytest.mark.asyncio
async def test_send_reports_retryable_error_on_request_failure(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._request = AsyncMock(side_effect=RuntimeError("network down"))

    result = await adapter.send("chan-1", "Hello")

    assert result.success is False
    assert "network down" in result.error
    assert result.retryable is True


@pytest.mark.asyncio
async def test_list_channels_enumerates_guild_channels_and_active_threads(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://api.fluxer.test/v1", "bot_token": "app.secret"})
    )
    adapter._request = AsyncMock(
        side_effect=[
            [{"id": "guild-1", "name": "Guild One"}],
            [
                {"id": "cat-1", "name": "Ops", "type": 4},
                {"id": "chan-1", "name": "general", "type": 0},
                {"id": "forum-1", "name": "ideas", "type": 15},
            ],
            {"threads": [{"id": "thread-1", "name": "Native thread", "parent_id": "chan-1", "type": 11}]},
        ]
    )

    channels = await adapter.list_channels()

    assert channels == [
        {"id": "chan-1", "name": "general", "guild": "Guild One", "guild_id": "guild-1", "type": "channel"},
        {"id": "forum-1", "name": "ideas", "guild": "Guild One", "guild_id": "guild-1", "type": "forum"},
        {"id": "thread-1", "name": "Native thread", "guild": "Guild One", "guild_id": "guild-1", "type": "thread", "parent_id": "chan-1"},
    ]
    assert adapter._request.await_args_list[0].args == ("GET", "/users/@me/guilds")
    assert adapter._request.await_args_list[1].args == ("GET", "/guilds/guild-1/channels")
    assert adapter._request.await_args_list[2].args == ("GET", "/guilds/guild-1/threads/active")
    assert adapter._request.await_args_list[2].kwargs == {"warn_on_error": False}


@pytest.mark.asyncio
async def test_list_channels_treats_active_thread_listing_as_optional(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://api.fluxer.test/v1", "bot_token": "app.secret"})
    )
    adapter._request = AsyncMock(
        side_effect=[
            [{"id": "guild-1", "name": "Guild One"}],
            [{"id": "chan-1", "name": "general", "type": 0}],
            RuntimeError("404 Not found"),
        ]
    )

    channels = await adapter.list_channels()

    assert channels == [
        {"id": "chan-1", "name": "general", "guild": "Guild One", "guild_id": "guild-1", "type": "channel"},
    ]
    assert adapter._request.await_args_list[2].args == ("GET", "/guilds/guild-1/threads/active")
    assert adapter._request.await_args_list[2].kwargs == {"warn_on_error": False}


@pytest.mark.asyncio
async def test_request_can_suppress_expected_rest_warning(monkeypatch, caplog):
    import httpx
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://api.fluxer.test/v1", "bot_token": "app.secret"})
    )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def request(self, method, url, headers=None, **kwargs):
            request = httpx.Request(method, url)
            return httpx.Response(404, json={"message": "Not found."}, request=request)

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    with caplog.at_level("WARNING"):
        with pytest.raises(httpx.HTTPStatusError):
            await adapter._request("GET", "/guilds/guild-1/threads/active", warn_on_error=False)

    assert "Fluxer REST GET" not in caplog.text


@pytest.mark.asyncio
async def test_create_thread_from_message_uses_fluxer_thread_route(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://api.fluxer.test/v1", "bot_token": "app.secret"})
    )
    adapter._request = AsyncMock(return_value={"id": "thread-1", "name": "Question"})

    result = await adapter.create_thread("chan-1", "Question", message_id="msg-1")

    assert result["id"] == "thread-1"
    adapter._request.assert_awaited_once_with(
        "POST",
        "/channels/chan-1/messages/msg-1/threads",
        json={"name": "Question", "auto_archive_duration": 1440, "rate_limit_per_user": 0},
    )


@pytest.mark.asyncio
async def test_send_exec_approval_posts_native_buttons_and_tracks_pending(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret", "allowed_users": "user-1"},
        )
    )
    adapter._request = AsyncMock(return_value={"id": "approval-1"})

    result = await adapter.send_exec_approval(
        "chan-1",
        "rm -rf /important",
        "agent:main:fluxer:channel:chan-1",
        description="dangerous deletion",
    )

    assert result.success is True
    assert result.message_id == "approval-1"
    assert "approval-1" in adapter._pending_exec_approvals
    first_call = adapter._request.await_args_list[0]
    assert first_call.args == ("POST", "/channels/chan-1/messages")
    payload = first_call.kwargs["json"]
    assert "rm -rf /important" in payload["content"]
    buttons = payload["components"][0]["components"]
    assert [button["label"] for button in buttons] == ["Allow once", "Session", "Always", "Deny"]
    assert [button["style"] for button in buttons] == [3, 1, 2, 4]
    assert len(adapter._pending_component_actions) == 4
    assert {state["choice"] for state in adapter._pending_component_actions.values()} == {"once", "session", "always", "deny"}


@pytest.mark.asyncio
async def test_component_interaction_resolves_pending_exec_approval(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret", "allowed_users": "user-1"},
        )
    )
    adapter._pending_exec_approvals["approval-1"] = {
        "session_key": "session-1",
        "channel_id": "chan-1",
        "content": "approval text",
        "resolved": False,
    }
    adapter._pending_component_actions["approve-session"] = {
        "kind": "exec_approval",
        "message_id": "approval-1",
        "session_key": "session-1",
        "channel_id": "chan-1",
        "choice": "session",
    }
    resolved = []
    monkeypatch.setattr("tools.approval.resolve_gateway_approval", lambda session_key, choice: resolved.append((session_key, choice)) or 1)
    adapter._request = AsyncMock(return_value={})

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "INTERACTION_CREATE",
            "d": {
                "id": "interaction-1",
                "token": "tok-1",
                "type": 3,
                "channel_id": "chan-1",
                "message": {"id": "approval-1"},
                "data": {"custom_id": "approve-session", "component_type": 2},
                "member": {"user": {"id": "user-1", "bot": False}},
            },
        }
    )

    assert resolved == [("session-1", "session")]
    assert "approval-1" not in adapter._pending_exec_approvals
    assert "approve-session" not in adapter._pending_component_actions
    call = adapter._request.await_args
    assert call is not None
    assert call.args == ("POST", "/interactions/interaction-1/tok-1/callback")


@pytest.mark.asyncio
async def test_component_interaction_ignores_unauthorized_without_consuming_action(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret", "allowed_users": "user-1"},
        )
    )
    adapter._pending_component_actions["approve-session"] = {
        "kind": "exec_approval",
        "message_id": "approval-1",
        "session_key": "session-1",
        "channel_id": "chan-1",
        "choice": "session",
    }

    await adapter._handle_component_interaction(
        {
            "id": "interaction-1",
            "token": "tok-1",
            "type": 3,
            "channel_id": "chan-1",
            "message": {"id": "approval-1"},
            "data": {"custom_id": "approve-session", "component_type": 2},
            "member": {"user": {"id": "intruder", "bot": False}},
        }
    )

    assert "approve-session" in adapter._pending_component_actions


@pytest.mark.asyncio
async def test_component_interaction_rejects_mismatched_message_or_channel(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret", "allowed_users": "user-1"},
        )
    )
    adapter._pending_component_actions["approve-session"] = {
        "kind": "exec_approval",
        "message_id": "approval-1",
        "session_key": "session-1",
        "channel_id": "chan-1",
        "choice": "session",
    }

    await adapter._handle_component_interaction(
        {
            "id": "interaction-1",
            "token": "tok-1",
            "type": 3,
            "channel_id": "other-chan",
            "message": {"id": "approval-1"},
            "data": {"custom_id": "approve-session", "component_type": 2},
            "member": {"user": {"id": "user-1", "bot": False}},
        }
    )

    assert "approve-session" in adapter._pending_component_actions


@pytest.mark.asyncio
async def test_attachment_download_omits_auth_for_off_origin_urls(monkeypatch):
    import httpx
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://api.fluxer.test/v1", "bot_token": "app.secret"})
    )
    captured = {}

    class FakeResponse:
        content = b"ok"

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers=None):
            captured["url"] = url
            captured["headers"] = headers or {}
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    data = await adapter._download_attachment_bytes("https://example.com/file.txt")

    assert data == b"ok"
    assert "Authorization" not in captured["headers"]


@pytest.mark.asyncio
async def test_unsafe_attachment_url_is_dropped(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://api.fluxer.test/v1", "bot_token": "app.secret"})
    )

    cached, content_type = await adapter._cache_attachment(
        {"url": "http://127.0.0.1:8080/secret.txt", "filename": "secret.txt", "content_type": "text/plain"}
    )

    assert cached is None
    assert content_type is None


@pytest.mark.asyncio
async def test_gateway_hello_starts_heartbeat_even_before_mark_connected(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"})
    )

    class FakeWS:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

    adapter._ws = FakeWS()
    monkeypatch.setattr(asyncio, "create_task", lambda coro, name=None: asyncio.ensure_future(coro))

    await adapter._handle_gateway_dispatch({"op": 10, "d": {"heartbeat_interval": 60000}})
    await asyncio.sleep(0)

    assert adapter._heartbeat_task is not None
    assert not adapter._heartbeat_task.done()
    adapter._heartbeat_task.cancel()


@pytest.mark.asyncio
async def test_register_native_commands_bulk_upserts_guild_slash_commands(monkeypatch):
    from gateway.config import PlatformConfig
    from hermes_cli.commands import CommandDef

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"})
    )
    adapter._request = AsyncMock(return_value=[{"id": "cmd-1"}])

    result = await adapter.register_native_commands(
        "app-1",
        guild_id="guild-1",
        commands=[CommandDef("model", "Choose model", "Config", args_hint="<model>")],
    )

    assert result == [{"id": "cmd-1"}]
    adapter._request.assert_awaited_once_with(
        "PUT",
        "/applications/app-1/guilds/guild-1/commands",
        json=[
            {
                "name": "model",
                "description": "Choose model",
                "type": 1,
                "options": [{"type": 3, "name": "args", "description": "<model>", "required": False}],
            }
        ],
    )


@pytest.mark.asyncio
async def test_thread_create_event_tracks_known_thread_channel(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"})
    )

    await adapter._handle_gateway_dispatch({"op": 0, "t": "THREAD_CREATE", "d": {"id": "thread-1", "parent_id": "chan-1", "name": "Native"}})

    assert "thread-1" in adapter._known_channel_ids


@pytest.mark.asyncio
async def test_send_slash_confirm_posts_native_buttons(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"})
    )
    adapter._request = AsyncMock(return_value={"id": "confirm-msg"})

    result = await adapter.send_slash_confirm(
        "chan-1",
        "Reload MCP?",
        "This reload may invalidate caches.",
        "session-1",
        "confirm-1",
    )

    assert result.success is True
    call = adapter._request.await_args
    assert call is not None
    payload = call.kwargs["json"]
    assert "Reload MCP?" in payload["content"]
    assert [button["label"] for button in payload["components"][0]["components"]] == ["Approve once", "Always approve", "Cancel"]
    assert {state["kind"] for state in adapter._pending_component_actions.values()} == {"slash_confirm"}


@pytest.mark.asyncio
async def test_application_command_interaction_dispatches_slash_text(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"})
    )
    adapter._request = AsyncMock(return_value={})
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "INTERACTION_CREATE",
            "d": {
                "id": "interaction-1",
                "token": "tok-1",
                "type": 2,
                "channel_id": "chan-1",
                "guild_id": "guild-1",
                "channel": {"id": "chan-1", "name": "bot-home", "type": 0},
                "member": {"user": {"id": "user-1", "username": "Elkim", "bot": False}},
                "data": {"name": "model", "options": [{"name": "model", "value": "gpt-5.5"}]},
            },
        }
    )

    assert [event.text for event in seen] == ["/model gpt-5.5"]
    assert seen[0].source.chat_id == "chan-1"
    adapter._request.assert_awaited_once_with(
        "POST",
        "/interactions/interaction-1/tok-1/callback",
        json={"type": 5, "data": {"flags": 64}},
    )


@pytest.mark.asyncio
async def test_reaction_add_resolves_pending_exec_approval(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret", "allowed_users": "user-1"},
        )
    )
    adapter._pending_exec_approvals["approval-1"] = {
        "session_key": "session-1",
        "channel_id": "chan-1",
        "content": "approval text",
        "resolved": False,
    }
    resolved = []
    monkeypatch.setattr("tools.approval.resolve_gateway_approval", lambda session_key, choice: resolved.append((session_key, choice)) or 1)
    adapter.edit_message = AsyncMock(return_value=type("Result", (), {"success": True})())

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_REACTION_ADD",
            "d": {
                "message_id": "approval-1",
                "channel_id": "chan-1",
                "user_id": "user-1",
                "emoji": {"name": "✅"},
                "member": {"user": {"id": "user-1", "bot": False}},
            },
        }
    )

    assert resolved == [("session-1", "once")]
    assert "approval-1" not in adapter._pending_exec_approvals
    adapter.edit_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_message_update_refreshes_pending_exec_approval_content(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._pending_exec_approvals["approval-1"] = {
        "session_key": "session-1",
        "channel_id": "chan-1",
        "content": "old approval text",
        "resolved": False,
    }

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_UPDATE",
            "d": {"id": "approval-1", "content": "edited approval text"},
        }
    )

    assert adapter._pending_exec_approvals["approval-1"]["content"] == "edited approval text"


@pytest.mark.asyncio
async def test_message_delete_denies_pending_exec_approval(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._seen_message_ids.add("approval-1")
    adapter._pending_exec_approvals["approval-1"] = {
        "session_key": "session-1",
        "channel_id": "chan-1",
        "content": "approval text",
        "resolved": False,
    }
    resolved = []
    monkeypatch.setattr("tools.approval.resolve_gateway_approval", lambda session_key, choice: resolved.append((session_key, choice)) or 1)

    await adapter._handle_gateway_dispatch(
        {"op": 0, "t": "MESSAGE_DELETE", "d": {"id": "approval-1", "channel_id": "chan-1"}}
    )

    assert resolved == [("session-1", "deny")]
    assert "approval-1" not in adapter._pending_exec_approvals
    assert "approval-1" not in adapter._seen_message_ids


@pytest.mark.asyncio
async def test_message_delete_for_non_approval_only_clears_seen_id(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._seen_message_ids.add("msg-1")
    resolved = []
    monkeypatch.setattr("tools.approval.resolve_gateway_approval", lambda session_key, choice: resolved.append((session_key, choice)) or 1)

    await adapter._handle_gateway_dispatch(
        {"op": 0, "t": "MESSAGE_DELETE", "d": {"id": "msg-1", "channel_id": "chan-1"}}
    )

    assert resolved == []
    assert "msg-1" not in adapter._seen_message_ids


@pytest.mark.asyncio
async def test_reaction_add_ignores_unauthorized_or_bot_users(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret", "allowed_users": "user-1"},
        )
    )
    adapter._pending_exec_approvals["approval-1"] = {"session_key": "session-1", "channel_id": "chan-1", "resolved": False}
    resolved = []
    monkeypatch.setattr("tools.approval.resolve_gateway_approval", lambda session_key, choice: resolved.append((session_key, choice)) or 1)

    await adapter._handle_reaction_add(
        {
            "message_id": "approval-1",
            "channel_id": "chan-1",
            "user_id": "user-2",
            "emoji": {"name": "❌"},
            "member": {"user": {"id": "user-2", "bot": False}},
        }
    )
    await adapter._handle_reaction_add(
        {
            "message_id": "approval-1",
            "channel_id": "chan-1",
            "user_id": "user-1",
            "emoji": {"name": "❌"},
            "member": {"user": {"id": "user-1", "bot": True}},
        }
    )

    assert resolved == []
    assert "approval-1" in adapter._pending_exec_approvals


@pytest.mark.asyncio
async def test_edit_message_uses_fluxer_patch_endpoint(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._request = AsyncMock(side_effect=[{"id": "msg-1", "content": "updated"}, {"id": "msg-1", "content": "updated"}])

    result = await adapter.edit_message("chan-1", "msg-1", "updated")

    assert result.success is True
    assert result.message_id == "msg-1"
    first_call = adapter._request.await_args_list[0]
    assert first_call.args == ("PATCH", "/channels/chan-1/messages/msg-1")
    assert first_call.kwargs == {"json": {"content": "updated", "allowed_mentions": {"parse": ["users"], "replied_user": True}}}
    assert adapter._request.await_args_list[1].args == ("GET", "/channels/chan-1/messages/msg-1")
    assert result.raw_response["delivery_verified"] is True


@pytest.mark.asyncio
async def test_edit_message_truncates_to_fluxer_limit(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._request = AsyncMock(side_effect=[{"id": "msg-1"}, {"id": "msg-1", "content": "a" * 3997 + "..."}])

    result = await adapter.edit_message("chan-1", "msg-1", "a" * 4500)

    assert result.success is True
    call = adapter._request.await_args_list[0]
    payload = call.kwargs["json"]
    assert len(payload["content"]) == 4000
    assert payload["content"].endswith("...")


@pytest.mark.asyncio
async def test_delete_message_uses_fluxer_delete_endpoint(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._request = AsyncMock(return_value={})

    assert await adapter.delete_message("chan-1", "msg-1") is True
    adapter._request.assert_awaited_once_with("DELETE", "/channels/chan-1/messages/msg-1")


@pytest.mark.asyncio
async def test_pin_helpers_use_fluxer_discord_shaped_pin_routes(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://api.fluxer.test/v1", "bot_token": "app.secret"})
    )
    adapter._request = AsyncMock(side_effect=[{"items": [{"id": "msg-1"}], "has_more": False}, {}, {}])

    pins = await adapter.list_pinned_messages("chan-1")
    pinned = await adapter.pin_message("chan-1", "msg-1")
    unpinned = await adapter.unpin_message("chan-1", "msg-1")

    assert pins == [{"id": "msg-1"}]
    assert pinned is True
    assert unpinned is True
    assert adapter._request.await_args_list[0].args == ("GET", "/channels/chan-1/messages/pins")
    assert adapter._request.await_args_list[1].args == ("PUT", "/channels/chan-1/messages/pins/msg-1")
    assert adapter._request.await_args_list[2].args == ("DELETE", "/channels/chan-1/messages/pins/msg-1")


@pytest.mark.asyncio
async def test_list_pinned_messages_accepts_bare_list_response(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"base_url": "https://api.fluxer.test/v1", "bot_token": "app.secret"})
    )
    adapter._request = AsyncMock(return_value=[{"id": "msg-1"}])

    assert await adapter.list_pinned_messages("chan-1") == [{"id": "msg-1"}]


@pytest.mark.asyncio
async def test_send_typing_starts_persistent_fluxer_typing_loop(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._request = AsyncMock(return_value={})
    sleep_calls = 0

    async def fake_sleep(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError()

    with patch("asyncio.sleep", new=fake_sleep):
        await adapter.send_typing("chan-1")
        task = adapter._typing_tasks.get("chan-1")
        assert task is not None
        await task

    assert adapter._request.await_count == 2
    adapter._request.assert_any_await("POST", "/channels/chan-1/typing", json={})
    assert "chan-1" not in adapter._typing_tasks


@pytest.mark.asyncio
async def test_send_typing_dedupes_and_stop_typing_cancels(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    blocker = asyncio.Event()
    adapter._request = AsyncMock(return_value={})

    real_sleep = asyncio.sleep

    async def fake_sleep(_seconds):
        await blocker.wait()

    with patch("asyncio.sleep", new=fake_sleep):
        await adapter.send_typing("chan-1")
        first_task = adapter._typing_tasks.get("chan-1")
        await adapter.send_typing("chan-1")
        assert adapter._typing_tasks.get("chan-1") is first_task
        await real_sleep(0)
        await adapter.stop_typing("chan-1")

    assert "chan-1" not in adapter._typing_tasks


@pytest.mark.asyncio
async def test_backlog_recovery_fetches_home_channel_and_dispatches_recent_messages(monkeypatch):
    from gateway.config import HomeChannel, Platform, PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            home_channel=HomeChannel(platform=Platform("fluxer"), chat_id="chan-1", name="Fluxer Home"),
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle
    adapter._request = AsyncMock(
        return_value={
            "messages": [
                {
                    "id": "old-msg",
                    "content": "too old",
                    "created_at": "2026-06-01T09:59:00Z",
                    "author": {"id": "user-1", "username": "Elkim", "bot": False},
                },
                {
                    "id": "new-msg",
                    "content": "recover me",
                    "created_at": "2026-06-01T10:00:30Z",
                    "author": {"id": "user-1", "username": "Elkim", "bot": False},
                },
            ]
        }
    )

    await adapter._recover_backlog(cutoff=_fluxer._parse_fluxer_timestamp("2026-06-01T10:00:00Z"))

    adapter._request.assert_awaited_once_with("GET", "/channels/chan-1/messages", params={"limit": 25})
    assert [event.message_id for event in seen] == ["new-msg"]
    assert seen[0].text == "recover me"
    assert seen[0].source.chat_id == "chan-1"


@pytest.mark.asyncio
async def test_backlog_recovery_dedupes_seen_messages(monkeypatch):
    from gateway.config import HomeChannel, Platform, PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            home_channel=HomeChannel(platform=Platform("fluxer"), chat_id="chan-1", name="Fluxer Home"),
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._seen_message_ids.add("msg-1")
    adapter.handle_message = AsyncMock()
    adapter._request = AsyncMock(
        return_value=[
            {
                "id": "msg-1",
                "channel_id": "chan-1",
                "content": "already handled",
                "created_at": "2026-06-01T10:00:30Z",
                "author": {"id": "user-1", "username": "Elkim", "bot": False},
            }
        ]
    )

    await adapter._recover_backlog(cutoff=_fluxer._parse_fluxer_timestamp("2026-06-01T10:00:00Z"))

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_hello_identifies_and_sends_immediate_heartbeat(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )

    class FakeWS:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

    ws = FakeWS()
    adapter._ws = ws
    adapter._mark_connected()
    monkeypatch.setattr(asyncio, "create_task", lambda coro, name=None: asyncio.ensure_future(coro))

    await adapter._handle_gateway_dispatch({"op": 10, "d": {"heartbeat_interval": 60000}})

    assert len(ws.sent) == 2
    assert '"op": 2' in ws.sent[0]
    assert '"op": 1' in ws.sent[1]
    assert adapter._awaiting_heartbeat_ack is True
    assert adapter._heartbeat_task is not None
    adapter._heartbeat_task.cancel()


@pytest.mark.asyncio
async def test_gateway_server_heartbeat_request_sends_heartbeat(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )

    class FakeWS:
        def __init__(self):
            self.send = AsyncMock()

    ws = FakeWS()
    adapter._ws = ws
    adapter._last_seq = 7

    await adapter._handle_gateway_dispatch({"op": 1})

    ws.send.assert_awaited_once()
    sent_payload = ws.send.await_args_list[0].args[0]
    assert '"op": 1' in sent_payload
    assert '"d": 7' in sent_payload


@pytest.mark.asyncio
async def test_gateway_heartbeat_ack_clears_pending_state(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._awaiting_heartbeat_ack = True

    await adapter._handle_gateway_dispatch({"op": 11})

    assert adapter._awaiting_heartbeat_ack is False
    assert adapter._last_heartbeat_ack_at is not None


@pytest.mark.asyncio
async def test_gateway_reconnect_opcode_closes_websocket(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    class FakeWS:
        def __init__(self):
            self.close = AsyncMock()

    ws = FakeWS()
    adapter._ws = ws

    await adapter._handle_gateway_dispatch({"op": 7, "d": {"reason": "server reconnect"}})

    ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_listener_schedules_reconnect_on_clean_websocket_close(monkeypatch):
    from gateway.config import PlatformConfig

    class EmptyWS:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._ws = EmptyWS()
    adapter._mark_connected()
    scheduled = []
    adapter._schedule_reconnect = lambda reason: scheduled.append(reason)

    await adapter._listen_loop()

    assert adapter._running is False
    assert scheduled == ["websocket closed"]


@pytest.mark.asyncio
async def test_send_message_tool_routes_fluxer_media_to_standalone_even_with_live_runner(monkeypatch):
    from gateway.config import Platform, PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry
    from tools.send_message_tool import _send_via_adapter

    platform = Platform("fluxer")
    pconfig = PlatformConfig(
        enabled=True,
        extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
    )
    live_adapter = AsyncMock()
    live_adapter.send = AsyncMock(return_value=type("Result", (), {"success": True, "message_id": "live", "error": None})())
    runner = type("Runner", (), {"adapters": {platform: live_adapter}})()
    standalone = AsyncMock(return_value={"success": True, "message_id": "standalone"})

    original = platform_registry.get("fluxer")
    platform_registry.register(
        PlatformEntry(
            name="fluxer",
            label="Fluxer",
            adapter_factory=lambda cfg: FluxerAdapter(cfg),
            check_fn=lambda: True,
            standalone_sender_fn=standalone,
            max_message_length=4000,
        )
    )
    try:
        with patch("gateway.run._gateway_runner_ref", return_value=runner):
            result = await _send_via_adapter(
                platform,
                pconfig,
                "chan-1",
                "",
                media_files=[("/tmp/reply.ogg", True)],
            )
    finally:
        if original is not None:
            platform_registry.register(original)
        else:
            platform_registry.unregister("fluxer")

    assert result == {"success": True, "message_id": "standalone"}
    live_adapter.send.assert_not_awaited()
    standalone.assert_awaited_once_with(
        pconfig,
        "chan-1",
        "",
        thread_id=None,
        media_files=[("/tmp/reply.ogg", True)],
        force_document=False,
    )


@pytest.mark.asyncio
async def test_message_create_dispatches_normalized_event(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_CREATE",
            "s": 42,
            "d": {
                "id": "msg-1",
                "channel_id": "chan-1",
                "channel_type": "dm",
                "content": "morning",
                "author": {"id": "user-1", "username": "Elkim", "bot": False},
                "guild_id": None,
            },
        }
    )

    assert len(seen) == 1
    event = seen[0]
    assert event.text == "morning"
    assert event.message_id == "msg-1"
    assert event.source.chat_id == "chan-1"
    assert event.source.chat_type == "dm"
    assert event.source.user_id == "user-1"
    assert event.source.user_name == "Elkim"
    assert event.source.message_id == "msg-1"


@pytest.mark.asyncio
async def test_message_create_classifies_numeric_thread_channels_and_remembers_thread_mention(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret", "bot_user_id": "bot-1"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_CREATE",
            "d": {
                "id": "msg-1",
                "channel_id": "thread-1",
                "channel_type": 11,
                "content": "<@bot-1> continue here",
                "author": {"id": "user-1", "username": "Elkim", "bot": False},
            },
        }
    )
    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_CREATE",
            "d": {
                "id": "msg-2",
                "channel_id": "thread-1",
                "channel_type": 11,
                "content": "follow-up without mention",
                "author": {"id": "user-1", "username": "Elkim", "bot": False},
            },
        }
    )

    assert [event.source.chat_type for event in seen] == ["thread", "thread"]
    assert [event.text for event in seen] == ["continue here", "follow-up without mention"]


@pytest.mark.asyncio
async def test_channel_message_requires_direct_mention_by_default(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret", "bot_user_id": "bot-1"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_CREATE",
            "d": {
                "id": "msg-quiet",
                "channel_id": "chan-1",
                "channel_type": "channel",
                "content": "I think Hermes mentioned that yesterday",
                "author": {"id": "user-1", "username": "Elkim", "bot": False},
            },
        }
    )

    assert seen == []


@pytest.mark.asyncio
async def test_channel_message_processes_and_strips_direct_address(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret", "bot_user_id": "bot-1"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_CREATE",
            "d": {
                "id": "msg-mention",
                "channel_id": "chan-1",
                "channel_type": "channel",
                "content": "Hermes, check this please",
                "author": {"id": "user-1", "username": "Elkim", "bot": False},
            },
        }
    )

    assert len(seen) == 1
    assert seen[0].text == "check this please"


@pytest.mark.asyncio
async def test_home_channel_is_free_response_even_with_require_mention(monkeypatch):
    from gateway.config import HomeChannel, Platform, PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            home_channel=HomeChannel(platform=Platform("fluxer"), chat_id="chan-1", name="Fluxer Home"),
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_CREATE",
            "d": {
                "id": "msg-home",
                "channel_id": "chan-1",
                "channel_type": "channel",
                "content": "continue here",
                "author": {"id": "user-1", "username": "Elkim", "bot": False},
            },
        }
    )

    assert len(seen) == 1
    assert seen[0].text == "continue here"


@pytest.mark.asyncio
async def test_message_create_dispatches_image_attachment(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import MessageType

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    with patch.object(
        _fluxer,
        "cache_image_from_url",
        new=AsyncMock(return_value="/tmp/fluxer-image.jpg"),
    ) as cache_image:
        await adapter._handle_gateway_dispatch(
            {
                "op": 0,
                "t": "MESSAGE_CREATE",
                "s": 43,
                "d": {
                    "id": "msg-img",
                    "channel_id": "chan-1",
                    "channel_type": "dm",
                    "content": "",
                    "author": {"id": "user-1", "username": "Elkim", "bot": False},
                    "attachments": [
                        {
                            "id": "att-1",
                            "filename": "photo.jpg",
                            "content_type": "image/jpeg",
                            "url": "https://fluxerusercontent.com/attachments/chan-1/att-1/photo.jpg",
                        }
                    ],
                },
            }
        )

    assert len(seen) == 1
    event = seen[0]
    assert event.text == ""
    assert event.message_type == MessageType.PHOTO
    assert event.media_urls == ["/tmp/fluxer-image.jpg"]
    assert event.media_types == ["image/jpeg"]
    cache_image.assert_awaited_once_with(
        "https://fluxerusercontent.com/attachments/chan-1/att-1/photo.jpg",
        ".jpg",
    )


@pytest.mark.asyncio
async def test_message_create_dispatches_voice_attachment_when_flagged(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import MessageType

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    with patch.object(
        _fluxer,
        "cache_audio_from_url",
        new=AsyncMock(return_value="/tmp/fluxer-voice.mp3"),
    ):
        await adapter._handle_gateway_dispatch(
            {
                "op": 0,
                "t": "MESSAGE_CREATE",
                "s": 44,
                "d": {
                    "id": "msg-voice",
                    "channel_id": "chan-1",
                    "channel_type": "dm",
                    "content": "",
                    "flags": 1 << 13,
                    "author": {"id": "user-1", "username": "Elkim", "bot": False},
                    "attachments": [
                        {
                            "id": "att-voice",
                            "filename": "voice.mp3",
                            "content_type": "audio/mpeg",
                            "url": "https://fluxerusercontent.com/attachments/chan-1/att-voice/voice.mp3",
                        }
                    ],
                },
            }
        )

    assert len(seen) == 1
    event = seen[0]
    assert event.message_type == MessageType.VOICE
    assert event.media_urls == ["/tmp/fluxer-voice.mp3"]
    assert event.media_types == ["audio/mpeg"]


@pytest.mark.asyncio
async def test_message_create_includes_embedded_reply_context(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_CREATE",
            "d": {
                "id": "msg-reply",
                "channel_id": "chan-1",
                "channel_type": "dm",
                "content": "yes, do this",
                "author": {"id": "user-1", "username": "Elkim", "bot": False},
                "referenced_message": {"id": "msg-parent", "content": "Please check this file"},
            },
        }
    )

    assert len(seen) == 1
    assert seen[0].reply_to_message_id == "msg-parent"
    assert seen[0].reply_to_text == "Please check this file"


@pytest.mark.asyncio
async def test_message_create_fetches_reply_context_from_message_reference(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle
    adapter._request = AsyncMock(return_value={"id": "msg-parent", "content": "Original quoted text"})

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_CREATE",
            "d": {
                "id": "msg-reply",
                "channel_id": "chan-1",
                "channel_type": "dm",
                "content": "that one",
                "author": {"id": "user-1", "username": "Elkim", "bot": False},
                "message_reference": {"message_id": "msg-parent"},
            },
        }
    )

    adapter._request.assert_awaited_once_with("GET", "/channels/chan-1/messages/msg-parent")
    assert len(seen) == 1
    assert seen[0].reply_to_message_id == "msg-parent"
    assert seen[0].reply_to_text == "Original quoted text"


@pytest.mark.asyncio
async def test_message_create_ignores_own_bot_messages(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter.bot_user_id = "bot-1"
    adapter.handle_message = AsyncMock()

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_CREATE",
            "d": {
                "id": "msg-1",
                "channel_id": "chan-1",
                "content": "echo",
                "author": {"id": "bot-1", "username": "Žofka", "bot": True},
            },
        }
    )

    adapter.handle_message.assert_not_awaited()


def test_register_metadata():
    calls = []

    class Ctx:
        def register_platform(self, **kwargs):
            calls.append(kwargs)

    register(Ctx())

    assert len(calls) == 1
    entry = calls[0]
    assert entry["name"] == "fluxer"
    assert entry["label"] == "Fluxer"
    assert entry["required_env"] == ["FLUXER_BOT_TOKEN"]
    assert entry["allowed_users_env"] == "FLUXER_ALLOWED_USERS"
    assert entry["allow_all_env"] == "FLUXER_ALLOW_ALL_USERS"
    assert entry["cron_deliver_env_var"] == "FLUXER_HOME_CHANNEL"
    assert entry["standalone_sender_fn"] is not None
    assert entry["max_message_length"] >= 2000
