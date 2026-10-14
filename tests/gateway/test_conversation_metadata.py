"""Conversation metadata must prove kind and preserve legacy key layouts."""

import pytest


@pytest.mark.asyncio
async def test_legacy_feishu_kind_cannot_come_from_a_failed_metadata_lookup():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from gateway.conversation import resolve_delivery_source
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(chat=SimpleNamespace(get=lambda: None))))
    adapter._chat_info_cache = {}
    adapter._build_get_chat_request = lambda chat_id: object()
    adapter._run_blocking = AsyncMock(return_value=SimpleNamespace(success=lambda: False, code=429, msg="rate limited"))
    with pytest.raises(ValueError):
        await resolve_delivery_source(adapter, "feishu", "oc_group", None, {"platform": "feishu", "chat_id": "oc_group", "user_id": "alice"})


def test_line_room_origin_is_capturable():
    from plugins.platforms.line.adapter import _resolve_chat
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.cronjob_job_args import _origin_from_env

    chat_id, kind = _resolve_chat({"type": "room", "roomId": "R123", "userId": "Ualice"})
    source = SessionSource(Platform("line"), chat_id, chat_type=kind, user_id="Ualice")
    tokens = set_session_vars(platform="line", chat_id=chat_id, chat_type=kind, user_id="Ualice", routing_source=source)
    try:
        assert _origin_from_env()["conversation"]["kind"] in {"group", "channel"}
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
async def test_normalized_feishu_forum_resolves_to_actual_inbound_key():
    from plugins.platforms.feishu.adapter import FeishuAdapter
    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key
    from gateway.conversation import conversation_origin, resolve_delivery_source

    kind = FeishuAdapter._resolve_source_chat_type(chat_info={"type": "forum"}, event_chat_type="group")
    inbound = SessionSource(Platform.FEISHU, "oc_forum", chat_type=kind, user_id="alice", thread_id="om_root")
    origin = {**inbound.to_dict(), "conversation": conversation_origin(inbound, thread_id="om_root")}
    resolved = await resolve_delivery_source(FeishuAdapter.__new__(FeishuAdapter), "feishu", "oc_forum", "om_root", origin)
    assert build_session_key(resolved) == build_session_key(inbound)


@pytest.mark.asyncio
@pytest.mark.parametrize("module,name,state", [
    ("plugins.platforms.feishu.adapter", "FeishuAdapter", {"_client": None}),
    ("plugins.platforms.whatsapp.adapter", "WhatsAppAdapter", {"_running": False}),
    ("plugins.platforms.discord.adapter", "DiscordAdapter", {"_client": None}),
    ("plugins.platforms.telegram.adapter", "TelegramAdapter", {"_bot": None}),
    ("plugins.platforms.buzz.adapter", "BuzzAdapter", {"_channel_state": {}, "_channel_names": {}, "cli_path": None}),
    ("plugins.platforms.photon.adapter", "PhotonAdapter", {}),
    ("plugins.platforms.line.adapter", "LineAdapter", {}),
])
async def test_cold_metadata_cannot_claim_a_conversation_kind(module, name, state):
    from importlib import import_module

    cls = getattr(import_module(module), name)
    adapter = cls.__new__(cls)
    for field, value in state.items():
        setattr(adapter, field, value)
    assert (await adapter.get_chat_info("opaque-chat"))["type"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, {}, {"type": "unexpected"}])
async def test_mattermost_failed_or_unknown_metadata_never_means_channel(payload):
    from unittest.mock import AsyncMock
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    adapter = MattermostAdapter.__new__(MattermostAdapter)
    adapter._api_get = AsyncMock(return_value=payload)
    assert (await adapter.get_chat_info("opaque-chat"))["type"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("info", [None, {"type": "group", "scope_id": "workspace"}])
async def test_relay_preserves_scope_and_error_without_defaulting_to_dm(info):
    from unittest.mock import AsyncMock
    from gateway.relay.ws_transport import WebSocketRelayTransport

    transport = WebSocketRelayTransport.__new__(WebSocketRelayTransport)
    envelope = {"error": "lookup failed"} if info is None else {"chat_info": info}
    transport._request_response = AsyncMock(return_value=envelope)
    actual = await transport.get_chat_info("opaque-chat")
    if info is None:
        assert actual["type"] == "unknown"
        assert actual["error"] == "lookup failed"
    else:
        assert actual["type"] == "group"
        assert actual["scope_id"] == "workspace"


@pytest.mark.asyncio
async def test_bluebubbles_normalizes_the_resolved_group_address():
    from unittest.mock import AsyncMock
    from gateway.platforms.bluebubbles import BlueBubblesAdapter
    from gateway.conversation import resolve_delivery_source

    adapter = BlueBubblesAdapter.__new__(BlueBubblesAdapter)
    adapter._resolve_chat_guid = AsyncMock(return_value="iMessage;+;chat000")
    adapter._api_get = AsyncMock(return_value={"data": {}})
    source = await resolve_delivery_source(adapter, "bluebubbles", "chat000", None, {})
    assert source.chat_type == "group"
    assert source.chat_id == "iMessage;+;chat000"


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [("p2p", "dm"), ("group", "group"), ("topic", "forum"), ("unexpected", "unknown"), ("", "unknown")])
async def test_feishu_successful_metadata_preserves_known_kinds_only(raw, expected):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(chat=SimpleNamespace(get=lambda: None))))
    adapter._chat_info_cache = {}
    adapter._build_get_chat_request = lambda chat_id: object()
    adapter._run_blocking = AsyncMock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(chat_type=raw)))
    assert (await adapter.get_chat_info("oc_chat"))["type"] == expected


@pytest.mark.asyncio
async def test_telegram_forum_metadata_uses_the_inbound_group_key_layout():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    ChatType = pytest.importorskip("telegram.constants").ChatType
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    chat = SimpleNamespace(type=ChatType.SUPERGROUP, is_forum=True, title="Forum", username=None)
    adapter._bot = SimpleNamespace(get_chat=AsyncMock(return_value=chat))
    info = await adapter.get_chat_info("-100123")
    assert info["type"] == "group"
    assert info["is_forum"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_thread", [None, "456", "other"])
async def test_legacy_discord_thread_metadata_proves_parent_scope_and_thread(requested_thread):
    from unittest.mock import AsyncMock, MagicMock
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.config import Platform
    from gateway.conversation import address_of, resolve_delivery_source
    from gateway.session import SessionSource, build_session_key

    discord = pytest.importorskip("discord")
    thread = MagicMock(spec=discord.Thread)
    thread.id, thread.parent_id, thread.name = 456, 123, "Thread"
    thread.guild = MagicMock(id=789, name="Guild")
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._client = object()
    adapter._resolve_channel = AsyncMock(return_value=thread)
    origin = {"platform": "discord", "chat_id": "456", "user_id": "alice"}
    if requested_thread == "other":
        with pytest.raises(PermissionError):
            await resolve_delivery_source(adapter, "discord", "456", requested_thread, origin)
        return
    source = await resolve_delivery_source(adapter, "discord", "456", requested_thread, origin)
    actual = SessionSource(Platform.DISCORD, "456", chat_type="thread", parent_chat_id="123",
                           scope_id="789", thread_id="456", user_id="alice")
    assert address_of(source) == address_of(actual)
    assert build_session_key(source) == build_session_key(actual)


@pytest.mark.asyncio
async def test_explicit_flat_target_does_not_inherit_the_captured_thread():
    from unittest.mock import MagicMock
    from gateway.config import Platform
    from gateway.conversation import conversation_origin, resolve_delivery_source
    from gateway.session import SessionSource

    source = SessionSource(Platform.MATRIX, "!room:example.org", chat_type="group", thread_id="$root", user_id="alice")
    origin = {**source.to_dict(), "conversation": conversation_origin(source, thread_id="$root")}
    resolved = await resolve_delivery_source(MagicMock(), "matrix", source.chat_id, None, origin)
    assert resolved.thread_id is None
