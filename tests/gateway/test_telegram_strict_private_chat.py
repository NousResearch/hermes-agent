"""Adapter gates for a dedicated owner-private Telegram bot."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType
from gateway.private_chat_routing import PrivateChatRouteRegistry
from gateway.topic_hooks import HookDecision, TopicHookRegistry, TopicPluginHook

OWNER_ID = 208214988


class AtlasHook(TopicPluginHook):
    profile = "atlas"
    callback_prefixes = ("sf1:",)

    def __init__(self, decision=HookDecision.CONTINUE):
        self.decision = decision
        self.calls = []

    async def on_message(self, route, origin, text, reply):
        self.calls.append(("message", route.profile, origin.thread_id))
        return self.decision

    async def on_media_pre_download(self, route, origin, media, reply):
        self.calls.append(("media", media.kind, origin.thread_id))
        return self.decision


def make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._bot = SimpleNamespace(
        id=900001,
        username="atlas_private_bot",
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=77)),
    )
    adapter._topic_route_registry = None
    adapter._private_chat_route_registry = PrivateChatRouteRegistry.from_config(
        {
            "mode": "strict",
            "chat_id": OWNER_ID,
            "user_id": OWNER_ID,
            "profile": "atlas",
            "expected_bot_id": 900001,
            "expected_bot_username": "atlas_private_bot",
            "hooks": [],
        }
    )
    adapter._strict_route_registry = adapter._private_chat_route_registry
    adapter._topic_hooks = TopicHookRegistry()
    adapter._strict_message_origins = {}
    return adapter


def message(
    *,
    chat_id=OWNER_ID,
    user_id=OWNER_ID,
    chat_type="private",
    thread=None,
    is_topic_message=False,
):
    return SimpleNamespace(
        chat=SimpleNamespace(
            id=chat_id,
            type=chat_type,
            is_forum=False,
            title=None,
            full_name="Owner",
        ),
        chat_id=chat_id,
        from_user=SimpleNamespace(
            id=user_id,
            first_name="Owner",
            full_name="Owner",
            is_bot=False,
        ),
        message_thread_id=thread,
        is_topic_message=is_topic_message,
        message_id=42,
        text="hello",
        caption=None,
        photo=None,
        video=None,
        sticker=None,
        voice=None,
        audio=None,
        document=None,
        media_group_id=None,
        reply_to_message=None,
        quote=None,
        date=None,
    )


def update(msg):
    return SimpleNamespace(update_id=500, message=msg, callback_query=None)


@pytest.mark.asyncio
async def test_owner_text_routes_to_atlas_without_thread():
    adapter = make_adapter()
    hook = AtlasHook(HookDecision.CONSUME)
    adapter.register_conversation_hook(hook)
    adapter._is_user_authorized_from_message = lambda msg: True
    adapter._should_process_message = MagicMock(return_value=True)
    await adapter._handle_text_message(update(message()), None)
    assert hook.calls == [("message", "atlas", None)]
    adapter._should_process_message.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "msg",
    [
        message(chat_id=999),
        message(user_id=999),
        message(chat_type="group"),
        message(thread=1, is_topic_message=True),
    ],
)
async def test_foreign_group_or_threaded_text_has_zero_pipeline_effect(msg):
    adapter = make_adapter()
    adapter._is_user_authorized_from_message = lambda incoming: True
    adapter._should_process_message = MagicMock(return_value=True)
    await adapter._handle_text_message(update(msg), None)
    adapter._should_process_message.assert_not_called()


@pytest.mark.asyncio
async def test_plain_private_reply_anchor_routes_as_unthreaded_owner_message():
    adapter = make_adapter()
    hook = AtlasHook(HookDecision.CONSUME)
    adapter.register_conversation_hook(hook)
    adapter._is_user_authorized_from_message = lambda incoming: True
    adapter._should_process_message = MagicMock(return_value=True)
    await adapter._handle_text_message(update(message(thread=777)), None)
    assert hook.calls == [("message", "atlas", None)]
    adapter._should_process_message.assert_not_called()


def test_private_event_is_stamped_with_admitted_route_profile():
    adapter = make_adapter()
    event = adapter._build_message_event(message(), MessageType.TEXT, update_id=500)
    assert event.source.profile == "atlas"


@pytest.mark.asyncio
async def test_command_auth_precedes_should_process():
    adapter = make_adapter()
    order = []
    adapter._is_user_authorized_from_message = lambda msg: order.append("auth") or False
    adapter._should_process_message = lambda msg, is_command=False: order.append("process") or True
    msg = message()
    msg.text = "/status"
    await adapter._handle_command(update(msg), None)
    assert order == ["auth"]


@pytest.mark.asyncio
async def test_foreign_command_route_precedes_should_process():
    adapter = make_adapter()
    adapter._is_user_authorized_from_message = lambda msg: True
    adapter._should_process_message = MagicMock(return_value=True)
    msg = message(chat_id=999)
    msg.text = "/status"
    await adapter._handle_command(update(msg), None)
    adapter._should_process_message.assert_not_called()


@pytest.mark.asyncio
async def test_origin_reply_never_sends_message_thread_id():
    adapter = make_adapter()
    gate = adapter._resolve_topic_route(
        OWNER_ID,
        None,
        500,
        42,
        chat_type="private",
        user_id=OWNER_ID,
    )
    route, origin = gate
    assert route.thread_id is None
    await adapter._origin_reply(origin)("ok")
    kwargs = adapter._bot.send_message.await_args.kwargs
    assert kwargs == {"chat_id": str(OWNER_ID), "text": "ok"}


@pytest.mark.asyncio
async def test_private_send_rejects_direct_message_topic_metadata():
    adapter = make_adapter()
    result = await adapter.send(
        str(OWNER_ID),
        "hello",
        metadata={"direct_messages_topic_id": "777"},
    )
    assert result.success is False
    assert result.error == "private_route_unexpected_thread"
    adapter._bot.send_message.assert_not_awaited()


def test_swapped_token_is_rejected_before_polling_start():
    adapter = make_adapter()
    adapter._validate_private_chat_bot_identity()
    adapter._bot.id = 900002
    with pytest.raises(ValueError, match="bot id"):
        adapter._validate_private_chat_bot_identity()
