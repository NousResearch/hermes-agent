"""Adapter-level strict topic routing: fail-closed gates on every path."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.topic_hooks import HookDecision, TopicHookRegistry, TopicPluginHook
from gateway.topic_routing import TopicRouteRegistry

OWNER_ID = 208214988
OWNER = str(OWNER_ID)


def make_adapter(strict=True):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._bot = SimpleNamespace(id=999000, send_message=AsyncMock())
    adapter._topic_hooks = TopicHookRegistry()
    if strict:
        adapter._topic_route_registry = TopicRouteRegistry.from_config(
            [
                {"chat_id": OWNER, "thread_id": 1, "profile": "sol"},
                {"chat_id": OWNER, "thread_id": 77, "profile": "atlas"},
            ]
        )
    else:
        adapter._topic_route_registry = None
    return adapter


def make_msg(
    *,
    chat_id=OWNER_ID,
    chat_type="private",
    thread_id=None,
    is_topic=False,
    text="hello",
    message_id=42,
    photo=None,
    media_group_id=None,
):
    chat = SimpleNamespace(id=chat_id, type=chat_type)
    return SimpleNamespace(
        chat=chat,
        chat_id=chat_id,
        message_thread_id=thread_id,
        is_topic_message=is_topic,
        text=text,
        caption=None,
        message_id=message_id,
        from_user=SimpleNamespace(id=OWNER_ID, first_name="German"),
        photo=photo,
        video=None,
        sticker=None,
        voice=None,
        audio=None,
        document=None,
        media_group_id=media_group_id,
    )


def make_update(msg, update_id=5000):
    return SimpleNamespace(update_id=update_id, message=msg, callback_query=None)


class RecordingHook(TopicPluginHook):
    profile = "sol"
    callback_prefixes = ("sf1:",)

    def __init__(self, decision=HookDecision.CONTINUE):
        self.decision = decision
        self.calls = []

    async def on_message(self, route, origin, text, reply):
        self.calls.append(("message", origin.thread_id, origin.update_id))
        return self.decision

    async def on_media_pre_download(self, route, origin, media, reply):
        self.calls.append(("media", media.kind, media.media_group_id))
        return self.decision

    async def on_callback(self, route, origin, callback_data, reply):
        self.calls.append(("callback", callback_data))
        return HookDecision.CONSUME


class TestThreadKey:
    def test_private_root_normalizes_to_general(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        msg = make_msg(thread_id=None, chat_type="private")
        assert TelegramAdapter._strict_thread_key(msg) == 1

    def test_explicit_thread_never_rewritten(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        msg = make_msg(thread_id=77, is_topic=True)
        assert TelegramAdapter._strict_thread_key(msg) == 77

    def test_group_root_stays_missing(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        msg = make_msg(chat_type="supergroup", thread_id=None)
        assert TelegramAdapter._strict_thread_key(msg) is None

    def test_topic_message_without_id_stays_missing(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        msg = make_msg(chat_type="private", thread_id=None, is_topic=True)
        assert TelegramAdapter._strict_thread_key(msg) is None


class TestInboundTextGate:
    @pytest.mark.asyncio
    async def test_foreign_chat_dropped_before_processing(self, monkeypatch):
        adapter = make_adapter()
        adapter._is_user_authorized_from_message = lambda msg: True
        should_process = MagicMock(return_value=True)
        adapter._should_process_message = should_process
        msg = make_msg(chat_id=31337)
        await adapter._handle_text_message(make_update(msg), None)
        should_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_topic_dropped(self):
        adapter = make_adapter()
        adapter._is_user_authorized_from_message = lambda msg: True
        should_process = MagicMock(return_value=True)
        adapter._should_process_message = should_process
        msg = make_msg(thread_id=555, is_topic=True)
        await adapter._handle_text_message(make_update(msg), None)
        should_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_general_message_reaches_hook_and_pipeline(self):
        adapter = make_adapter()
        hook = RecordingHook()
        adapter.register_topic_hook(hook)
        adapter._is_user_authorized_from_message = lambda msg: True
        adapter._should_process_message = MagicMock(return_value=False)
        adapter._should_observe_unmentioned_group_message = MagicMock(return_value=False)
        msg = make_msg(thread_id=None, chat_type="private")
        await adapter._handle_text_message(make_update(msg, update_id=6000), None)
        assert hook.calls == [("message", 1, 6000)]
        adapter._should_process_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_hook_consume_stops_pipeline(self):
        adapter = make_adapter()
        hook = RecordingHook(HookDecision.CONSUME)
        adapter.register_topic_hook(hook)
        adapter._is_user_authorized_from_message = lambda msg: True
        should_process = MagicMock(return_value=True)
        adapter._should_process_message = should_process
        msg = make_msg()
        await adapter._handle_text_message(make_update(msg), None)
        should_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_strict_off_is_untouched(self):
        adapter = make_adapter(strict=False)
        adapter._is_user_authorized_from_message = lambda msg: True
        adapter._should_process_message = MagicMock(return_value=False)
        adapter._should_observe_unmentioned_group_message = MagicMock(return_value=False)
        msg = make_msg(chat_id=31337)  # foreign chat: fine when off
        await adapter._handle_text_message(make_update(msg), None)
        adapter._should_process_message.assert_called_once()


class TestInboundMediaGate:
    @pytest.mark.asyncio
    async def test_media_hook_deny_blocks_download(self):
        adapter = make_adapter()
        hook = RecordingHook(HookDecision.DENY)
        adapter.register_topic_hook(hook)
        adapter._is_user_authorized_from_message = lambda msg: True
        should_process = MagicMock(return_value=True)
        adapter._should_process_message = should_process
        get_file = AsyncMock()
        photo = [SimpleNamespace(width=10, height=10, file_size=100, get_file=get_file)]
        msg = make_msg(photo=photo, media_group_id="album1", text=None)
        await adapter._handle_media_message(make_update(msg), None)
        get_file.assert_not_called()
        should_process.assert_not_called()
        assert hook.calls == [("media", "photo", "album1")]

    @pytest.mark.asyncio
    async def test_foreign_media_dropped_before_hook(self):
        adapter = make_adapter()
        hook = RecordingHook()
        adapter.register_topic_hook(hook)
        adapter._is_user_authorized_from_message = lambda msg: True
        should_process = MagicMock(return_value=True)
        adapter._should_process_message = should_process
        msg = make_msg(chat_id=31337, photo=[SimpleNamespace(width=1, height=1, file_size=1)], text=None)
        await adapter._handle_media_message(make_update(msg), None)
        assert hook.calls == []
        should_process.assert_not_called()


class TestCallbackGate:
    def _query(self, *, chat_id=OWNER_ID, thread_id=None, data="sf1:" + "A" * 22, message_id=77):
        chat = SimpleNamespace(id=chat_id, type="private")
        message = SimpleNamespace(
            chat=chat,
            chat_id=chat_id,
            message_thread_id=thread_id,
            is_topic_message=False,
            message_id=message_id,
        )
        return SimpleNamespace(
            data=data,
            message=message,
            from_user=SimpleNamespace(id=OWNER_ID, first_name="German"),
            answer=AsyncMock(),
        )

    @pytest.mark.asyncio
    async def test_sf1_dispatches_to_hook(self):
        adapter = make_adapter()
        hook = RecordingHook()
        adapter.register_topic_hook(hook)
        query = self._query()
        update = SimpleNamespace(update_id=7000, callback_query=query)
        await adapter._handle_callback_query(update, None)
        assert hook.calls == [("callback", "sf1:" + "A" * 22)]
        query.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_foreign_chat_callback_fails_closed(self):
        adapter = make_adapter()
        hook = RecordingHook()
        adapter.register_topic_hook(hook)
        query = self._query(chat_id=31337)
        update = SimpleNamespace(update_id=7001, callback_query=query)
        await adapter._handle_callback_query(update, None)
        assert hook.calls == []
        query.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_unknown_thread_callback_fails_closed(self):
        adapter = make_adapter()
        hook = RecordingHook()
        adapter.register_topic_hook(hook)
        query = self._query(thread_id=555)
        query.message.is_topic_message = True
        update = SimpleNamespace(update_id=7002, callback_query=query)
        await adapter._handle_callback_query(update, None)
        assert hook.calls == []


class TestOutboundGuard:
    @pytest.mark.asyncio
    async def test_fallback_metadata_refused(self):
        adapter = make_adapter()
        adapter._send_path_degraded = False
        result = await adapter.send(
            OWNER,
            "hello",
            metadata={"telegram_dm_topic_reply_fallback": True, "thread_id": "5"},
        )
        assert result.success is False
        assert result.error == "topic_route_send_fallback_denied"

    @pytest.mark.asyncio
    async def test_unregistered_destination_refused(self):
        adapter = make_adapter()
        adapter._send_path_degraded = False
        result = await adapter.send("31337", "hello", metadata={})
        assert result.success is False
        assert result.error == "topic_route_foreign_chat"

    @pytest.mark.asyncio
    async def test_unregistered_thread_refused(self):
        adapter = make_adapter()
        adapter._send_path_degraded = False
        result = await adapter.send(OWNER, "hello", metadata={"thread_id": "555"})
        assert result.success is False
        assert result.error == "topic_route_unknown_thread"

    def test_registered_thread_allowed(self):
        adapter = make_adapter()
        assert (
            adapter._strict_outbound_denied(OWNER, "77", {"thread_id": "77"}) is None
        )
        # Threadless send to the owner chat is the General/Sol lane.
        assert adapter._strict_outbound_denied(OWNER, None, None) is None

    def test_strict_off_never_blocks(self):
        adapter = make_adapter(strict=False)
        assert (
            adapter._strict_outbound_denied(
                "31337", "5", {"telegram_dm_topic_reply_fallback": True}
            )
            is None
        )


class TestHookRegistration:
    def test_requires_strict_mode(self):
        adapter = make_adapter(strict=False)
        with pytest.raises(ValueError):
            adapter.register_topic_hook(RecordingHook())

    def test_requires_registered_profile(self):
        adapter = make_adapter()
        hook = RecordingHook()
        hook.profile = "unrouted"
        with pytest.raises(ValueError):
            adapter.register_topic_hook(hook)
