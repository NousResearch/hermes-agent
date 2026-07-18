"""Adapter-level strict topic routing: fail-closed gates on every path."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
    is_forum=False,
):
    chat = SimpleNamespace(id=chat_id, type=chat_type, is_forum=is_forum)
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


class DownloadConsumingHook(RecordingHook):
    async def on_media_downloaded(
        self, route, origin, media, content, caption, reply
    ):
        self.calls.append(("downloaded", media.kind, bytes(content), caption))
        return HookDecision.CONSUME


class TestThreadKey:
    def test_private_root_without_explicit_topic_stays_missing(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        msg = make_msg(thread_id=None, chat_type="private")
        assert TelegramAdapter._strict_thread_key(msg) is None

    def test_general_topic_is_explicit_thread_one(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        msg = make_msg(thread_id=1, chat_type="private", is_topic=True)
        assert TelegramAdapter._strict_thread_key(msg) == "1"

    def test_explicit_thread_never_rewritten(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        msg = make_msg(thread_id=77, is_topic=True)
        assert TelegramAdapter._strict_thread_key(msg) == "77"

    def test_group_root_stays_missing(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        msg = make_msg(chat_type="supergroup", thread_id=None)
        assert TelegramAdapter._strict_thread_key(msg) is None

    def test_forum_general_without_raw_thread_normalizes_to_one(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        msg = make_msg(
            chat_type="supergroup",
            thread_id=None,
            is_topic=True,
            is_forum=True,
        )
        assert TelegramAdapter._strict_thread_key(msg) == "1"

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
        msg = make_msg(thread_id=1, chat_type="private", is_topic=True)
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

    @pytest.mark.asyncio
    async def test_explicit_food_command_reaches_strict_hook(self):
        adapter = make_adapter()
        hook = RecordingHook(HookDecision.CONSUME)
        adapter.register_topic_hook(hook)
        adapter._should_process_message = MagicMock(return_value=True)
        adapter._is_user_authorized_from_message = lambda msg: True
        adapter._ensure_forum_commands = AsyncMock()
        msg = make_msg(
            text="/food synthetic meal",
            thread_id=1,
            chat_type="private",
            is_topic=True,
        )
        await adapter._handle_command(make_update(msg, update_id=6010), None)
        assert hook.calls == [("message", 1, 6010)]
        adapter._ensure_forum_commands.assert_not_called()

    @pytest.mark.asyncio
    async def test_bot_addressed_food_command_is_normalized_before_strict_hook(self):
        adapter = make_adapter()
        adapter._bot.username = "HermesBot"
        hook = RecordingHook(HookDecision.CONSUME)
        hook.on_message = AsyncMock(return_value=HookDecision.CONSUME)
        adapter.register_topic_hook(hook)
        adapter._should_process_message = MagicMock(return_value=True)
        adapter._is_user_authorized_from_message = lambda msg: True
        adapter._ensure_forum_commands = AsyncMock()
        msg = make_msg(
            text="/food@HermesBot synthetic meal",
            thread_id=1,
            chat_type="private",
            is_topic=True,
        )

        await adapter._handle_command(make_update(msg, update_id=6011), None)

        assert hook.on_message.await_args.args[2] == "/food synthetic meal"
        adapter._ensure_forum_commands.assert_not_called()


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
        msg = make_msg(
            photo=photo,
            media_group_id="album1",
            text=None,
            thread_id=1,
            is_topic=True,
        )
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

    @pytest.mark.asyncio
    async def test_authenticated_post_download_hook_consumes_photo_bytes(self):
        adapter = make_adapter()
        hook = DownloadConsumingHook()
        adapter.register_topic_hook(hook)
        adapter._is_user_authorized_from_message = lambda msg: True
        adapter._should_process_message = MagicMock(return_value=True)
        adapter._media_message_type = MagicMock(return_value="image")
        adapter._build_message_event = MagicMock(
            return_value=SimpleNamespace(text=None, media_urls=[], media_types=[])
        )
        adapter._apply_telegram_group_observe_attribution = lambda event: event
        file_obj = SimpleNamespace(
            file_path="synthetic.jpg",
            download_as_bytearray=AsyncMock(return_value=bytearray(b"bounded-photo")),
        )
        photo = SimpleNamespace(
            width=10,
            height=10,
            file_size=13,
            get_file=AsyncMock(return_value=file_obj),
        )
        msg = make_msg(photo=[photo], text=None, thread_id=1, is_topic=True)
        msg.caption = "meal"
        await adapter._handle_media_message(make_update(msg), None)
        assert hook.calls[-1] == (
            "downloaded",
            "photo",
            b"bounded-photo",
            "meal",
        )


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
            edit_message_text=AsyncMock(),
        )

    @pytest.mark.asyncio
    async def test_sf1_dispatches_to_hook(self):
        adapter = make_adapter()
        adapter._is_callback_user_authorized = lambda *_a, **_kw: True
        hook = RecordingHook()
        adapter.register_topic_hook(hook)
        query = self._query(thread_id=1)
        query.message.is_topic_message = True
        update = SimpleNamespace(update_id=7000, callback_query=query)
        await adapter._handle_callback_query(update, None)
        assert hook.calls == [("callback", "sf1:" + "A" * 22)]
        query.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_unauthorized_sf1_never_reaches_hook(self):
        adapter = make_adapter()
        adapter._is_callback_user_authorized = lambda *_a, **_kw: False
        hook = RecordingHook()
        adapter.register_topic_hook(hook)
        query = self._query(thread_id=1)
        query.message.is_topic_message = True
        await adapter._handle_callback_query(
            SimpleNamespace(update_id=7003, callback_query=query), None
        )
        assert hook.calls == []
        query.answer.assert_awaited_once()

    def test_builtin_callback_state_is_exact_origin_bound(self):
        adapter = make_adapter()
        state = adapter._bind_callback_state("session", OWNER, "1")
        assert adapter._callback_state_value(state, OWNER, 1) == "session"
        assert adapter._callback_state_value(state, OWNER, 77) is None
        assert adapter._callback_state_value(state, "31337", 1) is None

    def test_model_picker_state_key_includes_thread(self):
        adapter = make_adapter()
        assert adapter._model_picker_key(OWNER, 1) != adapter._model_picker_key(
            OWNER, 77
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("data", "state_attr", "state_key"),
        [
            ("ea:once:7", "_approval_state", 7),
            ("sc:once:confirm-7", "_slash_confirm_state", "confirm-7"),
            ("cl:clarify-7:0", "_clarify_state", "clarify-7"),
        ],
    )
    async def test_foreign_topic_builtin_callback_does_not_consume_state(
        self, data, state_attr, state_key
    ):
        adapter = make_adapter()
        adapter._is_callback_user_authorized = lambda *_a, **_kw: True
        setattr(
            adapter,
            state_attr,
            {state_key: adapter._bind_callback_state("session", OWNER, "77")},
        )
        query = self._query(thread_id=1, data=data)
        query.message.is_topic_message = True
        await adapter._handle_callback_query(
            SimpleNamespace(update_id=7010, callback_query=query), None
        )
        assert state_key in getattr(adapter, state_attr)

    @pytest.mark.asyncio
    async def test_foreign_topic_model_callback_cannot_see_other_picker(self):
        adapter = make_adapter()
        adapter._is_callback_user_authorized = lambda *_a, **_kw: True
        adapter._model_picker_state = {
            adapter._model_picker_key(OWNER, 77): {"session_key": "atlas"}
        }
        query = self._query(thread_id=1, data="mx")
        query.message.is_topic_message = True
        await adapter._handle_callback_query(
            SimpleNamespace(update_id=7011, callback_query=query), None
        )
        assert adapter._model_picker_key(OWNER, 77) in adapter._model_picker_state
        query.answer.assert_awaited_with(text="Picker expired — use /model again.")

    @pytest.mark.asyncio
    async def test_general_raw_none_exec_callback_uses_thread_one_state(self):
        adapter = make_adapter()
        auth_calls = []
        adapter._is_callback_user_authorized = (
            lambda *_a, **kw: auth_calls.append(kw) or True
        )
        adapter._approval_state = {
            7: adapter._bind_callback_state("session", OWNER, "1")
        }
        adapter.resume_typing_for_chat = MagicMock()
        query = self._query(thread_id=None, data="ea:once:7")
        query.message.chat.type = "supergroup"
        query.message.chat.is_forum = True
        query.message.is_topic_message = True

        with patch("tools.approval.resolve_gateway_approval", return_value=1):
            await adapter._handle_callback_query(
                SimpleNamespace(update_id=7012, callback_query=query), None
            )

        assert 7 not in adapter._approval_state
        assert auth_calls[-1]["thread_id"] == "1"
        query.answer.assert_awaited_with(text="✅ Approved once")

    @pytest.mark.asyncio
    async def test_general_raw_none_model_callback_uses_thread_one_state(self):
        adapter = make_adapter()
        adapter._is_callback_user_authorized = lambda *_a, **_kw: True
        adapter._model_picker_state = {
            adapter._model_picker_key(OWNER, 1): {"session_key": "sol"}
        }
        query = self._query(thread_id=None, data="mx")
        query.message.chat.type = "supergroup"
        query.message.chat.is_forum = True
        query.message.is_topic_message = True

        await adapter._handle_callback_query(
            SimpleNamespace(update_id=7013, callback_query=query), None
        )

        assert adapter._model_picker_key(OWNER, 1) not in adapter._model_picker_state
        query.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_general_raw_none_slash_followup_preserves_strict_origin(self):
        adapter = make_adapter()
        adapter._is_callback_user_authorized = lambda *_a, **_kw: True
        adapter._slash_confirm_state = {
            "confirm-8": adapter._bind_callback_state("session", OWNER, "1")
        }
        adapter._reply_to_mode = "on"
        adapter._link_preview_kwargs = lambda: {}
        adapter._send_message_with_thread_fallback = AsyncMock(
            return_value=SimpleNamespace(message_id=93)
        )
        query = self._query(thread_id=None, data="sc:once:confirm-8")
        query.message.chat.type = "supergroup"
        query.message.chat.is_forum = True
        query.message.is_topic_message = True

        with patch(
            "tools.slash_confirm.resolve",
            new=AsyncMock(return_value="completed"),
        ):
            await adapter._handle_callback_query(
                SimpleNamespace(update_id=7014, callback_query=query), None
            )

        sent = adapter._send_message_with_thread_fallback.await_args.kwargs
        assert sent["message_thread_id"] is None
        assert adapter._strict_message_origins[(OWNER, "93")] == 1

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
    async def test_unregistered_private_topic_reply_metadata_refused(self):
        adapter = make_adapter()
        adapter._send_path_degraded = False
        result = await adapter.send(
            OWNER,
            "hello",
            metadata={"telegram_dm_topic_reply_fallback": True, "thread_id": "5"},
        )
        assert result.success is False
        assert result.error == "topic_route_unknown_thread"

    def test_registered_private_topic_reply_metadata_allowed(self):
        adapter = make_adapter()
        metadata = {
            "thread_id": "77",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "42",
        }

        assert adapter._strict_outbound_denied(OWNER, "77", metadata) is None

    def test_topic_creation_metadata_remains_refused(self):
        adapter = make_adapter()
        metadata = {
            "thread_id": "77",
            "telegram_dm_topic_created_for_send": True,
        }

        assert (
            adapter._strict_outbound_denied(OWNER, "77", metadata)
            == "topic_route_send_fallback_denied"
        )

    @pytest.mark.asyncio
    async def test_strict_private_overflow_never_retries_without_reply_anchor(self):
        adapter = make_adapter()
        adapter.truncate_message = MagicMock(return_value=["first", "second"])
        adapter.format_message = lambda text: text
        adapter._link_preview_kwargs = lambda: {}
        adapter._notification_kwargs = lambda metadata: {}
        adapter._bot.edit_message_text = AsyncMock()
        adapter._bot.send_message = AsyncMock(
            side_effect=RuntimeError("reply message not found")
        )
        metadata = {
            "thread_id": "77",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "42",
        }

        result = await adapter._edit_overflow_split(
            OWNER,
            "91",
            "oversized",
            finalize=False,
            metadata=metadata,
        )

        assert result.success is False
        assert adapter._bot.send_message.await_count == 1

    @pytest.mark.asyncio
    async def test_strict_private_media_never_retries_without_reply_anchor(self):
        class BadRequest(Exception):
            pass

        adapter = make_adapter()
        send = AsyncMock(side_effect=BadRequest("message to be replied not found"))
        metadata = {
            "thread_id": "77",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "42",
        }

        with pytest.raises(BadRequest):
            await adapter._send_with_dm_topic_reply_anchor_retry(
                send,
                {
                    "chat_id": OWNER,
                    "message_thread_id": 77,
                    "reply_to_message_id": 42,
                },
                metadata,
                42,
                "photo",
            )

        assert send.await_count == 1

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
        assert adapter._strict_outbound_denied(OWNER, "1", {"thread_id": "1"}) is None

    def test_message_mutation_requires_exact_recorded_origin(self):
        adapter = make_adapter()
        adapter._record_strict_message_origin(OWNER, "91", {"thread_id": "1"})

        assert (
            adapter._strict_message_mutation_denied(
                OWNER, "91", {"thread_id": "1"}
            )
            is None
        )
        assert (
            adapter._strict_message_mutation_denied(
                OWNER, "91", {"thread_id": "77"}
            )
            == "topic_route_message_origin_mismatch"
        )

    def test_message_origin_receipts_are_bounded(self):
        adapter = make_adapter()
        adapter._STRICT_MESSAGE_ORIGIN_CAP = 2

        for message_id in ("91", "92", "93"):
            adapter._record_strict_message_origin(
                OWNER, message_id, {"thread_id": "1"}
            )

        assert set(adapter._strict_message_origins) == {
            (OWNER, "92"),
            (OWNER, "93"),
        }
        assert (
            adapter._strict_message_mutation_denied(
                OWNER, "999", {"thread_id": "1"}
            )
            == "topic_route_message_origin_mismatch"
        )

    @pytest.mark.asyncio
    async def test_edit_and_delete_fail_before_bot_call_for_wrong_origin(self):
        adapter = make_adapter()
        adapter._bot.edit_message_text = AsyncMock()
        adapter._bot.delete_message = AsyncMock()
        adapter._record_strict_message_origin(OWNER, "91", {"thread_id": "1"})

        edit = await adapter.edit_message(
            OWNER,
            "91",
            "wrong topic",
            metadata={"thread_id": "77"},
        )
        deleted = await adapter.delete_message(
            OWNER, "91", metadata={"thread_id": "77"}
        )

        assert edit.success is False
        assert edit.error == "topic_route_message_origin_mismatch"
        assert deleted is False
        adapter._bot.edit_message_text.assert_not_called()
        adapter._bot.delete_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_exact_origin_succeeds_and_consumes_binding(self):
        adapter = make_adapter()
        adapter._bot.delete_message = AsyncMock()
        adapter._record_strict_message_origin(OWNER, "91", {"thread_id": "1"})

        assert await adapter.delete_message(
            OWNER, "91", metadata={"thread_id": "1"}
        )
        adapter._bot.delete_message.assert_awaited_once_with(
            chat_id=OWNER_ID, message_id=91
        )
        assert (
            adapter._strict_message_mutation_denied(
                OWNER, "91", {"thread_id": "1"}
            )
            == "topic_route_message_origin_mismatch"
        )

    @pytest.mark.asyncio
    async def test_scheduled_delete_preserves_exact_topic_origin(self):
        import gateway.platforms.base as base_module

        adapter = make_adapter()
        adapter._bot.delete_message = AsyncMock()
        adapter._record_strict_message_origin(OWNER, "91", {"thread_id": "1"})
        real_sleep = base_module.asyncio.sleep

        async def immediate_sleep(_duration):
            await real_sleep(0)

        with patch.object(base_module.asyncio, "sleep", immediate_sleep):
            adapter._schedule_ephemeral_delete(
                OWNER,
                "91",
                30,
                metadata={"thread_id": "1"},
            )
            for _ in range(5):
                await real_sleep(0)

        adapter._bot.delete_message.assert_awaited_once_with(
            chat_id=OWNER_ID, message_id=91
        )

        adapter._bot.delete_message.reset_mock()
        adapter._record_strict_message_origin(OWNER, "92", {"thread_id": "1"})
        with patch.object(base_module.asyncio, "sleep", immediate_sleep):
            adapter._schedule_ephemeral_delete(
                OWNER,
                "92",
                30,
                metadata={"thread_id": "77"},
            )
            for _ in range(5):
                await real_sleep(0)

        adapter._bot.delete_message.assert_not_called()

    def test_threadless_send_is_not_implicitly_sol(self):
        adapter = make_adapter()
        assert (
            adapter._strict_outbound_denied(OWNER, None, None)
            == "topic_route_missing_thread"
        )

    @pytest.mark.asyncio
    async def test_typing_never_falls_back_outside_registered_thread(self):
        adapter = make_adapter()
        adapter._bot.send_chat_action = AsyncMock(side_effect=RuntimeError("gone"))
        adapter._telegram_typing_cooldown_until = {}
        adapter._telegram_typing_cooldown_seconds = 30.0
        await adapter.send_typing(OWNER, {"thread_id": "1"})
        adapter._bot.send_chat_action.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_threadless_control_and_media_sends_make_no_bot_call(self, tmp_path):
        adapter = make_adapter()
        adapter._reply_to_mode = "on"
        adapter._link_preview_kwargs = lambda: {}
        adapter._notification_kwargs = lambda metadata: {}
        adapter._bot.send_message = AsyncMock()
        result = await adapter.send_exec_approval(OWNER, "cmd", "session")
        assert result.success is False
        assert result.error == "topic_route_missing_thread"
        adapter._bot.send_message.assert_not_called()

        image = tmp_path / "image.png"
        image.write_bytes(b"not-read-because-routing-denies-first")
        adapter._bot.send_photo = AsyncMock()
        result = await adapter.send_image_file(OWNER, str(image))
        assert result.success is False
        assert result.error == "topic_route_missing_thread"
        adapter._bot.send_photo.assert_not_called()

    @pytest.mark.asyncio
    async def test_origin_reply_normalizes_general_send_and_records_origin(self):
        adapter = make_adapter()
        adapter._bot.send_message = AsyncMock(
            return_value=SimpleNamespace(message_id=90)
        )
        from gateway.topic_routing import RouteOrigin

        reply = adapter._origin_reply(
            RouteOrigin(
                bot_id="999000",
                owner_chat_id=OWNER,
                thread_id=1,
                update_id=1,
                message_id=2,
            )
        )
        await reply("hello")
        adapter._bot.send_message.assert_awaited_once_with(
            chat_id=OWNER,
            text="hello",
            message_thread_id=None,
        )
        assert adapter._strict_message_origins[(OWNER, "90")] == 1

    @pytest.mark.asyncio
    async def test_origin_presenter_normalizes_general_send_and_records_origin(self):
        adapter = make_adapter()
        adapter._bot.send_message = AsyncMock(
            return_value=SimpleNamespace(message_id=91)
        )
        from gateway.topic_routing import RouteOrigin

        reply = adapter._origin_reply(
            RouteOrigin("999000", OWNER, 1, 1, 2)
        )
        message_id = await reply.present_actions(
            "Choose", [("Option 1", "sf1:" + "A" * 22)]
        )
        assert message_id == 91
        kwargs = adapter._bot.send_message.await_args.kwargs
        assert kwargs["chat_id"] == OWNER
        assert kwargs["message_thread_id"] is None
        assert kwargs["reply_markup"].inline_keyboard[0][0].callback_data.startswith(
            "sf1:"
        )
        assert adapter._strict_message_origins[(OWNER, "91")] == 1

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
