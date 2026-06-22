import asyncio
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from gateway.platforms.base import _thread_metadata_for_source
from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.run import GatewayRunner
import plugins.platforms.telegram.adapter as telegram_mod


class _GuestBot:
    id = 8573596292
    username = "hermes_iar_bot"
    do_api_request = AsyncMock(return_value={"inline_message_id": "inline-1"})


def _guest_raw_payload():
    return {
        "message_id": 1,
        "date": 1700000000,
        "chat": {"id": -100123, "type": "supergroup", "title": "External group"},
        "from": {"id": 42, "is_bot": False, "first_name": "Caller"},
        "text": "@hermes_iar_bot run a web search",
        "entities": [{"type": "mention", "offset": 0, "length": 15}],
        "guest_query_id": "guest-query-123",
    }


def _guest_message():
    return SimpleNamespace(
        message_id=1,
        date=None,
        chat=SimpleNamespace(id=-100123, type="supergroup", title="External group", is_forum=False),
        from_user=SimpleNamespace(id=42, full_name="Caller", first_name="Caller", is_bot=False),
        text="@hermes_iar_bot run a web search",
        caption=None,
        entities=[SimpleNamespace(type="mention", offset=0, length=15)],
        caption_entities=[],
        api_kwargs=MappingProxyType({"guest_query_id": "guest-query-123"}),
        message_thread_id=None,
        is_topic_message=False,
        reply_to_message=None,
    )


def _guest_update_api_kwargs():
    return SimpleNamespace(
        update_id=123,
        guest_message=None,
        effective_message=None,
        message=None,
        api_kwargs=MappingProxyType({"guest_message": _guest_raw_payload()}),
    )


def test_allowed_updates_include_guest_message_even_when_ptb_all_types_omits_it():
    assert "guest_message" in TelegramAdapter._telegram_allowed_updates()


def test_guest_update_api_kwargs_is_parsed_and_routed_to_answer_guest_query(monkeypatch):
    async def run():
        _GuestBot.do_api_request.reset_mock()
        monkeypatch.setattr(
            telegram_mod.Message,
            "de_json",
            staticmethod(lambda raw, bot: _guest_message()),
            raising=False,
        )
        adapter = TelegramAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"guest_mode": True, "require_mention": True},
            )
        )
        adapter._bot = _GuestBot()

        captured = []
        adapter._enqueue_text_event = lambda event: captured.append(event)

        update = _guest_update_api_kwargs()
        assert update.effective_message is None  # PTB 22.x does not model guest_message yet

        await adapter._handle_guest_update(update, None)

        assert len(captured) == 1
        event = captured[0]
        assert event.text == "run a web search"
        assert event.source.chat_id == "-100123"
        assert event.source.guest_query_id == "guest-query-123"

        metadata = _thread_metadata_for_source(event.source)
        assert metadata == {"telegram_guest_query_id": "guest-query-123"}

        result = await adapter.send(event.source.chat_id, "guest response", metadata=metadata)

        assert result.success is True
        _GuestBot.do_api_request.assert_awaited_once()
        (endpoint,) = _GuestBot.do_api_request.call_args.args
        payload = _GuestBot.do_api_request.call_args.kwargs["api_kwargs"]
        assert endpoint == "answerGuestQuery"
        assert payload["guest_query_id"] == "guest-query-123"
        assert payload["result"]["type"] == "article"
        assert payload["result"]["input_message_content"]["message_text"] == "guest response"

    asyncio.run(run())


def test_gateway_runner_thread_metadata_preserves_guest_query_id():
    runner = object.__new__(GatewayRunner)
    runner._thread_metadata_for_target = lambda *args, **kwargs: {"thread_id": "t-1"}
    source = SimpleNamespace(
        platform="telegram",
        chat_id="-100123",
        thread_id="t-1",
        chat_type="supergroup",
        message_id="777",
        guest_query_id="guest-query-789",
    )

    meta = GatewayRunner._thread_metadata_for_source(runner, source)

    assert meta == {"thread_id": "t-1", "telegram_guest_query_id": "guest-query-789"}


def test_guest_reply_disables_streaming():
    """Guest Mode replies are one-shot (answerGuestQuery) and must never stream.

    Private-chat guest replies were delivered purely through the streaming/edit
    path, whose intermediate sends drop the guest_query_id token and fall back
    to sendMessage in a chat the guest bot isn't a member of. Streaming must be
    forced off whenever the source carries a guest_query_id, so the reply goes
    out as a single consolidated answerGuestQuery.
    """
    dm_source = SimpleNamespace(
        platform="telegram",
        chat_type="dm",
        guest_query_id="guest-query-789",
    )
    group_source = SimpleNamespace(
        platform="telegram",
        chat_type="supergroup",
        guest_query_id="guest-query-789",
    )
    normal_source = SimpleNamespace(
        platform="telegram",
        chat_type="dm",
        guest_query_id=None,
    )

    # Guest replies (DM or group) must disable streaming even when enabled.
    assert GatewayRunner._streaming_allowed_for_source(dm_source, True) is False
    assert GatewayRunner._streaming_allowed_for_source(group_source, True) is False
    # Non-guest replies follow the incoming streaming config unchanged.
    assert GatewayRunner._streaming_allowed_for_source(normal_source, True) is True
    assert GatewayRunner._streaming_allowed_for_source(normal_source, False) is False


def test_guest_update_uses_caller_chat_when_present(monkeypatch):
    async def run():
        _GuestBot.do_api_request.reset_mock()

        def _caller_message():
            return SimpleNamespace(
                message_id=1,
                date=None,
                chat=SimpleNamespace(id=-100123, type="supergroup", title="Guest relay", is_forum=False),
                from_user=SimpleNamespace(id=42, full_name="Caller", first_name="Caller", is_bot=False),
                text="@hermes_iar_bot ping",
                caption=None,
                entities=[SimpleNamespace(type="mention", offset=0, length=15)],
                caption_entities=[],
                api_kwargs=MappingProxyType({
                    "guest_query_id": "guest-query-456",
                    "guest_bot_caller_chat": {"id": -200999, "type": "private", "first_name": "Target"},
                    "guest_bot_caller_user": {"id": 99, "is_bot": False, "first_name": "Target", "last_name": "User"},
                }),
                message_thread_id=None,
                is_topic_message=False,
                reply_to_message=None,
            )

        monkeypatch.setattr(
            telegram_mod.Message,
            "de_json",
            staticmethod(lambda raw, bot: _caller_message()),
            raising=False,
        )
        adapter = TelegramAdapter(
            PlatformConfig(
                enabled=True,
                token="fake-token",
                extra={"guest_mode": True, "require_mention": True},
            )
        )
        adapter._bot = _GuestBot()

        captured = []
        adapter._enqueue_text_event = lambda event: captured.append(event)

        update = _guest_update_api_kwargs()
        await adapter._handle_guest_update(update, None)

        assert len(captured) == 1
        event = captured[0]
        assert event.source.chat_id == "-200999"
        assert event.source.chat_type == "dm"
        assert event.source.user_id == "99"
        assert event.source.guest_query_id == "guest-query-456"

    asyncio.run(run())
