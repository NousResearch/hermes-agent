"""Regression coverage for Telegram Bot API Guest Mode queries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageType, _thread_metadata_for_source
from plugins.platforms.telegram.adapter import TelegramAdapter


class _RecordingApp:
    """Minimal stand-in for telegram.ext.Application.

    Records ``add_handler`` calls with their group so the registration
    wiring can be asserted without a live Application or network access.
    """

    def __init__(self) -> None:
        # PTB's default group is 0; add_handler defaults to it.
        self.handlers: list[tuple[object, int]] = []

    def add_handler(self, handler, group: int = 0) -> None:
        self.handlers.append((handler, group))


def _adapter() -> TelegramAdapter:
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter._bot = MagicMock()
    adapter._send_path_degraded = False
    return adapter


def test_guest_source_metadata_routes_to_answer_guest_query():
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_type="dm",
        thread_id="guest:5323706597129951744",
        message_id="17",
    )

    assert _thread_metadata_for_source(source, "17") == {
        "thread_id": "guest:5323706597129951744",
        "telegram_guest_query_id": "5323706597129951744",
    }


@pytest.mark.asyncio
async def test_guest_message_enters_normal_agent_queue_once():
    adapter = _adapter()
    event = SimpleNamespace(
        source=SimpleNamespace(thread_id=None, chat_topic=None),
        text="@darrenslavebot are you there?",
    )
    adapter._build_message_event = MagicMock(return_value=event)
    adapter._clean_bot_trigger_text = MagicMock(return_value="are you there?")
    adapter._enqueue_text_event = MagicMock()
    message = SimpleNamespace(
        text="@darrenslavebot are you there?",
        guest_query_id="5323706597129951744",
    )
    update = SimpleNamespace(update_id=91, guest_message=message)

    await adapter._handle_guest_message(update, None)

    adapter._build_message_event.assert_called_once_with(
        message, MessageType.TEXT, update_id=91
    )
    assert event.source.thread_id == "guest:5323706597129951744"
    assert event.source.chat_topic == "Guest Bot Mention"
    assert event.text == "are you there?"
    adapter._enqueue_text_event.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_unauthorized_guest_message_is_rejected():
    adapter = _adapter()
    adapter._is_user_authorized_from_message = MagicMock(return_value=False)
    adapter._build_message_event = MagicMock()
    adapter._enqueue_text_event = MagicMock()
    message = SimpleNamespace(text="summon", guest_query_id="guest-query-1")
    update = SimpleNamespace(update_id=92, guest_message=message)

    await adapter._handle_guest_message(update, None)

    adapter._is_user_authorized_from_message.assert_called_once_with(message)
    adapter._build_message_event.assert_not_called()
    adapter._enqueue_text_event.assert_not_called()


@pytest.mark.asyncio
async def test_generic_text_handler_ignores_guest_message():
    adapter = _adapter()
    adapter._effective_update_message = MagicMock()
    adapter._enqueue_text_event = MagicMock()
    update = SimpleNamespace(
        guest_message=SimpleNamespace(text="guest summon"),
        effective_message=SimpleNamespace(text="guest summon"),
    )

    await adapter._handle_text_message(update, None)

    adapter._effective_update_message.assert_not_called()
    adapter._enqueue_text_event.assert_not_called()


@pytest.mark.asyncio
async def test_generic_command_handler_ignores_guest_message():
    adapter = _adapter()
    adapter._effective_update_message = MagicMock()
    adapter._should_process_message = MagicMock()
    update = SimpleNamespace(
        guest_message=SimpleNamespace(text="/status", guest_query_id="guest-query-2"),
        effective_message=SimpleNamespace(text="/status"),
    )

    await adapter._handle_command(update, None)

    adapter._effective_update_message.assert_not_called()
    adapter._should_process_message.assert_not_called()


@pytest.mark.asyncio
async def test_intermediate_guest_send_is_suppressed():
    adapter = _adapter()
    adapter._bot.answer_guest_query = AsyncMock()

    result = await adapter.send(
        "1482299073",
        "Working…",
        metadata={"telegram_guest_query_id": "5323706597129951744"},
    )

    assert result.success is True
    assert result.message_id is None
    adapter._bot.answer_guest_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_guest_send_uses_answer_guest_query_once(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InputTextMessageContent",
        lambda text: SimpleNamespace(message_text=text),
    )
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InlineQueryResultArticle",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    adapter._bot.answer_guest_query = AsyncMock(
        return_value=SimpleNamespace(message_id=777)
    )

    result = await adapter.send(
        "1482299073",
        "Yes — I’m here.",
        metadata={
            "telegram_guest_query_id": "5323706597129951744",
            "notify": True,
        },
    )

    assert result.success is True
    assert result.message_id == "777"
    adapter._bot.answer_guest_query.assert_awaited_once()
    query_id, article = adapter._bot.answer_guest_query.await_args.args
    assert query_id == "5323706597129951744"
    assert article.input_message_content.message_text == "Yes — I’m here."


@pytest.mark.asyncio
async def test_guest_answer_failure_is_non_retryable(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InputTextMessageContent",
        lambda text: SimpleNamespace(message_text=text),
    )
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InlineQueryResultArticle",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    adapter._bot.answer_guest_query = AsyncMock(side_effect=RuntimeError("expired"))

    result = await adapter.send(
        "1482299073",
        "Too late",
        metadata={
            "telegram_guest_query_id": "5323706597129951744",
            "notify": True,
        },
    )

    assert result.success is False
    assert result.retryable is False
    assert result.error == "expired"


# ---------------------------------------------------------------------------
# Handler-registration regression coverage
#
# Guards against the Guest Mode catch-all being registered in a group that is
# evaluated before normal intake. python-telegram-bot evaluates each handler
# group independently and runs at most one matching handler per group, only
# stopping across groups on ApplicationHandlerStop. A TypeHandler(Update)
# matches every update, so it must live in a group that sorts AFTER the
# default (0) text/command/media handlers — never before them — otherwise it
# would occupy the single "one handler per group" slot ahead of normal intake.
#
# These tests drive the adapter's real _register_handlers() wiring and capture
# each handler's callback + group. They install lightweight recording stubs
# for the PTB handler constructors so the assertions hold whether the test run
# sees the real python-telegram-bot or the gateway conftest's telegram mock.
# ---------------------------------------------------------------------------


class _CapturedHandler:
    """Records the callback a handler constructor was given."""

    def __init__(self, callback):
        self.callback = callback


def _install_recording_handlers(monkeypatch, *, guest_symbols: bool):
    """Patch the adapter's handler constructors with capturing stubs.

    Returns nothing; call _register_handlers on a _RecordingApp afterwards and
    inspect ``app.handlers`` as ``[(handler, group), ...]``.
    """
    import plugins.platforms.telegram.adapter as adapter_mod

    def _message_handler(_filters, callback):
        return _CapturedHandler(callback)

    def _callback_query_handler(callback):
        return _CapturedHandler(callback)

    def _type_handler(_update_type, callback):
        return _CapturedHandler(callback)

    monkeypatch.setattr(adapter_mod, "TelegramMessageHandler", _message_handler)
    monkeypatch.setattr(adapter_mod, "CallbackQueryHandler", _callback_query_handler)
    monkeypatch.setattr(
        adapter_mod, "TypeHandler", _type_handler if guest_symbols else None
    )
    # The guest branch also requires these symbols to be non-None. Under the
    # gateway conftest telegram mock they already are; pin them explicitly so
    # the branch is deterministic regardless of the import path taken.
    if guest_symbols:
        monkeypatch.setattr(adapter_mod, "InlineQueryResultArticle", object())
        monkeypatch.setattr(adapter_mod, "InputTextMessageContent", object())


def _group_for_callback(handlers, callback):
    return [group for handler, group in handlers if handler.callback == callback]


def test_guest_catch_all_registers_in_later_group_than_normal_handlers(monkeypatch):
    adapter = _adapter()
    _install_recording_handlers(monkeypatch, guest_symbols=True)
    app = _RecordingApp()

    adapter._register_handlers(app)

    text_groups = _group_for_callback(app.handlers, adapter._handle_text_message)
    command_groups = _group_for_callback(app.handlers, adapter._handle_command)
    guest_groups = _group_for_callback(app.handlers, adapter._handle_guest_message)

    # Normal intake handlers are registered in the default group (0).
    assert text_groups == [0]
    assert command_groups == [0]

    # The guest catch-all is registered exactly once, in a strictly later
    # group than every normal handler — so it can never occupy the single
    # per-group handler slot ahead of ordinary text/command intake.
    guest_cb = adapter._handle_guest_message
    normal_groups = [g for h, g in app.handlers if h.callback != guest_cb]
    assert guest_groups == [1]
    assert normal_groups, "expected normal handlers to be registered"
    assert guest_groups[0] > max(normal_groups)


def test_no_guest_handler_registered_without_guest_symbols(monkeypatch):
    adapter = _adapter()
    _install_recording_handlers(monkeypatch, guest_symbols=False)
    app = _RecordingApp()

    adapter._register_handlers(app)

    # Normal intake is still fully wired…
    assert _group_for_callback(app.handlers, adapter._handle_text_message) == [0]
    assert _group_for_callback(app.handlers, adapter._handle_command) == [0]
    # …and no catch-all is added when Guest Mode is unavailable.
    assert _group_for_callback(app.handlers, adapter._handle_guest_message) == []


@pytest.mark.asyncio
async def test_group_dispatch_routes_ordinary_and_guest_updates(monkeypatch):
    """Simulate PTB's group dispatch over the adapter's real registration.

    python-telegram-bot processes handler groups in ascending order, running
    at most one matching handler per group and only stopping across groups on
    ApplicationHandlerStop. This drives that exact contract over the wiring
    produced by _register_handlers() and asserts:

      * an ordinary text update is handled by the group-0 text handler and is
        NOT consumed by the guest catch-all, and
      * a guest update is routed to the guest handler.
    """
    adapter = _adapter()
    _install_recording_handlers(monkeypatch, guest_symbols=True)
    app = _RecordingApp()
    adapter._register_handlers(app)

    fired: list[str] = []

    text_cb = adapter._handle_text_message
    guest_cb = adapter._handle_guest_message

    def matches(handler, update) -> bool:
        cb = handler.callback
        if cb == guest_cb:
            return True  # TypeHandler(Update) matches every update
        if cb == text_cb:
            return getattr(update, "message", None) is not None
        return False

    async def dispatch(update):
        # Group order mirrors PTB: ascending group number, one handler per
        # group, no cross-group stop (handlers here never raise
        # ApplicationHandlerStop — they return for irrelevant updates).
        for group in sorted({g for _, g in app.handlers}):
            for handler, g in app.handlers:
                if g != group or not matches(handler, update):
                    continue
                cb = handler.callback
                if cb == text_cb:
                    if getattr(update, "guest_message", None) is None:
                        fired.append("text")
                elif cb == guest_cb:
                    if getattr(update, "guest_message", None) is not None:
                        fired.append("guest")
                break  # only one handler per group

    ordinary = SimpleNamespace(message=SimpleNamespace(text="hello"), guest_message=None)
    guest = SimpleNamespace(
        message=None,
        guest_message=SimpleNamespace(text="summon", guest_query_id="gq-1"),
    )

    await dispatch(ordinary)
    await dispatch(guest)

    # Ordinary update reaches the normal text handler and is not swallowed by
    # the guest catch-all; the guest update routes to the guest handler.
    assert fired == ["text", "guest"]
