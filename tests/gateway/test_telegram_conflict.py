import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    telegram_mod.InlineKeyboardButton = lambda **kwargs: SimpleNamespace(**kwargs)
    telegram_mod.InlineKeyboardMarkup = lambda keyboard: SimpleNamespace(inline_keyboard=keyboard)

    for name in ("telegram", "telegram.ext", "telegram.constants"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_connect_rejects_same_host_token_lock(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="secret-token"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (False, {"pid": 4242}),
    )

    ok = await adapter.connect()

    assert ok is False
    assert adapter.fatal_error_code == "telegram_token_lock"
    assert adapter.has_fatal_error is True
    assert "already using this Telegram bot token" in adapter.fatal_error_message


@pytest.mark.asyncio
async def test_polling_conflict_retries_before_fatal(monkeypatch):
    """A single 409 should trigger a retry, not an immediate fatal error."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    captured = {}

    async def fake_start_polling(**kwargs):
        captured["error_callback"] = kwargs["error_callback"]

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=fake_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr("gateway.platforms.telegram.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

    # Speed up retries for testing
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    ok = await adapter.connect()

    assert ok is True
    assert callable(captured["error_callback"])

    conflict = type("Conflict", (Exception,), {})

    # First conflict: should retry, NOT be fatal
    captured["error_callback"](conflict("Conflict: terminated by other getUpdates request"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    # Give the scheduled task a chance to run
    for _ in range(10):
        await asyncio.sleep(0)

    assert adapter.has_fatal_error is False, "First conflict should not be fatal"
    assert adapter._polling_conflict_count == 0, "Count should reset after successful retry"


@pytest.mark.asyncio
async def test_polling_conflict_becomes_fatal_after_retries(monkeypatch):
    """After exhausting retries, the conflict should become fatal."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    captured = {}

    async def fake_start_polling(**kwargs):
        captured["error_callback"] = kwargs["error_callback"]

    # Make start_polling fail on retries to exhaust retries
    call_count = {"n": 0}

    async def failing_start_polling(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call (initial connect) succeeds
            captured["error_callback"] = kwargs["error_callback"]
        else:
            # Retry calls fail
            raise Exception("Connection refused")

    updater = SimpleNamespace(
        start_polling=AsyncMock(side_effect=failing_start_polling),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(set_my_commands=AsyncMock())
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr("gateway.platforms.telegram.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

    # Speed up retries for testing
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    ok = await adapter.connect()
    assert ok is True

    conflict = type("Conflict", (Exception,), {})

    # Directly call _handle_polling_conflict to avoid event-loop scheduling
    # complexity.  Each call simulates one 409 from Telegram.
    for i in range(4):
        await adapter._handle_polling_conflict(
            conflict("Conflict: terminated by other getUpdates request")
        )

    # After 3 failed retries (count 1-3 each enter the retry branch but
    # start_polling raises), the 4th conflict pushes count to 4 which
    # exceeds MAX_CONFLICT_RETRIES (3), entering the fatal branch.
    assert adapter.fatal_error_code == "telegram_polling_conflict", (
        f"Expected fatal after 4 conflicts, got code={adapter.fatal_error_code}, "
        f"count={adapter._polling_conflict_count}"
    )
    assert adapter.has_fatal_error is True
    fatal_handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_marks_retryable_fatal_error_for_startup_network_failure(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    builder = MagicMock()
    builder.token.return_value = builder
    app = SimpleNamespace(
        bot=SimpleNamespace(),
        updater=SimpleNamespace(),
        add_handler=MagicMock(),
        initialize=AsyncMock(side_effect=RuntimeError("Temporary failure in name resolution")),
        start=AsyncMock(),
    )
    builder.build.return_value = app
    monkeypatch.setattr("gateway.platforms.telegram.Application", SimpleNamespace(builder=MagicMock(return_value=builder)))

    ok = await adapter.connect()

    assert ok is False
    assert adapter.fatal_error_code == "telegram_connect_error"
    assert adapter.fatal_error_retryable is True
    assert "Temporary failure in name resolution" in adapter.fatal_error_message


@pytest.mark.asyncio
async def test_disconnect_skips_inactive_updater_and_app(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))

    updater = SimpleNamespace(running=False, stop=AsyncMock())
    app = SimpleNamespace(
        updater=updater,
        running=False,
        stop=AsyncMock(),
        shutdown=AsyncMock(),
    )
    adapter._app = app

    warning = MagicMock()
    monkeypatch.setattr("gateway.platforms.telegram.logger.warning", warning)

    await adapter.disconnect()

    updater.stop.assert_not_awaited()
    app.stop.assert_not_awaited()
    app.shutdown.assert_awaited_once()
    warning.assert_not_called()


@pytest.mark.asyncio
async def test_callback_query_is_routed_as_structured_interaction_event():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.handle_message = AsyncMock()

    button = SimpleNamespace(text="Approve", callback_data="hermes:approve")
    query = SimpleNamespace(
        data="hermes:approve",
        from_user=SimpleNamespace(id=42, full_name="Alice"),
        answer=AsyncMock(),
        message=SimpleNamespace(
            message_id=99,
            text="Choose one",
            caption=None,
            message_thread_id=None,
            date=None,
            chat=SimpleNamespace(id=123, type="private", title=None, full_name="Alice"),
            reply_markup=SimpleNamespace(inline_keyboard=[[button]]),
        ),
    )
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, None)

    query.answer.assert_awaited_once()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "Approve"
    assert event.reply_to_message_id == "99"
    assert event.reply_to_text == "Choose one"
    assert event.metadata["interaction"]["kind"] == "button"
    assert event.metadata["interaction"]["action"] == "approve"
    assert event.metadata["interaction"]["label"] == "Approve"


@pytest.mark.asyncio
async def test_callback_query_ignores_non_hermes_data():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.handle_message = AsyncMock()

    query = SimpleNamespace(
        data="foreign:approve",
        from_user=SimpleNamespace(id=42, full_name="Alice"),
        answer=AsyncMock(),
        message=SimpleNamespace(
            message_id=99,
            text="Choose one",
            caption=None,
            message_thread_id=None,
            date=None,
            chat=SimpleNamespace(id=123, type="private", title=None, full_name="Alice"),
            reply_markup=SimpleNamespace(inline_keyboard=[]),
        ),
    )
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, None)

    query.answer.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_skips_empty_formatted_chunks():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = SimpleNamespace(send_message=AsyncMock())

    result = await adapter.send("123", "")

    assert result.success is True
    assert result.raw_response == {"message_ids": []}
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_renders_controls_from_metadata(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)),
    )
    fake_markup = SimpleNamespace(inline_keyboard=[[SimpleNamespace(callback_data="hermes:approve")]])
    monkeypatch.setattr("gateway.platforms.telegram.build_telegram_reply_markup", lambda _controls: fake_markup)

    result = await adapter.send(
        "123",
        "Choose one",
        metadata={
            "controls": {
                "buttons": [
                    [{"label": "Approve", "action": "approve"}],
                    [{"label": "Docs", "url": "https://example.com"}],
                ]
            }
        },
    )

    assert result.success is True
    reply_markup = adapter._bot.send_message.await_args.kwargs["reply_markup"]
    assert reply_markup is fake_markup
