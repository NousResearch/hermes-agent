"""Tests for Telegram message reactions tied to processing lifecycle hooks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource


def _make_adapter(**extra):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=extra)
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
    adapter._intentional_reaction_targets = set()
    return adapter


def _make_event(chat_id: str = "123", message_id: str = "456") -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="private",
            user_id="42",
            user_name="TestUser",
        ),
        message_id=message_id,
    )


# ── _reactions_enabled ───────────────────────────────────────────────


def test_reactions_disabled_by_default(monkeypatch):
    """Telegram reactions should be disabled by default."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is False


def test_reactions_enabled_when_set_true(monkeypatch):
    """Setting TELEGRAM_REACTIONS=true enables reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is True


# ── _set_reaction ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_reaction_calls_bot_api(monkeypatch):
    """_set_reaction should call bot.set_message_reaction with correct args."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()

    result = await adapter._set_reaction("123", "456", "\U0001f440")

    assert result is True
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f440",
    )


@pytest.mark.asyncio
async def test_lifecycle_reaction_stays_disabled(monkeypatch):
    """The existing automatic processing reactions still honor the flag."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter()

    await adapter.on_processing_start(_make_event())
    await adapter.on_processing_complete(_make_event(), ProcessingOutcome.SUCCESS)

    adapter._bot.set_message_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_intentional_reaction_ignores_lifecycle_flag(monkeypatch):
    """The explicit Telegram reaction tool path is independent of lifecycle gating."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter()

    result = await adapter.add_reaction("123", "❤️", "456")

    assert result is True
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="❤",
    )


@pytest.mark.asyncio
async def test_lifecycle_completion_does_not_overwrite_intentional_reaction(monkeypatch):
    """A model-selected emoji must survive the automatic completion hook."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_start(event)
    assert await adapter.add_reaction("123", "❤️", "456") is True
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    bot = adapter._bot
    assert bot is not None
    assert bot.set_message_reaction.await_count == 2
    reactions = [call.kwargs["reaction"] for call in bot.set_message_reaction.await_args_list]
    assert reactions == ["👀", "❤"]
    assert adapter._intentional_reaction_targets == set()


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [(False, False), (True, True), ("yes", True), ("off", False)],
)
def test_application_handler_registration_gates_inbound_reactions(
    monkeypatch, enabled, expected,
):
    """Inbound reaction updates are registered only after explicit opt-in."""
    from plugins.platforms.telegram import adapter as telegram_module

    class FakeReactionHandler:
        MESSAGE_REACTION_UPDATED = "updated"

        def __init__(self, callback, **kwargs):
            self.callback = callback
            self.kwargs = kwargs

    monkeypatch.setattr(telegram_module, "MessageReactionHandler", FakeReactionHandler)
    adapter = _make_adapter(inbound_reactions=enabled)
    handlers = []
    app = SimpleNamespace(add_handler=lambda handler: handlers.append(handler))

    adapter._register_application_handlers(app)

    reaction_handlers = [
        handler for handler in handlers if isinstance(handler, FakeReactionHandler)
    ]
    assert bool(reaction_handlers) is expected
    if expected:
        assert reaction_handlers[0].callback == adapter._handle_message_reaction
        assert reaction_handlers[0].kwargs == {"message_reaction_types": "updated"}


def test_inbound_reactions_skip_cleanly_without_message_reaction_handler(
    monkeypatch, caplog,
):
    """Older PTB installs keep the rest of Telegram connected when opted in."""
    from plugins.platforms.telegram import adapter as telegram_module

    monkeypatch.setattr(telegram_module, "MessageReactionHandler", None)
    adapter = _make_adapter(inbound_reactions=True)
    handlers = []
    app = SimpleNamespace(add_handler=lambda handler: handlers.append(handler))

    with caplog.at_level("WARNING"):
        adapter._register_application_handlers(app)

    assert "MessageReactionHandler support" in caplog.text


def test_sent_index_edit_preserves_existing_thread_id(monkeypatch, tmp_path):
    from gateway import rich_sent_store

    monkeypatch.setattr(
        rich_sent_store,
        "_store_path",
        lambda: str(tmp_path / "state" / "rich_sent_index.json"),
    )
    rich_sent_store.record("123", "456", "first", thread_id="77")
    rich_sent_store.record("123", "456", "edited")

    assert rich_sent_store.lookup_entry("123", "456")["thread_id"] == "77"


def test_sent_index_is_not_expanded_when_inbound_reactions_are_disabled(monkeypatch):
    from gateway import rich_sent_store

    adapter = _make_adapter()
    adapter._rich_messages_enabled = False
    record = Mock()
    monkeypatch.setattr(rich_sent_store, "record", record)

    adapter._record_sent_message("123", "456", "answer")

    record.assert_not_called()


def test_sent_index_records_telegram_bot_owner(monkeypatch):
    from gateway import rich_sent_store

    adapter = _make_adapter(inbound_reactions=True)
    adapter._bot = SimpleNamespace(id=12345)
    record = Mock()
    monkeypatch.setattr(rich_sent_store, "record", record)

    adapter._record_sent_message("123", "456", "answer")

    assert record.call_args.kwargs["sender_id"] == 12345


@pytest.mark.asyncio
async def test_thread_fallback_indexes_returned_effective_thread():
    adapter = _make_adapter()
    adapter._is_bad_request_error = lambda error: True
    adapter._is_thread_not_found_error = lambda error: True
    adapter._prune_stale_dm_topic_binding = Mock()
    adapter._record_sent_message = Mock()
    returned = SimpleNamespace(
        message_id=999,
        message_thread_id=88,
        is_topic_message=True,
        chat=SimpleNamespace(is_forum=True),
        text="fallback message",
    )
    adapter._bot.send_message = AsyncMock(
        side_effect=[RuntimeError("Message thread not found"), returned]
    )

    result = await adapter._send_message_with_thread_fallback(
        chat_id="123",
        text="fallback message",
        message_thread_id=77,
    )

    assert result is returned
    assert adapter._record_sent_message.call_count == 1
    call = adapter._record_sent_message.call_args
    assert call.kwargs["effective_thread_id"] == 88


# ── on_processing_start ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_start_handles_missing_ids(monkeypatch):
    """Should handle events without chat_id or message_id gracefully."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SimpleNamespace(chat_id=None),
        message_id=None,
    )

    await adapter.on_processing_start(event)

    adapter._bot.set_message_reaction.assert_not_awaited()


# ── on_processing_complete ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_complete_cancelled_clears_reaction(monkeypatch):
    """Cancelled processing should clear the in-progress reaction.

    Without this clear, the 👀 reaction lingers on the user's message
    indefinitely (until another agent run swaps it for 👍/👎). On a
    ``/stop`` that ends a session, that reaction never gets cleaned up.
    """
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    # set_message_reaction with reaction=None clears all reactions on the
    # message (Bot API documented semantics; equivalent to Bot API 10.0's
    # deleteMessageReaction but works on PTB 22.6 already).
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction=None,
    )


@pytest.mark.asyncio
async def test_clear_reactions_handles_api_error_gracefully(monkeypatch):
    """API errors during clear should not propagate."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter._bot.set_message_reaction = AsyncMock(side_effect=RuntimeError("no perms"))

    result = await adapter._clear_reactions("123", "456")
    assert result is False


# ── config.py bridging ───────────────────────────────────────────────


def test_config_bridges_telegram_reactions(monkeypatch, tmp_path):
    """gateway/config.py bridges telegram.reactions to TELEGRAM_REACTIONS env var."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "telegram": {
            "reactions": True,
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Use setenv (not delenv) so monkeypatch registers cleanup even when
    # the var doesn't exist yet — load_gateway_config will overwrite it.
    monkeypatch.setenv("TELEGRAM_REACTIONS", "")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("TELEGRAM_REACTIONS") == "true"
