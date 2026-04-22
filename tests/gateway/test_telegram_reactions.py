"""Tests for Telegram message reactions tied to processing lifecycle hooks."""

import asyncio
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource


def _make_adapter(**extra_env):
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
    adapter._bot_message_cache = OrderedDict()
    adapter._bot_message_cache_lock = asyncio.Lock()
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


def test_reactions_enabled_with_1(monkeypatch):
    """TELEGRAM_REACTIONS=1 enables reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "1")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is True


def test_reactions_disabled_with_false(monkeypatch):
    """TELEGRAM_REACTIONS=false disables reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is False


def test_reactions_disabled_with_0(monkeypatch):
    """TELEGRAM_REACTIONS=0 disables reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "0")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is False


def test_reactions_disabled_with_no(monkeypatch):
    """TELEGRAM_REACTIONS=no disables reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "no")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is False


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
async def test_set_reaction_returns_false_without_bot(monkeypatch):
    """_set_reaction should return False when bot is not available."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter._bot = None

    result = await adapter._set_reaction("123", "456", "\U0001f440")
    assert result is False


@pytest.mark.asyncio
async def test_set_reaction_handles_api_error_gracefully(monkeypatch):
    """API errors during reaction should not propagate."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter._bot.set_message_reaction = AsyncMock(side_effect=RuntimeError("no perms"))

    result = await adapter._set_reaction("123", "456", "\U0001f440")
    assert result is False


# ── on_processing_start ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_start_adds_eyes_reaction(monkeypatch):
    """Processing start should add eyes reaction when enabled."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_start(event)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f440",
    )


@pytest.mark.asyncio
async def test_on_processing_start_skipped_when_disabled(monkeypatch):
    """Processing start should not react when reactions are disabled."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_start(event)

    adapter._bot.set_message_reaction.assert_not_awaited()


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
async def test_on_processing_complete_success(monkeypatch):
    """Successful processing should set thumbs-up reaction."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f44d",
    )


@pytest.mark.asyncio
async def test_on_processing_complete_failure(monkeypatch):
    """Failed processing should set thumbs-down reaction."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f44e",
    )


@pytest.mark.asyncio
async def test_on_processing_complete_skipped_when_disabled(monkeypatch):
    """Processing complete should not react when reactions are disabled."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.set_message_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_processing_complete_cancelled_keeps_existing_reaction(monkeypatch):
    """Expected cancellation should not replace the in-progress reaction."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    adapter._bot.set_message_reaction.assert_not_awaited()


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


def test_config_reactions_env_takes_precedence(monkeypatch, tmp_path):
    """Env var should take precedence over config.yaml for reactions."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "telegram": {
            "reactions": True,
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("TELEGRAM_REACTIONS") == "false"


# ── Inbound reactions: user reacts to a bot message ──────────────────


def _make_reaction_update(
    *,
    chat_id: int = 123,
    message_id: int = 456,
    old_emojis=(),
    new_emojis=("\U0001f44d",),
    user_id=42,
    user_name: str = "TestUser",
    chat_type: str = "private",
):
    """Build a minimal Update that looks like a message_reaction update."""
    old = tuple(SimpleNamespace(emoji=e) for e in old_emojis)
    new = tuple(SimpleNamespace(emoji=e) for e in new_emojis)
    chat = SimpleNamespace(id=chat_id, type=chat_type, title=None)
    user = (
        SimpleNamespace(id=user_id, full_name=user_name)
        if user_id is not None else None
    )
    reaction = SimpleNamespace(
        chat=chat,
        message_id=message_id,
        old_reaction=old,
        new_reaction=new,
        user=user,
        date=None,
    )
    return SimpleNamespace(message_reaction=reaction)


def test_inbound_reactions_disabled_by_default(monkeypatch):
    """TELEGRAM_INBOUND_REACTIONS unset → _inbound_reactions_enabled is False."""
    monkeypatch.delenv("TELEGRAM_INBOUND_REACTIONS", raising=False)
    adapter = _make_adapter()
    assert adapter._inbound_reactions_enabled() is False


def test_inbound_reactions_enabled_when_set_true(monkeypatch):
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "true")
    adapter = _make_adapter()
    assert adapter._inbound_reactions_enabled() is True


@pytest.mark.asyncio
async def test_remember_bot_message_noop_when_disabled(monkeypatch):
    """Cache stays empty when the feature is off so there's no overhead."""
    monkeypatch.delenv("TELEGRAM_INBOUND_REACTIONS", raising=False)
    adapter = _make_adapter()

    await adapter._remember_bot_message("chat", "msg", kind="regular")

    assert adapter._bot_message_cache == {}


@pytest.mark.asyncio
async def test_remember_bot_message_records_when_enabled(monkeypatch):
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "true")
    adapter = _make_adapter()

    await adapter._remember_bot_message(
        "chat-a",
        "987",
        kind="approval",
        thread_id="t-1",
        session_key="s-1",
    )

    cached = adapter._bot_message_cache[("chat-a", "987")]
    assert cached["kind"] == "approval"
    assert cached["thread_id"] == "t-1"
    assert cached["session_key"] == "s-1"


@pytest.mark.asyncio
async def test_remember_bot_message_evicts_oldest(monkeypatch):
    """Cache caps at _BOT_MSG_CACHE_MAX; oldest entries fall off."""
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "true")
    adapter = _make_adapter()
    from gateway.platforms.telegram import TelegramAdapter

    monkeypatch.setattr(TelegramAdapter, "_BOT_MSG_CACHE_MAX", 3)

    for mid in ("1", "2", "3", "4"):
        await adapter._remember_bot_message("c", mid, kind="regular")

    keys = list(adapter._bot_message_cache.keys())
    assert ("c", "1") not in keys
    assert ("c", "4") in keys
    assert len(adapter._bot_message_cache) == 3


@pytest.mark.asyncio
async def test_remember_bot_message_expires_ttl(monkeypatch):
    """Entries older than the TTL are pruned on the next insert."""
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "true")
    adapter = _make_adapter()
    from gateway.platforms.telegram import TelegramAdapter

    monkeypatch.setattr(TelegramAdapter, "_BOT_MSG_CACHE_TTL", 1)

    adapter._bot_message_cache[("c", "old")] = {
        "ts": time.time() - 10,
        "kind": "regular",
        "thread_id": None,
        "user_id": None,
        "session_key": None,
    }

    await adapter._remember_bot_message("c", "new", kind="regular")

    assert ("c", "old") not in adapter._bot_message_cache
    assert ("c", "new") in adapter._bot_message_cache


@pytest.mark.asyncio
async def test_inbound_reaction_unknown_message_is_ignored(monkeypatch):
    """Reaction on a message we didn't send must not reach handle_message."""
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "true")
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()

    update = _make_reaction_update(chat_id=999, message_id=111)
    await adapter._handle_message_reaction(update, context=None)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_inbound_reaction_on_cached_message_routes_added(monkeypatch):
    """👍 on a cached bot message emits a synthetic added event."""
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "true")
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    await adapter._remember_bot_message("123", "456", kind="approval", thread_id="99")

    update = _make_reaction_update(
        chat_id=123, message_id=456, old_emojis=(), new_emojis=("\U0001f44d",),
    )
    await adapter._handle_message_reaction(update, context=None)

    adapter.handle_message.assert_awaited_once()
    emitted = adapter.handle_message.await_args.args[0]
    assert isinstance(emitted, MessageEvent)
    assert emitted.text == "reaction:added:\U0001f44d"
    assert emitted.message_id == "456"
    assert emitted.message_type is MessageType.TEXT
    assert emitted.source.platform is Platform.TELEGRAM
    assert emitted.source.chat_id == "123"
    assert emitted.source.user_id == "42"
    assert emitted.source.thread_id == "99"


@pytest.mark.asyncio
async def test_inbound_reaction_removed_event(monkeypatch):
    """Removing an allowlisted emoji emits reaction:removed:EMOJI."""
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "true")
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    await adapter._remember_bot_message("1", "2", kind="regular")

    update = _make_reaction_update(
        chat_id=1, message_id=2, old_emojis=("\U0001f44d",), new_emojis=(),
    )
    await adapter._handle_message_reaction(update, context=None)

    adapter.handle_message.assert_awaited_once()
    emitted = adapter.handle_message.await_args.args[0]
    assert emitted.text == "reaction:removed:\U0001f44d"


@pytest.mark.asyncio
async def test_inbound_reaction_swap_emits_remove_then_add(monkeypatch):
    """old=[👍], new=[✅] → removed event then added event in order."""
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "true")
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    await adapter._remember_bot_message("1", "2", kind="regular")

    update = _make_reaction_update(
        chat_id=1,
        message_id=2,
        old_emojis=("\U0001f44d",),
        new_emojis=("✅",),
    )
    await adapter._handle_message_reaction(update, context=None)

    texts = [call.args[0].text for call in adapter.handle_message.await_args_list]
    assert texts == ["reaction:removed:\U0001f44d", "reaction:added:✅"]


@pytest.mark.asyncio
async def test_inbound_reaction_from_bot_itself_is_ignored(monkeypatch):
    """Lifecycle-style reactions from the bot's own account must not loop back."""
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "true")
    adapter = _make_adapter()
    adapter._bot.id = 42
    adapter.handle_message = AsyncMock()
    await adapter._remember_bot_message("123", "456", kind="regular")

    update = _make_reaction_update(
        chat_id=123, message_id=456, user_id=42, new_emojis=("\U0001f44d",),
    )
    await adapter._handle_message_reaction(update, context=None)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_inbound_reaction_unsupported_emoji_is_ignored(monkeypatch):
    """🎉 on a cached message emits nothing — it's outside the v1 allowlist."""
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "true")
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    await adapter._remember_bot_message("1", "2", kind="regular")

    update = _make_reaction_update(
        chat_id=1, message_id=2, old_emojis=(), new_emojis=("\U0001f389",),
    )
    await adapter._handle_message_reaction(update, context=None)

    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_inbound_reaction_anonymous_admin_routes_without_user(monkeypatch):
    """Anonymous group-admin reactions (user=None) must not crash and still route."""
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "true")
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    await adapter._remember_bot_message("1", "2", kind="regular")

    update = _make_reaction_update(
        chat_id=1,
        message_id=2,
        user_id=None,
        new_emojis=("\U0001f44d",),
    )
    await adapter._handle_message_reaction(update, context=None)

    adapter.handle_message.assert_awaited_once()
    emitted = adapter.handle_message.await_args.args[0]
    assert emitted.source.user_id is None


@pytest.mark.asyncio
async def test_inbound_reaction_custom_emoji_is_skipped(monkeypatch):
    """ReactionTypeCustomEmoji (no .emoji) is dropped without crashing."""
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "true")
    adapter = _make_adapter()
    adapter.handle_message = AsyncMock()
    await adapter._remember_bot_message("1", "2", kind="regular")

    custom = SimpleNamespace(custom_emoji_id="abc")  # no .emoji attribute
    reaction = SimpleNamespace(
        chat=SimpleNamespace(id=1, type="private", title=None),
        message_id=2,
        old_reaction=(),
        new_reaction=(custom,),
        user=SimpleNamespace(id=42, full_name="u"),
        date=None,
    )
    update = SimpleNamespace(message_reaction=reaction)

    await adapter._handle_message_reaction(update, context=None)

    adapter.handle_message.assert_not_awaited()


# ── config.py bridging for the new toggle ────────────────────────────


def test_config_bridges_telegram_inbound_reactions(monkeypatch, tmp_path):
    """gateway/config.py bridges telegram.inbound_reactions → TELEGRAM_INBOUND_REACTIONS."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "telegram": {
            "inbound_reactions": True,
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("TELEGRAM_INBOUND_REACTIONS") == "true"


def test_config_inbound_reactions_env_takes_precedence(monkeypatch, tmp_path):
    """Env var wins over config.yaml when both are set."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "telegram": {
            "inbound_reactions": True,
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_INBOUND_REACTIONS", "false")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("TELEGRAM_INBOUND_REACTIONS") == "false"
