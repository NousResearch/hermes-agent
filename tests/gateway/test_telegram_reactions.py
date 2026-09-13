"""Tests for Telegram message reactions tied to processing lifecycle hooks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource


def _make_adapter(**extra_env):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
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


def test_explicit_env_wins_over_materialized_yaml_default(monkeypatch):
    """TELEGRAM_REACTIONS=true must beat the stock ``reactions: false`` in config.yaml (#109032).

    Fresh installs materialize the whole default config tree, so ``_apply_yaml_config`` seeds
    ``extra["reactions"] = False`` even when the user never chose a value; the reader must still
    honour the explicitly set env var, like ``yaml_env_setter`` documents for the bridge.
    """
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter.config.extra["reactions"] = False
    assert adapter._reactions_enabled() is True


def test_bridged_yaml_false_without_explicit_env_still_disables(monkeypatch):
    """With no explicit env the YAML→env bridge writes 'false'; reactions stay off."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter()
    adapter.config.extra["reactions"] = False
    assert adapter._reactions_enabled() is False


def test_yaml_true_enables_when_env_unset(monkeypatch):
    """An explicit ``reactions: true`` in config.yaml enables reactions without any env var."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter()
    adapter.config.extra["reactions"] = True
    assert adapter._reactions_enabled() is True


def test_explicit_env_false_wins_over_yaml_true(monkeypatch):
    """An explicit TELEGRAM_REACTIONS=false also wins over a YAML ``reactions: true``."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter()
    adapter.config.extra["reactions"] = True
    assert adapter._reactions_enabled() is False


def test_scoped_miss_does_not_leak_default_profile_env(monkeypatch):
    """Under multiplex a scoped miss must not read another profile's process-env value (#72348)."""
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")  # default profile's bridged value
    adapter = _make_adapter()
    adapter.config.extra["reactions"] = False  # this profile's own YAML
    set_multiplex_active(True)
    token = set_secret_scope({"TELEGRAM_BOT_TOKEN": "222:b2"})
    try:
        assert adapter._reactions_enabled() is False
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


def test_scoped_env_hit_wins_over_own_yaml(monkeypatch):
    """A secondary profile's own scoped TELEGRAM_REACTIONS=true beats its YAML ``reactions: false``."""
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")  # default profile's value
    adapter = _make_adapter()
    adapter.config.extra["reactions"] = False
    set_multiplex_active(True)
    token = set_secret_scope({"TELEGRAM_BOT_TOKEN": "222:b2", "TELEGRAM_REACTIONS": "true"})
    try:
        assert adapter._reactions_enabled() is True
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


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


