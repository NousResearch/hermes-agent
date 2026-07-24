"""Tests for the ``gateway_platform_event`` observer hook (#64176's observer half).

Covers the normalized-envelope pattern that replaces raw-SDK handler args:
* the four ``gateway_*`` hooks are registered in ``VALID_HOOKS``
* ``BasePlatformAdapter._fire_gateway_hook`` routes to ``invoke_hook`` with a
  ``has_hook`` no-subscriber fast-path and per-call error isolation
* ``TelegramAdapter._normalize_platform_event`` maps an inbound PTB update to a
  stable ``{platform, event_type, payload}`` envelope (no raw SDK objects),
  including custom-emoji reactions
* ``_on_platform_update`` fires ``gateway_platform_event`` with that envelope
  and swallows normalization errors so the observer can't break the adapter
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# python-telegram-bot is an optional dep; mock it so the adapter imports
# (same shim as test_telegram_network_reconnect / test_telegram_plugin_handlers).
# ---------------------------------------------------------------------------
def _ensure_telegram_mock() -> None:
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402
from hermes_cli.plugins import VALID_HOOKS  # noqa: E402


def _adapter() -> TelegramAdapter:
    """Build a TelegramAdapter without the heavy __init__.

    _fire_gateway_hook / _normalize_platform_event only need self.name (a
    read-only property over self.platform), so set a stand-in platform.
    """
    a = object.__new__(TelegramAdapter)
    a.platform = SimpleNamespace(value="telegram")  # name -> "Telegram"
    return a


def _reaction(*, emoji=None, custom_emoji_id=None):
    """A PTB ReactionType stand-in.

    PTB exposes ``.emoji`` for standard-emoji reactions and
    ``.custom_emoji_id`` for custom-emoji reactions (one or the other). Set
    both explicitly so the MagicMock doesn't auto-supply a truthy attribute.
    """
    r = MagicMock()
    r.emoji = emoji
    r.custom_emoji_id = custom_emoji_id
    return r


def _reaction_update(reactions, chat_id=123, message_id=456):
    """A PTB Update stand-in carrying a message_reaction with ``reactions``."""
    update = MagicMock()
    update.message_reaction = MagicMock()
    update.message_reaction.chat.id = chat_id
    update.message_reaction.message_id = message_id
    update.message_reaction.new_reaction = list(reactions)
    return update


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

class TestHookRegistration:
    def test_gateway_hooks_are_valid(self):
        """register_hook rejects names not in VALID_HOOKS, so the four new
        platform-boundary hooks must be present there."""
        assert "gateway_platform_event" in VALID_HOOKS
        assert "gateway_session_titled" in VALID_HOOKS
        assert "gateway_message_delivered" in VALID_HOOKS
        assert "gateway_thread_created" in VALID_HOOKS


# ---------------------------------------------------------------------------
# BasePlatformAdapter._fire_gateway_hook — routing + isolation
# ---------------------------------------------------------------------------

class TestFireGatewayHook:
    def test_routes_to_invoke_hook_with_kwargs(self):
        a = _adapter()
        captured: dict = {}

        def fake_invoke(name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs

        mgr = MagicMock()
        mgr.has_hook.return_value = True
        mgr.invoke_hook.side_effect = fake_invoke

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            a._fire_gateway_hook(
                "gateway_platform_event",
                platform="telegram", event_type="reaction", payload={"emojis": ["x"]},
            )

        assert captured["name"] == "gateway_platform_event"
        assert captured["kwargs"] == {
            "platform": "telegram", "event_type": "reaction", "payload": {"emojis": ["x"]},
        }

    def test_skips_dispatch_when_no_subscriber(self):
        """has_hook False -> invoke_hook never called."""
        a = _adapter()
        mgr = MagicMock()
        mgr.has_hook.return_value = False

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            a._fire_gateway_hook("gateway_platform_event", platform="telegram")

        mgr.has_hook.assert_called_once_with("gateway_platform_event")
        mgr.invoke_hook.assert_not_called()

    def test_plugin_layer_error_is_isolated(self):
        """A raising invoke_hook OR get_plugin_manager must not propagate."""
        a = _adapter()
        mgr = MagicMock()
        mgr.has_hook.return_value = True
        mgr.invoke_hook.side_effect = RuntimeError("plugin boom")

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            a._fire_gateway_hook("gateway_platform_event", platform="telegram")  # no raise


# ---------------------------------------------------------------------------
# TelegramAdapter._normalize_platform_event — envelope normalization
# ---------------------------------------------------------------------------

class TestNormalizePlatformEvent:
    def test_standard_emoji_reaction_normalized(self):
        """A message_reaction update becomes {platform, event_type, payload} with
        exactly the fields a real plugin consumes — no raw SDK objects."""
        a = _adapter()
        update = _reaction_update([_reaction(emoji="\U0001F44E")], chat_id=123, message_id=456)

        assert a._normalize_platform_event(update) == {
            "platform": "telegram",
            "event_type": "reaction",
            "payload": {
                "emojis": ["\U0001F44E"],
                "custom_emoji_ids": [],
                "chat_id": "123",
                "message_id": "456",
                "thread_id": None,
            },
        }

    def test_custom_emoji_reaction_normalized(self):
        """Custom-emoji reactions expose custom_emoji_id (no .emoji) — captured
        separately so a string-joining consumer never sees None."""
        a = _adapter()
        update = _reaction_update([_reaction(custom_emoji_id="555123")])

        event = a._normalize_platform_event(update)
        assert event["payload"]["emojis"] == []
        assert event["payload"]["custom_emoji_ids"] == ["555123"]

    def test_mixed_reactions_split_correctly(self):
        """A reaction set with standard + custom emojis splits into both lists."""
        a = _adapter()
        update = _reaction_update([
            _reaction(emoji="\U0001F44D"),
            _reaction(custom_emoji_id="555"),
            _reaction(emoji="\U0001F525"),
        ])

        event = a._normalize_platform_event(update)
        assert event["payload"]["emojis"] == ["\U0001F44D", "\U0001F525"]
        assert event["payload"]["custom_emoji_ids"] == ["555"]

    def test_non_reaction_update_returns_none(self):
        """Unsupported update types return None (payload contracts pending #64231)."""
        a = _adapter()
        update = MagicMock()
        update.message_reaction = None  # e.g. an edited_message or chat_member update

        assert a._normalize_platform_event(update) is None


# ---------------------------------------------------------------------------
# TelegramAdapter._on_platform_update — fire-site
# ---------------------------------------------------------------------------

class TestOnPlatformUpdate:
    def test_fires_gateway_platform_event_with_envelope(self):
        a = _adapter()
        seen: list = []
        a._fire_gateway_hook = lambda name, **kw: seen.append((name, kw))  # type: ignore[assignment]

        asyncio.run(a._on_platform_update(
            _reaction_update([_reaction(emoji="\U0001F44E")], 123, 456), context=MagicMock(),
        ))

        assert len(seen) == 1
        name, kwargs = seen[0]
        assert name == "gateway_platform_event"
        assert kwargs["platform"] == "telegram"
        assert kwargs["event_type"] == "reaction"
        assert kwargs["payload"]["emojis"] == ["\U0001F44E"]
        assert kwargs["payload"]["chat_id"] == "123"

    def test_unsupported_update_does_not_fire(self):
        a = _adapter()
        seen: list = []
        a._fire_gateway_hook = lambda name, **kw: seen.append((name, kw))  # type: ignore[assignment]

        update = MagicMock()
        update.message_reaction = None
        asyncio.run(a._on_platform_update(update, context=MagicMock()))

        assert seen == []

    def test_normalize_error_does_not_propagate(self):
        """A malformed update that makes normalize raise must be swallowed — the
        observer can't break the adapter (regression guard for the try/except)."""
        a = _adapter()
        a._fire_gateway_hook = lambda *a_, **kw: pytest.fail("must not fire on normalize error")  # type: ignore[assignment]

        def boom(update):
            raise RuntimeError("malformed update")

        a._normalize_platform_event = boom  # type: ignore[assignment]
        asyncio.run(a._on_platform_update(MagicMock(), context=MagicMock()))  # must not raise
