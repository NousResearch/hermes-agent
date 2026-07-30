"""Tests for the ``gateway_platform_event`` observer hook (#64176's observer half).

Covers the normalized-envelope pattern that replaces raw-SDK handler args:
* only ``gateway_platform_event`` is registered in ``VALID_HOOKS`` (no inert
  hook surface pending #64231)
* ``BasePlatformAdapter._fire_gateway_hook`` routes to ``invoke_hook`` with a
  ``has_hook`` no-subscriber fast-path and per-call error isolation
* ``TelegramAdapter._normalize_platform_event`` maps an inbound PTB update to a
  stable ``{platform, event_type, payload}`` envelope (no raw SDK objects),
  including custom-emoji reactions
* ``_on_platform_update`` fires ``gateway_platform_event`` with that envelope,
  gated on the same authorization decision as inbound gateway traffic
  (unauthorized reactions never fire), and swallows errors so the observer
  can't break the adapter
* ``_register_handlers`` is the single PTB handler registration site, so a
  rebuild re-registers the observer alongside the core handlers
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


def _adapter(extra=None) -> TelegramAdapter:
    """Build a TelegramAdapter without the heavy __init__.

    _fire_gateway_hook / _normalize_platform_event / the post-auth gate only
    need self.name (a read-only property over self.platform) and self.config,
    so set stand-ins. The default config opens auth (allow_from=["*"]) so a
    normal reaction fires; pass a restrictive ``extra`` to exercise the gate.
    """
    a = object.__new__(TelegramAdapter)
    a.platform = SimpleNamespace(value="telegram")  # name -> "Telegram"
    a.config = SimpleNamespace(extra=extra if extra is not None else {"allow_from": ["*"]})
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


def _auth_reaction_update(user_id, chat_type="private", chat_id=123, message_id=456):
    """A PTB Update stand-in carrying a message_reaction with an actor identity.

    Wraps ``_reaction_update`` and pins the reactor's user id + chat type so the
    post-auth gate has an identity to authorize against.
    """
    update = _reaction_update(
        [_reaction(emoji="\U0001F44D")], chat_id=chat_id, message_id=message_id,
    )
    update.message_reaction.user.id = str(user_id)
    update.message_reaction.chat.type = chat_type
    return update


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

class TestHookRegistration:
    def test_gateway_platform_event_registered_reserved_absent(self):
        """register_hook rejects names not in VALID_HOOKS, so the implemented
        hook must be present. The reserved gateway_* names are deliberately
        absent (no inert surface pending #64231); lock that in."""
        assert "gateway_platform_event" in VALID_HOOKS
        assert "gateway_session_titled" not in VALID_HOOKS
        assert "gateway_message_delivered" not in VALID_HOOKS
        assert "gateway_thread_created" not in VALID_HOOKS


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


# ---------------------------------------------------------------------------
# TelegramAdapter._on_platform_update post-auth gate (#64176)
# ---------------------------------------------------------------------------

class TestOnPlatformUpdateAuthGate:
    """The catch-all sees every inbound update. A reaction from a sender the
    message intake would reject must NOT reach plugins, using the same
    authorization decision as inbound gateway traffic."""

    def test_unauthorized_reaction_does_not_fire(self):
        a = _adapter(extra={"allow_from": ["999"]})  # reactor 777 not allowed
        seen: list = []
        a._fire_gateway_hook = lambda name, **kw: seen.append((name, kw))  # type: ignore[assignment]

        asyncio.run(a._on_platform_update(
            _auth_reaction_update(user_id=777), context=MagicMock(),
        ))

        assert seen == []

    def test_authorized_reaction_fires(self):
        a = _adapter(extra={"allow_from": ["777"]})  # reactor 777 allowed
        seen: list = []
        a._fire_gateway_hook = lambda name, **kw: seen.append((name, kw))  # type: ignore[assignment]

        asyncio.run(a._on_platform_update(
            _auth_reaction_update(user_id=777), context=MagicMock(),
        ))

        assert len(seen) == 1
        assert seen[0][0] == "gateway_platform_event"

    def test_open_config_defers_to_pairing_flow(self):
        """With no allow_from and no env allowlist, intake defers (open) and the
        event fires, matching the message intake pairing flow."""
        a = _adapter(extra={})
        seen: list = []
        a._fire_gateway_hook = lambda name, **kw: seen.append((name, kw))  # type: ignore[assignment]
        cleared = {k: "" for k in (
            "TELEGRAM_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_USERS",
            "TELEGRAM_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
            "GATEWAY_ALLOW_ALL_USERS",
        )}

        with patch.dict("os.environ", cleared, clear=False):
            asyncio.run(a._on_platform_update(
                _auth_reaction_update(user_id=777), context=MagicMock(),
            ))

        assert len(seen) == 1

    def test_non_reaction_update_fails_closed(self):
        """A future event type whose update carries no message_reaction must
        NOT fire. _source_from_reaction_for_auth raises and the gate drops it
        (fail closed); without the guard the no-identity path would authorize
        and fire it despite the restrictive allow_from."""
        a = _adapter(extra={"allow_from": ["999"]})
        seen: list = []
        a._fire_gateway_hook = lambda name, **kw: seen.append((name, kw))  # type: ignore[assignment]

        update = MagicMock()
        update.message_reaction = None  # a future, not-yet-wired event type
        # Simulate that future normalization produced an event for it.
        a._normalize_platform_event = lambda u: {  # type: ignore[assignment]
            "platform": "telegram", "event_type": "future", "payload": {},
        }

        asyncio.run(a._on_platform_update(update, context=MagicMock()))

        assert seen == []  # fail closed: never fires without a real auth decision


# ---------------------------------------------------------------------------
# TelegramAdapter._register_handlers single registration site (#64176)
# ---------------------------------------------------------------------------

class TestRegisterHandlers:
    """_register_handlers is the sole PTB handler registration site, so a
    handler added there is registered on every (re)build that calls it. The
    #64176 review asked to share registration between the initial path and any
    rebuild; these tests pin that the observer (group 99) is included alongside
    the core handlers."""

    _HANDLER_ATTRS = (
        "_handle_text_message", "_handle_command", "_handle_location_message",
        "_handle_media_message", "_handle_callback_query", "_on_platform_update",
    )

    def _adapter_with_handlers(self) -> TelegramAdapter:
        a = _adapter()
        # Stand-ins for the bound handler methods. _register_handlers only
        # passes them to add_handler, it never calls them.
        for name in self._HANDLER_ATTRS:
            setattr(a, name, object())
        return a

    @staticmethod
    def _observer_calls(app):
        return [c for c in app.add_handler.call_args_list if c.kwargs.get("group") == 99]

    def test_registers_core_handlers_plus_observer(self):
        a = self._adapter_with_handlers()
        app = MagicMock()
        a._register_handlers(app)

        # Five core handlers (default group) plus the gateway_platform_event
        # observer in group 99.
        assert app.add_handler.call_count == 6
        assert len(self._observer_calls(app)) == 1

    def test_rebuild_re_registers_observer(self):
        """A second call on a fresh app (e.g. a future rebuild) re-registers
        every handler, observer included."""
        a = self._adapter_with_handlers()
        first_app = MagicMock()
        rebuilt_app = MagicMock()

        a._register_handlers(first_app)
        a._register_handlers(rebuilt_app)  # the rebuild path

        assert rebuilt_app.add_handler.call_count == 6
        assert len(self._observer_calls(rebuilt_app)) == 1
