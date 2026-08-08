"""Tests for the ``gateway_platform_event`` observer hook (#64176's observer half).

Covers the normalized-envelope pattern that replaces raw-SDK handler args:
* only ``gateway_platform_event`` is registered in ``VALID_HOOKS`` (no inert
  hook surface without a concrete fire site)
* the adapter forwards normalized events to a runner-owned callback; the runner
  performs the authoritative post-auth check before invoking plugins
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

from gateway.config import PlatformConfig


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
from gateway.run import GatewayRunner  # noqa: E402
from hermes_cli.plugins import (  # noqa: E402
    VALID_HOOKS,
    PluginContext,
    PluginManager,
    PluginManifest,
)


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
        absent (no inert surface without a concrete fire site); lock that in."""
        assert "gateway_platform_event" in VALID_HOOKS
        assert "gateway_session_titled" not in VALID_HOOKS
        assert "gateway_message_delivered" not in VALID_HOOKS
        assert "gateway_thread_created" not in VALID_HOOKS


# ---------------------------------------------------------------------------
# Runner-owned dispatch — authoritative post-auth gate + isolation
# ---------------------------------------------------------------------------

class TestRunnerDispatch:
    def test_authorized_event_routes_normalized_envelope(self):
        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized = lambda source: source.user_id == "777"
        mgr = MagicMock()
        mgr.has_hook.return_value = True
        source = _adapter()._source_from_reaction_for_auth(
            _auth_reaction_update(user_id=777)
        )
        event = {
            "platform": "telegram",
            "event_type": "reaction",
            "payload": {"chat_id": "123", "message_id": "456", "emojis": ["x"]},
        }

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            asyncio.run(runner._handle_gateway_platform_event(event, source))

        mgr.invoke_hook.assert_called_once_with("gateway_platform_event", **event)

    def test_unauthorized_event_never_reaches_hooks(self):
        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized = lambda source: False
        mgr = MagicMock()
        source = _adapter()._source_from_reaction_for_auth(
            _auth_reaction_update(user_id=777)
        )

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            asyncio.run(runner._handle_gateway_platform_event(
                {"platform": "telegram", "event_type": "reaction", "payload": {}},
                source,
            ))

        mgr.has_hook.assert_not_called()
        mgr.invoke_hook.assert_not_called()

    def test_skips_dispatch_when_no_subscriber(self):
        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized = lambda source: True
        mgr = MagicMock()
        mgr.has_hook.return_value = False
        source = _adapter()._source_from_reaction_for_auth(
            _auth_reaction_update(user_id=777)
        )

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            asyncio.run(runner._handle_gateway_platform_event(
                {"platform": "telegram", "event_type": "reaction", "payload": {}},
                source,
            ))

        mgr.has_hook.assert_called_once_with("gateway_platform_event")
        mgr.invoke_hook.assert_not_called()

    def test_plugin_layer_error_is_isolated(self):
        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized = lambda source: True
        mgr = MagicMock()
        mgr.has_hook.return_value = True
        mgr.invoke_hook.side_effect = RuntimeError("plugin boom")
        source = _adapter()._source_from_reaction_for_auth(
            _auth_reaction_update(user_id=777)
        )

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            asyncio.run(runner._handle_gateway_platform_event(
                {"platform": "telegram", "event_type": "reaction", "payload": {}},
                source,
            ))  # no raise


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
        """Unsupported update types return None until a concrete contract exists."""
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
        async def observe(event, source):
            seen.append((event, source))
        a.set_platform_event_handler(observe)

        asyncio.run(a._on_platform_update(
            _reaction_update([_reaction(emoji="\U0001F44E")], 123, 456), context=MagicMock(),
        ))

        assert len(seen) == 1
        event, source = seen[0]
        assert event["platform"] == "telegram"
        assert event["event_type"] == "reaction"
        assert event["payload"]["emojis"] == ["\U0001F44E"]
        assert event["payload"]["chat_id"] == "123"
        assert source.chat_id == "123"

    def test_unsupported_update_does_not_fire(self):
        a = _adapter()
        seen: list = []
        async def observe(event, source):
            seen.append((event, source))
        a.set_platform_event_handler(observe)

        update = MagicMock()
        update.message_reaction = None
        asyncio.run(a._on_platform_update(update, context=MagicMock()))

        assert seen == []

    def test_normalize_error_does_not_propagate(self):
        """A malformed update that makes normalize raise must be swallowed — the
        observer can't break the adapter (regression guard for the try/except)."""
        a = _adapter()
        async def observe(event, source):
            pytest.fail("must not fire on normalize error")
        a.set_platform_event_handler(observe)

        def boom(update):
            raise RuntimeError("malformed update")

        a._normalize_platform_event = boom  # type: ignore[assignment]
        asyncio.run(a._on_platform_update(MagicMock(), context=MagicMock()))  # must not raise


# ---------------------------------------------------------------------------
# TelegramAdapter._on_platform_update post-auth gate (#64176)
# ---------------------------------------------------------------------------

class TestOnPlatformUpdateAuthBoundary:
    def test_without_runner_callback_fails_closed(self):
        """A catch-all PTB handler must not expose pre-auth updates merely because
        adapter-local intake would defer an unknown DM to the pairing flow."""
        a = _adapter(extra={})
        seen: list = []
        # No set_platform_event_handler: the authoritative gateway boundary has
        # not been installed, so even a syntactically valid reaction is dropped.
        asyncio.run(a._on_platform_update(
            _auth_reaction_update(user_id=777), context=MagicMock(),
        ))

        assert seen == []

    def test_non_reaction_update_fails_closed(self):
        """A future event type whose update carries no message_reaction must
        NOT fire. _source_from_reaction_for_auth raises and the gate drops it
        (fail closed); without the guard the no-identity path would authorize
        and fire it despite the restrictive allow_from."""
        a = _adapter(extra={"allow_from": ["999"]})
        seen: list = []
        async def observe(event, source):
            seen.append((event, source))
        a.set_platform_event_handler(observe)

        update = MagicMock()
        update.message_reaction = None  # a future, not-yet-wired event type
        # Simulate that future normalization produced an event for it.
        a._normalize_platform_event = lambda u: {  # type: ignore[assignment]
            "platform": "telegram", "event_type": "future", "payload": {},
        }

        asyncio.run(a._on_platform_update(update, context=MagicMock()))

        assert seen == []  # fail closed: never fires without a real auth decision

    @pytest.mark.parametrize("missing", ["actor", "chat", "message_id"])
    def test_malformed_reaction_identity_fails_closed(self, missing):
        a = _adapter()
        seen = []

        async def observe(event, source):
            seen.append((event, source))

        a.set_platform_event_handler(observe)
        update = _auth_reaction_update(user_id=777)
        if missing == "actor":
            update.message_reaction.user = None
            update.message_reaction.actor_chat = None
        elif missing == "chat":
            update.message_reaction.chat = None
        else:
            update.message_reaction.message_id = None

        asyncio.run(a._on_platform_update(update, context=MagicMock()))

        assert seen == []


class TestProfileScopedPlatformEventHandler:
    def test_secondary_handler_stamps_profile_before_dispatch(self, monkeypatch):
        runner = object.__new__(GatewayRunner)
        captured = {}

        async def dispatch(event, source):
            captured["event"] = event
            captured["profile"] = source.profile

        runner._handle_gateway_platform_event = dispatch
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir", lambda name: None,
        )
        handler = runner._make_profile_platform_event_handler("work")
        source = _adapter()._source_from_reaction_for_auth(
            _auth_reaction_update(user_id=777)
        )

        asyncio.run(handler({"platform": "telegram", "event_type": "reaction", "payload": {}}, source))

        assert captured["profile"] == "work"


class TestFixturePluginObservationPath:
    def test_reaction_reaches_real_registered_plugin_callback(self):
        """Adapter -> normalized source/envelope -> runner auth -> real plugin bus."""
        manager = PluginManager()
        context = PluginContext(
            PluginManifest(name="reaction-fixture", source="user"), manager,
        )
        seen = []
        context.register_hook(
            "gateway_platform_event", lambda **event: seen.append(event),
        )

        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized = lambda source: source.user_id == "777"
        adapter = _adapter()
        adapter.set_platform_event_handler(runner._handle_gateway_platform_event)

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
            asyncio.run(adapter._on_platform_update(
                _auth_reaction_update(user_id=777), context=MagicMock(),
            ))

        assert seen == [{
            "platform": "telegram",
            "event_type": "reaction",
            "payload": {
                "emojis": ["\U0001F44D"],
                "custom_emoji_ids": [],
                "chat_id": "123",
                "message_id": "456",
                "thread_id": None,
            },
        }]


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

        # Five core handlers (default group, no group kwarg) plus the
        # gateway_platform_event observer alone in group 99, so it observes
        # alongside rather than displacing the core handlers.
        calls = app.add_handler.call_args_list
        assert len(calls) == 6
        assert len([c for c in calls if c.kwargs.get("group") == 99]) == 1
        assert len([c for c in calls if not c.kwargs]) == 5

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

    def test_transient_init_rebuild_uses_shared_registration(self, monkeypatch):
        """The real connect retry path must call the shared registration method
        for the rebuilt PTB Application, not duplicate only the core handlers."""
        a = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
        first_app = MagicMock()
        first_app.bot = MagicMock()
        first_app.initialize = MagicMock(side_effect=OSError("transient"))
        rebuilt_app = MagicMock()
        rebuilt_app.bot = MagicMock()
        rebuilt_app.initialize = MagicMock(side_effect=RuntimeError("stop after rebuild"))

        builder = MagicMock()
        builder.token.return_value = builder
        builder.request.return_value = builder
        builder.get_updates_request.return_value = builder
        builder.build.side_effect = [first_app, rebuilt_app]
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.Application",
            SimpleNamespace(builder=MagicMock(return_value=builder)),
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.HTTPXRequest", lambda **kwargs: MagicMock(),
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.discover_fallback_ips",
            lambda: _async_value([]),
        )
        monkeypatch.setattr("asyncio.sleep", MagicMock(return_value=_async_value(None)))
        monkeypatch.setattr(
            "gateway.status.acquire_scoped_lock",
            lambda scope, identity, metadata=None: (True, None),
        )
        a._register_handlers = MagicMock()

        result = asyncio.run(a.connect())

        assert result is False
        assert a._register_handlers.call_args_list == [
            ((first_app,), {}),
            ((rebuilt_app,), {}),
        ]


async def _async_value(value):
    return value
