"""Tests for plugin-registered python-telegram-bot handlers.

Covers the new plugin API that mirrors ``register_slack_action_handler``:
* ``PluginContext.register_telegram_handler`` validation + queuing
* ``PluginManager.get_telegram_handlers`` accessor
* the ``(handler, group, plugin_name)`` tuple shape the Telegram adapter
  consumes at connect time

(The adapter-side ``connect()`` wiring — ``for h, g, _ in
get_telegram_handlers(): app.add_handler(h, group=g)`` — is an 8-line mirror
of the Slack adapter's action-handler wiring; its defensive load/reject
behaviour matches the pattern covered by
``test_slack_plugin_action_handlers.py``.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Ensure the repo root is importable when this test runs directly
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from hermes_cli.plugins import (  # noqa: E402
    PluginContext,
    PluginManager,
    PluginManifest,
)


# ---------------------------------------------------------------------------
# A minimal BaseHandler stand-in. The real telegram.ext.BaseHandler (PTB v22,
# pinned in pyproject.toml) exposes check_update() + handle_update() and has
# NO `handle` attribute. We mirror that contract exactly so the test exercises
# what a real MessageHandler/TypeHandler instance looks like — this is what
# caught (and now guards against) the earlier `handle`-attr validation bug.
# ---------------------------------------------------------------------------

class _FakeHandler:
    def __init__(self, name: str = "fake"):
        self.name = name

    async def check_update(self, update):  # pragma: no cover - never dispatched
        return False

    async def handle_update(self, *a, **kw):  # pragma: no cover
        ...


class _NotAHandler:
    """Something a plugin might mistakenly register."""
    pass


def _make_ctx(name: str = "test_plugin") -> tuple[PluginManager, PluginContext]:
    """Build a fresh PluginManager + PluginContext bound to it."""
    mgr = PluginManager()
    manifest = PluginManifest(name=name, version="0.1.0", description="test")
    ctx = PluginContext(manifest=manifest, manager=mgr)
    return mgr, ctx


# ---------------------------------------------------------------------------
# PluginContext.register_telegram_handler — input validation + queuing
# ---------------------------------------------------------------------------

class TestRegisterTelegramHandlerAPI:
    """Behaviour of ctx.register_telegram_handler()."""

    def test_basehandler_is_queued_with_default_group(self):
        mgr, ctx = _make_ctx()
        h = _FakeHandler("reaction")

        ctx.register_telegram_handler(h)

        handlers = mgr.get_telegram_handlers()
        assert len(handlers) == 1
        handler, group, plugin_name = handlers[0]
        assert handler is h
        assert group == 0
        assert plugin_name == "test_plugin"

    def test_explicit_group_is_recorded(self):
        """Plugins use group=1 to run alongside built-ins without displacing them."""
        mgr, ctx = _make_ctx()
        h = _FakeHandler()
        ctx.register_telegram_handler(h, group=1)
        assert mgr.get_telegram_handlers()[0][1] == 1

    def test_accepts_real_ptb_v22_contract(self):
        """Regression: PTB v22 BaseHandler has check_update/handle_update, no `handle`.

        A real MessageHandler/TypeHandler instance carries no ``handle``
        attribute, so the validator must key on ``handle_update`` — not
        ``handle``. An earlier version required ``handle`` and silently
        rejected every real PTB handler; this pins the correct contract.
        """
        mgr, ctx = _make_ctx()
        h = _FakeHandler("reaction")
        assert not hasattr(h, "handle"), "fake must mirror real PTB (no handle attr)"

        # Must not raise — a real PTB handler would have failed the old check.
        ctx.register_telegram_handler(h)
        assert mgr.get_telegram_handlers()[0][0] is h

    def test_non_handler_raises(self):
        """A non-BaseHandler object must be rejected, not silently stored."""
        _mgr, ctx = _make_ctx()
        with pytest.raises(ValueError, match="BaseHandler"):
            ctx.register_telegram_handler(_NotAHandler())  # type: ignore[arg-type]

    def test_none_handler_raises(self):
        _mgr, ctx = _make_ctx()
        with pytest.raises(ValueError, match="BaseHandler"):
            ctx.register_telegram_handler(None)  # type: ignore[arg-type]

    def test_get_telegram_handlers_returns_copy(self):
        """The accessor returns a copy so callers can't mutate plugin state."""
        mgr, ctx = _make_ctx()
        ctx.register_telegram_handler(_FakeHandler())

        handlers = mgr.get_telegram_handlers()
        handlers.clear()
        assert len(mgr.get_telegram_handlers()) == 1

    def test_multiple_plugins_each_recorded(self):
        mgr = PluginManager()
        ctx_a = PluginContext(
            manifest=PluginManifest(name="plug_a", version="0", description=""),
            manager=mgr,
        )
        ctx_b = PluginContext(
            manifest=PluginManifest(name="plug_b", version="0", description=""),
            manager=mgr,
        )
        ctx_a.register_telegram_handler(_FakeHandler("a"), group=1)
        ctx_b.register_telegram_handler(_FakeHandler("b"), group=1)

        handlers = mgr.get_telegram_handlers()
        assert {h[2] for h in handlers} == {"plug_a", "plug_b"}
        # Registration order preserved (PTB group ordering is significant).
        assert [h[0].name for h in handlers] == ["a", "b"]


# ---------------------------------------------------------------------------
# Contract the Telegram adapter relies on at connect time
# ---------------------------------------------------------------------------

class TestTelegramHandlerContract:
    """The shape get_telegram_handlers() yields is exactly what connect() unpacks."""

    def test_tuple_is_handler_group_pluginname(self):
        mgr, ctx = _make_ctx()
        h = _FakeHandler()
        ctx.register_telegram_handler(h, group=2)

        entry = mgr.get_telegram_handlers()[0]
        assert len(entry) == 3
        handler, group, plugin_name = entry  # unpacks like the adapter loop
        assert handler is h
        assert group == 2
        assert isinstance(plugin_name, str)


# ---------------------------------------------------------------------------
# TelegramAdapter connect-path wiring
# ---------------------------------------------------------------------------
# Exercises TelegramAdapter._wire_plugin_handlers() — the connect-time code
# path that consumes get_telegram_handlers() and calls Application.add_handler.
# python-telegram-bot is an optional dep, so mock the telegram package the same
# way tests/gateway/test_telegram_network_reconnect.py does.

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


class _RecordingApp:
    """Stand-in PTB Application that records add_handler(handler, group=...)."""

    def __init__(self, raise_on: tuple = ()):
        self.added: list = []  # list of (handler, group)
        self._raise_on = set(raise_on)

    def add_handler(self, handler, group=None):
        if getattr(handler, "name", None) in self._raise_on:
            raise RuntimeError("PTB rejected handler")
        self.added.append((handler, group))


class TestTelegramAdapterPluginHandlerWiring:
    """_wire_plugin_handlers() (the connect path) registers handlers on the app."""

    def _adapter(self, app) -> TelegramAdapter:
        # object.__new__ skips the heavy __init__. _wire_plugin_handlers only
        # needs self._app and self.name; `name` is a read-only property over
        # self.platform (Platform.TELEGRAM.value.title() -> "Telegram"), so set
        # a stand-in platform rather than the property itself.
        a = object.__new__(TelegramAdapter)
        a.platform = SimpleNamespace(value="telegram")
        a._app = app
        return a

    def _mgr(self, handlers):
        mgr = MagicMock()
        mgr.get_telegram_handlers.return_value = handlers
        return mgr

    def test_handlers_wired_with_their_groups(self):
        """Each queued handler is add_handler()'d with its requested group."""
        app = _RecordingApp()
        adapter = self._adapter(app)
        h0, h1 = _FakeHandler("a"), _FakeHandler("b")

        with patch("hermes_cli.plugins.get_plugin_manager",
                   return_value=self._mgr([(h0, 0, "plug_a"), (h1, 1, "plug_b")])):
            adapter._wire_plugin_handlers()

        assert app.added == [(h0, 0), (h1, 1)]

    def test_no_handlers_is_a_noop(self):
        """Empty queue (the common case) wires nothing."""
        app = _RecordingApp()
        adapter = self._adapter(app)

        with patch("hermes_cli.plugins.get_plugin_manager",
                   return_value=self._mgr([])):
            adapter._wire_plugin_handlers()

        assert app.added == []

    def test_plugin_manager_load_failure_is_isolated(self):
        """If get_plugin_manager() raises, wiring is skipped — connect still safe."""
        app = _RecordingApp()
        adapter = self._adapter(app)

        with patch("hermes_cli.plugins.get_plugin_manager",
                   side_effect=RuntimeError("plugin layer down")):
            adapter._wire_plugin_handlers()  # must not raise

        assert app.added == []

    def test_one_rejected_handler_does_not_block_others(self):
        """A handler PTB rejects must not stop later handlers from wiring."""
        app = _RecordingApp(raise_on=("bad",))
        adapter = self._adapter(app)
        good, bad, other = _FakeHandler("good"), _FakeHandler("bad"), _FakeHandler("other")

        with patch("hermes_cli.plugins.get_plugin_manager",
                   return_value=self._mgr([(bad, 0, "buggy"), (good, 1, "g"), (other, 2, "o")])):
            adapter._wire_plugin_handlers()

        assert app.added == [(good, 1), (other, 2)]
