"""Tests for the Telegram platform-boundary observer hooks.

Covers the ``telegram:update`` / ``telegram:send`` / ``telegram:edit`` hooks
that extend the plugin hook bus to the Telegram adapter boundary:

* the hooks are present in ``VALID_HOOKS`` (so ``ctx.register_hook`` accepts them)
* ``_fire_observer_hook`` routes to ``invoke_hook`` and injects the adapter
* a plugin-layer error never propagates back into send/edit/connect
* ``_on_platform_update`` fires ``telegram:update`` per inbound update
* ``send()`` / ``edit_message()`` fire their hooks with the documented kwargs
"""

from __future__ import annotations

import asyncio
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


# ---------------------------------------------------------------------------
# python-telegram-bot is an optional dep; mock it so TelegramAdapter imports
# without the package (same shim as test_telegram_network_reconnect).
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _HookCalled(Exception):
    """Sentinel raised by the raise-spy to short-circuit send()/edit_message()
    at the hook fire — proving the call site + kwargs without depending on the
    send/edit internals or a live bot."""


def _make_adapter() -> TelegramAdapter:
    """Build a TelegramAdapter without the heavy __init__.

    Only the attributes the hook path touches are set: ``platform`` (so the
    ``name`` property resolves for debug logging) and ``_bot`` (truthy, so the
    send/edit not-connected guards pass).
    """
    a = object.__new__(TelegramAdapter)
    a.platform = SimpleNamespace(value="telegram")  # name -> "Telegram"
    a._bot = MagicMock()
    return a


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

class TestHookRegistration:
    def test_hooks_are_valid(self):
        """register_hook rejects names not in VALID_HOOKS, so the three new
        platform-boundary hooks must be present there."""
        assert "telegram:update" in VALID_HOOKS
        assert "telegram:send" in VALID_HOOKS
        assert "telegram:edit" in VALID_HOOKS


# ---------------------------------------------------------------------------
# _fire_observer_hook — routing + isolation
# ---------------------------------------------------------------------------

class TestFireObserverHook:
    def test_invokes_hook_with_adapter_injected(self):
        """_fire_observer_hook forwards to invoke_hook and injects adapter=self,
        so plugins get platform access without the adapter passing self at every
        call site."""
        captured: dict = {}

        def fake_invoke(name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs

        mgr = MagicMock()
        mgr.invoke_hook.side_effect = fake_invoke
        a = _make_adapter()

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            a._fire_observer_hook("telegram:send", chat_id="1", content="hi")

        assert captured["name"] == "telegram:send"
        assert captured["kwargs"]["chat_id"] == "1"
        assert captured["kwargs"]["content"] == "hi"
        assert captured["kwargs"]["adapter"] is a  # auto-injected

    def test_plugin_layer_error_does_not_propagate(self):
        """If invoke_hook (or the plugin manager) raises, the adapter must swallow
        it so a misbehaving plugin can never break send/edit/connect."""
        a = _make_adapter()
        mgr = MagicMock()
        mgr.invoke_hook.side_effect = RuntimeError("plugin boom")

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            a._fire_observer_hook("telegram:send", chat_id="1", content="x")  # no raise

    def test_skips_dispatch_when_no_subscriber(self):
        """Hot path: when no plugin subscribes, invoke_hook is never called.
        send/edit fire per streaming chunk, so the no-subscriber common case
        must short-circuit before any dispatch machinery runs."""
        a = _make_adapter()
        mgr = MagicMock()
        mgr.has_hook.return_value = False

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            a._fire_observer_hook("telegram:send", chat_id="1", content="x")

        mgr.has_hook.assert_called_once_with("telegram:send")
        mgr.invoke_hook.assert_not_called()


# ---------------------------------------------------------------------------
# _on_platform_update — inbound fire-site
# ---------------------------------------------------------------------------

class TestOnPlatformUpdate:
    def test_fires_telegram_update_with_update_bot_context(self):
        a = _make_adapter()
        captured: dict = {}

        def fake_invoke(name, **kw):
            captured["name"] = name
            captured["kw"] = kw

        mgr = MagicMock()
        mgr.invoke_hook.side_effect = fake_invoke
        update = MagicMock(name="update")
        context = MagicMock(name="context")

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr):
            asyncio.run(a._on_platform_update(update, context))

        assert captured["name"] == "telegram:update"
        assert captured["kw"]["update"] is update
        assert captured["kw"]["context"] is context
        assert captured["kw"]["bot"] is a._bot
        assert captured["kw"]["adapter"] is a


# ---------------------------------------------------------------------------
# send() / edit_message() outbound fire-sites
# ---------------------------------------------------------------------------

class TestSendAndEditFireSites:
    """send() and edit_message() must call _fire_observer_hook with the documented
    name + kwargs. A raise-spy short-circuits at the fire site (the first thing
    each method does after its guards), proving the call without depending on
    the send/edit internals or a live bot."""

    def test_send_fires_telegram_send_hook(self):
        a = _make_adapter()
        seen: list = []

        def spy(name, **kwargs):
            seen.append((name, kwargs))
            raise _HookCalled

        a._fire_observer_hook = spy  # type: ignore[assignment]

        with pytest.raises(_HookCalled):
            asyncio.run(a.send("123", "hello", reply_to="456", metadata={"k": 1}))

        assert len(seen) == 1
        name, kwargs = seen[0]
        assert name == "telegram:send"
        assert kwargs == {
            "chat_id": "123", "content": "hello",
            "reply_to": "456", "metadata": {"k": 1},
        }

    def test_edit_fires_telegram_edit_hook(self):
        a = _make_adapter()
        seen: list = []

        def spy(name, **kwargs):
            seen.append((name, kwargs))
            raise _HookCalled

        a._fire_observer_hook = spy  # type: ignore[assignment]

        with pytest.raises(_HookCalled):
            asyncio.run(
                a.edit_message("123", "m1", "hello", finalize=True, metadata={"k": 1})
            )

        assert len(seen) == 1
        name, kwargs = seen[0]
        assert name == "telegram:edit"
        assert kwargs == {
            "chat_id": "123", "message_id": "m1", "content": "hello",
            "finalize": True, "metadata": {"k": 1},
        }

    def test_send_skips_hook_for_empty_content(self):
        """The whitespace guard fires before the hook, so an empty send (which
        returns early) must NOT fire telegram:send."""
        a = _make_adapter()
        seen: list = []
        a._fire_observer_hook = lambda *a_, **kw: seen.append((a_, kw))  # type: ignore[assignment]

        result = asyncio.run(a.send("123", "   "))  # whitespace-only

        assert result.success is True
        assert seen == []
