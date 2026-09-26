"""A plugin command handler must see the chat's current HERMES_SESSION_ID inside its scope.

``_hm_dispatch_quick_and_plugin_commands`` binds HERMES_SESSION_* via ``_session_env_scope`` but
``_set_session_env`` never passed ``session_id``, and the dispatch built its context without a
session entry — so ``set_session_vars`` bound ``HERMES_SESSION_ID=""`` and a handler keying state
by session id had nothing to match the ``old_session_id`` the next ``/new`` reports in
``on_session_reset`` (#123245). The routing entry is now peeked read-only (never minted) so the
handler records the live id, and the scope still clears it on exit.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_context import get_session_env


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_runner(entries=None):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._draining = False
    runner._hm_quick_commands = lambda: {}
    if entries is not None:
        runner.session_store = SimpleNamespace(
            _generate_session_key=lambda _source: "sk-test",
            _entries=entries,
        )
    return runner


@pytest.mark.asyncio
async def test_plugin_command_binds_the_routing_sessions_id(monkeypatch):
    """With a live routing entry the handler sees its session id, and only inside the scope."""
    from hermes_cli import plugins as _plugins_mod

    seen = []

    async def _handler(args: str) -> str:
        seen.append(get_session_env("HERMES_SESSION_ID"))
        return "ok"

    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_command_handler",
        lambda name: _handler if name == "planmode" else None,
    )

    runner = _make_runner(
        entries={"sk-test": SimpleNamespace(session_id="sess-live-42")}
    )
    event = MessageEvent(text="/planmode on", source=_make_source(), message_id="m1")

    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(
        event, event.source, "planmode"
    )

    assert (handled, result, command) == (True, "ok", "planmode")
    assert seen == ["sess-live-42"]
    # The scope is exited after dispatch: the var is explicitly cleared, not left behind.
    assert get_session_env("HERMES_SESSION_ID") == ""


@pytest.mark.asyncio
async def test_plugin_command_without_entry_binds_empty_not_minted(monkeypatch):
    """No routing entry yet: the handler reads "" and no session is created on its behalf."""
    from hermes_cli import plugins as _plugins_mod

    seen = []

    async def _handler(args: str) -> str:
        seen.append(get_session_env("HERMES_SESSION_ID"))
        return "ok"

    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_command_handler",
        lambda name: _handler if name == "planmode" else None,
    )

    runner = _make_runner(entries={})
    event = MessageEvent(text="/planmode on", source=_make_source(), message_id="m1")

    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(
        event, event.source, "planmode"
    )

    assert (handled, result, command) == (True, "ok", "planmode")
    assert seen == [""]


@pytest.mark.asyncio
async def test_plugin_command_without_session_store_binds_empty(monkeypatch):
    """A runner with no session store at all still dispatches and binds "" (#108698 parity)."""
    from hermes_cli import plugins as _plugins_mod

    seen = []

    async def _handler(args: str) -> str:
        seen.append(get_session_env("HERMES_SESSION_ID"))
        return "ok"

    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_command_handler",
        lambda name: _handler if name == "planmode" else None,
    )

    runner = _make_runner(entries=None)
    event = MessageEvent(text="/planmode on", source=_make_source(), message_id="m1")

    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(
        event, event.source, "planmode"
    )

    assert (handled, result, command) == (True, "ok", "planmode")
    assert seen == [""]
