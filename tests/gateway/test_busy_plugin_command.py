"""A PLUGIN slash command must be dispatched while an agent turn is in flight.

The busy fast-path resolves commands through the CORE ``hermes_cli.commands`` registry only
(``_hm_busy_slash_or_photo``), and its own comment says "Unrecognized commands and plain text fall
through." A plugin command (``PluginContext.register_command``) is not in that registry, so it fell
through into the queue / steer / interrupt paths — where the slash-command safety net DISCARDS
command text. The user got a command that did nothing: no reply, no error, no handler log, no retry.

Core commands were already routed around that trap by ``should_bypass_active_session``; plugin
commands were simply missed. These tests pin the busy path's behavior for all four cases: a plugin
command dispatches, its reply reaches the user, an access-denied plugin command is gated, and a
genuinely unknown command still falls through untouched.
"""

import contextlib
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from hermes_cli import plugins as _plugins_mod

PLUGIN_COMMAND = "bridge-start"


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    """Minimal runner: enough for the busy fast-path, no session store or agent."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._session_key_for_source = lambda source: "k"
    runner._session_env_scope = lambda context: contextlib.nullcontext()
    runner._check_slash_access = lambda source, canonical_cmd: None

    async def _run_inline(func, *args):
        return func(*args)

    runner._run_in_executor_with_context = _run_inline
    return runner


@pytest.fixture
def plugin(monkeypatch):
    """A registered plugin command, resolved through the real lookup path."""
    calls = []

    def _handler(args: str) -> str:
        calls.append(args)
        return f"bridge started {args}".strip()

    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_commands",
        lambda: {PLUGIN_COMMAND: {"description": "Start the bridge", "args_hint": ""}},
    )
    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_command_handler",
        lambda name: _handler if name.replace("_", "-") == PLUGIN_COMMAND else None,
    )
    return calls


@pytest.mark.asyncio
async def test_plugin_command_is_dispatched_while_busy(plugin):
    """The regression: mid-turn, the plugin command must be handled, not fall through."""
    runner = _make_runner()

    handled, result = await runner._hm_busy_slash_or_photo(
        _make_event(f"/{PLUGIN_COMMAND}"), _make_source(), "k"
    )

    assert handled is True, "plugin command fell through to the busy queue/steer paths"
    assert result == "bridge started"
    assert plugin == [""], "handler must have run exactly once"


@pytest.mark.asyncio
async def test_plugin_command_reply_reaches_the_user(plugin):
    """Through the real entry point, the reply is returned for delivery — not swallowed."""
    runner = _make_runner()

    reply = await runner._hm_handle_running_session_message(
        _make_event(f"/{PLUGIN_COMMAND} tail"), _make_source(), "k"
    )

    assert reply == "bridge started tail"
    assert plugin == ["tail"], "args must be passed through"


@pytest.mark.asyncio
async def test_plugin_command_args_are_passed_through(plugin):
    runner = _make_runner()

    handled, result = await runner._hm_busy_slash_or_photo(
        _make_event(f"/{PLUGIN_COMMAND} one two"), _make_source(), "k"
    )

    assert handled and result == "bridge started one two"


@pytest.mark.asyncio
async def test_denied_plugin_command_is_gated_while_busy(plugin):
    """Admin gating mirrors the cold path: a non-admin cannot run it mid-turn either."""
    runner = _make_runner()
    runner._check_slash_access = lambda source, canonical_cmd: f"denied:{canonical_cmd}"

    handled, result = await runner._hm_busy_slash_or_photo(
        _make_event(f"/{PLUGIN_COMMAND}"), _make_source(), "k"
    )

    assert handled is True
    assert result == f"denied:{PLUGIN_COMMAND}"
    assert plugin == [], "the handler must not run for a denied caller"


@pytest.mark.asyncio
async def test_unknown_command_still_falls_through(plugin):
    """The fix must not swallow everything: an unregistered name keeps the old behavior."""
    runner = _make_runner()

    handled, result = await runner._hm_busy_slash_or_photo(
        _make_event("/definitely-not-a-command"), _make_source(), "k"
    )

    assert handled is False
    assert result is None


@pytest.mark.asyncio
async def test_photo_followup_still_queues_without_interrupt(plugin):
    """The PHOTO branch after the plugin branch must stay reachable."""
    runner = _make_runner()
    merged = []
    runner._hm_merge_pending_for_source = lambda source, _quick_key, event, **kw: merged.append(_quick_key)

    event = MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=_make_source(),
        message_id="m1",
        media_urls=["/tmp/a.jpg"],
        media_types=["image/jpeg"],
    )
    handled, result = await runner._hm_busy_slash_or_photo(event, _make_source(), "k")

    assert (handled, result) == (True, None)
    assert merged == ["k"]