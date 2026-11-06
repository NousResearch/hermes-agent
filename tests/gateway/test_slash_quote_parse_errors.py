"""Messaging-side slash commands with an unclosed quote answer by name.

The bug class under test (#140803): the gateway's messaging handlers parsed the
user's raw command argument with ``shlex.split`` and no guard, so an apostrophe
or unclosed quote (``/kanban show it's``) raised ``ValueError: No closing
quotation`` and escaped to the message loop's catch-all as a generic internal
error. The bound-session classification path had already been fixed
(``226d854e8d``, Refs #134889 CMD-1); these tests pin the messaging-side
handlers to the same contract: a named parse-error reply, never an exception.
"""

import asyncio
from types import SimpleNamespace

import pytest


def _slash_host():
    """Minimal host exposing the messaging handlers without gateway.run's import chain."""
    from gateway.slash_commands import GatewaySlashCommandsMixin

    class _Stub(GatewaySlashCommandsMixin):
        pass

    return _Stub()


def _telegram_event(text: str) -> SimpleNamespace:
    from gateway.config import Platform

    source = SimpleNamespace(
        platform=Platform["TELEGRAM"],
        chat_id="chat1",
        chat_type="dm",
        thread_id=None,
        user_id="u1",
    )
    return SimpleNamespace(text=text, source=source, message_id="1", reply_to_message_id=None)


@pytest.mark.asyncio
async def test_kanban_unclosed_quote_is_a_named_refusal():
    """``/kanban show it's`` answers with the parse-error text, not an exception."""
    event = _telegram_event("/kanban show it's")

    out = await _slash_host()._handle_kanban_command(event)

    assert isinstance(out, str)
    assert "No closing quotation" in out  # the shlex cause rides the named reply
    assert "Could not parse" in out


@pytest.mark.asyncio
async def test_curator_unclosed_quote_is_a_named_refusal():
    """``/curator it's`` answers with the parse-error text, not an exception."""
    event = SimpleNamespace(get_command_args=lambda: "it's")

    out = await _slash_host()._handle_curator_command(event)

    assert isinstance(out, str)
    assert "No closing quotation" in out
    assert "Could not parse" in out


@pytest.mark.asyncio
async def test_kanban_paired_quotes_still_reach_the_cli(monkeypatch):
    """A paired quote keeps parsing: the argument reaches run_slash unchanged."""
    seen = {}

    async def fake_to_thread(fn, text):
        seen["fn"], seen["text"] = fn, text
        return "ok"

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    host = _slash_host()
    host._owns_kanban_dispatcher_lock = lambda: True
    event = _telegram_event('/kanban show "it\'s"')

    out = await host._handle_kanban_command(event)

    assert seen["text"] == 'show "it\'s"'
    assert out == "ok"
