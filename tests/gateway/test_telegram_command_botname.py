"""``/command@ourbot args`` from Telegram's group command menu reaches the gateway as ``/command args``.

Driven through the adapter's real command entry point (``_handle_command``).
"""

import asyncio
from types import SimpleNamespace

import pytest

from tests.gateway.test_telegram_group_gating import _bot_command_entity, _group_message, _make_adapter


def _dispatched_command(text):
    adapter = _make_adapter(require_mention=True)
    seen = []

    async def _capture(event):
        seen.append(event)

    adapter.handle_message = _capture
    command = text.split()[0].rstrip(":")
    msg = _group_message(text, entities=[_bot_command_entity(text, command)])
    asyncio.run(adapter._handle_command(SimpleNamespace(update_id=7, message=msg, effective_message=msg), SimpleNamespace()))
    assert len(seen) == 1, "the addressed command was not dispatched"
    return seen[0]


@pytest.mark.parametrize("text, command, args", [
    ("/reasoning@hermes_bot medium", "reasoning", "medium"),
    ("/reasoning@hermes_bot: high --global", "reasoning", "high --global"),
    ("/memory@hermes_bot approve abc123", "memory", "approve abc123"),
    ("/new@hermes_bot", "new", ""),
])
def test_command_menu_form_keeps_its_arguments(text, command, args):
    event = _dispatched_command(text)
    assert event.get_command() == command
    assert event.get_command_args() == args


def test_prose_mention_cleanup_is_unchanged():
    adapter = _make_adapter(require_mention=True)
    assert adapter._clean_bot_trigger_text("@hermes_bot, summarize this") == "summarize this"
    assert adapter._clean_bot_trigger_text("thanks @hermes_bot") == "thanks"
