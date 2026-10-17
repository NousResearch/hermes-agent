"""A Slack ``!command`` needs the command name to touch the ``!``.

``! model opus`` is chat, not a command: rewriting it produced ``/ model opus``,
a COMMAND event with an empty command name that neither dispatched ``/model``
nor reached the agent as the text the user typed.
"""

import importlib
import sys
from importlib.machinery import PathFinder
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType


def _load_installed_package(name):
    if PathFinder.find_spec(name) is None:
        return None
    prefix = f"{name}."
    displaced = {
        m: sys.modules.pop(m)
        for m in tuple(sys.modules)
        if (m == name or m.startswith(prefix)) and not isinstance(sys.modules[m], ModuleType)
    }
    try:
        return importlib.import_module(name)
    except ImportError:
        sys.modules.update(displaced)
        return None


_load_installed_package("slack_bolt")
_load_installed_package("slack_sdk")

_slack_mod = importlib.import_module("plugins.platforms.slack.adapter")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("!model opus", "/model opus"),
        ("! model opus", "! model opus"),
        ("!  stop", "!  stop"),
        ("!\tstop", "!\tstop"),
        ("!\nstop", "!\nstop"),
        ("!\u00a0stop", "!\u00a0stop"),
        ("!", "!"),
        ("! nice work", "! nice work"),
    ],
)
def test_bang_rewrite_requires_command_touching_the_bang(text, expected):
    assert _slack_mod._rewrite_known_bang_command(text) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("typed", "delivered_text"),
    [("! model opus", "! model opus"), ("<@U_BOT> ! model opus", "! model opus")],
)
async def test_spaced_bang_reaches_the_agent_as_typed_text(typed, delivered_text):
    adapter = _slack_mod.SlackAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = "U_BOT"
    adapter.handle_message = AsyncMock()

    await adapter._handle_slack_message({
        "text": typed,
        "user": "U_USER",
        "channel": "D123",
        "channel_type": "im",
        "ts": "1234567890.000001",
    })

    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.text == delivered_text
    assert delivered.message_type == MessageType.TEXT
    assert delivered.get_command() is None
