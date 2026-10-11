"""Regression coverage for Discord mock identity across test directories."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.e2e.conftest import DiscordAdapter, discord
from tests.gateway.conftest import _ensure_discord_mock


def test_gateway_setup_keeps_e2e_discord_module_for_message_admission():
    """A gateway helper must not replace the Discord module imported by e2e."""
    e2e_discord = discord

    _ensure_discord_mock()

    assert discord is e2e_discord
    assert sys.modules["discord"] is e2e_discord

    adapter = SimpleNamespace(
        _dedup=SimpleNamespace(contains=MagicMock(return_value=False)),
        _client=SimpleNamespace(user=object()),
        _is_allowed_user=MagicMock(return_value=True),
        _warn_if_fail_closed_default=MagicMock(),
        _allowed_role_ids=set(),
        _self_is_explicitly_mentioned=MagicMock(return_value=False),
    )
    message = SimpleNamespace(
        id="message-1",
        author=SimpleNamespace(id="user-1", bot=False),
        type=e2e_discord.MessageType.default,
        channel=e2e_discord.DMChannel(),
        guild=None,
        mentions=[],
    )

    assert DiscordAdapter._discord_message_admission(adapter, message, claim=False) == (True, False)
