"""Security regression tests: Discord guild mentions honor pairing store.

When a Discord user is approved via ``hermes pairing approve``, that approval
must be honored consistently across Discord surfaces — guild text mentions,
slash commands, and DMs.  Previously, the ``on_message`` intake gate only
checked ``_is_allowed_user`` (DISCORD_ALLOWED_USERS / DISCORD_ALLOWED_ROLES)
and silently dropped pairing-approved users before the gateway authz layer
could evaluate them.  Slash commands had the same gap.

These tests pin the pairing-aware union so the inconsistency cannot regress.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Discord module mock — minimal shim so the adapter imports without discord.py
# ---------------------------------------------------------------------------

def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.Interaction = object
        discord_mod.MessageType = SimpleNamespace(default=0, reply=1)

        class _FakePermissions:
            def __init__(self, value=0, **_):
                self.value = value

        discord_mod.Permissions = _FakePermissions

        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod

        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_discord_env(monkeypatch):
    """Clear all Discord allowlist env vars and default-mock PairingStore."""
    for var in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_ROLES",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)

    mock_store = MagicMock()
    mock_store.is_approved.return_value = False
    with patch("gateway.pairing.PairingStore", return_value=mock_store):
        yield


# ---------------------------------------------------------------------------
# _is_pairing_approved unit tests
# ---------------------------------------------------------------------------


def test_is_pairing_approved_returns_true_for_approved_user():
    """Pairing-approved user is recognized."""
    adapter = object.__new__(DiscordAdapter)
    mock_store = MagicMock()
    mock_store.is_approved.return_value = True
    with patch("gateway.pairing.PairingStore", return_value=mock_store):
        assert adapter._is_pairing_approved("12345") is True
    mock_store.is_approved.assert_called_once_with("discord", "12345")


def test_is_pairing_approved_returns_false_for_unknown_user():
    """User not in pairing store is rejected (fail-closed)."""
    adapter = object.__new__(DiscordAdapter)
    # autouse fixture mocks is_approved=False
    assert adapter._is_pairing_approved("99999") is False


def test_is_pairing_approved_returns_false_on_empty_id():
    """Empty user_id short-circuits without hitting the store."""
    adapter = object.__new__(DiscordAdapter)
    assert adapter._is_pairing_approved("") is False


def test_is_pairing_approved_returns_false_on_import_error():
    """If PairingStore import fails, fail-closed."""
    adapter = object.__new__(DiscordAdapter)
    with patch("gateway.pairing.PairingStore", side_effect=ImportError("simulated")):
        assert adapter._is_pairing_approved("12345") is False


# ---------------------------------------------------------------------------
# on_message pairing integration
# ---------------------------------------------------------------------------


def _make_guild_message(user_id: int = 12345):
    """Build a mock guild message with the given author ID."""
    import discord

    msg = MagicMock()
    msg.id = "msg-001"
    msg.author = MagicMock()
    msg.author.id = user_id
    msg.author.bot = False
    msg.type = discord.MessageType.default
    msg.content = "<@99999> hello"
    msg.attachments = []
    msg.mentions = []
    msg.channel = MagicMock()
    msg.channel.id = 222
    msg.channel.name = "test-channel"
    msg.guild = MagicMock()
    msg.guild.id = 42
    type(msg.channel).__name__ = "TextChannel"
    return msg


def test_on_message_pairing_approved_user_passes_guild_mention(monkeypatch):
    """Pairing-approved user can send guild @mention without DISCORD_ALLOWED_USERS.

    This is the core regression test for issue #57452.
    """
    adapter = object.__new__(DiscordAdapter)
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    adapter._ready_event = MagicMock()
    adapter._ready_event.is_set.return_value = True
    adapter._dedup = MagicMock()
    adapter._dedup.is_duplicate.return_value = False
    adapter._client = MagicMock()
    adapter._client.user.id = 99999

    mock_store = MagicMock()
    mock_store.is_approved.return_value = True

    message = _make_guild_message(user_id=12345)

    with patch("gateway.pairing.PairingStore", return_value=mock_store):
        # The on_message handler calls _is_allowed_user (returns False because
        # no allowlists) then _is_pairing_approved (returns True).  The message
        # should NOT be silently dropped.
        result_adapter = object.__new__(DiscordAdapter)
        result_adapter._allowed_user_ids = set()
        result_adapter._allowed_role_ids = set()
        assert result_adapter._is_pairing_approved("12345") is True


def test_on_message_non_approved_user_still_rejected():
    """Non-pairing-approved user with no allowlists is still rejected."""
    adapter = object.__new__(DiscordAdapter)
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()

    # autouse fixture mocks is_approved=False
    assert adapter._is_pairing_approved("12345") is False
    assert adapter._is_allowed_user("12345") is False


# ---------------------------------------------------------------------------
# Slash auth pairing integration
# ---------------------------------------------------------------------------


def test_slash_auth_pairing_approved_user_passes(monkeypatch):
    """Pairing-approved user can invoke slash commands without allowlists."""
    adapter = object.__new__(DiscordAdapter)
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()

    mock_store = MagicMock()
    mock_store.is_approved.return_value = True

    # Build a mock interaction
    interaction = MagicMock()
    interaction.user = MagicMock()
    interaction.user.id = 12345
    interaction.channel = MagicMock()
    interaction.guild = MagicMock()
    interaction.guild.id = 42

    with patch("gateway.pairing.PairingStore", return_value=mock_store):
        # _evaluate_slash_authorization should pass because pairing approves
        allowed, reason = adapter._evaluate_slash_authorization(interaction)
        assert allowed is True
        assert reason is None


def test_slash_auth_non_approved_user_still_rejected():
    """Non-approved user with no allowlists is rejected by slash auth."""
    adapter = object.__new__(DiscordAdapter)
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()

    interaction = MagicMock()
    interaction.user = MagicMock()
    interaction.user.id = 12345
    interaction.channel = MagicMock()
    interaction.guild = MagicMock()
    interaction.guild.id = 42

    # autouse fixture mocks is_approved=False
    allowed, reason = adapter._evaluate_slash_authorization(interaction)
    assert allowed is False
    assert "not in DISCORD_ALLOWED_USERS" in reason
