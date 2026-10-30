"""Native slash interactions must deliver a command reply EXACTLY ONCE.

A Discord native slash command answered with the same text twice: once as the
ephemeral interaction reply (a hardcoded ``followup_msg`` in
``_run_simple_slash``) and once as a public channel message
(``BasePlatformAdapter._dispatch_inline_reply``, which publishes the gateway's
own return value on the active-session bypass paths).

Reported for ``/queue`` — "Queued for the next turn." appeared both ephemerally
and in the channel — but every ``_run_simple_slash`` caller that passes a
hardcoded ``followup_msg`` has the identical shape. The hardcoded string also
dropped the gateway's "(N queued)" depth suffix, so the ephemeral copy was
wrong as well as duplicated.

These tests drive the real ``_run_simple_slash`` -> ``handle_message`` ->
active-session bypass chain and count deliveries on BOTH surfaces.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig

from tests.gateway.test_discord_slash_commands import _ensure_discord_mock

_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _FakeTextChannel:
    """A channel that is NOT a discord.Thread or discord.DMChannel."""

    def __init__(self, channel_id=123, name="general", guild_name="TestGuild"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name, id=1)
        self.topic = None


@pytest.fixture
def adapter():
    a = DiscordAdapter(PlatformConfig(enabled=True, token="x"))
    a._client = SimpleNamespace(
        tree=MagicMock(), get_channel=lambda _id: None, fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    a._text_batch_delay_seconds = 0
    a._check_slash_authorization = AsyncMock(return_value=True)
    a._send_with_retry = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    a._busy_session_handler = None
    return a


def _interaction():
    return SimpleNamespace(
        channel=_FakeTextChannel(), channel_id=123, guild_id=456,
        user=SimpleNamespace(id=42, name="Jezza", display_name="Jezza"),
        response=SimpleNamespace(defer=AsyncMock()),
        edit_original_response=AsyncMock(), delete_original_response=AsyncMock(),
    )


def _mark_busy(adapter, interaction, command_text):
    """Install an active-session guard for the session this slash targets."""
    probe = adapter._build_slash_event(interaction, command_text)
    session_key = adapter._event_session_key(probe)
    adapter._active_sessions[session_key] = asyncio.Event()
    return session_key


def _deliveries(adapter, interaction):
    return (adapter._send_with_retry.await_count
            + interaction.edit_original_response.await_count
            + interaction.delete_original_response.await_count)


def _ephemeral_text(interaction):
    if interaction.edit_original_response.await_args is None:
        return None
    return interaction.edit_original_response.await_args.kwargs.get("content")


@pytest.mark.asyncio
async def test_native_queue_while_busy_delivers_exactly_once(adapter):
    """Native /queue on a BUSY session: ONE delivery, on the interaction.

    Before the fix: 2 (public _send_with_retry + ephemeral edit_original_response).
    """
    interaction = _interaction()
    _mark_busy(adapter, interaction, "/queue do the thing")
    adapter._message_handler = AsyncMock(return_value="Queued for the next turn.")

    await adapter._run_simple_slash(interaction, "/queue do the thing", "Queued for the next turn.")

    assert _deliveries(adapter, interaction) == 1, (
        f"expected ONE delivery, got public={adapter._send_with_retry.await_count} "
        f"ephemeral={interaction.edit_original_response.await_count}")
    adapter._send_with_retry.assert_not_awaited()
    assert _ephemeral_text(interaction) == "Queued for the next turn."


@pytest.mark.asyncio
async def test_native_queue_ephemeral_carries_gateway_depth_suffix(adapter):
    """The hardcoded followup dropped "(N queued)"; the interaction must show the
    gateway's own text."""
    interaction = _interaction()
    _mark_busy(adapter, interaction, "/queue second")
    adapter._message_handler = AsyncMock(return_value="Queued for the next turn. (2 queued)")

    await adapter._run_simple_slash(interaction, "/queue second", "Queued for the next turn.")

    assert _ephemeral_text(interaction) == "Queued for the next turn. (2 queued)"
    adapter._send_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_slash_ephemeral_shows_gateway_usage_error(adapter):
    """A usage error must reach the user, not be masked by the optimistic followup."""
    interaction = _interaction()
    _mark_busy(adapter, interaction, "/queue")
    adapter._message_handler = AsyncMock(return_value="Usage: /queue <prompt>")

    await adapter._run_simple_slash(interaction, "/queue", "Queued for the next turn.")

    assert _ephemeral_text(interaction) == "Usage: /queue <prompt>"
    adapter._send_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_typed_queue_still_publishes_one_public_message(adapter):
    """A text-typed /queue has no interaction to answer, so its public echo must
    survive untouched."""
    interaction = _interaction()
    session_key = _mark_busy(adapter, interaction, "/queue typed")
    event = adapter._build_slash_event(interaction, "/queue typed")
    adapter._message_handler = AsyncMock(return_value="Queued for the next turn.")

    await adapter.handle_message(event)

    assert session_key in adapter._active_sessions
    assert adapter._send_with_retry.await_count == 1
    assert adapter._send_with_retry.await_args.kwargs["content"] == "Queued for the next turn."
    interaction.edit_original_response.assert_not_awaited()


@pytest.mark.parametrize("command_text,followup_msg,gateway_reply", [
    ("/queue x", "Queued for the next turn.", "Queued for the next turn."),
    ("/reset", "New conversation started~", "\u2728 New session started!"),
    ("/reset", "Session reset~", "\u2728 New session started!"),
    ("/retry", "Retrying~", "Agent is running \u2014 `/retry` can't run mid-turn."),
    ("/status", "Status sent~", "Status: running"),
    ("/stop", "Stop requested~", "Stopped."),
    ("/update", "Update initiated~", "Updating..."),
    ("/restart", "Restart requested~", "Restarting..."),
    ("/bg x", "Background task started~", "Background task started."),
    ("/btw x", "Side question dispatched~", "Side question dispatched."),
])
@pytest.mark.asyncio
async def test_every_hardcoded_followup_caller_delivers_once_when_busy(
    adapter, command_text, followup_msg, gateway_reply
):
    """The whole class, not just /queue: every caller passing a hardcoded
    followup_msg doubled on the active-session bypass paths."""
    interaction = _interaction()
    _mark_busy(adapter, interaction, command_text)
    adapter._message_handler = AsyncMock(return_value=gateway_reply)

    await adapter._run_simple_slash(interaction, command_text, followup_msg)

    assert _deliveries(adapter, interaction) == 1, f"{command_text}: doubled"
    assert _ephemeral_text(interaction) == gateway_reply


@pytest.mark.asyncio
async def test_hardcoded_followup_survives_when_gateway_returns_nothing(adapter):
    """followup_msg is not dead code: with no gateway text it is the user's only
    feedback and must still render."""
    interaction = _interaction()
    _mark_busy(adapter, interaction, "/queue x")
    adapter._message_handler = AsyncMock(return_value=None)

    await adapter._run_simple_slash(interaction, "/queue x", "Queued for the next turn.")

    adapter._send_with_retry.assert_not_awaited()
    assert _ephemeral_text(interaction) == "Queued for the next turn."


@pytest.mark.asyncio
async def test_no_followup_and_no_gateway_text_deletes_placeholder(adapter):
    """A caller with no followup_msg and no gateway text keeps existing behavior:
    delete the deferred placeholder."""
    interaction = _interaction()
    _mark_busy(adapter, interaction, "/model")
    adapter._message_handler = AsyncMock(return_value=None)

    await adapter._run_simple_slash(interaction, "/model")

    adapter._send_with_retry.assert_not_awaited()
    interaction.edit_original_response.assert_not_awaited()
    interaction.delete_original_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_expired_interaction_keeps_public_echo(adapter):
    """When defer() fails there is no ephemeral surface left, so suppressing the
    public echo would deliver NOTHING."""

    class UnknownInteraction(Exception):
        status = 404
        code = 10062

    interaction = _interaction()
    interaction.response.defer = AsyncMock(side_effect=UnknownInteraction("Unknown interaction"))
    _mark_busy(adapter, interaction, "/queue x")
    adapter._message_handler = AsyncMock(return_value="Queued for the next turn.")

    await adapter._run_simple_slash(interaction, "/queue x", "Queued for the next turn.")

    assert adapter._send_with_retry.await_count == 1
    assert adapter._send_with_retry.await_args.kwargs["content"] == "Queued for the next turn."
    interaction.edit_original_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_oversized_gateway_reply_publishes_instead_of_truncating(adapter):
    """Discord rejects an interaction response over 2000 chars (error 50035); an
    oversized reply is published on the channel, where send() chunks it."""
    interaction = _interaction()
    _mark_busy(adapter, interaction, "/status")
    long_reply = "x" * (adapter.MAX_MESSAGE_LENGTH + 500)
    adapter._message_handler = AsyncMock(return_value=long_reply)

    await adapter._run_simple_slash(interaction, "/status", "Status sent~")

    assert adapter._send_with_retry.await_count == 1
    assert adapter._send_with_retry.await_args.kwargs["content"] == long_reply
    assert _ephemeral_text(interaction) == "Status sent~"


@pytest.mark.asyncio
async def test_dispatch_inline_reply_publishes_by_default(adapter):
    """_dispatch_inline_reply must be unchanged for any event that did not opt in."""
    interaction = _interaction()
    event = adapter._build_slash_event(interaction, "/status")
    adapter._message_handler = AsyncMock(return_value="hello")

    await adapter._dispatch_inline_reply(event)

    assert adapter._send_with_retry.await_count == 1
    assert adapter._send_with_retry.await_args.kwargs["content"] == "hello"
    assert event._deferred_reply_text is None


@pytest.mark.asyncio
async def test_dispatch_inline_reply_hands_back_when_suppressed(adapter):
    interaction = _interaction()
    event = adapter._build_slash_event(interaction, "/status")
    event._suppress_public_echo = True
    adapter._message_handler = AsyncMock(return_value="hello")

    await adapter._dispatch_inline_reply(event)

    adapter._send_with_retry.assert_not_awaited()
    assert event._deferred_reply_text == "hello"
