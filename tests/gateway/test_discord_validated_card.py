"""Regression tests for the Discord-native validated card/action vertical slice (#436)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.card import CardValidationError, parse_discord_card


def _card(*, actions=None):
    return {
        "title": "Daily check-in",
        "body": "Protein and training are on plan.",
        "fields": [{"name": "Calories", "value": "2,100 kcal"}],
        "actions": actions if actions is not None else [
            {"id": "log", "label": "Log meal", "style": "primary"},
        ],
    }


def _adapter():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._allowed_user_ids = {"42"}
    adapter._allowed_role_ids = set()
    adapter._client = MagicMock()
    return adapter


def _interaction(user_id="42"):
    embed = MagicMock()
    embed.color = None
    embed.set_footer = MagicMock()
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, display_name="Tester", roles=[]),
        response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock()),
        message=SimpleNamespace(embeds=[embed]),
        channel_id=9001,
    )


def test_parse_discord_card_produces_plain_content_and_typed_actions():
    card = parse_discord_card(_card())

    assert card.title == "Daily check-in"
    assert card.content == "**Daily check-in**\n\nProtein and training are on plan.\n\n**Calories**\n2,100 kcal"
    assert [action.id for action in card.actions] == ["log"]


@pytest.mark.parametrize(
    ("payload", "path", "code"),
    [
        ({**_card(), "unsafe": {"raw": "discord payload"}}, "$.unsafe", "unknown_field"),
        (_card(actions=[{"id": "log", "label": "One"}, {"id": "log", "label": "Two"}]), "$.actions[1].id", "duplicate_action_id"),
        (_card(actions=[{"id": "log", "label": "x" * 81}]), "$.actions[0].label", "too_long"),
    ],
)
def test_parse_discord_card_rejects_unrepresentable_or_unsafe_input(payload, path, code):
    with pytest.raises(CardValidationError) as raised:
        parse_discord_card(payload)

    assert raised.value.path == path
    assert raised.value.code == code


@pytest.mark.asyncio
async def test_send_card_validates_before_transport_side_effects():
    adapter = _adapter()
    channel = MagicMock()
    channel.send = AsyncMock()
    adapter._client.get_channel.return_value = channel

    result = await adapter.send_card(
        "9001", {"title": "Daily check-in", "unsafe": True}, handlers={}
    )

    assert result.success is False
    assert "$.unsafe" in result.error
    channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_card_renders_validated_actions_and_invokes_the_selected_handler_once():
    adapter = _adapter()
    channel = MagicMock()
    channel.send = AsyncMock(return_value=SimpleNamespace(id=1234))
    adapter._client.get_channel.return_value = channel
    seen = []

    async def log_meal(interaction):
        seen.append((interaction.user.id, interaction.channel_id))

    result = await adapter.send_card("9001", _card(), handlers={"log": log_meal})

    assert result.success is True
    assert result.message_id == "1234"
    kwargs = channel.send.await_args.kwargs
    assert kwargs["content"] == parse_discord_card(_card()).content
    assert kwargs["embed"].title == "Daily check-in"
    view = kwargs["view"]
    interaction = _interaction()

    await view._on_action(interaction, "log")
    await view._on_action(interaction, "log")

    assert seen == [("42", 9001)]
    interaction.response.edit_message.assert_awaited_once()
    interaction.response.send_message.assert_awaited_once()
    assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_card_action_rejects_an_unauthorized_click_without_invoking_the_handler():
    adapter = _adapter()
    channel = MagicMock()
    channel.send = AsyncMock(return_value=SimpleNamespace(id=1234))
    adapter._client.get_channel.return_value = channel
    handler = AsyncMock()

    result = await adapter.send_card("9001", _card(), handlers={"log": handler})

    assert result.success is True
    view = channel.send.await_args.kwargs["view"]
    interaction = _interaction(user_id="99")
    await view._on_action(interaction, "log")

    handler.assert_not_awaited()
    interaction.response.edit_message.assert_not_awaited()
    interaction.response.send_message.assert_awaited_once()
    assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_display_only_card_sends_without_a_view_or_action_handlers():
    adapter = _adapter()
    channel = MagicMock()
    channel.send = AsyncMock(return_value=SimpleNamespace(id=5678))
    adapter._client.get_channel.return_value = channel

    result = await adapter.send_card("9001", _card(actions=[]), handlers={})

    assert result.success is True
    assert result.message_id == "5678"
    assert "view" not in channel.send.await_args.kwargs


@pytest.mark.asyncio
async def test_send_card_rejects_handlers_that_do_not_exactly_match_validated_actions():
    adapter = _adapter()
    channel = MagicMock()
    channel.send = AsyncMock()
    adapter._client.get_channel.return_value = channel

    result = await adapter.send_card("9001", _card(), handlers={"other": AsyncMock()})

    assert result.success is False
    assert "handler_mismatch" in result.error
    channel.send.assert_not_awaited()
