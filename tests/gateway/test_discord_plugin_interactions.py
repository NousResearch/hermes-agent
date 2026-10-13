"""Behavior contract for explicitly private/public plugin Discord interactions."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from hermes_cli.plugin_interactions import (
    DiscordInteractionReply,
    PrivateInteractionUnavailable,
    dispatch_plugin_interaction,
)


@dataclass
class SendResult:
    success: bool = True


class FakeResponse:
    def __init__(self, *, fail_defer=False):
        self.events = []
        self.fail_defer = fail_defer
        self.done = False

    def is_done(self):
        return self.done

    async def defer(self, *, ephemeral):
        self.events.append(("defer", ephemeral))
        if self.fail_defer:
            raise RuntimeError("simulated private transport failure")
        self.done = True

    async def send_message(self, content, *, ephemeral):
        self.events.append(("initial", content, ephemeral))
        self.done = True


class FakeInteraction:
    def __init__(self, *, invalid_user=False, fail_defer=False, fail_private=False):
        self.id = 1001
        self.user = SimpleNamespace(
            id="bad" if invalid_user else 2002, name="test-user", display_name="Test User",
        )
        self.guild_id = 3003
        self.channel_id = 4004
        self.channel = SimpleNamespace(id=4004, guild=SimpleNamespace(id=3003))
        self.response = FakeResponse(fail_defer=fail_defer)
        self.private_fail = fail_private
        self.events = self.response.events
        self.followup = SimpleNamespace(send=self._followup)

    async def _followup(self, *, content, ephemeral):
        self.events.append(("followup", content, ephemeral))
        if self.private_fail:
            raise RuntimeError("simulated private followup failure")

    async def edit_original_response(self, *, content):
        self.events.append(("edit_original", content, True))
        if self.private_fail:
            raise RuntimeError("simulated private initial response failure")

    async def delete_original_response(self):
        self.events.append(("delete_original",))


class FakeAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, channel_id, content):
        self.sent.append((channel_id, content))
        return SendResult()


@pytest.mark.asyncio
async def test_private_reply_and_followups_only_use_ephemeral_transport():
    interaction = FakeInteraction()
    adapter = FakeAdapter()
    seen = []

    async def handler(args, context):
        seen.append((args, context.id, context.user.id, context.server_id, context.channel_id))
        return [DiscordInteractionReply.private("private initial"),
                DiscordInteractionReply.private("private followup")]

    ok = await dispatch_plugin_interaction(
        interaction=interaction, adapter=adapter, command_name="account", raw_args="show", handler=handler,
    )

    assert ok is True
    assert seen == [("show", "1001", "2002", "3003", "4004")]
    assert ("defer", True) in interaction.events
    assert ("edit_original", "private initial", True) in interaction.events
    assert ("followup", "private followup", True) in interaction.events
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_public_roleplay_reply_uses_channel_transport():
    interaction = FakeInteraction()
    adapter = FakeAdapter()

    async def handler(_args, _context):
        return DiscordInteractionReply.public("Sarfina steps into the room.")

    ok = await dispatch_plugin_interaction(
        interaction=interaction, adapter=adapter, command_name="scene", raw_args="", handler=handler,
    )

    assert ok is True
    assert adapter.sent == [("4004", "Sarfina steps into the room.")]
    assert ("defer", True) in interaction.events
    assert ("delete_original",) in interaction.events
    assert not any(event[0] in {"edit_original", "followup"} for event in interaction.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_user", [True, False])
async def test_missing_or_invalid_interaction_identity_is_rejected_without_dispatch(invalid_user):
    interaction = FakeInteraction(invalid_user=invalid_user)
    if not invalid_user:
        interaction.id = None
    adapter = FakeAdapter()
    called = []

    async def handler(*_args):
        called.append(True)
        return DiscordInteractionReply.public("must not be sent")

    ok = await dispatch_plugin_interaction(
        interaction=interaction, adapter=adapter, command_name="private", raw_args="", handler=handler,
    )

    assert ok is False
    assert called == []
    assert adapter.sent == []
    assert interaction.events == [("initial", "Invalid interaction context.", True)]


@pytest.mark.asyncio
async def test_private_ack_failure_rejects_before_plugin_and_never_falls_back_public():
    interaction = FakeInteraction(fail_defer=True)
    adapter = FakeAdapter()
    called = []

    async def handler(*_args):
        called.append(True)
        return DiscordInteractionReply.public("must not be sent")

    with pytest.raises(PrivateInteractionUnavailable):
        await dispatch_plugin_interaction(
            interaction=interaction, adapter=adapter, command_name="private", raw_args="", handler=handler,
        )

    assert called == []
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_private_delivery_failure_suppresses_explicit_public_outputs():
    interaction = FakeInteraction(fail_private=True)
    adapter = FakeAdapter()

    async def handler(_args, _context):
        return [DiscordInteractionReply.private("secret"), DiscordInteractionReply.public("narration")]

    with pytest.raises(PrivateInteractionUnavailable):
        await dispatch_plugin_interaction(
            interaction=interaction, adapter=adapter, command_name="private", raw_args="", handler=handler,
        )

    assert adapter.sent == []


def test_context_requires_discord_snowflake_ids():
    from hermes_cli.plugin_interactions import interaction_context_from_sdk

    interaction = FakeInteraction()
    interaction.channel_id = "not-a-snowflake"
    with pytest.raises(ValueError):
        interaction_context_from_sdk(interaction, "private")


def test_legacy_plugin_command_registration_remains_separate_from_interaction_api():
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    context = PluginContext(PluginManifest(name="contract-test", source="user"), manager)
    legacy_handler = lambda raw: raw
    interaction_handler = lambda raw, _context: DiscordInteractionReply.private(raw)

    context.register_command("legacy", legacy_handler, description="Existing command")
    context.register_discord_interaction_command(
        "private", interaction_handler, description="Private interaction", args_hint="<action>",
    )

    assert manager._plugin_commands["legacy"]["handler"] is legacy_handler
    assert manager._plugin_commands["legacy"]["description"] == "Existing command"
    assert manager._plugin_discord_interaction_commands["private"]["handler"] is interaction_handler
    assert "private" not in manager._plugin_commands
    assert "legacy" not in manager._plugin_discord_interaction_commands


@pytest.mark.asyncio
async def test_native_callback_runs_adapter_auth_gate_before_private_defer(monkeypatch):
    from plugins.platforms.discord.plugin_interactions import register_plugin_interaction_commands

    called = []
    handler = lambda args, _context: called.append(args) or DiscordInteractionReply.private("ok")
    monkeypatch.setattr(
        "hermes_cli.plugin_interaction_api.get_plugin_discord_interaction_commands",
        lambda: {"private": {"handler": handler, "description": "Private", "args_hint": "<action>"}},
    )

    class FakeTree:
        def __init__(self):
            self.commands = []

        def add_command(self, command):
            self.commands.append(command)

    class FakeAppCommands:
        @staticmethod
        def Command(*, name, description, callback):
            return SimpleNamespace(name=name, description=description, callback=callback)

        @staticmethod
        def describe(**_kwargs):
            return lambda callback: callback

    discord_module = SimpleNamespace(Interaction=FakeInteraction, app_commands=FakeAppCommands)
    tree = FakeTree()
    adapter = SimpleNamespace(_check_slash_authorization=lambda *_args: _deny())

    async def _deny():
        return False

    assert register_plugin_interaction_commands(adapter, tree, discord_module, set(), 99) == 1
    interaction = FakeInteraction()
    await tree.commands[0].callback(interaction, "show")

    assert called == []
    assert interaction.events == []
