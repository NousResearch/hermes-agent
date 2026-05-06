"""Tests for plugin-provided Slack native UI extension hooks."""

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.slack import SlackAdapter
from hermes_cli.commands import slack_native_slashes
from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
    SlackActionHandler,
    SlackSlashCommand,
    SlackViewHandler,
    get_slack_extension_action_handlers,
    get_slack_extension_slash_commands,
    get_slack_extension_view_handlers,
)


def _install_manager(monkeypatch, mgr: PluginManager) -> None:
    mgr._discovered = True
    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", mgr)


def test_register_slack_extension_stores_handlers(monkeypatch):
    mgr = PluginManager()
    _install_manager(monkeypatch, mgr)
    ctx = PluginContext(PluginManifest(name="slack-ui"), mgr)

    def slash_handler(**_kwargs):
        return None

    def action_handler(**_kwargs):
        return None

    def view_handler(**_kwargs):
        return None

    ctx.register_slack_extension(
        slash_commands=[
            SlackSlashCommand(
                name="/Board UI",
                description="Open the board",
                usage_hint="[filters]",
                handler=slash_handler,
                ack_text="Opening board...",
            )
        ],
        actions=[SlackActionHandler("board_action", action_handler)],
        views=[SlackViewHandler("board_view", view_handler)],
    )

    slashes = get_slack_extension_slash_commands()
    actions = get_slack_extension_action_handlers()
    views = get_slack_extension_view_handlers()

    assert [command.name for command in slashes] == ["board-ui"]
    assert slashes[0].description == "Open the board"
    assert slashes[0].usage_hint == "[filters]"
    assert slashes[0].ack_text == "Opening board..."
    assert actions == [SlackActionHandler("board_action", action_handler)]
    assert views == [SlackViewHandler("board_view", view_handler)]


def test_register_slack_extension_rejects_builtin_command_collision(monkeypatch):
    mgr = PluginManager()
    _install_manager(monkeypatch, mgr)
    ctx = PluginContext(PluginManifest(name="bad-slack-ui"), mgr)

    ctx.register_slack_extension(
        slash_commands=[SlackSlashCommand(name="model", handler=lambda **_: None)]
    )

    assert get_slack_extension_slash_commands() == []


def test_register_slack_extension_rejects_duplicate_handlers(monkeypatch):
    mgr = PluginManager()
    _install_manager(monkeypatch, mgr)
    first = PluginContext(PluginManifest(name="first"), mgr)
    second = PluginContext(PluginManifest(name="second"), mgr)

    first.register_slack_extension(
        slash_commands=[SlackSlashCommand(name="board", handler=lambda **_: None)],
        actions=[SlackActionHandler("board_action", lambda **_: None)],
        views=[SlackViewHandler("board_view", lambda **_: None)],
    )
    second.register_slack_extension(
        slash_commands=[SlackSlashCommand(name="board", handler=lambda **_: None)],
        actions=[SlackActionHandler("board_action", lambda **_: None)],
        views=[SlackViewHandler("board_view", lambda **_: None)],
    )

    assert len(get_slack_extension_slash_commands()) == 1
    assert len(get_slack_extension_action_handlers()) == 1
    assert len(get_slack_extension_view_handlers()) == 1


def test_slack_native_slashes_includes_extension_commands(monkeypatch):
    mgr = PluginManager()
    _install_manager(monkeypatch, mgr)
    ctx = PluginContext(PluginManifest(name="slack-ui"), mgr)
    ctx.register_slack_extension(
        slash_commands=[
            SlackSlashCommand(
                name="workflow",
                description="Open an interactive workflow",
                handler=lambda **_: None,
            )
        ],
    )

    slashes = {name: (description, hint) for name, description, hint in slack_native_slashes()}

    assert "workflow" in slashes
    assert slashes["workflow"][0] == "Open an interactive workflow"


@pytest.mark.asyncio
async def test_slack_adapter_invokes_extension_handler_with_compatible_kwargs():
    adapter = SlackAdapter(PlatformConfig(token="xoxb-test"))
    seen = {}

    async def handler(adapter, command):
        seen["adapter"] = adapter
        seen["command"] = command

    await adapter._invoke_slack_extension_handler(
        handler,
        adapter=adapter,
        command={"command": "/workflow"},
        ack=lambda **_: None,
        ignored=True,
    )

    assert seen == {"adapter": adapter, "command": {"command": "/workflow"}}
