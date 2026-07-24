"""Gateway command help rendering tests."""

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text: str, platform: Platform) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=platform,
            chat_id="chat-1",
            user_id="user-1",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner():
    from gateway.run import GatewayRunner

    return object.__new__(GatewayRunner)


def test_start_is_known_gateway_command():
    """Telegram sends /start automatically; gateway should intercept it as a no-op."""
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    cmd = resolve_command("start")
    assert "start" in GATEWAY_KNOWN_COMMANDS
    assert cmd is not None
    assert cmd.name == "start"


@pytest.mark.asyncio
async def test_help_sanitizes_slash_command_mentions_for_telegram(monkeypatch):
    """Telegram help output must not expose invalid uppercase/hyphenated slashes."""
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {
            "/Linear": {"description": "Open Linear"},
            "/Custom-Thing": {"description": "Run a custom thing"},
        },
    )

    result = await _make_runner()._handle_help_command(
        _make_event("/help", Platform.TELEGRAM)
    )

    assert "`/linear`" in result
    assert "`/custom_thing`" in result
    assert "`/Linear`" not in result
    assert "`/Custom-Thing`" not in result


@pytest.mark.asyncio
async def test_commands_sanitizes_slash_command_mentions_for_telegram(monkeypatch):
    """Paginated Telegram /commands output uses Telegram-valid slash mentions."""
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/Linear": {"description": "Open Linear"}},
    )

    result = await _make_runner()._handle_commands_command(
        _make_event("/commands 999", Platform.TELEGRAM)
    )

    assert "`/linear`" in result
    assert "`/Linear`" not in result


@pytest.mark.asyncio
async def test_help_keeps_non_telegram_slash_command_mentions_unchanged(monkeypatch):
    """Only Telegram needs slash mentions rewritten to Telegram command names."""
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/Linear": {"description": "Open Linear"}},
    )

    result = await _make_runner()._handle_help_command(
        _make_event("/help", Platform.DISCORD)
    )

    assert "`/Linear`" in result


def test_gateway_help_labels_goal_file_as_local_backend_only():
    """The ``/goal --file`` help line must mark file input as local/backend
    only. The gateway runtime rejects ``--file`` for messaging platforms
    (see test_goal_file_command.py); the help text must tell the same story
    rather than implying a remote chat user can read a backend file.

    Reads from the same ``gateway_help_lines()`` the gateway help command
    renders, so a drift between the hint and the runtime rejection surfaces
    here instead of at a confused user.
    """
    from hermes_cli.commands import gateway_help_lines

    goal_lines = [ln for ln in gateway_help_lines() if ln.startswith("`/goal")]
    assert goal_lines, "no /goal line rendered in gateway help"
    goal_line = goal_lines[0]
    assert "--file" in goal_line
    # The label must signal this surface is unavailable on messaging
    # gateways, not advertise file reading to remote chat users.
    assert "CLI/TUI/Desktop only" in goal_line
