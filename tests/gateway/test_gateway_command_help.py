"""Gateway command help rendering tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.asyncio
async def test_commands_telegram_uses_interactive_browser_when_supported(monkeypatch):
    """Telegram /commands should use the adapter's interactive browser when available."""
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/Linear": {"description": "Open Linear"}},
    )

    runner = _make_runner()
    adapter = MagicMock()
    adapter.send_command_browser = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="42")
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    runner._reply_anchor_for_event = MagicMock(return_value=None)

    result = await runner._handle_commands_command(
        _make_event("/commands", Platform.TELEGRAM)
    )

    assert result is None
    adapter.send_command_browser.assert_awaited_once()
    kwargs = adapter.send_command_browser.await_args.kwargs
    entries = kwargs["entries"]
    assert any(entry["command_text"] == "/help" for entry in entries)
    assert any(entry["command_text"] == "/linear" for entry in entries)


@pytest.mark.asyncio
async def test_menu_telegram_uses_interactive_browser_when_supported():
    """Telegram /menu should use the adapter's interactive browser with curated entries."""
    runner = _make_runner()
    adapter = MagicMock()
    adapter.send_command_browser = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="84")
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    runner._reply_anchor_for_event = MagicMock(return_value=None)

    result = await runner._handle_menu_command(
        _make_event("/menu", Platform.TELEGRAM)
    )

    assert result is None
    adapter.send_command_browser.assert_awaited_once()
    kwargs = adapter.send_command_browser.await_args.kwargs
    assert kwargs["title"] == "Quick Menu"
    entries = kwargs["entries"]
    assert [entry["command_text"] for entry in entries] == [
        "/status",
        "/model",
        "/menu inbox",
        "/menu session",
        "/menu ops",
        "/menu help",
    ]


@pytest.mark.asyncio
async def test_menu_inbox_telegram_uses_interactive_browser_when_supported():
    """Telegram /menu inbox should open the curated Ronii inbox submenu."""
    runner = _make_runner()
    adapter = MagicMock()
    adapter.send_command_browser = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="85")
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    runner._reply_anchor_for_event = MagicMock(return_value=None)

    result = await runner._handle_menu_command(
        _make_event("/menu inbox", Platform.TELEGRAM)
    )

    assert result is None
    kwargs = adapter.send_command_browser.await_args.kwargs
    assert kwargs["title"] == "Quick Menu • Inbox"
    assert [entry["command_text"] for entry in kwargs["entries"]] == [
        "/inbox recent",
        "/inbox domain list",
        "/inbox help",
        "/menu main",
    ]


@pytest.mark.asyncio
async def test_inbox_recent_telegram_renders_recent_item_buttons():
    """Telegram /inbox recent should render recent-item action buttons."""
    runner = _make_runner()
    adapter = MagicMock()
    adapter.send_command_browser = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="86")
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._load_ronii_recent_inbox_items = MagicMock(
        return_value=[
            {"id": 17, "title": "A very important inbox item", "classification": "link_only", "status": "done"},
            {"id": 18, "title": "Second item", "classification": "dlq", "status": "failed"},
        ]
    )

    result = await runner._handle_inbox_command(
        _make_event("/inbox recent", Platform.TELEGRAM)
    )

    assert result is None
    kwargs = adapter.send_command_browser.await_args.kwargs
    assert kwargs["title"] == "Quick Menu • Inbox Recent"
    assert [entry["command_text"] for entry in kwargs["entries"]] == [
        "/inbox actions 17",
        "/inbox actions 18",
        "/menu inbox",
    ]


@pytest.mark.asyncio
async def test_inbox_actions_text_menu_contains_item_commands():
    """Non-Telegram /inbox actions should expose view/dig/reanalyze commands."""
    runner = _make_runner()

    result = await runner._handle_inbox_command(
        _make_event("/inbox actions 17", Platform.DISCORD)
    )

    assert result is not None
    assert "**Inbox Actions • 17**" in result
    assert "`/inbox view 17`" in result
    assert "`/inbox dig 17`" in result
    assert "`/inbox reanalyze 17`" in result


@pytest.mark.asyncio
async def test_inbox_view_routes_to_ronii_runner():
    """/inbox view delegates to the Ronii runner wrapper."""
    runner = _make_runner()
    runner._run_ronii_runner_command = MagicMock(return_value="VIEW OUTPUT")

    result = await runner._handle_inbox_command(
        _make_event("/inbox view 17", Platform.DISCORD)
    )

    runner._run_ronii_runner_command.assert_called_once_with("inbox_view", "17")
    assert result == "VIEW OUTPUT"


@pytest.mark.asyncio
async def test_menu_text_fallback_renders_compact_sections():
    """Non-Telegram /menu falls back to a compact text menu."""
    runner = _make_runner()

    result = await runner._handle_menu_command(
        _make_event("/menu ops", Platform.DISCORD)
    )

    assert result is not None
    assert "**Quick Menu • Ops**" in result
    assert "`/restart`" in result
    assert "`/menu main`" in result
