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


@pytest.mark.asyncio
async def test_persona_command_lists_donna_personas():
    runner = _make_runner()
    runner._session_key_for_source = lambda _source: "telegram:c1:u1"

    result = await runner._handle_persona_command(
        _make_event("/persona list", Platform.TELEGRAM)
    )

    assert "Donna personas:" in result
    assert "`pm`" in result
    assert "Linear and ClickUp" in result
    assert "`eng`" in result
    assert "Use `/persona <name>`" in result


@pytest.mark.asyncio
async def test_persona_command_switches_and_clears_chat_persona():
    runner = _make_runner()
    runner._session_key_for_source = lambda _source: "telegram:c1:u1"

    switched = await runner._handle_persona_command(
        _make_event("/persona pm", Platform.TELEGRAM)
    )
    assert "Switched this chat to `pm`" in switched
    assert runner._active_personas["telegram:c1:u1"] == "pm"

    status = await runner._handle_persona_command(
        _make_event("/persona", Platform.TELEGRAM)
    )
    assert "Active persona for this chat: `pm`" in status

    cleared = await runner._handle_persona_command(
        _make_event("/persona cos", Platform.TELEGRAM)
    )
    assert "default Donna gateway" in cleared
    assert "telegram:c1:u1" not in runner._active_personas


@pytest.mark.asyncio
async def test_persona_command_one_shot_prompt_uses_specialist(monkeypatch):
    runner = _make_runner()
    runner._session_key_for_source = lambda _source: "telegram:c1:u1"
    calls = []

    async def fake_run(persona, prompt):
        calls.append((persona, prompt))
        return "specialist result"

    runner._run_persona_prompt = fake_run

    result = await runner._handle_persona_command(
        _make_event("/persona pm Show tomorrow's tasks", Platform.TELEGRAM)
    )

    assert result == "specialist result"
    assert calls == [("pm", "Show tomorrow's tasks")]
    assert not getattr(runner, "_active_personas", {})


def test_persona_is_known_gateway_command():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    cmd = resolve_command("persona")
    assert "persona" in GATEWAY_KNOWN_COMMANDS
    assert cmd is not None
    assert cmd.name == "persona"
    assert cmd.gateway_only is True
