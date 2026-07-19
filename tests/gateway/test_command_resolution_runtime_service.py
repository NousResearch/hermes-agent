"""DEAD path: not imported by gateway/run.py — contract-only unit tests.

Tests for gateway non-built-in slash command resolution helpers.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.dead_runtime_service

from gateway.command_resolution_runtime_service import (
    resolve_gateway_non_builtin_command,
)

def _make_event(command: str, args: str = "") -> MagicMock:
    event = MagicMock()
    event.get_command_args.return_value = args
    event.text = f"/{command} {args}".strip()
    return event

def _make_source(platform: str = "telegram") -> SimpleNamespace:
    return SimpleNamespace(platform=SimpleNamespace(value=platform))

@pytest.mark.asyncio
async def test_plugin_command_with_empty_result_is_still_handled(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda _name: (lambda _args: None),
        raising=False,
    )

    result = await resolve_gateway_non_builtin_command(
        runner=SimpleNamespace(config={}),
        event=_make_event("plug"),
        source=_make_source(),
        session_key="session-1",
        command="plug",
        logger=MagicMock(),
        unavailable_skill_checker=lambda _command: None,
    )

    assert result.handled is True
    assert result.response is None

@pytest.mark.asyncio
async def test_skill_command_rewrites_event_text_and_falls_through(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda _name: None,
        raising=False,
    )
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/foo-skill": {"name": "foo-skill"}},
    )
    monkeypatch.setattr(
        "agent.skill_commands.resolve_skill_command_key",
        lambda command: "/foo-skill" if command == "foo_skill" else None,
    )
    monkeypatch.setattr(
        "agent.skill_commands.build_skill_invocation_message",
        lambda *_args, **_kwargs: "INVOKE SKILL",
    )
    monkeypatch.setattr(
        "agent.skill_utils.get_disabled_skill_names",
        lambda platform=None: set(),
    )

    event = _make_event("foo_skill", "investigate this")
    result = await resolve_gateway_non_builtin_command(
        runner=SimpleNamespace(config={}),
        event=event,
        source=_make_source(),
        session_key="session-1",
        command="foo_skill",
        logger=MagicMock(),
        unavailable_skill_checker=lambda _command: None,
    )

    assert result.handled is False
    assert event.text == "INVOKE SKILL"

@pytest.mark.asyncio
async def test_disabled_skill_returns_platform_guidance(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda _name: None,
        raising=False,
    )
    monkeypatch.setattr(
        "agent.skill_commands.get_skill_commands",
        lambda: {"/foo-skill": {"name": "foo-skill"}},
    )
    monkeypatch.setattr(
        "agent.skill_commands.resolve_skill_command_key",
        lambda command: "/foo-skill" if command == "foo-skill" else None,
    )
    monkeypatch.setattr(
        "agent.skill_utils.get_disabled_skill_names",
        lambda platform=None: {"foo-skill"} if platform == "telegram" else set(),
    )

    result = await resolve_gateway_non_builtin_command(
        runner=SimpleNamespace(config={}),
        event=_make_event("foo-skill"),
        source=_make_source(),
        session_key="session-1",
        command="foo-skill",
        logger=MagicMock(),
        unavailable_skill_checker=lambda _command: None,
    )

    assert result.handled is True
    assert "disabled for telegram" in str(result.response)

@pytest.mark.asyncio
async def test_unknown_slash_command_returns_guidance(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command_handler",
        lambda _name: None,
        raising=False,
    )
    monkeypatch.setattr("agent.skill_commands.get_skill_commands", lambda: {})
    monkeypatch.setattr(
        "agent.skill_commands.resolve_skill_command_key",
        lambda _command: None,
    )

    result = await resolve_gateway_non_builtin_command(
        runner=SimpleNamespace(config={}),
        event=_make_event("not-real"),
        source=_make_source(),
        session_key="session-1",
        command="not-real",
        logger=MagicMock(),
        unavailable_skill_checker=lambda _command: None,
    )

    assert result.handled is True
    assert "Unknown command" in str(result.response)
