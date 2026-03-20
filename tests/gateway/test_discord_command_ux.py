"""Tests for Discord command UX additions: /approve and /think."""

from datetime import datetime
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(**overrides) -> SessionSource:
    data = {
        "platform": Platform.DISCORD,
        "user_id": "u1",
        "chat_id": "c1",
        "user_name": "alan",
        "chat_name": "Hermes / #general",
        "chat_type": "group",
    }
    data.update(overrides)
    return SessionSource(**data)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.send_exec_approval = AsyncMock()
    runner.adapters = {Platform.DISCORD: adapter}
    runner.session_store = MagicMock()
    runner.session_store._generate_session_key = lambda source: build_session_key(source)
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._effective_model = None
    runner._effective_provider = None
    runner._resolve_model_runtime_details = lambda _provider: ({}, None)
    runner._load_current_model_selection = lambda: {
        "current_model": "anthropic/claude-opus-4.6",
        "current_provider": "openrouter",
    }
    runner._reasoning_config = {"enabled": True, "effort": "medium"}
    runner._show_reasoning = False
    return runner


@pytest.mark.asyncio
async def test_approve_command_resolves_current_pending_from_command_target(monkeypatch):
    runner = _make_runner()
    target_source = _make_source()
    slash_source = _make_source(session_namespace="slash:u1")
    session_key = build_session_key(target_source)
    runner._pending_approvals[session_key] = {
        "approval_id": "appr-1234",
        "command": "rm -rf /tmp/test",
        "pattern_keys": ["recursive delete"],
        "session_key": session_key,
    }

    approve_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "tools.approval.approve_session",
        lambda key, pattern: approve_calls.append((key, pattern)),
    )
    terminal_module = importlib.import_module("tools.terminal_tool")
    monkeypatch.setattr(
        terminal_module,
        "terminal_tool",
        lambda **kwargs: "command output",
    )

    event = MessageEvent(
        text="/approve allow-once",
        message_type=MessageType.COMMAND,
        source=target_source,
        message_id="m1",
        metadata={
            "session_source": slash_source,
            "command_target_source": target_source,
        },
    )

    result = await runner._handle_approve_command(event)

    assert "approved (allow-once)" in result
    assert "command output" in result
    assert approve_calls == [(session_key, "recursive delete")]
    assert session_key not in runner._pending_approvals


@pytest.mark.asyncio
async def test_approve_command_allow_always_uses_id_lookup(monkeypatch):
    runner = _make_runner()
    target_source = _make_source()
    session_key = build_session_key(target_source)
    runner._pending_approvals[session_key] = {
        "approval_id": "appr-9999",
        "command": "rm -rf /tmp/test",
        "pattern_keys": ["recursive delete", "tirith:shortened_url"],
        "session_key": session_key,
    }

    approval_calls = {
        "session": [],
        "permanent": [],
        "saved": [],
    }
    monkeypatch.setattr(
        "tools.approval.approve_session",
        lambda key, pattern: approval_calls["session"].append((key, pattern)),
    )
    monkeypatch.setattr(
        "tools.approval.approve_permanent",
        lambda pattern: approval_calls["permanent"].append(pattern),
    )
    monkeypatch.setattr(
        "tools.approval.save_permanent_allowlist",
        lambda patterns: approval_calls["saved"].append(set(patterns)),
    )
    monkeypatch.setattr(
        "tools.approval._permanent_approved",
        {"recursive delete", "tirith:shortened_url"},
    )
    terminal_module = importlib.import_module("tools.terminal_tool")
    monkeypatch.setattr(
        terminal_module,
        "terminal_tool",
        lambda **kwargs: "command output",
    )

    event = MessageEvent(
        text="/approve appr-9999 allow-always",
        message_type=MessageType.COMMAND,
        source=_make_source(user_id="u2"),
        message_id="m1",
    )

    result = await runner._handle_approve_command(event)

    assert "approved (allow-always)" in result
    assert approval_calls["session"] == [
        (session_key, "recursive delete"),
        (session_key, "tirith:shortened_url"),
    ]
    assert approval_calls["permanent"] == ["recursive delete", "tirith:shortened_url"]
    assert approval_calls["saved"] == [{"recursive delete", "tirith:shortened_url"}]


@pytest.mark.asyncio
async def test_think_command_routes_to_reasoning_effort(monkeypatch):
    runner = _make_runner()

    monkeypatch.setattr(
        runner,
        "_handle_reasoning_command",
        AsyncMock(return_value="🧠 reasoning updated"),
    )
    event = MessageEvent(
        text="/think high",
        message_type=MessageType.COMMAND,
        source=_make_source(),
        message_id="m1",
    )

    result = await runner._handle_think_command(event)

    assert result == "🧠 reasoning updated"
    think_event = runner._handle_reasoning_command.await_args.args[0]
    assert think_event.text == "/reasoning high"
    assert think_event.source.chat_id == "c1"
