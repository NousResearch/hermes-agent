"""Tests for Discord long-tail command catalog parity work."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
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


def _make_event(text: str, *, source: SessionSource | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=source or _make_source(),
        message_id="m1",
    )


def _make_runner(history: list[dict] | None = None) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.DISCORD: MagicMock()}
    source = _make_source()
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=datetime(2026, 3, 19, 12, 0),
        updated_at=datetime(2026, 3, 19, 12, 30),
        platform=Platform.DISCORD,
        chat_type="group",
    )
    runner.session_store = MagicMock()
    runner.session_store._generate_session_key = lambda session_source: build_session_key(session_source)
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history or []
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._session_send_policies = {}
    runner._session_docks = {}
    runner._effective_model = None
    runner._effective_provider = None
    runner._reasoning_config = {"enabled": True, "effort": "medium"}
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._load_current_model_selection = lambda: {
        "current_model": "anthropic/claude-opus-4.6",
        "current_provider": "openrouter",
    }
    runner._resolve_model_runtime_details = lambda _provider: ({}, None)
    runner._is_user_authorized = lambda _source: True
    return runner


@pytest.mark.asyncio
async def test_allowlist_command_adds_lists_and_removes(monkeypatch):
    runner = _make_runner()
    store: set[str] = set()
    import tools.approval as approval_mod

    monkeypatch.setattr(approval_mod, "load_permanent_allowlist", lambda: set(store))
    monkeypatch.setattr(
        approval_mod,
        "load_permanent",
        lambda patterns: store.update(patterns),
    )
    monkeypatch.setattr(
        approval_mod,
        "save_permanent_allowlist",
        lambda patterns: store.clear() or store.update(patterns),
    )
    monkeypatch.setattr(approval_mod, "_permanent_approved", set())

    add_result = await runner._handle_allowlist_command(_make_event("/allowlist add recursive delete"))
    list_result = await runner._handle_allowlist_command(_make_event("/allowlist"))
    remove_result = await runner._handle_allowlist_command(_make_event("/allowlist remove recursive delete"))

    assert "Added `recursive delete`" in add_result
    assert "🛡️ **Command Allowlist**" in list_result
    assert "`recursive delete`" in list_result
    assert "Removed `recursive delete`" in remove_result


@pytest.mark.asyncio
async def test_config_command_sets_gets_and_unsets_values(monkeypatch):
    runner = _make_runner()

    set_result = await runner._handle_config_command(
        _make_event("/config set terminal.backend docker")
    )
    get_result = await runner._handle_config_command(
        _make_event("/config get terminal.backend")
    )
    unset_result = await runner._handle_config_command(
        _make_event("/config unset terminal.backend")
    )

    assert "Updated `terminal.backend`" in set_result
    assert "docker" in get_result
    assert "Removed `terminal.backend`" in unset_result


@pytest.mark.asyncio
async def test_context_command_renders_list_detail_and_json():
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "tool", "content": "done"},
        {"role": "assistant", "content": "all set"},
    ]
    runner = _make_runner(history)

    list_result = await runner._handle_context_command(_make_event("/context"))
    detail_result = await runner._handle_context_command(_make_event("/context detail"))
    json_result = await runner._handle_context_command(_make_event("/context json"))

    assert "🧠 **Hermes Context**" in list_result
    assert "Messages: 4" in list_result
    assert "**Prompt Sources**" in detail_result
    payload = json.loads(json_result.removeprefix("```json\n").removesuffix("\n```"))
    assert payload["message_count"] == 4
    assert payload["role_counts"]["assistant"] == 2


@pytest.mark.asyncio
async def test_export_session_writes_html_and_json(tmp_path, monkeypatch):
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    runner = _make_runner(history)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    html_result = await runner._handle_export_session_command(_make_event("/export-session"))
    json_path = tmp_path / "session.json"
    json_result = await runner._handle_export_session_command(
        _make_event(f"/export-session {json_path}")
    )

    html_path = next((tmp_path / ".hermes-exports").glob("*.html"))
    assert html_path.exists()
    assert "Hermes Session Export" in html_path.read_text(encoding="utf-8")
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["session"]["session_id"] == "sess-1"
    assert len(payload["messages"]) == 2
    assert "Exported the current session" in html_result
    assert "Format: JSON" in json_result


def test_prepare_skill_command_lists_and_rewrites(monkeypatch):
    runner = _make_runner()
    event = _make_event("/skill summarize docs")

    monkeypatch.setattr(
        "agent.skill_commands.build_skill_invocation_message",
        lambda cmd_key, user_instruction, task_id=None: f"prepared:{cmd_key}:{user_instruction}:{task_id}",
    )
    prepared, error = runner._prepare_skill_command(event, task_id="sess-123")

    assert error is None
    assert prepared == "prepared:/summarize:docs:sess-123"
    assert event.text == prepared


@pytest.mark.asyncio
async def test_subagents_command_lists_empty_runtime():
    runner = _make_runner()
    result = await runner._handle_message(_make_event("/subagents list"))

    assert "No subagents recorded" in result


@pytest.mark.asyncio
async def test_compact_command_passes_instructions_to_compressor(monkeypatch):
    history = [
        {"role": "user", "content": "message one"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "message two"},
        {"role": "assistant", "content": "reply two"},
    ]
    runner = _make_runner(history)
    captured: dict[str, str] = {}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def _compress_context(self, msgs, instructions, approx_tokens=None):
            captured["instructions"] = instructions
            return ([{"role": "system", "content": "summary"}], None)

    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"api_key": "secret"})
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "anthropic/claude-opus-4.6")
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("agent.model_metadata.estimate_messages_tokens_rough", lambda _msgs: 123)

    result = await runner._handle_compress_command(_make_event("/compact keep TODOs and decisions"))

    assert captured["instructions"] == "keep TODOs and decisions"
    assert "Preserved guidance" in result
    runner.session_store.rewrite_transcript.assert_called_once()
