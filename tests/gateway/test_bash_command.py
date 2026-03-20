"""Tests for gateway /bash command parity."""

from __future__ import annotations

from datetime import datetime
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


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {
        Platform.DISCORD: SimpleNamespace(send_exec_approval=AsyncMock())
    }
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
    runner._bash_jobs = {}
    runner._pending_approvals = {}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._voice_mode = {}
    runner._session_send_policies = {}
    runner._session_docks = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    return runner


def test_rewrite_bash_shortcut_supports_command_poll_and_stop():
    assert GatewayRunner._rewrite_bash_shortcut("! echo hello") == "/bash echo hello"
    assert GatewayRunner._rewrite_bash_shortcut("!poll") == "/bash poll"
    assert GatewayRunner._rewrite_bash_shortcut("!stop proc-1") == "/bash stop proc-1"
    assert GatewayRunner._rewrite_bash_shortcut("hello") is None


@pytest.mark.asyncio
async def test_handle_bash_command_completes_fast_process(monkeypatch):
    runner = _make_runner()

    monkeypatch.setattr(
        runner,
        "_run_bash_process",
        lambda **_kwargs: {"session_id": "proc-1"},
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        SimpleNamespace(
            wait=lambda session_id, timeout=None: {
                "status": "exited",
                "exit_code": 0,
                "output": "hello\nworld",
            },
            poll=lambda _sid: {"status": "exited"},
        ),
    )

    result = await runner._handle_bash_command(_make_event("/bash echo hello"))

    assert "Bash Complete" in result
    assert "`0`" in result
    assert "hello" in result
    assert runner._bash_jobs == {}


@pytest.mark.asyncio
async def test_handle_bash_command_backgrounds_long_process(monkeypatch):
    runner = _make_runner()

    monkeypatch.setattr(
        runner,
        "_run_bash_process",
        lambda **_kwargs: {"session_id": "proc-2"},
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        SimpleNamespace(
            wait=lambda session_id, timeout=None: {"status": "timeout"},
            poll=lambda _sid: {
                "status": "running",
                "pid": 4242,
                "output_preview": "still running",
            },
        ),
    )

    result = await runner._handle_bash_command(_make_event("/bash sleep 30"))

    assert "Bash Running" in result
    assert "`proc-2`" in result
    assert "still running" in result
    assert runner._bash_jobs[build_session_key(_make_source())] == "proc-2"


@pytest.mark.asyncio
async def test_bash_poll_and_stop_use_current_job(monkeypatch):
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._bash_jobs[session_key] = "proc-9"

    registry = SimpleNamespace(
        get=lambda _sid: SimpleNamespace(exited=False),
        poll=lambda _sid: {
            "status": "running",
            "pid": 999,
            "uptime_seconds": 8,
        },
        read_log=lambda _sid, limit=40: {"output": "line 1\nline 2"},
        kill_process=lambda _sid: {"status": "killed"},
    )
    monkeypatch.setattr("tools.process_registry.process_registry", registry)

    poll_result = await runner._handle_bash_command(_make_event("/bash poll"))
    stop_result = await runner._handle_bash_command(_make_event("/bash stop"))

    assert "Bash Status" in poll_result
    assert "line 1" in poll_result
    assert "stopped" in stop_result
    assert session_key not in runner._bash_jobs


@pytest.mark.asyncio
async def test_bash_command_records_pending_approval(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(
        runner,
        "_run_bash_process",
        lambda **_kwargs: {
            "status": "approval_required",
            "error": "Need approval",
            "description": "dangerous command",
            "pattern_key": "rm-rf",
        },
    )

    result = await runner._handle_bash_command(_make_event("/bash rm -rf /tmp/demo"))
    pending = next(iter(runner._pending_approvals.values()))

    assert "requires approval" in result
    assert pending["pattern_key"] == "rm-rf"
    assert callable(pending["on_approve"])
    runner.adapters[Platform.DISCORD].send_exec_approval.assert_awaited_once()


def test_resolve_pending_approval_uses_callback_when_present():
    runner = _make_runner()
    source = _make_source()
    session_key = build_session_key(source)
    runner._pending_approvals[session_key] = {
        "approval_id": "appr-1",
        "command": "echo hi",
        "pattern_key": "echo",
        "on_approve": lambda decision: f"approved:{decision}",
    }

    result = runner._resolve_pending_approval(decision="allow-once", source=source)

    assert result == "approved:allow-once"
    assert session_key not in runner._pending_approvals
