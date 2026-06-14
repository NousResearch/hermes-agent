from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    source = _make_source()
    session_key = build_session_key(source)
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._running_agents = {}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = [
        {"role": "assistant", "content": "latest answer"}
    ]
    return runner


@pytest.mark.asyncio
async def test_codex_status_renders_cockpit(monkeypatch):
    from hermes_cli import codex_cockpit as cc

    runner = _make_runner()
    agent = SimpleNamespace(
        model="gpt-test",
        provider="openai-codex",
        api_mode="codex_app_server",
        session_cwd="/tmp/repo",
        _last_codex_thread_id="thread-1",
        _last_codex_turn_id="turn-1",
    )
    runner._running_agents[build_session_key(_make_source())] = agent

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {"openai_runtime": "codex_app_server"},
            "codex_cockpit": {"readout": {"include_git_status": False}},
        },
    )
    monkeypatch.setattr(cc, "_codex_binary_status", lambda: (True, "0.130.0"))
    monkeypatch.setattr(cc, "_codex_auth_line", lambda: "logged in (test)")

    result = await runner._handle_codex_command(_make_event("/codex status"))

    assert "**Codex Cockpit**" in result
    assert "codex_app_server" in result
    assert "thread-1" in result
    assert "latest answer" in result


@pytest.mark.asyncio
async def test_codex_launch_spawns_background_process(monkeypatch):
    from hermes_cli import codex_cockpit as cc

    runner = _make_runner()
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"codex_cockpit": {}})
    plan = cc.LaunchPlan(
        repo_root="/tmp/repo",
        worktree_path="/tmp/worktree",
        branch="codex/test",
        command="codex exec -C /tmp/worktree test",
        task_id="codex_test",
        prompt_preview="test prompt",
    )
    monkeypatch.setattr(cc, "prepare_launch", lambda *args, **kwargs: (plan, None))

    spawned = {}

    class FakeRegistry:
        def list_sessions(self):
            return []

        def spawn_local(self, command, **kwargs):
            spawned["command"] = command
            spawned.update(kwargs)
            return SimpleNamespace(id="proc-1")

    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())

    result = await runner._handle_codex_command(
        _make_event('/codex launch /tmp/repo "test prompt"')
    )

    assert "Codex driver launched" in result
    assert "proc-1" in result
    assert spawned["command"] == plan.command
    assert spawned["cwd"] == plan.repo_root
    assert spawned["task_id"] == plan.task_id
    assert spawned["use_pty"] is True
