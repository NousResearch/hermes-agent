"""A native Discord /stop must escape a route previously poisoned onto a delegate.

The legacy route is constructed from durable rows, then the real Discord adapter,
gateway runner, session store, and background-delegation registry exercise recovery.
"""

import asyncio
from datetime import datetime
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import tools.async_delegation as delegation
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionEntry
from hermes_state import SessionDB


def _ensure_discord_mock():
    """Import the adapter without a Discord connection or optional discord.py install."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    if sys.modules.get("discord") is None:
        discord = MagicMock()
        discord.Intents.default.return_value = MagicMock()
        discord.DMChannel = type("DMChannel", (), {})
        discord.Thread = type("Thread", (), {})
        discord.ForumChannel = type("ForumChannel", (), {})
        discord.Interaction = object
        discord.app_commands = SimpleNamespace(
            describe=lambda **kwargs: (lambda fn: fn),
            choices=lambda **kwargs: (lambda fn: fn),
            autocomplete=lambda **kwargs: (lambda fn: fn),
            Choice=lambda **kwargs: SimpleNamespace(**kwargs),
            Group=MagicMock,
            Command=MagicMock,
        )
        sys.modules["discord"] = discord
        sys.modules.setdefault("discord.ext", MagicMock())
        sys.modules.setdefault("discord.ext.commands", MagicMock())


_ensure_discord_mock()

import discord  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _Thread(discord.Thread):
    def __init__(self):
        self.id = 106742
        self.name = "work"
        self.guild = SimpleNamespace(id=7, name="Test Guild")
        self.parent = SimpleNamespace(id=10, name="general", guild=self.guild)
        self.topic = None


def _stamp(db, session_id, **values):
    columns = ", ".join(f"{column} = ?" for column in values)
    with db._lock:
        db._conn.execute(f"UPDATE sessions SET {columns} WHERE id = ?", (*values.values(), session_id))
        db._conn.commit()


@pytest.fixture(autouse=True)
def _clear_delegations():
    delegation._reset_for_tests()
    yield
    delegation._reset_for_tests()


@pytest.mark.asyncio
async def test_native_stop_escapes_poisoned_route_and_next_inbound_restores_owner(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "gateway_voice_mode.json")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    config = PlatformConfig(enabled=True, token="test-token")
    runner = GatewayRunner(GatewayConfig(
        platforms={Platform.DISCORD: config}, sessions_dir=tmp_path / "sessions",
    ))
    adapter = DiscordAdapter(config)
    runner.adapters = {Platform.DISCORD: adapter}
    adapter.gateway_runner = runner
    runner._wire_adapter_handlers(adapter)
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="stop-ack"))

    channel = _Thread()
    interaction = SimpleNamespace(
        channel=channel, channel_id=channel.id, guild_id=channel.guild.id,
        user=SimpleNamespace(id=42, name="Axl", display_name="Axl"),
        response=SimpleNamespace(defer=AsyncMock()),
        edit_original_response=AsyncMock(), delete_original_response=AsyncMock(),
    )
    source = adapter._build_slash_event(interaction, "/stop").source
    key = runner.session_store._generate_session_key(source)
    assert key == adapter._source_session_key(source)

    db = SessionDB(db_path=tmp_path / "state.db")
    runner.session_store._db = db
    peer = dict(
        source="discord", session_key=key, user_id=source.user_id,
        chat_id=source.chat_id, chat_type=source.chat_type,
        thread_id=source.thread_id, origin_json=json.dumps(source.to_dict()),
    )
    db.create_session("human-owner", **peer)
    _stamp(db, "human-owner", started_at=100.0, ended_at=300.0, end_reason="session_switch")
    db.create_session("delegate-child", "subagent", parent_session_id="human-owner")
    _stamp(db, "delegate-child", started_at=200.0, **peer)
    now = datetime.now()
    store = runner.session_store
    store._ensure_loaded()
    with store._lock:
        store._entries[key] = SessionEntry(
            session_key=key, session_id="delegate-child", created_at=now,
            updated_at=now, origin=source, platform=Platform.DISCORD, chat_type="thread",
        )
        store._save()

    interrupted = MagicMock()
    with delegation._records_lock:
        delegation._records["live-child"] = {
            "delegation_id": "live-child", "status": "running", "session_key": key,
            "origin_ui_session_id": "", "parent_session_id": "human-owner",
            "interrupt_fn": interrupted,
        }
    running_agent = MagicMock()
    running_agent.session_id = "human-owner"
    runner._session_state(key).turn.agent = running_agent
    blocked = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)
    adapter._active_sessions[key] = asyncio.Event()
    adapter._session_tasks[key] = blocked

    try:
        await asyncio.wait_for(adapter._run_simple_slash(interaction, "/stop"), timeout=2)
        assert blocked.cancelled()
        interrupted.assert_called_once()
        assert runner.session_store.peek_session_id(key) == "delegate-child"
        assert not runner._is_session_running(key)
        assert len(adapter.send.await_args_list) == 1
        assert "Stopped" in adapter.send.await_args.kwargs["content"]

        # The next ordinary Discord inbound resolves the route before agent work.
        resolved_ids = []

        async def _run_after_route_resolution(_event, inbound_source, _key, _generation):
            entry = await runner.async_session_store.get_or_create_session(inbound_source)
            resolved_ids.append(entry.session_id)
            return "resumed"

        runner._handle_message_with_agent = _run_after_route_resolution
        runner._run_post_turn_hooks = AsyncMock()
        await adapter.handle_message(adapter._build_slash_event(interaction, "continue"))
        resumed_task = adapter._session_tasks[key]
        await asyncio.wait_for(resumed_task, timeout=2)
        assert resolved_ids == ["human-owner"]
        assert store.peek_session_id(key) == "human-owner"
        assert db.get_session("delegate-child")["created_source"] == "subagent"
        assert len(adapter.send.await_args_list) == 2
        assert adapter.send.await_args_list[1].kwargs["content"] == "resumed"
    finally:
        if not blocked.done():
            blocked.cancel()
            await asyncio.gather(blocked, return_exceptions=True)
        db.close()
