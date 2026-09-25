"""Joined threads observe human conversation before either gateway busy guard."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import discord

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner
from tests.gateway.test_discord_free_response import (
    adapter, FakeDMChannel, FakeThread, make_message,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize("kind,verdict,quiet", [
    ("fyi", "CONTEXT_ONLY", True),
    ("request", "ADMIT", False),
    ("mention", "CONTEXT_ONLY", False),
    ("reply", "CONTEXT_ONLY", False),
    ("command", "CONTEXT_ONLY", False),
    ("dm", "CONTEXT_ONLY", False),
    ("uncertain", "uncertain", False),
    ("failure", TimeoutError(), False),
])
async def test_context_observation_precedes_busy_and_idle_dispatch(
    adapter, monkeypatch, tmp_path, busy, kind, verdict, quiet,
):
    from agent import secret_scope
    from hermes_constants import get_hermes_home

    home = tmp_path / ".hermes"
    homes = {"default": home, "b": home / "profiles" / "b"}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name, path in homes.items():
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.yaml").write_text("{}\n")
        (path / ".env").write_text(f"QUIET_TEST_SECRET={name}\n")
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda **kwargs: list(homes.items()))
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: homes[name])
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in homes)
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
    runner._primary_profile_name = "default"
    runner.config.profile_routes = parse_profile_routes([
        {"name": "route-b", "platform": "discord", "profile": "b", "chat_id": "202"},
    ])
    runner.adapters = {Platform.DISCORD: adapter}
    runner._profile_adapters = {"b": {}}
    adapter.gateway_runner = runner
    adapter.config.extra.update({"missed_message_backfill": {"enabled": True}})
    adapter._ready_event.set()
    adapter._is_allowed_user = Mock(return_value=True)
    adapter._message_handler = AsyncMock()
    adapter._busy_session_handler = AsyncMock(return_value=True)
    adapter._start_session_processing = Mock(return_value=True)
    adapter.handle_message = BasePlatformAdapter.handle_message.__get__(adapter)
    adapter.send = AsyncMock()
    scopes = []

    async def remote_model(**kwargs):
        scopes.append((get_hermes_home(), secret_scope.get_secret("QUIET_TEST_SECRET")))
        payload = json.loads(kwargs["messages"][1]["content"])
        assert payload["recent_thread"][-1]["text"] == "What model should I use?"
        if isinstance(verdict, Exception):
            raise verdict
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=verdict))])

    model = AsyncMock(side_effect=remote_model)
    monkeypatch.setattr("agent.auxiliary_client.async_call_llm", model)
    was_multiplex = secret_scope._MULTIPLEX_ACTIVE
    secret_scope.set_multiplex_active(True)
    try:
        for index, (profile, channel_id) in enumerate([("default", 101), ("b", 202), ("default", 101)]):
            channel = FakeDMChannel(channel_id) if kind == "dm" else FakeThread(channel_id)
            prior = SimpleNamespace(id=1, author=SimpleNamespace(id=999, name="AgentGrid", bot=True),
                                    content="What model should I use?", attachments=[], type=discord.MessageType.default)
            passive = SimpleNamespace(id=0, author=SimpleNamespace(id=820, name="Souren", bot=False),
                                      content="FYI Michael, the deployment is ready", attachments=[],
                                      type=discord.MessageType.default)
            async def history(**kwargs):
                yield prior
                yield passive
            channel.history = history
            await adapter._threads.mark_async(str(channel_id))
            content = {
                "fyi": "FYI <@820> it is on sonnet now ran out of Astra, we can probably put it on fable/opus too",
                "request": "Can you switch it to opus? Ask <@820> to confirm.",
                "mention": "<@999> please check with <@820>",
                "reply": "sonnet please", "command": "/stop", "dm": "FYI <@820> model changed",
                "uncertain": "AgentGrid, could you check?", "failure": "Please continue",
            }[kind]
            mentions = [SimpleNamespace(id=999 if kind == "mention" else 820, bot=False)]
            message = make_message(channel=channel, content=content, mentions=mentions)
            message.id = 1000 + index
            message.guild = None if kind == "dm" else SimpleNamespace(id=42)
            if kind == "reply":
                message.reference = SimpleNamespace(message_id=1, resolved=prior)
            source = adapter.build_source(chat_id=str(channel_id), chat_type="dm" if kind == "dm" else "thread",
                                          thread_id=None if kind == "dm" else str(channel_id), user_id="42")
            key = adapter._source_session_key(source)
            if busy:
                adapter._active_sessions[key] = asyncio.Event()
            await adapter._dispatch_discord_message(message)
            if quiet:
                adapter._start_session_processing.assert_not_called()
                adapter._busy_session_handler.assert_not_awaited()
                assert not adapter._pending_messages
                assert adapter._discord_message_is_persistently_complete(str(message.id))
                adapter._record_discord_message_seen(message, status="discovered")
                assert not await adapter._should_backfill_discord_message(message)
            elif kind != "command":
                target = adapter._busy_session_handler if busy else adapter._start_session_processing
                assert target.call_count == index + 1
                event = target.call_args.args[0]
                if kind != "dm":
                    assert "What model should I use?" in event.channel_context
                    assert passive.content in event.channel_context
            if kind != "command":
                adapter.send.assert_not_awaited()
        if kind in {"mention", "reply", "command", "dm"}:
            model.assert_not_awaited()
        else:
            assert scopes == [(homes[p], p) for p in ("default", "b", "default")]
    finally:
        secret_scope.set_multiplex_active(was_multiplex)
