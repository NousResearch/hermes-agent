"""A spoken turn from a member admitted by ``DISCORD_ALLOWED_ROLES`` reaches the agent.

The listen loop admits a speaker through the adapter's role check, but the voice source is rebuilt
from the ``/voice join`` copy, which never carries the per-event role grant, so the gateway's own
check refused every role-only speaker. The grant is recomputed for the current speaker against the
guild's current member, never inherited from whoever ran ``/voice join``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.discord.adapter import DiscordAdapter

_ROLE, _GUILD, _TEXT, _JOINER = 1234, 1, 700, 42


@pytest.mark.asyncio
@pytest.mark.parametrize("speaker_roles,dispatched", [
    ([_ROLE], 1),   # role member: admitted on both gates
    ([], 0),        # the joiner holds the role, this speaker does not
    (None, 0),      # speaker no longer in the guild
])
async def test_role_member_speech_passes_the_gateway_gate(
    monkeypatch: pytest.MonkeyPatch, speaker_roles: list[int] | None, dispatched: int,
) -> None:
    for var in ("DISCORD_ALLOWED_USERS", "DISCORD_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
                "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", str(_ROLE))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._allowed_user_ids, adapter._allowed_role_ids = set(), {_ROLE}
    speaker = 43
    members = {_JOINER: SimpleNamespace(id=_JOINER, roles=[SimpleNamespace(id=_ROLE)])}
    if speaker_roles is not None:
        members[speaker] = SimpleNamespace(id=speaker, roles=[SimpleNamespace(id=r) for r in speaker_roles])
    guild = SimpleNamespace(id=_GUILD, name="Hermes Server", get_member=members.get)
    adapter._client = SimpleNamespace(get_guild=lambda _id: guild, get_channel=lambda _id: None)
    adapter._voice_text_channels = {_GUILD: _TEXT}
    joined = SessionSource(platform=Platform.DISCORD, chat_id=str(_TEXT), chat_type="group",
                           user_id=str(_JOINER), user_name="joiner", guild_id=str(_GUILD),
                           role_authorized=True)
    adapter._voice_sources = {_GUILD: joined.to_dict()}
    adapter.handle_message = AsyncMock()
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._voice_mode, runner._session_db, runner.session_store = {}, None, MagicMock()

    await runner._handle_voice_channel_input(_GUILD, speaker, "what broke?", adapter=adapter)

    assert adapter.handle_message.await_count == dispatched
    for call in adapter.handle_message.await_args_list:
        assert runner._is_user_authorized(call.args[0].source)
