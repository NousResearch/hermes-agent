"""Canonical slash-command gates must read the routed profile (#148248)."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from hermes_constants import get_hermes_home


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["new", "undo"])
async def test_canonical_gate_follows_routed_config_and_restores_launch_home(tmp_path, monkeypatch, command):
    launch = tmp_path / "launch"
    profiles = {name: tmp_path / "profiles" / name for name in ("alpha", "beta")}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    for home, required in ((launch, True), (profiles["alpha"], False), (profiles["beta"], True)):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f"approvals:\n  destructive_slash_confirm: {str(required).lower()}\n", encoding="utf-8",
        )
    original_configs = {home: (home / "config.yaml").read_bytes() for home in (launch, *profiles.values())}
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = lambda source: profiles[source.profile]
    runner._session_key_for_source = lambda source: source.profile
    runner._is_telegram_topic_root_lobby = lambda source: False
    runner._request_slash_confirm = AsyncMock(return_value="confirmation required")
    executed_in = []

    async def execute(event):
        executed_in.append(get_hermes_home())
        return "executed"

    runner._handle_reset_command = execute
    runner._handle_undo_command = execute
    replies = []
    with _profile_runtime_scope(launch):
        for profile in ("alpha", "beta", "alpha"):
            source = SessionSource(
                platform=Platform.TELEGRAM, user_id="u", user_name="u",
                chat_id="c", chat_type="dm", profile=profile,
            )
            event = MessageEvent(text=f"/{command}", message_type=MessageType.TEXT, source=source, message_id="m")
            handled, reply = await runner._hm_dispatch_canonical_command(event, source, profile, command)
            assert handled
            replies.append(reply)
            assert get_hermes_home() == launch

    assert replies == ["executed", "confirmation required", "executed"]
    assert executed_in == [profiles["alpha"], profiles["alpha"]]
    runner._request_slash_confirm.assert_awaited_once()
    assert {home: (home / "config.yaml").read_bytes() for home in original_configs} == original_configs
