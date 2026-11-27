"""/new must honour display.in_app_tips: the desktop Appearance switch that already
withdraws the show_tip tool should also silence the tip line on the reset reply."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_runner(source):
    from gateway.run import GatewayRunner

    class _Runner(GatewayRunner):
        persistence = None

    runner = object.__new__(_Runner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="x")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()

    session_key = build_session_key(source)
    entry = SessionEntry(
        session_key=session_key,
        session_id="sess-new",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.reset_session.return_value = entry
    runner.session_store._entries = {session_key: entry}
    runner.session_store._generate_session_key.return_value = session_key
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._session_db = AsyncMock()
    runner.persistence = MagicMock()
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""
    return runner


@pytest.mark.parametrize("tips_on", [True, False])
@pytest.mark.asyncio
async def test_reset_tip_respects_in_app_tips(monkeypatch, tips_on):
    from hermes_cli.config import load_config_readonly

    def _config():
        cfg = load_config_readonly()
        cfg.setdefault("display", {})["in_app_tips"] = tips_on
        return cfg

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _config)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="12345",
        chat_id="67890",
        user_name="testuser",
    )
    runner = _make_runner(source)
    event = MessageEvent(text="/new", source=source)

    reply = str(await runner._handle_reset_command(event))

    # U+2726 is the tip-line marker in every locale; the reset headers use U+2728 instead.
    assert ("✦" in reply) is tips_on
