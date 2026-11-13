"""The /new paused-queue note names a discard command the chat can actually send (#145177).

Slack registers no native ``/discard`` (its slash-command cap); the umbrella ``/hermes discard``
reaches it, as the lost-turn pause notice already tells Slack users.
"""
import pytest

from agent.i18n import t
from hermes_agent.gateway.config import Platform
from hermes_agent.gateway.session import SessionSource
from hermes_agent.gateway.slash_commands_session import discard_command_for


def _note(platform):
    source = SessionSource(platform=platform, chat_id="c1", chat_type="dm", user_id="u1")
    return t("gateway.reset.paused_left", count=2, session_id="sid-1", discard=discard_command_for(source))


def test_slack_is_told_the_umbrella_discard_command():
    note = _note(Platform.SLACK)
    assert "/hermes discard" in note and "/resume sid-1" in note


def test_telegram_is_told_the_native_discard_command():
    note = _note(Platform.TELEGRAM)
    assert "then /discard" in note and "/hermes" not in note


@pytest.mark.asyncio
@pytest.mark.parametrize("platform,discard", [(Platform.SLACK, "/hermes discard"), (Platform.TELEGRAM, "/discard")])
async def test_reset_reply_names_the_platform_discard_command(monkeypatch, platform, discard):
    """The /new handler itself fills {discard} per platform, not only the helper."""
    from datetime import datetime
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import hermes_agent.gateway.session_authorities as authorities
    import hermes_agent.gateway.slash_commands_session as session_commands
    from hermes_agent.gateway.config import GatewayConfig
    from hermes_agent.gateway.platforms.event import MessageEvent
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.session import SessionEntry, build_session_key

    source = SessionSource(platform=platform, chat_id="c1", chat_type="dm", user_id="u1")
    key = build_session_key(source)

    def entry(sid):
        return SessionEntry(session_key=key, session_id=sid, created_at=datetime.now(),
                            updated_at=datetime.now(), platform=platform, chat_type="dm")
    runner = object.__new__(GatewayRunner)
    runner.config, runner.adapters, runner._background_tasks = GatewayConfig(), {}, set()
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides, runner._pending_model_notes, runner._voice_mode = {}, {}, {}
    runner._running_agents, runner._pending_messages, runner._session_db = {}, {}, None
    runner._agent_cache_lock, runner._format_session_info = None, lambda: ""
    runner.session_store = MagicMock(_entries={key: entry("sid-1")})
    runner.session_store.reset_session.return_value = entry("sid-2")

    async def paused(_runner, session_id):
        return 2 if session_id == "sid-1" else 0
    monkeypatch.setattr(session_commands, "paused_followers", paused)
    monkeypatch.setattr(authorities, "active_authority", lambda _runner: MagicMock(logical_owner=lambda sid: sid, sessions={}))

    reply = await runner._handle_reset_command(MessageEvent(text="/new", source=source, message_id="m1"))
    assert "/resume sid-1" in reply and f"then {discard}." in reply, reply
