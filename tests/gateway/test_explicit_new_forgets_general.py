"""/new in a forum topic also resets General, and a no-shrink compress does not latch."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.slash_commands_session import forum_general_sibling_source


def test_forum_topic_new_names_general_sibling():
    topic = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-100999", chat_type="group",
        user_id="10001", thread_id="11",
    )
    sibling = forum_general_sibling_source(topic)
    assert sibling is not None
    assert sibling.thread_id == "1"
    assert build_session_key(sibling) == "agent:main:telegram:group:-100999:1"
    assert forum_general_sibling_source(sibling) is None
    dm = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="u1")
    assert forum_general_sibling_source(dm) is None


def _entry(source: SessionSource, session_id: str) -> SessionEntry:
    key = build_session_key(source)
    return SessionEntry(
        session_key=key, session_id=session_id, created_at=datetime.now(), updated_at=datetime.now(),
        platform=source.platform, chat_type=source.chat_type,
    )


@pytest.mark.asyncio
async def test_forum_new_resets_topic_and_general_only():
    from gateway.run import GatewayRunner

    topic = SessionSource(
        platform=Platform.TELEGRAM, chat_id="-100999", chat_type="group",
        user_id="10001", thread_id="11",
    )
    general = forum_general_sibling_source(topic)
    projects = dataclasses_replace_thread(topic, "106")
    topic_entry = _entry(topic, "topic-old")
    general_entry = _entry(general, "general-old")
    projects_entry = _entry(projects, "projects-old")

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner.adapters = {Platform.TELEGRAM: MagicMock(send=AsyncMock())}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""
    runner.session_store = MagicMock()
    runner.session_store._entries = {
        topic_entry.session_key: topic_entry,
        general_entry.session_key: general_entry,
        projects_entry.session_key: projects_entry,
    }
    runner.session_store._generate_session_key.side_effect = lambda source: build_session_key(source)

    def _reset(session_key, display_name=None):
        old = runner.session_store._entries[session_key]
        fresh = _entry(old.origin or topic, "fresh-" + session_key[-1])
        fresh.session_key = session_key
        return fresh

    runner.session_store.reset_session.side_effect = _reset
    event = MessageEvent(text="/new", source=topic, message_id="m1")
    with patch("hermes_cli.lifecycle.invoke_hook", return_value=[]):
        await runner._handle_reset_command(event)

    reset_keys = [call.args[0] for call in runner.session_store.reset_session.call_args_list]
    assert topic_entry.session_key in reset_keys
    assert general_entry.session_key in reset_keys
    assert projects_entry.session_key not in reset_keys


def dataclasses_replace_thread(source: SessionSource, thread_id: str) -> SessionSource:
    import dataclasses
    return dataclasses.replace(source, thread_id=thread_id)
