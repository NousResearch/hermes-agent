from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_cli.commands import resolve_command


class _FakeDB:
    def __init__(self, messages):
        self.messages = messages
        self.requested_session_id = None

    def get_messages(self, session_id):
        self.requested_session_id = session_id
        return list(self.messages)


class _FakeSessionStore:
    def __init__(self, source, messages):
        self._entries = {}
        self._db = _FakeDB(messages)
        self._key = build_session_key(source)
        self._entries[self._key] = SessionEntry(
            session_key=self._key,
            session_id="session-ctx-1",
            created_at=datetime(2026, 5, 9, 12, 0, 0),
            updated_at=datetime(2026, 5, 9, 12, 5, 0),
            origin=source,
            display_name=source.chat_name,
            platform=source.platform,
            chat_type=source.chat_type,
        )

    def _ensure_loaded(self):
        return None

    def _generate_session_key(self, source):
        return build_session_key(source)


def _make_runner(source, messages):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.emit_collect = AsyncMock(return_value=[])
    runner.session_store = _FakeSessionStore(source, messages)
    return runner


def _make_source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100123",
        chat_name="Hermes Agent - Mina",
        chat_type="group",
        user_id="u1",
        user_name="Joohyun Kim",
        thread_id="2226",
    )


def _event(text="/ctx"):
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_make_source())


def test_ctx_command_registered_for_gateway():
    cmd = resolve_command("ctx")
    assert cmd is not None
    assert cmd.name == "ctx"
    assert cmd.gateway_only is True


@pytest.mark.asyncio
async def test_ctx_command_returns_compact_thread_brief_from_current_session():
    source = _make_source()
    messages = [
        {"role": "user", "content": "Startup memory context for Mina: staged memory stack and local-first cmem."},
        {"role": "assistant", "content": "Understood. I will use the memory stack explicitly."},
        {"role": "user", "content": "Telegram Quick Action: delegate a follow-up from the previous assistant reply."},
        {"role": "assistant", "content": "Created telegram-thread-context-brief skill and added /ctx implementation todo."},
        {"role": "user", "content": "게이트웨이 재시작 때문에 끊겼어. 계속 진행."},
    ]
    runner = _make_runner(source, messages)

    reply = await runner._handle_message(_event("/ctx"))

    assert "## Thread context" in reply
    assert "**전체 맥락:**" in reply
    assert "**최근 맥락:**" in reply
    assert "**다음 액션:**" in reply
    assert "**결정 필요:**" in reply
    assert "staged memory stack" in reply
    assert "telegram-thread-context-brief" in reply
    assert runner.session_store._db.requested_session_id == "session-ctx-1"


@pytest.mark.asyncio
async def test_ctx_command_bypasses_running_agent_guard():
    source = _make_source()
    runner = _make_runner(source, [
        {"role": "user", "content": "첫 메시지"},
        {"role": "assistant", "content": "첫 응답"},
    ])
    runner._running_agents[build_session_key(source)] = object()

    reply = await runner._handle_message(_event("/ctx"))

    assert "## Thread context" in reply
    assert "can't run mid-turn" not in reply
