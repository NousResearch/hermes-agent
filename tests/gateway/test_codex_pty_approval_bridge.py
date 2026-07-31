"""Gateway routing tests for managed native Codex PTY approvals."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from tools.process_registry import ProcessSession


class _DiscordAdapter:
    def __init__(self):
        self.calls = []
        self.paused = []

    def pause_typing_for_chat(self, chat_id):
        self.paused.append(chat_id)

    async def send_exec_approval(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(success=True)


class _TextDiscordAdapter:
    typed_command_prefix = "/"

    def __init__(self):
        self.sent = []

    def pause_typing_for_chat(self, _chat_id):
        pass

    async def send(self, chat_id, message, metadata=None):
        self.sent.append((chat_id, message, metadata))
        return SimpleNamespace(success=True)


@pytest.mark.asyncio
async def test_managed_codex_prompt_uses_process_session_origin_and_queue_key():
    runner = object.__new__(GatewayRunner)
    source = SimpleNamespace(
        platform=Platform.DISCORD,
        chat_id="channel-123",
        thread_id="thread-9",
        message_id=None,
        profile=None,
    )
    adapter = _DiscordAdapter()
    captured_event = {}

    def build_source(event):
        captured_event.update(event)
        return source

    runner._build_process_event_source = build_source
    runner._adapter_for_source = lambda actual: adapter if actual is source else None
    runner._thread_metadata_for_source = lambda actual: {"thread_id": actual.thread_id}
    session = ProcessSession(
        id="proc_codex",
        command="codex",
        session_key="agent:main:discord:thread:thread-9:user:42",
    )

    sent = await runner._send_managed_codex_approval(
        session,
        {"command": "$ make deploy", "description": "Codex command execution"},
    )

    assert sent is True
    assert captured_event["session_key"] == session.session_key
    assert adapter.paused == [source.chat_id]
    assert adapter.calls == [
        {
            "chat_id": source.chat_id,
            "command": "$ make deploy",
            "session_key": session.session_key,
            "description": "Codex command execution",
            "metadata": {"thread_id": source.thread_id},
            "allow_permanent": False,
            "allow_session": True,
            "smart_denied": False,
        }
    ]


@pytest.mark.asyncio
async def test_managed_codex_prompt_refuses_non_discord_origin():
    runner = object.__new__(GatewayRunner)
    runner._build_process_event_source = lambda _event: SimpleNamespace(
        platform=Platform.TELEGRAM,
    )
    session = ProcessSession(
        id="proc_codex",
        command="codex",
        session_key="agent:main:telegram:dm:123:user:42",
    )

    assert await runner._send_managed_codex_approval(session, {}) is False


@pytest.mark.asyncio
async def test_managed_codex_prompt_text_fallback_names_channel_commands():
    runner = object.__new__(GatewayRunner)
    source = SimpleNamespace(
        platform=Platform.DISCORD,
        chat_id="channel-123",
        thread_id=None,
    )
    adapter = _TextDiscordAdapter()
    runner._build_process_event_source = lambda _event: source
    runner._adapter_for_source = lambda _source: adapter
    runner._thread_metadata_for_source = lambda _source: None
    session = ProcessSession(
        id="proc_codex",
        command="codex",
        session_key="agent:main:discord:group:channel-123:user:42",
    )

    assert await runner._send_managed_codex_approval(
        session, {"command": "$ make deploy"}
    )
    assert "`/approve`" in adapter.sent[0][1]
    assert "`/deny`" in adapter.sent[0][1]
