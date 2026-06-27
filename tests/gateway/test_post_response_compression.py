import threading
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource


def _entry(session_key: str = "agent:main:discord:group:chan") -> SessionEntry:
    return SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="group",
    )


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan",
        chat_type="group",
        user_id="user-1",
    )


def _messages(content_size: int = 400) -> list[dict]:
    return [
        {"role": "user", "content": "u" * content_size},
        {"role": "assistant", "content": "a" * content_size},
        {"role": "user", "content": "u2" * content_size},
        {"role": "assistant", "content": "a2" * content_size},
    ]


class _RotatingAgent:
    compression_enabled = True
    tools = []
    _cached_system_prompt = ""
    _session_db = object()

    def __init__(self):
        self.session_id = "sess-1"
        self.context_compressor = SimpleNamespace(
            threshold_tokens=50,
            context_length=100,
        )
        self._last_compaction_in_place = False
        self.calls = []

    def _compress_context(self, messages, system_message, **kwargs):
        self.calls.append((messages, system_message, kwargs))
        self.session_id = "sess-2"
        return ([{"role": "assistant", "content": "summary"}], "system")


class _NoopAgent(_RotatingAgent):
    def _compress_context(self, messages, system_message, **kwargs):
        self.calls.append((messages, system_message, kwargs))
        return (messages, "system")


class _FailingAgent(_RotatingAgent):
    def _compress_context(self, messages, system_message, **kwargs):
        raise RuntimeError("compression broke")


def _runner(agent, entry):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = OrderedDict({entry.session_key: (agent, "sig", 4)})
    runner._session_db = MagicMock()
    runner.session_store = MagicMock()
    runner.session_store._entries = {entry.session_key: entry}
    runner.session_store._save = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._sync_telegram_topic_binding = MagicMock()
    runner._refresh_agent_cache_message_count = MagicMock()
    return runner


@pytest.mark.asyncio
async def test_post_response_compression_rotates_and_persists_live_context(monkeypatch):
    from gateway import run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "compression": {
                "enabled": True,
                "post_response_enabled": True,
                "post_response_threshold": None,
            }
        },
    )
    entry = _entry()
    agent = _RotatingAgent()
    runner = _runner(agent, entry)

    await runner._run_post_response_compression(
        session_key=entry.session_key,
        session_entry=entry,
        source=_source(),
        agent_result={"messages": _messages()},
    )

    assert entry.session_id == "sess-2"
    runner.session_store._save.assert_called_once()
    runner.session_store.rewrite_transcript.assert_called_once_with(
        "sess-2",
        [{"role": "assistant", "content": "summary"}],
    )
    runner.session_store.update_session.assert_called_once_with(
        entry.session_key,
        last_prompt_tokens=0,
    )
    runner._refresh_agent_cache_message_count.assert_called_once_with(
        entry.session_key,
        "sess-2",
    )


@pytest.mark.asyncio
async def test_post_response_compression_skips_below_threshold(monkeypatch):
    from gateway import run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "compression": {
                "enabled": True,
                "post_response_enabled": True,
                "post_response_threshold": None,
            }
        },
    )
    entry = _entry()
    agent = _RotatingAgent()
    agent.context_compressor.threshold_tokens = 1_000_000
    runner = _runner(agent, entry)

    await runner._run_post_response_compression(
        session_key=entry.session_key,
        session_entry=entry,
        source=_source(),
        agent_result={"messages": _messages(content_size=10)},
    )

    assert agent.calls == []
    assert entry.session_id == "sess-1"
    runner.session_store.rewrite_transcript.assert_not_called()
    runner.session_store.update_session.assert_not_called()


@pytest.mark.asyncio
async def test_post_response_compression_noop_does_not_rewrite(monkeypatch):
    from gateway import run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "compression": {
                "enabled": True,
                "post_response_enabled": True,
                "post_response_threshold": 0.01,
            }
        },
    )
    entry = _entry()
    agent = _NoopAgent()
    runner = _runner(agent, entry)

    await runner._run_post_response_compression(
        session_key=entry.session_key,
        session_entry=entry,
        source=_source(),
        agent_result={"messages": _messages()},
    )

    assert entry.session_id == "sess-1"
    runner.session_store.rewrite_transcript.assert_not_called()
    runner.session_store.update_session.assert_not_called()


@pytest.mark.asyncio
async def test_post_response_compression_failure_logs_and_continues(monkeypatch, caplog):
    from gateway import run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "compression": {
                "enabled": True,
                "post_response_enabled": True,
                "post_response_threshold": 0.01,
            }
        },
    )
    entry = _entry()
    runner = _runner(_FailingAgent(), entry)

    await runner._run_post_response_compression(
        session_key=entry.session_key,
        session_entry=entry,
        source=_source(),
        agent_result={"messages": _messages()},
    )

    assert "Post-response compression failed for session sess-1" in caplog.text
    runner.session_store.rewrite_transcript.assert_not_called()
