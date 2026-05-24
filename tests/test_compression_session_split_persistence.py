"""Regression tests for context compression session splits.

When automatic compression creates a continuation session, both persistence
layers must treat the compressed message list as the new session's complete
history.  Keeping the old history offset causes the compressed summary/tail to
be skipped, so the next turn resumes an effectively empty continuation.
"""

import sys
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _mock_response(content="OK", finish_reason="stop"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model", usage=None)
    return resp


def _make_413_error():
    err = Exception("Request entity too large")
    err.status_code = 413
    return err


class TestAgentSessionSplitPersistence:
    def _make_agent(self, session_db):
        from run_agent import AIAgent

        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="x",
                base_url="https://example.invalid/v1",
                model="test/model",
                quiet_mode=True,
                session_id="original-session",
                session_db=session_db,
                enabled_toolsets=[],
                skip_context_files=True,
                skip_memory=True,
                save_trajectories=False,
            )
        agent.client = MagicMock()
        agent.tools = []
        agent._cached_system_prompt = "system prompt"
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        return agent

    def test_413_compression_split_flushes_compressed_messages_to_new_session(self, tmp_path):
        """A stale pre-compression history offset must not skip the new session."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "sessions.db")
        agent = self._make_agent(db)

        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"old message {i}"}
            for i in range(40)
        ]
        compressed_messages = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary of the old session"},
            {"role": "user", "content": "fresh question after compression"},
        ]

        def fake_compress(messages, system_message, **kwargs):
            agent.session_id = "compressed-session"
            db.create_session(
                session_id="compressed-session",
                source="qqbot",
                model="test/model",
                parent_session_id="original-session",
            )
            agent._last_flushed_db_idx = 0
            return list(compressed_messages), "compressed system prompt"

        agent.client.chat.completions.create.side_effect = [
            _make_413_error(),
            _mock_response("final answer after compression"),
        ]

        with (
            patch.object(agent, "_compress_context", side_effect=fake_compress),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time.sleep", return_value=None),
        ):
            result = agent.run_conversation(
                "fresh question after compression",
                conversation_history=long_history,
            )

        assert result["completed"] is True
        rows = db.get_messages("compressed-session")
        contents = [row.get("content") for row in rows]
        assert any(
            "[CONTEXT COMPACTION] summary of the old session" in str(content)
            for content in contents
        )
        assert "final answer after compression" in contents


def test_gateway_history_offset_resets_when_session_split():
    from gateway.run import _history_offset_for_session_split

    assert _history_offset_for_session_split(
        original_session_id="original-session",
        effective_session_id="compressed-session",
        original_history_len=40,
    ) == 0


def test_gateway_history_offset_preserves_normal_session():
    from gateway.run import _history_offset_for_session_split

    assert _history_offset_for_session_split(
        original_session_id="same-session",
        effective_session_id="same-session",
        original_history_len=40,
    ) == 40


@pytest.mark.asyncio
async def test_gateway_error_response_preserves_session_split(monkeypatch, tmp_path):
    """Even no-final-response paths must persist compressed continuations."""
    from gateway.config import Platform
    from gateway.session import SessionSource
    import gateway.run as gateway_run
    import run_agent

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]
            self.tools = []
            self.context_compressor = SimpleNamespace(last_prompt_tokens=123)

        def run_conversation(self, message, conversation_history=None, task_id=None):
            self.session_id = "compressed-session"
            return {
                "final_response": None,
                "messages": [
                    {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                    {"role": "user", "content": message},
                ],
                "api_calls": 1,
                "error": "compression retry failed",
                "completed": False,
            }

    class FakeSessionStore:
        def __init__(self):
            self._entries = {
                "session-key": SimpleNamespace(session_id="original-session")
            }
            self.saved = False

        def _save(self):
            self.saved = True

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False, emit=AsyncMock())
    runner.session_store = FakeSessionStore()
    runner._provider_routing = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._get_or_create_gateway_honcho = lambda _session_key: (None, None)

    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *a, **k: "test/model")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***", "base_url": "https://example.invalid/v1"},
    )
    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)

    result = await runner._run_agent(
        message="fresh question",
        context_prompt="",
        history=[
            {"role": "session_meta", "tools": []},
            {"role": "user", "content": "old message"},
            {"role": "assistant", "content": "old answer"},
        ],
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm"),
        session_id="original-session",
        session_key="session-key",
    )

    assert result["final_response"] == "⚠️ compression retry failed"
    assert result["session_id"] == "compressed-session"
    assert result["history_offset"] == 0
    assert runner.session_store._entries["session-key"].session_id == "compressed-session"
    assert runner.session_store.saved is True

def test_gateway_split_failed_path_uses_compressed_messages_not_duplicate_user():
    from gateway.run import _messages_to_persist_after_agent_run

    agent_messages = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "fresh question"},
    ]
    messages, skip_db = _messages_to_persist_after_agent_run(
        agent_result={
            "failed": True,
            "session_id": "compressed-session",
            "history_offset": 0,
            "messages": agent_messages,
        },
        current_session_id="compressed-session",
        original_session_id="original-session",
        agent_messages=agent_messages,
        message_text="fresh question",
        timestamp="2026-01-01T00:00:00",
        platform_message_id="msg-1",
        session_db_available=True,
    )

    assert [msg["content"] for msg in messages] == [
        "[CONTEXT COMPACTION] summary",
        "fresh question",
    ]
    assert messages[1]["message_id"] == "msg-1"
    assert skip_db is True


def test_gateway_split_failed_path_does_not_bind_message_id_to_compaction_summary():
    from gateway.run import _messages_to_persist_after_agent_run

    agent_messages = [
        {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary"},
        {"role": "user", "content": [{"type": "text", "text": "fresh question"}]},
    ]
    messages, skip_db = _messages_to_persist_after_agent_run(
        agent_result={
            "failed": True,
            "session_id": "compressed-session",
            "history_offset": 0,
            "messages": agent_messages,
        },
        current_session_id="compressed-session",
        original_session_id="original-session",
        agent_messages=agent_messages,
        message_text="fresh question",
        timestamp="2026-01-01T00:00:00",
        platform_message_id="msg-1",
        session_db_available=True,
    )

    assert "message_id" not in messages[0]
    assert messages[1]["message_id"] == "msg-1"
    assert skip_db is True


def _make_split_persistence_runner(monkeypatch, tmp_path, agent_result):
    import gateway.run as gateway_run
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.session import SessionEntry

    entry = SessionEntry(
        session_key="agent:main:telegram:dm:chat:user",
        session_id="original-session",
        created_at=datetime(2026, 1, 1, 0, 0, 0),
        updated_at=datetime(2026, 1, 1, 0, 0, 1),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    appended = []

    class FakeSessionStore:
        config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")}
        )

        def get_or_create_session(self, source):
            return entry

        def load_transcript(self, session_id):
            return [
                {"role": "user", "content": "old message"},
                {"role": "assistant", "content": "old answer"},
            ]

        def append_to_transcript(self, session_id, message, skip_db=False):
            appended.append((session_id, dict(message), skip_db))

        def update_session(self, *args, **kwargs):
            pass

        def clear_resume_pending(self, *args, **kwargs):
            pass

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = FakeSessionStore.config
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = FakeSessionStore()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = object()
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._get_or_create_gateway_honcho = lambda _session_key: (None, None)
    runner._is_session_run_current = lambda _session_key, _generation: True

    async def fake_run_agent(*args, **kwargs):
        # Simulate _run_agent's real split side effect: it mutates the shared
        # SessionEntry before _handle_message_with_agent persists rows.
        entry.session_id = "compressed-session"
        return dict(agent_result)

    runner._run_agent = fake_run_agent
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    return runner, entry, appended


@pytest.mark.asyncio
async def test_gateway_failed_split_persistence_uses_original_session_id_before_entry_mutation(monkeypatch, tmp_path):
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    runner, entry, appended = _make_split_persistence_runner(
        monkeypatch,
        tmp_path,
        {
            "final_response": "",
            "failed": True,
            "error": "compression retry failed",
            "messages": [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "fresh question"},
            ],
            "history_offset": 0,
            "tools": [],
            "last_prompt_tokens": 0,
            "session_id": "compressed-session",
        },
    )

    event = MessageEvent(
        text="fresh question",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user"),
        message_id="msg-1",
    )

    result = await runner._handle_message_with_agent(event, event.source, entry.session_key, 1)

    assert "compression retry failed" in result
    persisted = [row for _sid, row, _skip_db in appended]
    assert [row["content"] for row in persisted] == [
        "[CONTEXT COMPACTION] summary",
        "fresh question",
    ]
    assert "message_id" not in persisted[0]
    assert persisted[1]["message_id"] == "msg-1"
    assert all(sid == "compressed-session" for sid, _row, _skip_db in appended)
    assert all(skip_db is True for _sid, _row, skip_db in appended)


@pytest.mark.asyncio
async def test_gateway_success_split_message_id_skips_compaction_summary(monkeypatch, tmp_path):
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    runner, entry, appended = _make_split_persistence_runner(
        monkeypatch,
        tmp_path,
        {
            "final_response": "final answer",
            "messages": [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": [{"type": "text", "text": "fresh question"}]},
                {"role": "assistant", "content": "final answer"},
            ],
            "history_offset": 0,
            "tools": [],
            "last_prompt_tokens": 0,
            "session_id": "compressed-session",
        },
    )

    event = MessageEvent(
        text="fresh question",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user"),
        message_id="msg-1",
    )

    result = await runner._handle_message_with_agent(event, event.source, entry.session_key, 1)

    assert result == "final answer"
    persisted = [row for _sid, row, _skip_db in appended]
    assert [row["content"] for row in persisted] == [
        "[CONTEXT COMPACTION] summary",
        [{"type": "text", "text": "fresh question"}],
        "final answer",
    ]
    assert "message_id" not in persisted[0]
    assert persisted[1]["message_id"] == "msg-1"
