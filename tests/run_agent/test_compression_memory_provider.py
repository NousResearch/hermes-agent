"""Tests for memory provider lifecycle during context compression.

Verifies that external memory providers (e.g. OpenViking) are properly
notified when a session is split due to compression.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


@pytest.fixture
def agent_with_mem_mgr():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
    ):
        a = AIAgent(
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="original-session",
        )
        # Replace the real memory manager with a mock so we can assert calls
        a._memory_manager = MagicMock()
        a._memory_manager.build_system_prompt.return_value = ""
        a._cached_system_prompt = "You are helpful."
        yield a


class TestCompressContextNotifiesMemoryProvider:
    """_compress_context must call on_session_end and re-initialize providers."""

    def test_compress_calls_on_session_end_before_split(self, agent_with_mem_mgr):
        """When compression splits the session, the memory provider must receive
        on_session_end with the full messages so it can commit/archive them."""
        agent = agent_with_mem_mgr
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent._session_db = db
            db.create_session(session_id=agent.session_id, source="test")

            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "what is my name"},
            ]

            with patch.object(
                agent.context_compressor, "compress", return_value=[
                    {"role": "user", "content": "[SUMMARY] earlier chat"},
                    {"role": "user", "content": "what is my name"},
                ]
            ):
                compressed, _ = agent._compress_context(
                    messages, "system", approx_tokens=1000
                )

            agent._memory_manager.on_session_end.assert_called_once()
            call_args = agent._memory_manager.on_session_end.call_args
            assert call_args.args[0] == messages

    def test_compress_reinitializes_memory_provider_with_new_session(self, agent_with_mem_mgr):
        """After splitting, the memory provider must be re-initialized with the
        new session_id so future sync_turn calls target the new session."""
        agent = agent_with_mem_mgr
        original_session_id = agent.session_id
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent._session_db = db
            db.create_session(session_id=agent.session_id, source="test")

            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]

            with patch.object(
                agent.context_compressor, "compress", return_value=[
                    {"role": "user", "content": "[SUMMARY] earlier chat"},
                ]
            ):
                agent._compress_context(messages, "system", approx_tokens=1000)

            agent._memory_manager.initialize_all.assert_called_once()
            call_kwargs = agent._memory_manager.initialize_all.call_args.kwargs
            assert call_kwargs["session_id"] != original_session_id
            assert agent.session_id == call_kwargs["session_id"]
