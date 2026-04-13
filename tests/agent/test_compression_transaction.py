"""W1 / F-010 regression tests.

Compression must be transactional: on any mid-sequence failure the agent's
in-memory snapshot is restored and a CompressionFailed is raised so the
caller can surface "try /new" cleanly. Pre-fix a partial DB-split failure
left the agent with a cached system prompt rebuilt for a session that was
never fully persisted — the poisoned lineage then replayed on resume.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.compression_errors import CompressionFailed
from agent.core import AIAgent


def _make_tool_defs(*names):
    from openai.types.chat import ChatCompletionToolParam
    return [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": n,
                "description": f"{n} stub",
                "parameters": {"type": "object", "properties": {}},
            },
        )
        for n in names
    ]


@pytest.fixture()
def agent():
    with (
        patch("agent.core.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("agent.core.check_toolset_requirements", return_value={}),
        patch("agent.client_manager.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a.client_manager.client = a.client
        # Pre-set a stable baseline that snapshot should restore to.
        a.session_id = "pre_compress_session"
        a._cached_system_prompt = "PRE-COMPRESS PROMPT"
        a._last_flushed_db_idx = 7
        a.session_log_file = Path("/tmp/pre_compress_session.json")
        return a


class TestSnapshotRestore:
    def test_compressor_exception_restores_state(self, agent):
        """If context_compressor.compress raises, agent state is
        pre-attempt and CompressionFailed(stage="compress") surfaces."""
        messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        boom = RuntimeError("compressor exploded")

        with patch.object(agent.context_compressor, "compress", side_effect=boom):
            with pytest.raises(CompressionFailed) as excinfo:
                agent._compress_context(messages, "system msg")

        assert excinfo.value.stage == "compress"
        assert excinfo.value.original_message_count == 2
        # Snapshot restored.
        assert agent.session_id == "pre_compress_session"
        assert agent._cached_system_prompt == "PRE-COMPRESS PROMPT"
        assert agent._last_flushed_db_idx == 7

    def test_db_split_exception_restores_state(self, agent):
        """If session-DB split fails partway, session_id/cached_prompt
        roll back to pre-attempt values; no poisoned lineage."""
        fake_db = MagicMock()
        fake_db.get_session_title.return_value = None
        # end_session succeeds, create_session blows up halfway through.
        fake_db.create_session.side_effect = RuntimeError("sqlite locked")
        agent._session_db = fake_db

        messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        with patch.object(agent.context_compressor, "compress", return_value=[{"role": "user", "content": "compressed"}]):
            with patch.object(agent, "_build_system_prompt", return_value="NEW PROMPT"):
                with pytest.raises(CompressionFailed) as excinfo:
                    agent._compress_context(messages, "sys")

        assert excinfo.value.stage == "db_split"
        # Critical: agent's session_id must be the ORIGINAL, not the new ID
        # that was assigned mid-transaction before create_session failed.
        assert agent.session_id == "pre_compress_session"
        # Cached prompt must be the ORIGINAL — not the rebuilt "NEW PROMPT"
        # that was published just before the DB exception.
        assert agent._cached_system_prompt == "PRE-COMPRESS PROMPT"
        assert agent._last_flushed_db_idx == 7

    def test_prompt_rebuild_exception_restores_state(self, agent):
        """If _build_system_prompt raises, we haven't touched the DB yet
        but cached prompt may have been invalidated — must restore."""
        messages = [{"role": "user", "content": "a"}]
        with patch.object(agent.context_compressor, "compress", return_value=[]):
            with patch.object(agent, "_build_system_prompt", side_effect=RuntimeError("bad template")):
                with pytest.raises(CompressionFailed) as excinfo:
                    agent._compress_context(messages, "sys")

        assert excinfo.value.stage == "prompt_rebuild"
        assert agent._cached_system_prompt == "PRE-COMPRESS PROMPT"

    def test_happy_path_updates_state(self, agent):
        """Success path: session_id changes, cached prompt rebuilt,
        flush cursor reset — no CompressionFailed raised."""
        fake_db = MagicMock()
        fake_db.get_session_title.return_value = None
        agent._session_db = fake_db

        messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        compressed = [{"role": "user", "content": "summary"}]
        with patch.object(agent.context_compressor, "compress", return_value=compressed):
            with patch.object(agent, "_build_system_prompt", return_value="NEW PROMPT"):
                result_msgs, result_prompt = agent._compress_context(messages, "sys")

        assert result_msgs == compressed
        assert result_prompt == "NEW PROMPT"
        # Mutations published.
        assert agent.session_id != "pre_compress_session"
        assert agent._cached_system_prompt == "NEW PROMPT"
        assert agent._last_flushed_db_idx == 0

    def test_memory_flush_failure_is_compression_failed(self, agent):
        """flush_memories raising is not a transient bug — it means the
        model's own memory-save step errored; surface as CompressionFailed
        so the caller can ask the user to /new rather than compressing
        against half-flushed memory state."""
        messages = [{"role": "user", "content": "a"}]
        with patch.object(agent, "flush_memories", side_effect=RuntimeError("memory write failed")):
            with pytest.raises(CompressionFailed) as excinfo:
                agent._compress_context(messages, "sys")
        assert excinfo.value.stage == "memory_flush"
        # flush_memories runs before the snapshot, so session_id must still
        # be the original — the whole point of the typed raise is that the
        # CALLER surfaces /new, not that we try to "undo" the memory write.
        assert agent.session_id == "pre_compress_session"
