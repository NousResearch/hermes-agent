from unittest.mock import patch

from hermes_state import SessionDB
from run_agent import AIAgent


class TestCompressionContinuationNotice:
    def test_compression_emits_continuation_status_when_session_id_rotates(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        statuses = []

        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key-12345678",
                base_url="https://example.test/v1",
                provider="custom",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_db=db,
                session_id="compression-session",
                status_callback=lambda kind, message: statuses.append((kind, message)),
            )

        old_session_id = agent.session_id
        original_messages = [
            {"role": "user", "content": "Please continue the task."},
            {"role": "assistant", "content": "Working on it."},
            {"role": "user", "content": "Add the regression tests too."},
            {"role": "assistant", "content": "On it."},
        ]
        compressed_messages = [
            {"role": "user", "content": "[CONTEXT COMPACTION] earlier work summary"},
            {"role": "assistant", "content": "Recent state"},
        ]

        with (
            patch.object(agent, "flush_memories"),
            patch.object(agent, "commit_memory_session"),
            patch.object(agent, "_invalidate_system_prompt"),
            patch.object(agent, "_build_system_prompt", return_value="system prompt"),
            patch.object(agent.context_compressor, "compress", return_value=compressed_messages),
            patch.object(agent._todo_store, "format_for_injection", return_value=""),
        ):
            result_messages, new_system_prompt = agent._compress_context(
                original_messages,
                "system prompt",
                approx_tokens=12345,
                task_id="compression-session",
            )

        assert result_messages == compressed_messages
        assert new_system_prompt == "system prompt"
        assert agent.session_id != old_session_id
        assert db.get_session(agent.session_id)["parent_session_id"] == old_session_id
        assert statuses, "expected a lifecycle status message for compression continuation"
        kind, message = statuses[-1]
        assert kind == "lifecycle"
        assert "same conversation" in message
        assert "not a reset" in message
        assert old_session_id in message
        assert agent.session_id in message

        db.close()
