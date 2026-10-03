"""A refused idle publication must not announce a compression boundary."""

from unittest.mock import MagicMock

from agent.conversation_compression import _finish_compaction_boundary


def test_non_durable_in_place_boundary_keeps_existing_notifications():
    agent = MagicMock()
    agent.session_id = "session"
    agent.model = "test"
    agent.tools = []
    agent._post_reply_idle_claim = None
    agent.context_compressor.compression_count = 1
    _finish_compaction_boundary(
        agent, [{"role": "user", "content": "head"}],
        new_system_prompt="pinned", old_session_id=None, in_place=True,
        compacted_in_place=False, session_commit_succeeded=False,
        defer_context_engine_notification=False, compression_made_progress=False,
        compression_used_fallback=False, compression_feasibility_skip=False,
        task_id="session",
    )
    agent._memory_manager.on_session_switch.assert_called_once()
    agent.event_callback.assert_called_once()


def test_failed_in_place_commit_does_not_notify_memory_or_hooks():
    agent = MagicMock()
    agent.session_id = "session"
    agent.model = "test"
    agent.tools = []
    agent.context_compressor.compression_count = 1
    agent.context_compressor.threshold_tokens = 100000
    agent.context_compressor.summary_target_ratio = 0.2
    agent._memory_manager = MagicMock()
    agent.event_callback = MagicMock()
    _finish_compaction_boundary(
        agent, [{"role": "user", "content": "unchanged"}],
        new_system_prompt="pinned", old_session_id=None, in_place=True,
        compacted_in_place=False, session_commit_succeeded=False,
        defer_context_engine_notification=False, compression_made_progress=False,
        compression_used_fallback=False, compression_feasibility_skip=False,
        task_id="session",
    )
    agent._memory_manager.on_session_switch.assert_not_called()
    agent.event_callback.assert_not_called()
