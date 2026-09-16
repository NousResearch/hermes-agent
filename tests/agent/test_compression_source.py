"""Compression rotation preserves the durable session source."""

import os
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def test_compression_child_preserves_parent_source(tmp_path):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        db = SessionDB(db_path=tmp_path / "state.db")
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id="original-session",
            skip_context_files=True,
            skip_memory=True,
        )
    try:
        agent.compression_in_place = False
        db.create_session(agent.session_id, source="oneshot")
        agent._session_db_created = True
        compressor = MagicMock()
        compressor.compress.return_value = [{"role": "user", "content": "summary"}]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        agent.context_compressor = compressor

        agent._compress_context(
            [{"role": "user", "content": "request"}],
            "sys",
            approx_tokens=100,
        )

        assert db.get_session(agent.session_id)["source"] == "oneshot"
    finally:
        agent.close()
        db.close()
