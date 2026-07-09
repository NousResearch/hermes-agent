"""Workspace metadata stamping at the runtime compression boundary."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def test_compress_context_stamps_workspace_metadata_on_summary(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PARENT_WORKSPACE_STAMP"
    db.create_session(session_id, source="cli", cwd=str(repo))
    with db._lock:
        assert db._conn is not None
        db._conn.execute(
            "UPDATE sessions SET cwd=?, git_repo_root=? WHERE id=?",
            (str(repo), str(repo), session_id),
        )
        db._conn.commit()

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()
    compressor.compress.return_value = [
        {
            "role": "user",
            "content": "[CONTEXT COMPACTION] summary",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    setattr(agent, "context_compressor", compressor)
    setattr(agent, "compression_in_place", True)

    compressed, _ = agent._compress_context(
        [{"role": "user", "content": f"m{i}"} for i in range(20)],
        "sys",
        approx_tokens=120_000,
    )

    summary = next(m for m in compressed if m.get(COMPRESSED_SUMMARY_METADATA_KEY))
    assert f"HERMES_WORKSPACE:{repo}" in summary["content"]
