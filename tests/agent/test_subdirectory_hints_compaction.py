"""Subdirectory hints re-arm at a compaction boundary.

The tracker remembers which directories' AGENTS.md / CLAUDE.md hints were already appended to tool
results. Compaction summarises those tool results away, so a tracker that still counts them as delivered
never sends the hint again for the rest of the session.
"""

import os
from unittest.mock import MagicMock, patch

from agent.subdirectory_hints import SubdirectoryHintTracker
from hermes_state import SessionDB


def _project(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Root project instructions", encoding="utf-8")
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "AGENTS.md").write_text("Backend rules: use FastAPI", encoding="utf-8")
    return tmp_path


def _agent(db, session_id, project):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            platform="cli", quiet_mode=True, session_db=db, session_id=session_id,
            skip_context_files=True, skip_memory=True,
        )
    agent._subdirectory_hints = SubdirectoryHintTracker(working_dir=str(project))
    compressor = MagicMock()
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] earlier work summarised"},
        {"role": "assistant", "content": "acknowledged"},
        {"role": "user", "content": "tail"},
    ]
    agent.context_compressor = compressor
    agent.compression_in_place = False
    return agent


def _msgs(n=20):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * 400} for i in range(n)]


def test_hint_is_delivered_again_after_compaction(tmp_path):
    project = _project(tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("SUBDIR_HINTS_COMPACTION", source="cli")
    agent = _agent(db, "SUBDIR_HINTS_COMPACTION", project)
    call = ("read_file", {"path": str(project / "backend" / "app.py")})

    first = agent._subdirectory_hints.check_tool_call(*call)
    assert first is not None and "Backend rules" in first
    assert agent._subdirectory_hints.check_tool_call(*call) is None

    agent._compress_context(_msgs(), "sys", approx_tokens=120_000, task_id="t-hints")
    db.close()

    again = agent._subdirectory_hints.check_tool_call(*call)
    assert again is not None and "Backend rules" in again
    # The working dir's own hint file lives in the system prompt, which the boundary rebuilds: it stays
    # pre-marked and is never spliced into a tool result.
    root = agent._subdirectory_hints.check_tool_call("read_file", {"path": str(project / "notes.txt")})
    assert root is None
