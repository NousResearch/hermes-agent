"""Regression: compaction must move the LOGGING session context with the id.

When ``compress_context`` rotates ``agent.session_id`` it updates the
gateway/tools session context (``gateway.session_context.set_current_session_id``,
which moves ``HERMES_SESSION_ID`` env + ContextVar). The ``[session_id]`` tag on
log lines comes from a SEPARATE mechanism — ``hermes_logging._session_context``
(a threading.local read by the global LogRecord factory), set once per turn in
``conversation_loop.py``. Before the fix, the rotation block never updated it, so
log lines emitted after a mid-turn compaction carried the STALE old id while the
message body / session DB / gateway state carried the new one (see #34089). This
asserts the logging context follows the rotation.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import hermes_logging
from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str):
    """Mirror tests/agent/test_compression_concurrent_fork.py's harness."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

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
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    return agent


def test_logging_session_context_follows_compression_rotation(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "PARENT_LOGCTX_SESSION"
    db.create_session(parent_sid, source="cli")

    agent = _build_agent_with_db(db, parent_sid)

    # conversation_loop.py pins the logging tag to the ORIGINAL id at turn start.
    hermes_logging.set_session_context(parent_sid)
    try:
        messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(messages, "sys", approx_tokens=120_000)

        # The id actually rotated (sanity — otherwise the assertion is vacuous).
        assert agent.session_id != parent_sid

        # The logging context must now match the NEW id, not the stale one.
        current = getattr(hermes_logging._session_context, "session_id", None)
        assert current == agent.session_id, (
            "Logging session context did not follow the compaction rotation: "
            f"log tag still {current!r}, agent.session_id is {agent.session_id!r} "
            "(see #34089)."
        )
    finally:
        hermes_logging.clear_session_context()


def test_compression_continuation_inherits_parent_cwd(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "PARENT_CWD_SESSION"
    parent_cwd = str(tmp_path / "project")
    Path(parent_cwd).mkdir()
    db.create_session(parent_sid, source="cli", cwd=parent_cwd)

    agent = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    agent._compress_context(messages, "sys", approx_tokens=120_000)

    child_sid = getattr(agent, "session_id")
    assert child_sid != parent_sid
    child = db.get_session(child_sid)
    assert child is not None
    assert child["parent_session_id"] == parent_sid
    assert child["cwd"] == parent_cwd
