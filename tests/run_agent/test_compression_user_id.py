"""Test: the context-compression rollover carries the owner's user_id into
the continuation session row (parity with the normal gateway session-create).

Regression guard for the bug where conversation_compression rotated onto a fresh
session_id but wrote the new state.db row via create_session(...) WITHOUT user_id,
orphaning gateway (Telegram/Discord) sessions to user_id=NULL on every compaction.

Mirrors the real driving pattern in tests/run_agent/test_compression_boundary_hook.py.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestCompressionCarriesUserId:
    def _make_agent(self, session_db, user_id):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )
        # Force the ROTATION path (mint a new session_id) so the rollover
        # create_session runs — mirrors test_compression_boundary_hook.py (#38763).
        agent.compression_in_place = False
        agent._user_id = user_id
        return agent

    def _stub_compressor(self, agent):
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail question"},
        ]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        # Clear the abort flag so the post-compress abort branch does not
        # short-circuit before the session-id rotation we assert on.
        compressor._last_compress_aborted = False
        agent.context_compressor = compressor

    def test_rollover_persists_gateway_user_id(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db, "U-12345")
            self._stub_compressor(agent)

            old_sid = agent.session_id
            messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]
            agent._compress_context(messages, "sys", approx_tokens=10_000)

            assert agent.session_id != old_sid, "compression should rotate session_id"
            row = db.get_session(agent.session_id)
            assert row is not None, "continuation session row was not persisted"
            assert row["user_id"] == "U-12345", (
                f"continuation row must carry owner user_id, got {row['user_id']!r}"
            )

    def test_rollover_userless_stays_none(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "test.db")
            agent = self._make_agent(db, None)  # CLI / user-less
            self._stub_compressor(agent)

            old_sid = agent.session_id
            messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]
            agent._compress_context(messages, "sys", approx_tokens=10_000)

            assert agent.session_id != old_sid
            row = db.get_session(agent.session_id)
            assert row is not None
            assert row["user_id"] is None, (
                f"user-less session must stay NULL (no behavior change), got {row['user_id']!r}"
            )
