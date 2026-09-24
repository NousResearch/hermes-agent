"""SQLite oracle for the future PostgreSQL atomic rotation contract.

This deliberately drives AIAgent's real rotation path with a deterministic
provider-shaped compressor and a real temporary SessionDB.  It does not patch
the publisher and therefore cannot turn a missing publication into a pass.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from agent.context_compressor import _DB_PERSISTED_MARKER
from hermes_state import SessionDB


class DeterministicCompressionProvider:
    """Offline provider seam with the state exposed by the real compressor."""

    compression_count = 1
    last_prompt_tokens = 17
    last_completion_tokens = 5
    _last_summary_error = None
    _last_compress_aborted = False
    _last_summary_auth_failure = False
    _last_aux_model_failure_model = None
    _last_aux_model_failure_error = None

    def compress(self, messages, **_kwargs):
        # Return new objects: publication, rather than fake-provider mutation,
        # must apply persistence markers and preserve the original list.
        assert messages
        return [
            {"role": "assistant", "content": "[CONTEXT COMPACTION] deterministic summary"},
            {"role": "user", "content": "deterministic live tail"},
        ]


def _agent(db: SessionDB, session_id: str):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.invalid/v1",
            model="oracle/model", platform="telegram", quiet_mode=True,
            session_db=db, session_id=session_id, skip_context_files=True,
            skip_memory=True,
        )
    agent.context_compressor = DeterministicCompressionProvider()
    agent.compression_in_place = False
    return agent


def test_sqlite_rotation_oracle_preserves_publication_and_metadata(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    parent = "sqlite-oracle-parent"
    try:
        db.create_session(
            parent, source="telegram", user_id="oracle-user",
            session_key="telegram:oracle-user:chat", chat_id="chat",
            chat_type="private", model="oracle/model",
            model_config={"provider": "deterministic"},
            system_prompt="oracle prompt", profile_name="oracle-profile",
        )
        db.update_session_cwd(parent, "/oracle/repo", git_branch="main", git_repo_root="/oracle/repo")
        db.set_session_title(parent, "Oracle title")
        db.append_message(parent, "user", "persisted parent question")
        db.append_message(parent, "assistant", "persisted parent answer")
        db.touch_session_activity(parent, 1234.0, description="oracle activity", provenance="tool")
        db.record_compression_failure_cooldown(parent, 2345.0, "oracle cooldown")
        db.set_compression_fallback_streak(parent, 3)
        db.set_compression_ineffective_count(parent, 2)
        db.set_compression_recovery_deadline(parent, 3456.0)

        agent = _agent(db, parent)
        original = [
            *db.get_messages_as_conversation(parent),
            {"role": "user", "content": "deterministic live tail"},
        ]
        agent._persist_user_message_idx = len(original) - 1
        returned, _prompt = agent._compress_context(original, "oracle prompt", approx_tokens=120_000)
        child = agent.session_id

        assert child != parent
        parent_row, child_row = db.get_session(parent), db.get_session(child)
        assert parent_row is not None and child_row is not None
        assert parent_row["end_reason"] == "compression" and parent_row["ended_at"] is not None
        assert child_row["parent_session_id"] == parent
        for key, expected in {
            "source": "telegram", "user_id": "oracle-user",
            "session_key": "telegram:oracle-user:chat", "chat_id": "chat",
            "chat_type": "private", "model": "oracle/model",
            "profile_name": "oracle-profile", "cwd": "/oracle/repo",
            "git_repo_root": "/oracle/repo",
            "git_branch": "main", "title": "Oracle title",
        }.items():
            assert child_row[key] == expected
        child_model_config = child_row["model_config"]
        assert isinstance(child_model_config, str)
        assert json.loads(child_model_config) == {
            "max_iterations": 9223372036854775807,
            "reasoning_config": None,
            "max_tokens": None,
        }
        # AIAgent owns the live prompt snapshot; rotation must persist that exact
        # cached value rather than the stale create_session seed.
        assert child_row["system_prompt"] == agent._cached_system_prompt
        # A successful real compression resets anti-thrash state on the new
        # generation; stale cooldown/counter state must not poison the child.
        assert db.get_compression_fallback_streak(child) == 0
        assert db.get_compression_ineffective_count(child) == 0
        assert db.get_compression_recovery_deadline(child) == 0.0
        assert db.get_compression_failure_cooldown(child) is None
        assert db.get_compression_tip(parent) == child

        rows = db.get_messages_as_conversation(child, include_inactive=True)
        assert [(row["role"], row["content"]) for row in rows] == [
            ("assistant", "[CONTEXT COMPACTION] deterministic summary"),
            ("user", "deterministic live tail"),
        ]
        assert all(row.get(_DB_PERSISTED_MARKER) for row in returned)
        assert original[-1]["content"] == "deterministic live tail"
    finally:
        db.close()
