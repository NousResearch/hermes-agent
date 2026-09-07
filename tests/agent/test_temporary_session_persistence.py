"""Behavioral persistence policy tests for temporary chats vs normal control sessions.

Verifies:
1. Zero session/message/accounting/title rows in SessionDB for temporary sessions.
2. Normal control sessions persist rows, tokens, and titles as expected.
3. Compression rotation on temporary agents is purely in-memory without child publication.
4. Ephemeral registry reference counting and lifecycle ordering on agent close.
"""

import pytest
from unittest.mock import MagicMock, patch
from hermes_state import SessionDB
from agent.session_policy import (
    mark_session_ephemeral,
    unmark_session_ephemeral,
    is_session_ephemeral,
)
from agent.title_generator import apply_instant_title, maybe_auto_title


def test_temporary_session_db_zero_rows(tmp_path):
    """Temporary session ID must have zero rows across all SessionDB tables."""
    db = SessionDB(db_path=tmp_path / "state.db")
    temp_sid = "temp-session-test-01"
    norm_sid = "normal-session-test-01"

    try:
        mark_session_ephemeral(temp_sid)
        assert is_session_ephemeral(temp_sid)

        # 1. create_session / _insert_session_row
        db.create_session(temp_sid, "cli", model="test-model")
        db.create_session(norm_sid, "cli", model="test-model")

        with db._read_ctx() as conn:
            temp_rows = conn.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", (temp_sid,)).fetchone()[0]
            norm_rows = conn.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", (norm_sid,)).fetchone()[0]
        assert temp_rows == 0, "Temporary session created a row in sessions table"
        assert norm_rows == 1, "Normal control session failed to persist in sessions table"

        # 2. update_token_counts / usage accounting
        db.update_token_counts(temp_sid, input_tokens=100, output_tokens=50, model="test-model")
        db.update_token_counts(norm_sid, input_tokens=100, output_tokens=50, model="test-model")

        with db._read_ctx() as conn:
            temp_rows = conn.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", (temp_sid,)).fetchone()[0]
            temp_usage = conn.execute("SELECT COUNT(*) FROM session_model_usage WHERE session_id = ?", (temp_sid,)).fetchone()[0]
            norm_usage = conn.execute("SELECT COUNT(*) FROM session_model_usage WHERE session_id = ?", (norm_sid,)).fetchone()[0]
        assert temp_rows == 0, "Token accounting resurrected a session row for temporary chat"
        assert temp_usage == 0, "Token accounting recorded usage row for temporary chat"
        assert norm_usage == 1, "Normal control failed to record usage row"

        # 3. title generation
        persisted_temp = apply_instant_title(db, temp_sid, "Fix login button on mobile")
        persisted_norm = apply_instant_title(db, norm_sid, "Fix login button on mobile")
        assert persisted_temp is None, "Instant title persisted for temporary chat"
        assert persisted_norm is not None, "Instant title failed for normal session"

        # 4. publish_compression_child refuses temporary session
        with pytest.raises(RuntimeError, match="temporary"):
            db.publish_compression_child(
                parent_session_id=temp_sid,
                child_session_id="temp-child-sid",
                source="cli",
                messages=[{"role": "user", "content": "summary"}],
                require_compression_lease=False,
            )

        with db._read_ctx() as conn:
            total_temp = conn.execute("SELECT COUNT(*) FROM sessions WHERE id IN (?, ?)", (temp_sid, "temp-child-sid")).fetchone()[0]
        assert total_temp == 0, "publish_compression_child created durable rows for temporary chat"

    finally:
        unmark_session_ephemeral(temp_sid, force=True)
        db.close()


def test_ephemeral_registry_ref_counting():
    """Multiple agents sharing a session id maintain the ephemeral registration until all close."""
    sid = "shared-ephemeral-sid"
    unmark_session_ephemeral(sid, force=True)

    mark_session_ephemeral(sid)
    assert is_session_ephemeral(sid)

    mark_session_ephemeral(sid)
    assert is_session_ephemeral(sid)

    # First unmark (e.g. child or first agent closes)
    unmark_session_ephemeral(sid)
    assert is_session_ephemeral(sid), "Session policy unmarked too early while another owner is active"

    # Second unmark (last agent closes)
    unmark_session_ephemeral(sid)
    assert not is_session_ephemeral(sid), "Session policy failed to unmark when all owners closed"


def test_aiagent_ephemeral_constructor_and_close():
    """AIAgent with ephemeral=True registers session_id and unregisters on close."""
    from run_agent import AIAgent

    agent = AIAgent(
        model="test-model",
        api_key="mock-key",
        base_url="http://127.0.0.1:9999/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        ephemeral=True,
    )
    sid = agent.session_id
    assert sid, "Agent must have session_id"
    assert agent.ephemeral is True
    assert agent._persist_disabled is True
    assert agent._session_json_enabled is False
    assert is_session_ephemeral(sid), "AIAgent failed to register session_id in ephemeral registry"

    agent.close()
    assert not is_session_ephemeral(sid), "AIAgent failed to unmark session_id on close()"
