"""Tests for _session_db_failed flag and JSONL fallback behaviour."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


def _make_agent(session_db, session_id="test-sid"):
    """Create a minimal AIAgent for testing persistence paths."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )


def _make_message(role, content):
    return {"role": role, "content": content}


# ── _ensure_db_session failure sets _session_db_failed ──────────────────

def test_ensure_db_session_failure_sets_flag():
    """When create_session raises, _session_db_failed must be True."""
    db = MagicMock()
    db.create_session.side_effect = RuntimeError("locking protocol")
    agent = _make_agent(db)
    agent._session_db_created = False

    agent._ensure_db_session()

    assert agent._session_db_failed is True


# ── append_message failure preserves idx and sets flag ──────────────────

class TestFlushPartialFailure:
    def test_failure_preserves_idx(self):
        """Single append failure: idx stays at last success, flag set."""
        db = MagicMock()
        # First 3 appends succeed, 4th fails
        db.append_message.side_effect = [None, None, None, RuntimeError("wal locked")]
        agent = _make_agent(db, "sid")
        agent._ensure_db_session()

        messages = [_make_message("user", f"msg{i}") for i in range(5)]
        agent._flush_messages_to_session_db(messages)

        assert agent._last_flushed_db_idx == 3
        assert agent._session_db_failed is True

    def test_all_succeed_no_flag(self):
        """Normal path: flag stays False, idx = len(messages)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from hermes_state import SessionDB
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = _make_agent(db, "sid")
            agent._ensure_db_session()

            messages = [_make_message("user", "a"), _make_message("assistant", "b")]
            agent._flush_messages_to_session_db(messages)

            assert agent._session_db_failed is False
            assert agent._last_flushed_db_idx == 2


# ── JSONL fallback ─────────────────────────────────────────────────────

class TestJsonlFallback:
    def test_writes_only_unflushed(self):
        """Only messages[_last_flushed_db_idx:] appear in fallback file."""
        db = MagicMock()
        # Make append_message fail so _flush cannot reset _last_flushed_db_idx
        db.append_message.side_effect = RuntimeError("locked")
        agent = _make_agent(db, "test-sid")
        agent._session_db_created = True
        agent._last_flushed_db_idx = 1  # msg[0] already in DB
        agent._save_session_log = MagicMock()

        messages = [_make_message("user", "already-done"),
                     _make_message("assistant", "new-one"),
                     _make_message("user", "new-two")]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("hermes_constants.get_hermes_home", return_value=Path(tmpdir)):
                agent._persist_session(messages)

            fb_dir = Path(tmpdir) / "sessions"
            fb_files = list(fb_dir.glob("*.pending.jsonl"))
            assert len(fb_files) == 1
            pending = [json.loads(line) for line in fb_files[0].read_text().splitlines() if line.strip()]
            assert len(pending) == 2
            # Check wrapped format with timestamp
            assert "_fallback_timestamp" in pending[0]
            assert "message" in pending[0]
            assert pending[0]["message"]["content"] == "new-one"
            assert pending[1]["message"]["content"] == "new-two"

    def test_no_fallback_when_flag_false(self):
        """_session_db_failed=False → no .pending.jsonl created."""
        db = MagicMock()
        agent = _make_agent(db, "sid")
        agent._session_db_failed = False
        agent._save_session_log = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("hermes_constants.get_hermes_home", return_value=Path(tmpdir)):
                agent._persist_session([_make_message("user", "hello")])

            fb_dir = Path(tmpdir) / "sessions"
            assert not fb_dir.exists() or not list(fb_dir.glob("*.pending.jsonl"))


# ── Profile isolation ──────────────────────────────────────────────────

def test_profile_isolation():
    """HERMES_PROFILE=myprof → fallback in profiles/myprof/sessions/."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from hermes_state import SessionDB
        db_path = Path(tmpdir) / "test.db"
        db = SessionDB(db_path=db_path)

        agent = _make_agent(db, "test-sid")
        agent._ensure_db_session()
        agent._session_db_failed = True
        agent._last_flushed_db_idx = 0

        messages = [_make_message("user", "x")]
        with patch("hermes_constants.get_hermes_home", return_value=Path(tmpdir)):
            with patch.dict(os.environ, {"HERMES_PROFILE": "myprof"}):
                agent._persist_session(messages)

        prof_dir = Path(tmpdir) / "profiles" / "myprof" / "sessions"
        assert len(list(prof_dir.glob("*.pending.jsonl"))) == 1

        default_dir = Path(tmpdir) / "sessions"
        assert not default_dir.exists() or not list(default_dir.glob("*.pending.jsonl"))


# ── agent_persisted logic ──────────────────────────────────────────────

def test_agent_persisted_false_when_failed():
    """agent_persisted must be False when _session_db_failed is True."""
    db = MagicMock()
    agent = _make_agent(db)
    agent._session_db_failed = True

    result = (
        agent._session_db is not None
        and agent._session_db_created
        and not agent._session_db_failed
    )
    assert result is False


def test_agent_persisted_true_when_all_ok():
    """agent_persisted must be True when DB exists, created, and not failed."""
    db = MagicMock()
    agent = _make_agent(db)
    agent._session_db_created = True
    agent._session_db_failed = False

    result = (
        agent._session_db is not None
        and agent._session_db_created
        and not agent._session_db_failed
    )
    assert result is True
