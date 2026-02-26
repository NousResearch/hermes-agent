"""Tests for agent.session_persister — SessionPersister extraction from AIAgent.

TDD Phase 4: Tests written BEFORE implementation to define expected behavior.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from agent.session_persister import SessionPersister


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logs_dir(tmp_path):
    """Temporary logs directory."""
    d = tmp_path / "sessions"
    d.mkdir()
    return d


@pytest.fixture
def mock_session_db():
    """MagicMock standing in for the SQLite session database."""
    db = MagicMock()
    db.append_message = MagicMock()
    db.create_session = MagicMock()
    db.end_session = MagicMock()
    db.update_system_prompt = MagicMock()
    return db


@pytest.fixture
def persister(logs_dir, mock_session_db):
    """SessionPersister with a mock DB and a temp logs_dir."""
    return SessionPersister(
        session_id="test_session_001",
        session_db=mock_session_db,
        logs_dir=logs_dir,
        model="anthropic/claude-opus-4.6",
        base_url="https://openrouter.ai/api/v1",
        platform="cli",
        session_start=datetime(2025, 6, 1, 12, 0, 0),
        save_trajectories=False,
        save_interval=1,
    )


@pytest.fixture
def persister_no_db(logs_dir):
    """SessionPersister without a session database."""
    return SessionPersister(
        session_id="test_session_no_db",
        session_db=None,
        logs_dir=logs_dir,
        model="anthropic/claude-opus-4.6",
        base_url="https://openrouter.ai/api/v1",
        save_interval=1,
    )


# ---------------------------------------------------------------------------
# 1. log_message is no-op when session_db is None
# ---------------------------------------------------------------------------


class TestLogMessageNoopWithoutDB:
    def test_log_message_noop_without_db(self, persister_no_db):
        """log_message does nothing when session_db is None -- no error raised."""
        msg = {"role": "user", "content": "hello"}
        # Should not raise
        persister_no_db.log_message(msg)


# ---------------------------------------------------------------------------
# 2. log_message calls append_message with correct args
# ---------------------------------------------------------------------------


class TestLogMessageCallsAppend:
    def test_log_message_calls_append(self, persister, mock_session_db):
        """log_message delegates to session_db.append_message with correct kwargs."""
        msg = {
            "role": "assistant",
            "content": "Sure, I can help.",
            "tool_calls": [{"name": "search", "arguments": "{}"}],
            "finish_reason": "stop",
        }
        persister.log_message(msg)
        mock_session_db.append_message.assert_called_once()
        kw = mock_session_db.append_message.call_args
        assert kw.kwargs["session_id"] == "test_session_001"
        assert kw.kwargs["role"] == "assistant"
        assert kw.kwargs["content"] == "Sure, I can help."
        assert kw.kwargs["finish_reason"] == "stop"

    def test_log_message_with_dict_tool_calls(self, persister, mock_session_db):
        """log_message handles dict-style tool_calls (from message dicts)."""
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"name": "run_code", "arguments": '{"code": "print(1)"}'}],
        }
        persister.log_message(msg)
        mock_session_db.append_message.assert_called_once()
        kw = mock_session_db.append_message.call_args
        assert kw.kwargs["tool_calls"] == msg["tool_calls"]


# ---------------------------------------------------------------------------
# 3. persist() calls save_session_log + flush_to_db
# ---------------------------------------------------------------------------


class TestPersistWritesJsonAndDB:
    def test_persist_writes_json_and_db(self, persister, mock_session_db, logs_dir):
        """persist() writes the JSON log file AND flushes to DB."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        persister.persist(messages)
        # JSON file should exist
        assert persister.session_log_file.exists()
        # DB should have been called (flush_to_db path)
        # start_idx = 0 + 1 = 1 when no conversation_history
        # So messages[1:] = [assistant msg] => 1 call
        assert mock_session_db.append_message.call_count >= 1


# ---------------------------------------------------------------------------
# 4. save_session_log overwrites the log file
# ---------------------------------------------------------------------------


class TestSaveSessionLogOverwrites:
    def test_save_session_log_overwrites(self, persister, logs_dir):
        """Session log file is created and then overwritten on second call."""
        msgs1 = [{"role": "user", "content": "first"}]
        persister.save_session_log(msgs1)
        assert persister.session_log_file.exists()
        content1 = persister.session_log_file.read_text()

        msgs2 = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        persister.save_session_log(msgs2)
        content2 = persister.session_log_file.read_text()
        # Second write should have more messages
        data2 = json.loads(content2)
        assert data2["message_count"] == 2


# ---------------------------------------------------------------------------
# 5. save_session_log metadata fields
# ---------------------------------------------------------------------------


class TestSaveSessionLogMetadata:
    def test_save_session_log_metadata(self, persister, logs_dir):
        """JSON includes session_id, model, timestamps, message_count."""
        messages = [
            {"role": "user", "content": "what time is it?"},
            {"role": "assistant", "content": "It is noon."},
        ]
        persister.save_session_log(messages)
        data = json.loads(persister.session_log_file.read_text())
        assert data["session_id"] == "test_session_001"
        assert data["model"] == "anthropic/claude-opus-4.6"
        assert data["base_url"] == "https://openrouter.ai/api/v1"
        assert data["platform"] == "cli"
        assert data["session_start"] == "2025-06-01T12:00:00"
        assert "last_updated" in data
        assert data["message_count"] == 2
        assert len(data["messages"]) == 2


# ---------------------------------------------------------------------------
# 6. flush_to_db skips already-logged messages
# ---------------------------------------------------------------------------


class TestFlushToDBSkipsLogged:
    def test_flush_to_db_skips_logged(self, persister, mock_session_db):
        """start_idx = len(conversation_history) + 1 skips pre-logged messages."""
        conversation_history = [
            {"role": "user", "content": "earlier turn 1"},
            {"role": "assistant", "content": "earlier turn 2"},
        ]
        messages = [
            {"role": "system", "content": "system prompt"},  # idx 0
            {"role": "user", "content": "earlier turn 1"},  # idx 1
            {"role": "assistant", "content": "earlier turn 2"},  # idx 2
            {"role": "user", "content": "new turn"},  # idx 3 -- should be logged
            {"role": "assistant", "content": "new reply"},  # idx 4 -- should be logged
        ]
        persister.flush_to_db(messages, conversation_history)
        # start_idx = len(conversation_history) + 1 = 2 + 1 = 3
        # messages[3:] has 2 items
        assert mock_session_db.append_message.call_count == 2
        roles = [
            c.kwargs["role"] for c in mock_session_db.append_message.call_args_list
        ]
        assert roles == ["user", "assistant"]

    def test_flush_to_db_no_conversation_history(self, persister, mock_session_db):
        """When conversation_history is None, start_idx = 0 + 1 = 1."""
        messages = [
            {"role": "system", "content": "system prompt"},  # idx 0, skipped
            {"role": "user", "content": "hello"},  # idx 1
        ]
        persister.flush_to_db(messages, conversation_history=None)
        assert mock_session_db.append_message.call_count == 1
        assert mock_session_db.append_message.call_args.kwargs["role"] == "user"


# ---------------------------------------------------------------------------
# 7. save_trajectory is no-op when disabled
# ---------------------------------------------------------------------------


class TestSaveTrajectoryNoopWhenDisabled:
    def test_save_trajectory_noop_when_disabled(self, persister):
        """save_trajectories=False means save_trajectory does nothing."""
        assert persister._save_trajectories is False
        # Should not raise, should not write anything
        persister.save_trajectory(
            messages=[{"role": "user", "content": "hi"}],
            user_query="hi",
            completed=True,
        )
        # No trajectory file should be created (we'd need to mock to verify,
        # but the key check is that it returns early without error)


# ---------------------------------------------------------------------------
# 8. session_id setter atomically updates BOTH session_id AND session_log_file
# ---------------------------------------------------------------------------


class TestSessionIdSetterUpdatesLogFile:
    def test_session_id_setter_updates_log_file(self, persister, logs_dir):
        """Setting session_id also updates session_log_file -- THE BUG FIX."""
        assert persister.session_id == "test_session_001"
        expected_original = logs_dir / "session_test_session_001.json"
        assert persister.session_log_file == expected_original

        # Update session_id
        persister.session_id = "new_session_002"
        assert persister.session_id == "new_session_002"
        expected_new = logs_dir / "session_new_session_002.json"
        assert persister.session_log_file == expected_new

    def test_session_id_setter_write_uses_new_path(self, persister, logs_dir):
        """After session_id change, save_session_log writes to the new file."""
        persister.session_id = "renamed_session"
        msgs = [{"role": "user", "content": "post-rename"}]
        persister.save_session_log(msgs)

        new_file = logs_dir / "session_renamed_session.json"
        assert new_file.exists()
        old_file = logs_dir / "session_test_session_001.json"
        assert not old_file.exists()


# ---------------------------------------------------------------------------
# 9. create_compression_session ends old and creates new
# ---------------------------------------------------------------------------


class TestCreateCompressionSession:
    def test_create_compression_session(self, persister, mock_session_db):
        """Ends old session, creates a new one, returns new session ID."""
        old_id = persister.session_id
        new_id = persister.create_compression_session()
        # Old session should be ended
        mock_session_db.end_session.assert_called_once_with(old_id, "compression")
        # New session should be created
        mock_session_db.create_session.assert_called_once()
        create_kw = mock_session_db.create_session.call_args.kwargs
        assert create_kw["session_id"] == new_id
        assert create_kw["source"] == "cli"
        assert create_kw["model"] == "anthropic/claude-opus-4.6"
        # Return value is the new ID
        assert new_id != old_id
        assert isinstance(new_id, str)
        # Format: YYYYMMDD_HHMMSS_<6hex>
        assert re.match(r"\d{8}_\d{6}_[0-9a-f]{6}", new_id)


# ---------------------------------------------------------------------------
# 10. create_compression_session updates session_id property
# ---------------------------------------------------------------------------


class TestCreateCompressionSessionUpdatesId:
    def test_create_compression_session_updates_session_id(
        self, persister, mock_session_db, logs_dir
    ):
        """session_id and session_log_file reflect the new ID after compression."""
        old_id = persister.session_id
        new_id = persister.create_compression_session()
        assert persister.session_id == new_id
        assert persister.session_log_file == logs_dir / f"session_{new_id}.json"
        assert persister.session_id != old_id

    def test_create_compression_session_without_db(self, persister_no_db):
        """create_compression_session still updates session_id even without DB."""
        old_id = persister_no_db.session_id
        new_id = persister_no_db.create_compression_session()
        assert new_id != old_id
        assert persister_no_db.session_id == new_id


# ---------------------------------------------------------------------------
# 11. maybe_save_session_log respects save_interval
# ---------------------------------------------------------------------------


class TestMaybeSaveSessionLogInterval:
    def test_maybe_save_session_log_interval(self, logs_dir):
        """With save_interval=3, only writes every 3rd call."""
        p = SessionPersister(
            session_id="interval_test",
            session_db=None,
            logs_dir=logs_dir,
            model="test-model",
            base_url="http://localhost",
            save_interval=3,
        )
        msgs = [{"role": "user", "content": "tick"}]

        # Calls 1 and 2 should NOT write
        p.maybe_save_session_log(msgs)
        assert not p.session_log_file.exists()
        p.maybe_save_session_log(msgs)
        assert not p.session_log_file.exists()

        # Call 3 should write
        p.maybe_save_session_log(msgs)
        assert p.session_log_file.exists()
        data = json.loads(p.session_log_file.read_text())
        assert data["message_count"] == 1

        # Delete the file to test the next cycle
        p.session_log_file.unlink()

        # Calls 4 and 5 should NOT write
        p.maybe_save_session_log(msgs)
        assert not p.session_log_file.exists()
        p.maybe_save_session_log(msgs)
        assert not p.session_log_file.exists()

        # Call 6 should write again
        p.maybe_save_session_log(msgs)
        assert p.session_log_file.exists()


# ---------------------------------------------------------------------------
# 12. _clean_session_content converts scratchpad tags
# ---------------------------------------------------------------------------


class TestCleanSessionContent:
    def test_content_scratchpad_to_think_conversion(self):
        """<REASONING_SCRATCHPAD> tags are converted to <think> tags."""
        content = "Some text\n\n\n<REASONING_SCRATCHPAD>internal thoughts</REASONING_SCRATCHPAD>\n\n\nMore text"
        result = SessionPersister._clean_session_content(content)
        assert "<REASONING_SCRATCHPAD>" not in result
        assert "<think>" in result
        assert "</think>" in result
        assert "internal thoughts" in result

    def test_clean_session_content_none(self):
        """None content is returned as-is."""
        assert SessionPersister._clean_session_content(None) is None

    def test_clean_session_content_empty(self):
        """Empty string is returned as-is."""
        assert SessionPersister._clean_session_content("") == ""

    def test_clean_session_content_strips_extra_newlines(self):
        """Extra newlines around think blocks are collapsed."""
        content = "Hello\n\n\n<think>reasoning</think>\n\n\nWorld"
        result = SessionPersister._clean_session_content(content)
        # Should not have triple newlines around think blocks
        assert "\n\n\n<think>" not in result
        assert "</think>\n\n\n" not in result

    def test_save_session_log_cleans_assistant_content(self, persister, logs_dir):
        """save_session_log applies _clean_session_content to assistant messages."""
        messages = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "<REASONING_SCRATCHPAD>thinking</REASONING_SCRATCHPAD>answer",
            },
        ]
        persister.save_session_log(messages)
        data = json.loads(persister.session_log_file.read_text())
        assistant_msg = data["messages"][1]
        assert "<REASONING_SCRATCHPAD>" not in assistant_msg["content"]
        assert "<think>" in assistant_msg["content"]
