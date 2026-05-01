"""Tests for issue #8038 — Silent message persistence failure.

Verifies that:
1. When _flush_messages_to_session_db fails, _flush_failed is set to True
2. The flush failure is logged at ERROR level (not WARNING)
3. _persist_session emits a user-visible warning when the flag is set
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestFlushFailureFlag:
    """Verify _flush_failed flag behavior on DB write errors."""

    def _make_agent(self, session_db=None):
        """Create a minimal AIAgent with an optional session DB."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="test-session-8038",
                skip_context_files=True,
                skip_memory=True,
            )
        return agent

    def test_flush_failed_flag_initially_false(self):
        """Agent starts with _flush_failed = False."""
        agent = self._make_agent()
        assert agent._flush_failed is False

    def test_flush_failed_set_on_exception(self):
        """When flush raises, _flush_failed is set to True."""
        mock_db = MagicMock()
        mock_db.ensure_session = MagicMock(side_effect=RuntimeError("DB locked"))

        agent = self._make_agent(session_db=mock_db)
        assert agent._flush_failed is False

        messages = [{"role": "user", "content": "hello"}]
        agent._flush_messages_to_session_db(messages, [])

        assert agent._flush_failed is True

    def test_flush_failure_logged_at_error_level(self, caplog):
        """Flush failure is logged at ERROR, not WARNING."""
        mock_db = MagicMock()
        mock_db.ensure_session = MagicMock(side_effect=RuntimeError("disk full"))

        agent = self._make_agent(session_db=mock_db)

        messages = [{"role": "user", "content": "hello"}]
        with caplog.at_level(logging.DEBUG, logger="run_agent"):
            agent._flush_messages_to_session_db(messages, [])

        # Should have an ERROR record, not WARNING
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1, (
            f"Expected at least one ERROR log, got levels: "
            f"{[r.levelname for r in caplog.records]}"
        )
        assert "Session DB append_message failed" in error_records[0].message

        # Should NOT have a WARNING for the same message
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "Session DB append_message failed" in r.message
        ]
        assert len(warning_records) == 0, (
            "Flush failure should be logged at ERROR, not WARNING"
        )

    def test_persist_session_warns_on_flush_failure(self, caplog):
        """_persist_session logs a user-visible warning when _flush_failed is set."""
        mock_db = MagicMock()
        mock_db.ensure_session = MagicMock(side_effect=RuntimeError("DB locked"))

        agent = self._make_agent(session_db=mock_db)
        agent._save_session_log = MagicMock()  # stub file I/O

        messages = [{"role": "user", "content": "test"}]
        with caplog.at_level(logging.DEBUG, logger="run_agent"):
            agent._persist_session(messages, [])

        # The user-visible warning should be present
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "some messages may not have been saved" in r.message
        ]
        assert len(warning_records) >= 1, (
            f"Expected user-visible WARNING about unsaved messages, got: "
            f"{[(r.levelname, r.message) for r in caplog.records]}"
        )

    def test_persist_session_no_warning_on_success(self, caplog):
        """_persist_session does NOT warn when flush succeeds."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(session_db=db)
            agent._save_session_log = MagicMock()

            messages = [{"role": "user", "content": "test"}]
            with caplog.at_level(logging.DEBUG, logger="run_agent"):
                agent._persist_session(messages, [])

            warning_records = [
                r for r in caplog.records
                if r.levelno == logging.WARNING
                and "some messages may not have been saved" in r.message
            ]
            assert len(warning_records) == 0, (
                "No warning should be emitted when flush succeeds"
            )
            assert agent._flush_failed is False

    def test_flush_failed_flag_set_on_append_message_error(self):
        """Flag is set even when ensure_session succeeds but append_message fails."""
        mock_db = MagicMock()
        mock_db.ensure_session = MagicMock()  # succeeds
        mock_db.append_message = MagicMock(side_effect=RuntimeError("write failed"))

        agent = self._make_agent(session_db=mock_db)

        messages = [{"role": "user", "content": "hello"}]
        agent._flush_messages_to_session_db(messages, [])

        assert agent._flush_failed is True
