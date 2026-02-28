"""Tests for the /retry, /undo, and /reset gateway command fixes (issue #210)."""

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestSessionDBTruncateAfter(unittest.TestCase):
    """Tests for SessionDB.truncate_after()"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_state.db"

        from hermes_state import SessionDB
        self.db = SessionDB(db_path=self.db_path)
        self.session_id = "test_session_001"
        self.db.create_session(self.session_id, source="test")

    def tearDown(self):
        self.db.close()
        try:
            os.unlink(self.db_path)
        except Exception:
            pass

    def _add_messages(self, count: int):
        for i in range(count):
            role = "user" if i % 2 == 0 else "assistant"
            self.db.append_message(self.session_id, role, f"msg_{i}")

    def test_truncate_keeps_first_n(self):
        self._add_messages(6)
        deleted = self.db.truncate_after(self.session_id, keep_count=3)
        self.assertEqual(deleted, 3)
        msgs = self.db.get_messages_as_conversation(self.session_id)
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["content"], "msg_0")
        self.assertEqual(msgs[2]["content"], "msg_2")

    def test_truncate_updates_message_count(self):
        self._add_messages(5)
        self.db.truncate_after(self.session_id, keep_count=2)
        session = self.db.get_session(self.session_id)
        self.assertEqual(session["message_count"], 2)

    def test_truncate_noop_when_within_limit(self):
        self._add_messages(3)
        deleted = self.db.truncate_after(self.session_id, keep_count=10)
        self.assertEqual(deleted, 0)
        msgs = self.db.get_messages_as_conversation(self.session_id)
        self.assertEqual(len(msgs), 3)

    def test_truncate_zero_removes_all(self):
        self._add_messages(4)
        deleted = self.db.truncate_after(self.session_id, keep_count=0)
        self.assertEqual(deleted, 4)
        msgs = self.db.get_messages_as_conversation(self.session_id)
        self.assertEqual(len(msgs), 0)


class TestSessionStoreRewriteTranscript(unittest.TestCase):
    """Tests for SessionStore.rewrite_transcript()"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sessions_dir = Path(self.tmpdir) / "sessions"
        self.sessions_dir.mkdir()
        self.session_id = "test_rewrite_001"

        # Create a mock config
        mock_config = MagicMock()
        mock_config.get_reset_policy.return_value = MagicMock(mode="none")

        from gateway.session import SessionStore
        self.store = SessionStore(self.sessions_dir, mock_config)
        # Disable SQLite for these tests to focus on JSONL
        self.store._db = None

    def _write_jsonl(self, messages):
        path = self.store.get_transcript_path(self.session_id)
        with open(path, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

    def test_rewrite_truncates_jsonl(self):
        original = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "retry this"},
            {"role": "assistant", "content": "old response"},
        ]
        self._write_jsonl(original)
        truncated = original[:2]
        self.store.rewrite_transcript(self.session_id, truncated)

        loaded = self.store.load_transcript(self.session_id)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["content"], "hello")
        self.assertEqual(loaded[1]["content"], "hi")

    def test_rewrite_atomic_no_partial(self):
        """After rewrite, no .tmp file should remain."""
        original = [{"role": "user", "content": "test"}]
        self._write_jsonl(original)
        self.store.rewrite_transcript(self.session_id, [])

        tmp_path = self.store.get_transcript_path(self.session_id).with_suffix(".tmp")
        self.assertFalse(tmp_path.exists())

    def test_rewrite_noop_when_no_file(self):
        """rewrite_transcript should not crash when JSONL file doesn't exist."""
        self.store.rewrite_transcript(self.session_id, [{"role": "user", "content": "x"}])
        # No JSONL was created (file didn't exist, nothing to rewrite)
        path = self.store.get_transcript_path(self.session_id)
        self.assertFalse(path.exists())


class TestResetEntryLookup(unittest.TestCase):
    """Test that /reset uses _entries (not _sessions) and canonical key generation."""

    def test_reset_uses_entries_attribute(self):
        """Confirm _handle_reset_command accesses _entries, not _sessions."""
        import gateway.run as run_module
        import inspect

        source = inspect.getsource(run_module.GatewayRunner._handle_reset_command)
        self.assertNotIn("._sessions", source,
                         "Bug 2: /reset should use ._entries, not ._sessions")
        self.assertIn("._entries", source)

    def test_reset_uses_generate_session_key(self):
        """Confirm /reset uses _generate_session_key instead of manual construction."""
        import gateway.run as run_module
        import inspect

        source = inspect.getsource(run_module.GatewayRunner._handle_reset_command)
        self.assertIn("_generate_session_key", source,
                      "/reset should use _generate_session_key for correct WhatsApp DM keys")


class TestRetryUndoUseRewriteTranscript(unittest.TestCase):
    """Test that /retry and /undo call rewrite_transcript instead of setting dangling attrs."""

    def test_retry_does_not_set_conversation_history(self):
        import gateway.run as run_module
        import inspect

        source = inspect.getsource(run_module.GatewayRunner._handle_retry_command)
        self.assertNotIn("conversation_history", source,
                         "Bug 1: /retry should not assign to session_entry.conversation_history")
        self.assertIn("rewrite_transcript", source)

    def test_undo_does_not_set_conversation_history(self):
        import gateway.run as run_module
        import inspect

        source = inspect.getsource(run_module.GatewayRunner._handle_undo_command)
        self.assertNotIn("conversation_history", source,
                         "Bug 1: /undo should not assign to session_entry.conversation_history")
        self.assertIn("rewrite_transcript", source)


class TestTruncateAfterWithDB(unittest.TestCase):
    """Integration test: rewrite_transcript with SQLite backend."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_integration.db"
        self.sessions_dir = Path(self.tmpdir) / "sessions"
        self.sessions_dir.mkdir()
        self.session_id = "integration_001"

        from hermes_state import SessionDB
        self.db = SessionDB(db_path=self.db_path)
        self.db.create_session(self.session_id, source="test")

        mock_config = MagicMock()
        mock_config.get_reset_policy.return_value = MagicMock(mode="none")

        from gateway.session import SessionStore
        self.store = SessionStore(self.sessions_dir, mock_config)
        self.store._db = self.db

    def tearDown(self):
        self.db.close()

    def test_rewrite_transcript_truncates_db(self):
        """rewrite_transcript should truncate SQLite messages via truncate_after."""
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            self.db.append_message(self.session_id, role, f"msg_{i}")

        # Keep first 2 messages
        keep = [{"role": "user", "content": "msg_0"},
                {"role": "assistant", "content": "msg_1"}]
        self.store.rewrite_transcript(self.session_id, keep)

        msgs = self.db.get_messages_as_conversation(self.session_id)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["content"], "msg_0")
        self.assertEqual(msgs[1]["content"], "msg_1")

        session = self.db.get_session(self.session_id)
        self.assertEqual(session["message_count"], 2)


if __name__ == "__main__":
    unittest.main()
