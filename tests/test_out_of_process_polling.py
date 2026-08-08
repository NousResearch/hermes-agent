"""Tests for out-of-process message polling.

Covers:
1. SessionDB.get_messages_after — the core query used by the poller
2. HermesCLI._poll_and_display_out_of_process_messages — render + history update
3. _last_seen_msg_id initialization on fresh and resumed sessions
"""

import pathlib
import threading

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: pathlib.Path):
    from hermes_state import SessionDB
    return SessionDB(db_path=tmp_path / "state.db")


# ---------------------------------------------------------------------------
# 1. SessionDB.get_messages_after
# ---------------------------------------------------------------------------

class TestGetMessagesAfter:
    def test_returns_empty_when_no_new_messages(self, tmp_path):
        db = _make_db(tmp_path)
        db.create_session("s1", source="cli", model="m")
        id1 = db.append_message("s1", role="user", content="hello")
        result = db.get_messages_after("s1", id1)
        assert result == []

    def test_returns_messages_after_watermark(self, tmp_path):
        db = _make_db(tmp_path)
        db.create_session("s1", source="cli", model="m")
        id1 = db.append_message("s1", role="user", content="hello")
        id2 = db.append_message("s1", role="assistant", content="world")
        id3 = db.append_message("s1", role="user", content="again")

        result = db.get_messages_after("s1", id1)
        assert len(result) == 2
        assert result[0]["id"] == id2
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "world"
        assert result[1]["id"] == id3
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "again"

    def test_zero_watermark_returns_all(self, tmp_path):
        db = _make_db(tmp_path)
        db.create_session("s1", source="cli", model="m")
        db.append_message("s1", role="user", content="a")
        db.append_message("s1", role="assistant", content="b")

        result = db.get_messages_after("s1", 0)
        assert len(result) == 2

    def test_wrong_session_returns_empty(self, tmp_path):
        db = _make_db(tmp_path)
        db.create_session("s1", source="cli", model="m")
        db.append_message("s1", role="user", content="hello")

        result = db.get_messages_after("other-session", 0)
        assert result == []

    def test_ordering_is_ascending_by_id(self, tmp_path):
        db = _make_db(tmp_path)
        db.create_session("s1", source="cli", model="m")
        ids = [db.append_message("s1", role="user", content=f"msg{i}") for i in range(5)]

        result = db.get_messages_after("s1", 0)
        returned_ids = [r["id"] for r in result]
        assert returned_ids == sorted(returned_ids)
        assert returned_ids == ids

    def test_decodes_json_encoded_content(self, tmp_path):
        db = _make_db(tmp_path)
        db.create_session("s1", source="cli", model="m")
        id1 = db.append_message("s1", role="user", content=[{"type": "text", "text": "hi"}])
        id2 = db.append_message("s1", role="user", content="plain")

        result = db.get_messages_after("s1", 0)
        assert len(result) == 2
        # Multimodal content is decoded back to a list
        assert isinstance(result[0]["content"], list)
        assert result[1]["content"] == "plain"

    def test_thread_safe_concurrent_reads(self, tmp_path):
        """Two threads calling get_messages_after simultaneously must not raise."""
        db = _make_db(tmp_path)
        db.create_session("s1", source="cli", model="m")
        for i in range(20):
            db.append_message("s1", role="user", content=f"msg{i}")

        errors = []

        def _read():
            try:
                db.get_messages_after("s1", 0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_read) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# 2. _poll_and_display_out_of_process_messages
# ---------------------------------------------------------------------------

class TestPollAndDisplayOutOfProcessMessages:
    """Unit tests that mock the CLI plumbing so we only test the poller logic."""

    def _make_cli(self, tmp_path):
        """Return a minimal HermesCLI-like object with just the attributes the
        poller reads.  We don't instantiate the real class because that pulls
        in prompt_toolkit, Rich, and the agent — just duck-type the minimum."""
        from hermes_state import SessionDB
        from unittest.mock import MagicMock

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess", source="cli", model="m")

        class FakeCLI:
            _session_db = db
            session_id = "sess"
            _last_seen_msg_id = 0
            _last_oop_poll = 0.0   # epoch 0 → always past the throttle window
            conversation_history = []
            final_response_markdown = "strip"

            # Methods the poller calls — will be recorded
            _print_user_message_preview = MagicMock()
            _render_panel = MagicMock()
            _invalidate = MagicMock()

            def _scrollback_box_width(self):
                return 80

        # Bind the real method to our fake object
        from cli import HermesCLI
        fake = FakeCLI()
        fake._poll_and_display_out_of_process_messages = (
            HermesCLI._poll_and_display_out_of_process_messages.__get__(fake)
        )
        return fake, db

    def test_no_op_when_no_new_messages(self, tmp_path):
        fake, db = self._make_cli(tmp_path)
        db.append_message("sess", role="user", content="old")
        fake._last_seen_msg_id = db._conn.execute(
            "SELECT MAX(id) FROM messages WHERE session_id='sess'"
        ).fetchone()[0]

        fake._poll_and_display_out_of_process_messages()

        fake._print_user_message_preview.assert_not_called()
        assert fake.conversation_history == []

    def test_user_message_rendered_and_appended(self, tmp_path):
        fake, db = self._make_cli(tmp_path)
        id1 = db.append_message("sess", role="user", content="hello from outside")

        fake._poll_and_display_out_of_process_messages()

        fake._print_user_message_preview.assert_called_once_with("hello from outside")
        assert len(fake.conversation_history) == 1
        assert fake.conversation_history[0] == {"role": "user", "content": "hello from outside"}
        assert fake._last_seen_msg_id == id1

    def test_watermark_advances(self, tmp_path):
        fake, db = self._make_cli(tmp_path)
        id1 = db.append_message("sess", role="user", content="first")
        id2 = db.append_message("sess", role="assistant", content="second")

        fake._poll_and_display_out_of_process_messages()
        assert fake._last_seen_msg_id == id2

        # No new messages — calling again is a no-op
        pre_history_len = len(fake.conversation_history)
        fake._poll_and_display_out_of_process_messages()
        assert len(fake.conversation_history) == pre_history_len

    def test_throttle_suppresses_db_read(self, tmp_path):
        """Poller must not hit SQLite within OOP_POLL_INTERVAL of the last read."""
        import time
        fake, db = self._make_cli(tmp_path)
        db.append_message("sess", role="user", content="msg")

        # First call — runs and renders
        fake._poll_and_display_out_of_process_messages()
        assert len(fake.conversation_history) == 1

        # Second message written immediately after
        db.append_message("sess", role="user", content="msg2")

        # Second call within the throttle window — must be suppressed
        fake._poll_and_display_out_of_process_messages()
        assert len(fake.conversation_history) == 1  # still 1, throttled

        # Expire the throttle by winding back _last_oop_poll
        fake._last_oop_poll = 0.0
        fake._poll_and_display_out_of_process_messages()
        assert len(fake.conversation_history) == 2  # now sees msg2

    def test_tool_message_appended_silently(self, tmp_path):
        fake, db = self._make_cli(tmp_path)
        db.append_message("sess", role="tool", content="tool output")

        fake._poll_and_display_out_of_process_messages()

        # Not rendered visually
        fake._print_user_message_preview.assert_not_called()
        # But added to history
        assert len(fake.conversation_history) == 1
        assert fake.conversation_history[0]["role"] == "tool"

    def test_no_session_db_is_noop(self, tmp_path):
        """When _session_db is None the poller must not crash."""
        from cli import HermesCLI
        from unittest.mock import MagicMock

        class FakeCLI:
            _session_db = None
            session_id = "sess"
            _last_seen_msg_id = 0
            _last_oop_poll = 0.0   # epoch 0 → always past the throttle window
            conversation_history = []
            final_response_markdown = "strip"
            _print_user_message_preview = MagicMock()
            _invalidate = MagicMock()

        fake = FakeCLI()
        fake._poll_and_display_out_of_process_messages = (
            HermesCLI._poll_and_display_out_of_process_messages.__get__(fake)
        )
        # Should not raise
        fake._poll_and_display_out_of_process_messages()
        assert fake.conversation_history == []


# ---------------------------------------------------------------------------
# 3. _last_seen_msg_id initialization
# ---------------------------------------------------------------------------

class TestLastSeenMsgIdInit:
    def test_fresh_session_starts_at_zero(self, tmp_path):
        """A brand-new session with no messages should start at 0."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")

        # Verify the query returns 0 for an empty session
        with db._lock:
            row = db._conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = ?",
                ("brand-new-session",),
            ).fetchone()
        assert row[0] == 0

    def test_resumed_session_starts_at_tip(self, tmp_path):
        """A resumed session should start at the highest existing message id."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old-sess", source="cli", model="m")
        id1 = db.append_message("old-sess", role="user", content="a")
        id2 = db.append_message("old-sess", role="assistant", content="b")
        id3 = db.append_message("old-sess", role="user", content="c")

        with db._lock:
            row = db._conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = ?",
                ("old-sess",),
            ).fetchone()
        assert row[0] == id3
