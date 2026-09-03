"""Tests for stale tool-call marker session repair (hermes_state, #78148).

Before the root-cause fix in ``agent.conversation_loop``, a local tool-call
template could emit a bare bracketed marker (e.g. "[memory]") as assistant
content alongside a real tool call. The loop cached that marker as a
fallback and, when the following turn came back empty, replayed it as the
"final response" — persisting it into the session as if the model had
actually answered.

``_strip_stale_tool_call_markers`` is the load-on-read defense-in-depth
that clears any such stray marker content from sessions written before the
fix, so resuming a polluted session doesn't re-teach the model to keep
emitting the marker. Unaffected sessions pass through unchanged.
"""

import pytest

from hermes_state import (
    _is_stale_tool_call_marker_message,
    _strip_stale_tool_call_markers,
)


class TestIsStaleToolCallMarkerMessage:
    def test_matches_bare_marker_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "[memory]",
            "tool_calls": [{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}],
        }
        assert _is_stale_tool_call_marker_message(msg) is True

    def test_matches_dotted_marker(self):
        msg = {
            "role": "assistant",
            "content": "[foo.bar]",
            "tool_calls": [{"id": "1", "function": {"name": "foo.bar", "arguments": "{}"}}],
        }
        assert _is_stale_tool_call_marker_message(msg) is True

    def test_ignores_marker_without_tool_calls(self):
        # A genuine final response of "[memory]" with no tool call is not
        # the contamination signature — leave it alone.
        msg = {"role": "assistant", "content": "[memory]"}
        assert _is_stale_tool_call_marker_message(msg) is False

    def test_ignores_real_content_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "I'll check that for you.",
            "tool_calls": [{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}],
        }
        assert _is_stale_tool_call_marker_message(msg) is False

    def test_ignores_user_role(self):
        msg = {
            "role": "user",
            "content": "[memory]",
            "tool_calls": [{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}],
        }
        assert _is_stale_tool_call_marker_message(msg) is False


class TestStripStaleToolCallMarkers:
    def test_clears_contaminated_content_keeps_tool_calls(self):
        messages = [
            {"role": "user", "content": "do the full task"},
            {
                "role": "assistant",
                "content": "[memory]",
                "tool_calls": [{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "1"},
        ]
        out = _strip_stale_tool_call_markers(messages)
        assert out[1]["content"] == ""
        # Tool call itself must survive — provider tool_call/result pairing.
        assert out[1]["tool_calls"] == [{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}]

    def test_unaffected_session_passes_through_unchanged(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "It's sunny."},
        ]
        out = _strip_stale_tool_call_markers(messages)
        assert out == messages


class TestGetMessagesAsConversationStripsStaleMarkers:
    """The load-on-read wiring: get_messages_as_conversation must actually
    call _strip_stale_tool_call_markers, so a session polluted with a stale
    "[memory]" marker resumes clean end-to-end (not just the pure helper in
    isolation)."""

    def test_polluted_session_resumes_without_marker(self):
        import tempfile
        from pathlib import Path
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                db.create_session(session_id="s1", source="cli")
                db.append_message("s1", role="user", content="do the full task")
                # Stray contamination written by an older build (pre-#78148 fix).
                db.append_message(
                    "s1", role="assistant", content="[memory]",
                    tool_calls=[{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}],
                )
                db.append_message("s1", role="tool", content="ok", tool_call_id="1")
                db.append_message("s1", role="assistant", content="Here is the result.")

                conv = db.get_messages_as_conversation("s1")
                contents = [m.get("content") for m in conv if m.get("role") == "assistant"]

                assert "[memory]" not in contents
                assert "Here is the result." in contents
            finally:
                db.close()

    def test_clean_session_resumes_unaffected(self):
        import tempfile
        from pathlib import Path
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                db.create_session(session_id="s1", source="cli")
                db.append_message("s1", role="user", content="What's the weather?")
                db.append_message("s1", role="assistant", content="It's sunny.")

                conv = db.get_messages_as_conversation("s1")
                contents = [m.get("content") for m in conv]

                assert contents == ["What's the weather?", "It's sunny."]
            finally:
                db.close()


class TestPurgeStaleToolCallMarkers:
    """SessionDB.purge_stale_tool_call_markers: the permanent, one-time DB
    rewrite. Complements the load-on-read repair — this variant edits the
    stored rows in place so long-lived sessions stop re-scanning/re-repairing
    the same contaminated rows on every resume."""

    def _seed_polluted_db(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="do the full task")
        db.append_message(
            "s1", role="assistant", content="[memory]",
            tool_calls=[{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}],
        )
        db.append_message("s1", role="tool", content="ok", tool_call_id="1")
        db.append_message("s1", role="assistant", content="Here is the result.")

    def test_dry_run_reports_without_writing(self):
        import tempfile
        from pathlib import Path
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                self._seed_polluted_db(db)

                report = db.purge_stale_tool_call_markers(dry_run=True)
                assert report["dry_run"] is True
                assert report["rows_affected"] == 1

                # Nothing written: the raw row still has the marker.
                raw = db._conn.execute(
                    "SELECT content FROM messages WHERE role = 'assistant' "
                    "AND tool_calls IS NOT NULL AND tool_calls != ''"
                ).fetchone()
                assert raw["content"] == "[memory]"
            finally:
                db.close()

    def test_purge_clears_content_keeps_tool_calls(self):
        import tempfile
        from pathlib import Path
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                self._seed_polluted_db(db)

                report = db.purge_stale_tool_call_markers(dry_run=False)
                assert report["dry_run"] is False
                assert report["rows_affected"] == 1
                # Backup defaults to on for a destructive, irreversible write.
                assert report["backup_path"] is not None
                assert Path(report["backup_path"]).exists()

                row = db._conn.execute(
                    "SELECT content, tool_calls FROM messages WHERE role = 'assistant' "
                    "AND tool_calls IS NOT NULL AND tool_calls != ''"
                ).fetchone()
                assert row["content"] == ""
                # tool_calls column itself must survive the rewrite untouched.
                assert row["tool_calls"]

                # Running again finds nothing left to clean — idempotent.
                second = db.purge_stale_tool_call_markers(dry_run=False)
                assert second["rows_affected"] == 0
                assert second["backup_path"] is None  # nothing to change, nothing to back up
            finally:
                db.close()

    def test_purge_bumps_each_visible_lineage_once_and_invalidates_conditional_page(self):
        import tempfile
        from pathlib import Path
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                db.create_session("root", source="cli")
                db.append_message(
                    "root", role="assistant", content="[memory]",
                    tool_calls=[{"id": "1", "function": {"name": "a", "arguments": "{}"}}],
                )
                db.append_message(
                    "root", role="assistant", content="[memory]",
                    tool_calls=[{"id": "2", "function": {"name": "b", "arguments": "{}"}}],
                )
                db.create_session("other", source="cli")
                db.append_message(
                    "other", role="assistant", content="[memory]",
                    tool_calls=[{"id": "3", "function": {"name": "c", "arguments": "{}"}}],
                )
                root_before = db.get_display_revision("root")
                other_before = db.get_display_revision("other")

                assert db.get_display_message_page(
                    "root", limit=120, known_display_revision=root_before
                )["unchanged"] is True
                report = db.purge_stale_tool_call_markers(backup=False)

                assert report["rows_affected"] == 3
                assert db.get_display_revision("root") == root_before + 1
                assert db.get_display_revision("other") == other_before + 1
                assert db.get_display_message_page(
                    "root", limit=120, known_display_revision=root_before
                )["unchanged"] is False
                assert db.purge_stale_tool_call_markers(backup=False)["rows_affected"] == 0
                assert db.get_display_revision("root") == root_before + 1
            finally:
                db.close()

    def test_purge_rolls_back_marker_write_when_revision_bump_fails(self, monkeypatch):
        import tempfile
        from pathlib import Path
        import sqlite3
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                db.create_session("root", source="cli")
                db.append_message(
                    "root", role="assistant", content="[memory]",
                    tool_calls=[{"id": "1", "function": {"name": "a", "arguments": "{}"}}],
                )

                def fail_bump(conn, root_id):
                    raise sqlite3.OperationalError("display revision write failed")

                monkeypatch.setattr(db, "_bump_display_revision_root", fail_bump)
                with pytest.raises(sqlite3.OperationalError, match="display revision write failed"):
                    db.purge_stale_tool_call_markers(backup=False)

                row = db._conn.execute("SELECT content FROM messages").fetchone()
                assert row["content"] == "[memory]"
            finally:
                db.close()

    def test_no_backup_when_flag_false(self):
        import tempfile
        from pathlib import Path
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                self._seed_polluted_db(db)

                report = db.purge_stale_tool_call_markers(dry_run=False, backup=False)
                assert report["rows_affected"] == 1
                assert report["backup_path"] is None
                # No extra file created beside the DB.
                siblings = list(Path(tmp).glob("t.db.*backup*"))
                assert siblings == []
            finally:
                db.close()

    def test_dry_run_never_backs_up(self):
        import tempfile
        from pathlib import Path
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                self._seed_polluted_db(db)

                report = db.purge_stale_tool_call_markers(dry_run=True)
                assert report["backup_path"] is None
                siblings = list(Path(tmp).glob("t.db.*backup*"))
                assert siblings == []
            finally:
                db.close()

    def test_no_affected_rows_on_clean_db(self):
        import tempfile
        from pathlib import Path
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            try:
                db.create_session(session_id="s1", source="cli")
                db.append_message("s1", role="user", content="What's the weather?")
                db.append_message("s1", role="assistant", content="It's sunny.")

                report = db.purge_stale_tool_call_markers(dry_run=False)
                assert report["rows_affected"] == 0
                assert report["row_ids"] == []
            finally:
                db.close()
