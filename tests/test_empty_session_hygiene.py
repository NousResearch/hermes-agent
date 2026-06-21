"""Tests for empty-session hygiene — gemini-cli#27770 port.

Starting the CLI and immediately quitting (or rotating sessions with /new)
used to leave empty untitled rows in the session DB that clutter /resume
and `hermes sessions list`. ``SessionDB.delete_session_if_empty`` removes
a just-ended session row only when it never gained resumable content:
no messages, no title, and no child sessions.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


class TestDeleteSessionIfEmpty:
    def test_deletes_empty_untitled_session(self, db):
        db.create_session(session_id="empty", source="cli", model="test")
        db.end_session("empty", "cli_close")

        assert db.delete_session_if_empty("empty") is True
        assert db.get_session("empty") is None

    def test_keeps_session_with_messages(self, db):
        db.create_session(session_id="busy", source="cli", model="test")
        db.append_message("busy", role="user", content="hello")
        db.end_session("busy", "cli_close")

        assert db.delete_session_if_empty("busy") is False
        assert db.get_session("busy") is not None

    def test_keeps_titled_session(self, db):
        """A user-assigned title is resumable content even without messages."""
        db.create_session(session_id="titled", source="cli", model="test")
        db.set_session_title("titled", "Important plans")
        db.end_session("titled", "cli_close")

        assert db.delete_session_if_empty("titled") is False
        assert db.get_session("titled") is not None

    def test_keeps_session_with_children(self, db):
        """A parent that spawned delegate subagent runs is not empty."""
        db.create_session(session_id="parent", source="cli", model="test")
        db.create_session(
            session_id="child",
            source="tool",
            model="test",
            parent_session_id="parent",
        )
        db.end_session("parent", "cli_close")

        assert db.delete_session_if_empty("parent") is False
        assert db.get_session("parent") is not None
        assert db.get_session("child") is not None

    def test_unknown_session_returns_false(self, db):
        assert db.delete_session_if_empty("nope") is False

    def test_removes_on_disk_transcripts(self, db, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "empty.json").write_text("{}", encoding="utf-8")
        (sessions_dir / "empty.jsonl").write_text("", encoding="utf-8")

        db.create_session(session_id="empty", source="cli", model="test")
        db.end_session("empty", "cli_close")

        assert db.delete_session_if_empty("empty", sessions_dir=sessions_dir)
        assert not (sessions_dir / "empty.json").exists()
        assert not (sessions_dir / "empty.jsonl").exists()

    def test_no_file_cleanup_when_kept(self, db, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "busy.json").write_text("{}", encoding="utf-8")

        db.create_session(session_id="busy", source="cli", model="test")
        db.append_message("busy", role="user", content="hello")

        assert not db.delete_session_if_empty("busy", sessions_dir=sessions_dir)
        assert (sessions_dir / "busy.json").exists()

    def test_empty_session_disappears_from_listing(self, db):
        """The user-facing symptom: empty rows polluting session lists."""
        db.create_session(session_id="real", source="cli", model="test")
        db.append_message("real", role="user", content="do the thing")
        db.end_session("real", "cli_close")

        db.create_session(session_id="ghost", source="cli", model="test")
        db.end_session("ghost", "cli_close")

        ids_before = {s["id"] for s in db.list_sessions_rich(source="cli")}
        assert {"real", "ghost"} <= ids_before

        db.delete_session_if_empty("ghost")

        ids_after = {s["id"] for s in db.list_sessions_rich(source="cli")}
        assert "real" in ids_after
        assert "ghost" not in ids_after


class TestCLIDiscardSessionIfEmpty:
    """Wiring tests for HermesCLI._discard_session_if_empty."""

    def _make_cli(self, db):
        from cli import HermesCLI

        cli = HermesCLI.__new__(HermesCLI)
        cli._session_db = db
        cli.conversation_history = []
        return cli

    def test_discards_empty(self, db):
        db.create_session(session_id="empty", source="cli", model="test")
        db.end_session("empty", "cli_close")

        cli = self._make_cli(db)
        assert cli._discard_session_if_empty("empty") is True
        assert db.get_session("empty") is None

    def test_keeps_nonempty(self, db):
        db.create_session(session_id="busy", source="cli", model="test")
        db.append_message("busy", role="user", content="hi")

        cli = self._make_cli(db)
        assert cli._discard_session_if_empty("busy") is False
        assert db.get_session("busy") is not None

    def test_no_db_is_noop(self):
        cli = self._make_cli(None)
        assert cli._discard_session_if_empty("anything") is False

    def test_none_session_id_is_noop(self, db):
        cli = self._make_cli(db)
        assert cli._discard_session_if_empty(None) is False

    def test_db_error_swallowed(self, db):
        class Boom:
            def delete_session_if_empty(self, *a, **k):
                raise RuntimeError("locked")

        cli = self._make_cli(Boom())
        assert cli._discard_session_if_empty("x") is False

    def test_in_memory_history_blocks_prune(self, db):
        """The live transcript is authoritative: even if the DB row has no
        flushed messages yet, a CLI holding conversation history must not
        prune the session (covers flush-failed / not-yet-flushed turns)."""
        db.create_session(session_id="unflushed", source="cli", model="test")
        db.end_session("unflushed", "new_session")

        cli = self._make_cli(db)
        cli.conversation_history = [{"role": "user", "content": "hello"}]
        assert cli._discard_session_if_empty("unflushed") is False
        assert db.get_session("unflushed") is not None
