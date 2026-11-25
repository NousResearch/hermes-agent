"""Tests for empty-session hygiene — gemini-cli#27770 port.

Starting the CLI and immediately quitting (or rotating sessions with /new)
used to leave empty untitled rows in the session DB that clutter /resume
and `hermes sessions list`. ``sessions.delete_if_empty`` removes
a just-ended session row only when it never gained resumable content:
no messages, no title, and no child sessions; ``after_sessions_removed``
then removes its transcript files.
"""

import pytest

from gateway.session_ingress_media import native_inputs_root
from session_retention import after_sessions_removed
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.fixture()
def store(tmp_path):
    return storage_handle(tmp_path)


def _delete_if_empty(store, session_id, sessions_dir=None):
    """The empty-session delete, then its post-commit work (transcript files under *sessions_dir*); the removed ids."""
    removed = store.sessions.delete_if_empty(session_id, media_root=str(native_inputs_root()))
    after_sessions_removed(store, store.scope.profile_dir, removed, reason="delete", primary=session_id,
                           sessions_dir=sessions_dir)
    return removed


class TestDeleteSessionIfEmpty:

    def test_keeps_session_with_messages(self, store):
        seed_session(store, "busy", "cli", model="test")
        seed_messages(store, "busy", {"role": "user", "content": "hello"})
        store.sessions.end("busy", "cli_close")

        assert _delete_if_empty(store, "busy") == ()
        assert session_row(store, "busy") is not None

    def test_keeps_titled_session(self, store):
        """A user-assigned title is resumable content even without messages."""
        seed_session(store, "titled", "cli", model="test", title="Important plans", end_reason="cli_close")

        assert _delete_if_empty(store, "titled") == ()
        assert session_row(store, "titled") is not None

    def test_keeps_session_with_children(self, store):
        """A parent that spawned delegate subagent runs is not empty."""
        seed_session(store, "parent", "cli", model="test")
        seed_session(store, "child", "tool", model="test", parent_session_id="parent")
        store.sessions.end("parent", "cli_close")

        assert _delete_if_empty(store, "parent") == ()
        assert session_row(store, "parent") is not None
        assert session_row(store, "child") is not None


    def test_removes_on_disk_transcripts(self, store, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "empty.json").write_text("{}", encoding="utf-8")
        (sessions_dir / "empty.jsonl").write_text("", encoding="utf-8")

        seed_session(store, "empty", "cli", model="test", end_reason="cli_close")

        assert _delete_if_empty(store, "empty", sessions_dir=sessions_dir)
        assert not (sessions_dir / "empty.json").exists()
        assert not (sessions_dir / "empty.jsonl").exists()

    def test_no_file_cleanup_when_kept(self, store, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "busy.json").write_text("{}", encoding="utf-8")

        seed_session(store, "busy", "cli", model="test")
        seed_messages(store, "busy", {"role": "user", "content": "hello"})

        assert not _delete_if_empty(store, "busy", sessions_dir=sessions_dir)
        assert (sessions_dir / "busy.json").exists()



class _PruningStore:
    """The CLI's session store as these tests drive it: its empty-session prune deletes through the interface."""

    def __init__(self, persistence):
        self.persistence = persistence

    def delete_session_if_empty(self, session_id, sessions_dir=None):
        return bool(_delete_if_empty(self.persistence, session_id, sessions_dir))


class TestCLIDiscardSessionIfEmpty:
    """Wiring tests for HermesCLI._discard_session_if_empty."""

    def _make_cli(self, db):
        from cli import HermesCLI

        cli = HermesCLI.__new__(HermesCLI)
        cli._session_db = db
        cli.conversation_history = []
        return cli





    def test_db_error_swallowed(self, store):
        class Boom:
            def delete_session_if_empty(self, *a, **k):
                raise RuntimeError("locked")

        cli = self._make_cli(Boom())
        assert cli._discard_session_if_empty("x") is False

    def test_in_memory_history_blocks_prune(self, store):
        """The live transcript is authoritative: even if the DB row has no
        flushed messages yet, a CLI holding conversation history must not
        prune the session (covers flush-failed / not-yet-flushed turns)."""
        seed_session(store, "unflushed", "cli", model="test", end_reason="new_session")

        cli = self._make_cli(_PruningStore(store))
        cli.conversation_history = [{"role": "user", "content": "hello"}]
        assert cli._discard_session_if_empty("unflushed") is False
        assert session_row(store, "unflushed") is not None
