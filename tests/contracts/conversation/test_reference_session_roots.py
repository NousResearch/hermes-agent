"""Reference session mutations and their lifecycle behaviour."""

import pytest

from persistence.conversation.sessions import NewSession, ReplaceSessionConfig, SessionLineageKind, TitleConflict, TitleSource
from tests.contracts._reference import open_reference_backend
from tests.contracts._scopes import profile_scope


def test_title_transfers_from_compression_ancestor():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        backend.open(scope).sessions.create(NewSession("parent", "cli"))
        backend.open(scope).sessions.set_title("parent", "taken")
        backend.open(scope).sessions.end("parent", "compression")
        backend.open(scope).sessions.create(NewSession("child", "cli", parent_session_id="parent"))
        assert backend.open(scope).sessions.set_title("child", "taken")
        assert (backend.open(scope).sessions.get("parent")).title is None
        assert (backend.open(scope).sessions.get("parent")).title_source is TitleSource.USER
        assert backend.open(scope).sessions.get_title_source("parent") is None
        assert backend.open(scope).sessions.get_title_source("child") is TitleSource.USER
        backend.open(scope).sessions.create(NewSession("other", "cli"))
        with pytest.raises(TitleConflict) as exc:
            backend.open(scope).sessions.set_title("other", "taken")
        assert exc.value.holder == "child"
        assert str(exc.value) == "Title 'taken' is already in use by session child"



def test_session_records_carry_counter_cells_as_stored():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        sessions = backend.open(scope).sessions
        sessions.create(NewSession("cells", "cli"))
        row = sessions._table(scope)["cells"]
        row.update(compression_fallback_streak="2", compression_overload_streak=1.5, cwd_generation="x")
        record = sessions.get("cells")
        # The row's cells are handed over without coercion: the record carries them as stored.
        assert (record.compression_fallback_streak, record.compression_overload_streak, record.git_metadata_generation) == ("2", 1.5, "x")


def test_automatic_title_refusal_message():
    with open_reference_backend() as backend:
        with pytest.raises(ValueError, match="invalid automatic title source: 'user'"):
            backend.open(profile_scope("default")).sessions.set_auto_title("absent", "title", source=TitleSource.USER)


def test_reopen_preserves_implicit_reset_children():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        backend.open(scope).sessions.create(NewSession("parent", "telegram", session_key="peer"))
        backend.open(scope).sessions.end("parent", "session_reset")
        backend.open(scope).sessions.create(NewSession("child", "telegram", parent_session_id="parent", session_key="peer"))
        backend.open(scope).sessions.create(NewSession("other", "telegram", parent_session_id="parent", session_key="other"))
        backend.open(scope).sessions.reopen("parent")
        child = backend.open(scope).sessions.get("child")
        assert child.lineage_kind is SessionLineageKind.RESET
        assert child.lineage_origin_id == "parent"
        assert child.parent_session_id == "parent"
        assert (backend.open(scope).sessions.get("other")).lineage_kind is SessionLineageKind.NONE


def test_upsert_keeps_explicit_empty_config_and_replaces_unknown_source():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        backend.open(scope).sessions.create(NewSession("s", "unknown"))
        backend.open(scope).sessions.replace_session_config("s", ReplaceSessionConfig({}))
        backend.open(scope).sessions.create(NewSession("s", "cli", model_config={"late": 2}))
        row = backend.open(scope).sessions.get("s")
        assert row.source == "cli"
        assert row.model_config == {}


def test_missing_parent_creation_refuses_without_inserting():
    from sqlite3 import IntegrityError
    from persistence import PermanentStoreError

    with open_reference_backend() as backend:
        scope = profile_scope("default")
        with pytest.raises(PermanentStoreError, match="FOREIGN KEY constraint failed") as exc:
            backend.open(scope).sessions.create(NewSession("child", "cli", parent_session_id="absent"))
        assert isinstance(exc.value.__cause__, IntegrityError)
        assert backend.open(scope).sessions.get("child") is None


def test_listing_preview_uses_first_visible_user_message():
    from datetime import UTC, datetime

    from persistence.conversation.messages import NewMessage
    from persistence.conversation.sessions import SessionQuery, SessionWindow

    with open_reference_backend() as backend:
        scope = profile_scope("default")
        backend.open(scope).sessions.create(NewSession("s", "cli"))
        for text, stamp, hidden in [("late", 1700000002, False), ("hidden", 1700000000, True), ("  line one\nline two  ", 1700000001, False)]:
            backend.open(scope).messages.append("s", NewMessage("user", text, datetime.fromtimestamp(stamp, UTC), display_kind="hidden" if hidden else None))
        rows = backend.open(scope).sessions.list_sessions(SessionQuery(), SessionWindow(limit=10))
        assert rows[0].preview == "line one line two"


@pytest.mark.parametrize("kind", [SessionLineageKind.BRANCH, SessionLineageKind.DELEGATE])
def test_explicit_forks_inherit_workspace_but_not_peer_routing(kind):
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        backend.open(scope).sessions.create(NewSession("parent", "cli", cwd="/parent", chat_id="chat"))
        backend.open(scope).sessions.end("parent", "compression")
        backend.open(scope).sessions.create(NewSession("child", "cli", parent_session_id="parent", lineage_kind=kind))
        child = backend.open(scope).sessions.get("child")
        assert child.cwd == "/parent"
        assert child.chat_id is None


def test_exclusive_existing_id_precedes_oversized_title_validation():
    from persistence.conversation.sessions import SessionExists

    with open_reference_backend() as backend:
        scope = profile_scope("default")
        backend.open(scope).sessions.create(NewSession("exists", "cli"))
        with pytest.raises(SessionExists):
            backend.open(scope).sessions.create_exclusive("exists", source="cli", title="x" * 101)


def test_missing_session_still_validates_the_title():
    with open_reference_backend() as backend:
        with pytest.raises(ValueError, match="Title too long"):
            backend.open(profile_scope("default")).sessions.set_title("missing", "x" * 101)
