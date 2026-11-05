"""``hermes_state`` stand-in for outside plugins: each name through the stand-in and through its replacement gives the
same result on one seeded store; the deprecation warning fires once per process; a writable ``SessionDB`` for a file
not named ``state.db`` and a name outside the plugin guide's list are refused."""
import asyncio
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import hermes_constants
import hermes_state
import persistence
import sqlite_support
import storage_errors
from hermes_cli.session_export import to_legacy_export_dict
from persistence.conversation.messages import ConversationRequest, ResumeTooLarge, TranscriptRead, TranscriptView
from persistence.conversation.search import SearchSort
from persistence.conversation.sessions import SessionCountQuery, SessionOrder, SessionQuery, SessionWindow
from persistence.housekeeping.maintenance import PruneCriteria
from persistence_assembly import open_persistence, profile_scope, storage_health, store_location
from tests.fixtures.storage_seed import seed_messages, seed_session

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def quiet(monkeypatch):
    """The stand-in's once-per-process warning, consumed so a test's own accesses stay silent."""
    monkeypatch.setattr(hermes_state, "_warned", True)


@pytest.fixture
def store():
    handle = open_persistence(profile_scope(), eager=True)
    seed_session(handle, "s1", "cli", model="m1", title="alpha plan")
    seed_messages(handle, "s1", {"role": "user", "content": "find the zebra notes"},
                  {"role": "assistant", "content": "the zebra notes are here"})
    seed_session(handle, "s2", "telegram", model="m2", title="alpha plan #2", end_reason="user_exit")
    seed_messages(handle, "s2", {"role": "user", "content": "zebra again"})
    seed_session(handle, "s3", "cli", end_reason="user_exit")
    yield handle
    handle.close()


@pytest.fixture
def legacy(quiet, store):
    db = hermes_state.SessionDB()
    yield db
    db.close()


def test_session_reads_match_their_replacements(store, legacy):
    assert legacy.get_session("s1") == store.sessions.get("s1").as_session_row()
    assert legacy.get_session("missing") is None
    assert legacy.list_sessions_rich() == [e.as_list_row() for e in store.sessions.list_sessions(SessionQuery())]
    assert legacy.list_sessions_rich(source="cli", limit=1, offset=1) == [
        e.as_list_row() for e in store.sessions.list_sessions(SessionQuery(sources=("cli",)), SessionWindow(1, 1))]
    assert legacy.list_sessions_rich(order_by_last_active=True, search_query="alpha", compact_rows=True) == [
        e.as_list_row() for e in store.sessions.list_sessions(
            SessionQuery(order=SessionOrder.LAST_ACTIVE, search_text="alpha"), compact_rows=True)]
    assert legacy.session_count() == store.sessions.count_sessions() == 3
    assert legacy.session_count(sources=["telegram"]) == store.sessions.count_sessions(
        SessionCountQuery(sources=("telegram",))) == 1
    assert legacy.resolve_session_by_title("alpha plan") == store.sessions.resolve_by_title("alpha plan") == "s2"
    assert legacy.get_next_title_in_lineage("alpha plan") == store.sessions.next_title_in_lineage("alpha plan")


def test_message_reads_match_their_replacements(store, legacy):
    assert legacy.get_messages("s1") == [row.as_wire_row() for row in store.messages.read_transcript("s1")]
    assert legacy.get_messages("s1", latest=True, limit=1) == [
        row.as_wire_row() for row in store.messages.read_transcript("s1", TranscriptRead(limit=1, latest=True))]
    assert legacy.get_messages("s1", include_compacted=True) == [
        row.as_wire_row() for row in store.messages.read_transcript("s1", TranscriptRead(view=TranscriptView.DISPLAY))]
    assert legacy.get_messages_as_conversation("s1", include_row_ids=True) == [
        m.as_openai_message() for m in store.messages.read_conversation("s1", ConversationRequest(include_row_ids=True))]
    hits = store.search.search_messages("zebra", sort=SearchSort.NEWEST)
    rows = legacy.search_messages("zebra", sort="newest")
    assert [(row["id"], row["session_id"], row["snippet"]) for row in rows] == [
        (hit.message_id, hit.session_id, hit.snippet) for hit in hits]
    assert list(rows[0]) == ["id", "session_id", "role", "snippet", "timestamp", "tool_name", "source", "model",
                             "session_started", "context"]
    assert rows[0]["context"] == [{"role": c.role, "content": c.content} for c in hits[0].context]
    projected = legacy.search_messages("zebra", source_filter=["telegram"], fields=("source", "session_id"))
    assert projected == [{"session_id": "s2", "source": "telegram"}]


def test_writes_land_as_their_replacements_read_them(store, legacy):
    assert legacy.create_session("s4", "cli", model="m4", origin_json='{"platform": "cli"}') == "s4"
    record = store.sessions.get("s4")
    assert (record.source, record.model, dict(record.origin)) == ("cli", "m4", {"platform": "cli"})
    first = legacy.append_message("s4", "user", "hello")
    second = legacy.append_message("s4", "assistant", "", tool_calls=[{"id": "c1", "type": "function",
                                                                       "function": {"name": "f", "arguments": "{}"}}])
    rows = store.messages.read_transcript("s4")
    assert [row.id for row in rows] == [first, second]
    assert rows[1].tool_calls == [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    assert legacy.set_session_title("s4", "beta") is True
    assert store.sessions.get("s4").title == "beta"
    legacy.end_session("s4", "user_exit")
    assert store.sessions.get("s4").end_reason == "user_exit"
    legacy.reopen_session("s4")
    assert store.sessions.get("s4").ended_at is None
    legacy.clear_messages("s4")
    assert store.messages.read_transcript("s4") == ()
    assert store.sessions.get("s4").message_count == 0
    legacy.set_meta("goal:s4", "x")
    assert legacy.get_meta("goal:s4") == store.state_meta.get("goal:s4") == "x"
    assert legacy.list_meta_prefix("goal:") == [(e.key, e.value) for e in store.state_meta.list_prefix("goal:")]
    assert legacy.list_meta_prefix("") == []


def test_a_title_conflict_is_still_a_value_error(store, legacy):
    with pytest.raises(ValueError):
        legacy.set_session_title("s3", "alpha plan")


def test_export_matches_its_replacement(store, legacy):
    assert legacy.export_session("s1") == to_legacy_export_dict(store.maintenance.export_session("s1"))
    assert legacy.export_session("missing") is None
    assert legacy.export_all(source="cli") == [to_legacy_export_dict(e) for e in store.maintenance.export_all("cli")]


def test_delete_and_prune_remove_what_their_replacements_select(store, legacy):
    expected = {row.session_id for row in store.maintenance.list_prune_candidates(0, "telegram", criteria=PruneCriteria())}
    assert expected == {"s2"}
    deleted = []
    assert legacy.prune_sessions(older_than_days=0, source="telegram", deleted_ids=deleted) == 1
    assert deleted == ["s2"] and store.sessions.get("s2") is None
    assert legacy.delete_session("s3") is True
    assert store.sessions.get("s3") is None
    assert legacy.delete_session("s3") is False


def test_read_only_and_async_doors_read_the_same_rows(store, legacy):
    with hermes_state.SessionDB(read_only=True) as reader:
        assert reader.get_session("s1") == legacy.get_session("s1")
        assert reader.get_messages("s1") == legacy.get_messages("s1")
    with hermes_state.SessionDB(db_path=store_location(profile_scope()), read_only=True) as reader:
        assert reader.session_count() == 3
    door = hermes_state.AsyncSessionDB(legacy)
    assert asyncio.run(door.get_session("s1")) == legacy.get_session("s1")
    assert asyncio.run(door.get_messages_as_conversation("s1")) == legacy.get_messages_as_conversation("s1")


def test_a_named_state_db_opens_that_profile_folder(quiet, tmp_path):
    home = tmp_path / "elsewhere"
    home.mkdir()
    with hermes_state.SessionDB(db_path=home / "state.db") as db:
        db.create_session("x", "cli")
    with open_persistence(profile_scope(home=home)) as other:
        assert other.sessions.get("x").source == "cli"


def test_module_names_are_their_replacements(quiet, store):
    assert hermes_state.StateDbCorruptError is persistence.StoreQuarantined
    assert hermes_state.StateDbReplacedError is persistence.StoreReplaced
    assert hermes_state.SessionCompressionInProgressError is persistence.CompressionInProgress
    assert hermes_state.DeletedWalGenerationError is persistence.DeletedWalGeneration
    assert hermes_state.SessionResumeTooLargeError is ResumeTooLarge
    assert hermes_state.SessionExportTooLargeError is persistence.ExportTooLarge
    for name in ("classify_persistence_error", "is_sqlite_lock_error", "is_malformed_db_error"):
        assert getattr(hermes_state, name) is getattr(storage_errors, name)
    for name in ("apply_wal_with_fallback", "apply_database_pragmas"):
        assert getattr(hermes_state, name) is getattr(sqlite_support, name)
    for name in ("get_hermes_home", "mkdir_under_hermes_home"):
        assert getattr(hermes_state, name) is getattr(hermes_constants, name)
    db_path = store_location(profile_scope())
    assert hermes_state.DEFAULT_DB_PATH == db_path
    assert hermes_state.storage_state(db_path) == storage_health(profile_scope()).storage_state().value == "ok"
    assert hermes_state.storage_corrupt_reason(db_path) is None


def test_the_first_name_used_warns_once_naming_its_replacement_and_the_removal_release(monkeypatch):
    monkeypatch.setattr(hermes_state, "_warned", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = hermes_state.DEFAULT_DB_PATH, hermes_state.SessionDB, hermes_state.is_sqlite_lock_error
    assert [w.category for w in caught] == [DeprecationWarning]
    text = str(caught[0].message)
    assert "hermes_state.DEFAULT_DB_PATH" in text and "persistence_assembly.store_location" in text
    assert "v0.26.0" in text and hermes_state._MIGRATION_GUIDE in text
    assert caught[0].filename == __file__


def test_a_fresh_process_warns_once_for_its_imports():
    script = ("import warnings\n"
              "with warnings.catch_warnings(record=True) as caught:\n"
              "    warnings.simplefilter('always')\n"
              "    from hermes_state import SessionDB, StateDbCorruptError\n"
              "    from hermes_state import AsyncSessionDB\n"
              "print(len(caught), caught[0].category.__name__, 'hermes_state.SessionDB' in str(caught[0].message))\n")
    out = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
                         check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["1", "DeprecationWarning", "True"]


def test_a_writable_store_for_a_file_not_named_state_db_is_refused(quiet, tmp_path):
    target = tmp_path / "backup.db"
    with pytest.raises(ValueError, match=r"open_persistence\(profile_scope\(home=\.\.\.\)\)"):
        hermes_state.SessionDB(db_path=target)
    assert not target.exists() and not (tmp_path / "state.db").exists()


def test_a_name_outside_the_list_points_at_the_migration_guide(quiet, store, legacy):
    with pytest.raises(AttributeError, match="hermes_state-deprecated-use-the-storage-interface"):
        _ = hermes_state.load_fts5_cjk_extension
    with pytest.raises(ImportError):
        from hermes_state import SessionSessionsMixin
    with pytest.raises(AttributeError, match="hermes_state-deprecated-use-the-storage-interface"):
        legacy.get_session_title("s1")
    assert not [name for name in dir(hermes_state) if not name.startswith("_") and name not in hermes_state.__all__]
