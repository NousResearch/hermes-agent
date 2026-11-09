"""``hermes_state`` stand-in for outside plugins: each name through the stand-in and through its replacement gives the
same result on one seeded store; the deprecation warning fires once per process; a writable ``SessionDB`` for a file
not named ``state.db`` and a name outside the plugin guide's list are refused; a writable ``SessionDB`` used after
``close()`` opens its store again with a warning, unless the store is quarantined or its file was replaced, and a
read-only one refuses; an agent given only a writable ``SessionDB`` saves its turn to that store; the stand-in is gone
before the release its warning names."""
import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import hermes_constants
import hermes_dir_acl
import hermes_state
import persistence
import process_identity
import sqlite_support
import storage_errors
import storage_limits
import storage_user_copy
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


def test_a_named_source_wins_over_a_source_list(store, legacy):
    assert legacy.session_count(source="cli", sources=["telegram"]) == 2
    assert {row["id"] for row in legacy.list_sessions_rich(source="cli", sources=["telegram"])} == {"s1", "s3"}
    assert legacy.session_count(source="", sources=["telegram"]) == 1


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


def test_clearing_a_session_ended_by_compression_deletes_its_messages(store, legacy):
    legacy.end_session("s1", "compression")
    assert legacy.clear_messages("s1") is None
    assert store.messages.read_transcript("s1", TranscriptRead(view=TranscriptView.ALL)) == ()
    record = store.sessions.get("s1")
    assert (record.message_count, record.tool_call_count, record.end_reason) == (0, 0, "compression")


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


_MOVED = {
    "format_session_db_unavailable": storage_user_copy, "get_last_init_error": storage_user_copy,
    "holder_pid_checkable": process_identity, "make_user_dir": hermes_dir_acl,
    "preflight_db_writability": sqlite_support, "read_only_db_uri": sqlite_support,
    "resolved_max_export_messages": storage_limits, "resolved_max_resume_messages": storage_limits,
}


def test_the_moved_helpers_are_their_replacements(quiet):
    for name, module in _MOVED.items():
        assert getattr(hermes_state, name) is getattr(module, name), name


def test_a_diverted_transcript_lands_in_the_profiles_sessions_folder(quiet):
    path = hermes_state.divert_session_transcript_jsonl("s9", [{"role": "user", "content": "kept"}, None, "plain"])
    assert path == hermes_constants.get_hermes_home() / "sessions" / "s9.jsonl"
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [
        {"role": "user", "content": "kept"}, {"content": "plain"}]
    assert hermes_state.divert_session_transcript_jsonl("s9", []) is None


def test_the_plugin_api_lists_each_carried_name_and_not_the_whole_module():
    data = json.loads((REPO_ROOT / "hermes_cli" / "data" / "plugin_api.json").read_text(encoding="utf-8-sig"))
    entries = {entry for group in data["groups"] for entry in group["entries"]}
    assert "hermes_state" not in entries
    assert {entry.partition(":")[2] for entry in entries if entry.startswith("hermes_state:")} == set(
        hermes_state.__all__)


def test_validate_flags_a_name_the_stand_in_does_not_carry(tmp_path):
    from hermes_cli.plugin_validate_private_imports import private_import_findings

    plugin = tmp_path / "legacy"
    plugin.mkdir()
    (plugin / "__init__.py").write_text("from hermes_state import SessionDB, resolved_max_resume_messages\n"
                                        "from hermes_state import SessionMessagesMixin\n", encoding="utf-8")
    assert private_import_findings(plugin) == ["__init__.py:2 hermes_state.SessionMessagesMixin"]


def test_the_registry_stand_in_lends_a_session_db_that_reads_gateway_sessions(monkeypatch, store):
    import hermes_state_registry

    seed_session(store, "g1", "telegram", session_key="agent:main:telegram:dm:42", chat_id="42", chat_type="dm")
    seed_messages(store, "g1", {"role": "user", "content": "hello from telegram"})
    seed_session(store, "g2", "discord", session_key="agent:main:discord:dm:7", end_reason="user_exit")
    monkeypatch.setattr(hermes_state, "_warned", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        db = hermes_state_registry.acquire()
    assert [w.category for w in caught] == [DeprecationWarning]
    text = str(caught[0].message)
    assert "hermes_state_registry.acquire" in text and "persistence_assembly.open_persistence" in text
    assert "v0.27.0" in text and caught[0].filename == __file__
    assert isinstance(db, hermes_state.SessionDB)
    assert [row["content"] for row in db.get_messages("g1")] == ["hello from telegram"]
    rows = db.list_gateway_sessions()
    assert rows == [{**e.session.as_session_row(), "last_active": e.last_active}
                    for e in store.sessions.list_gateway_sessions()]
    assert [row["session_key"] for row in rows] == ["agent:main:telegram:dm:42"]
    assert db.list_gateway_sessions(platform="TELEGRAM") == rows
    assert {row["id"] for row in db.list_gateway_sessions(active_only=False)} == {"g1", "g2"}
    closes = []
    monkeypatch.setattr(db, "close", lambda: closes.append("close"))
    assert hermes_state_registry.release(db) is True
    hermes_state_registry.release_or_close(db)
    assert closes == ["close", "close"]


def test_the_plugin_api_lists_the_registry_stand_ins_names():
    data = json.loads((REPO_ROOT / "hermes_cli" / "data" / "plugin_api.json").read_text(encoding="utf-8-sig"))
    entries = {entry for group in data["groups"] for entry in group["entries"]}
    assert {entry for entry in entries if entry.startswith("hermes_state_registry")} == {
        "hermes_state_registry:acquire", "hermes_state_registry:release", "hermes_state_registry:release_or_close"}


def test_the_registry_stand_in_is_silent_once_the_stand_in_has_warned(quiet, tmp_path):
    import hermes_state_registry

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        db = hermes_state_registry.acquire(tmp_path / "state.db")
        hermes_state_registry.release_or_close(db)
    assert caught == []


def test_the_first_name_used_warns_once_naming_its_replacement_and_the_removal_release(monkeypatch):
    monkeypatch.setattr(hermes_state, "_warned", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = hermes_state.DEFAULT_DB_PATH, hermes_state.SessionDB, hermes_state.is_sqlite_lock_error
    assert [w.category for w in caught] == [DeprecationWarning]
    text = str(caught[0].message)
    assert "hermes_state.DEFAULT_DB_PATH" in text and "persistence_assembly.store_location" in text
    assert "v0.27.0" in text and hermes_state._MIGRATION_GUIDE in text
    assert caught[0].filename == __file__


def test_the_stand_in_is_removed_before_its_removal_release_ships(monkeypatch):
    from hermes_cli import version_info
    from hermes_cli.plugins_manifest import running_hermes_version, version_satisfies

    monkeypatch.setattr(version_info, "_cached_version_info", None)  # the suite's conftest seeds "unknown"
    running = running_hermes_version()
    if running == "unknown":
        pytest.skip("no install stamp or release tag names the running release")
    assert not version_satisfies(f">={hermes_state._REMOVAL_RELEASE}", running), (
        f"Hermes {running} has reached {hermes_state._REMOVAL_RELEASE}, the release that removes the hermes_state "
        "stand-in. Remove together: hermes_state.py, hermes_state_registry.py and tests/test_hermes_state_standin.py; "
        "the session_db= argument "
        "plugin context engines get (context_engine_session_db and close_context_engine_session_db in "
        "agent/conversation_compression.py, and their callers); _REMOVED_STORE_PRIVATE_METHODS in "
        "hermes_cli/plugin_validate_private_imports.py; the hermes_state entries in hermes_cli/data/plugin_api.json "
        "and website/docs/developer-guide/plugins/public-api.md; the section \"hermes_state deprecated: use the "
        "storage interface\" in website/docs/developer-guide/plugins/index.md and session_db in "
        "website/docs/developer-guide/context-engine-plugin.md.")


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


@pytest.fixture
def sole(quiet, tmp_path):
    """A writable stand-in holding the only handle on its store, so its ``close()`` releases the store."""
    db = hermes_state.SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", "cli")
    yield db
    db.close()


def test_a_write_after_close_opens_the_store_again_with_a_warning(sole, caplog):
    sole.append_message("s1", "user", content="hello")
    sole.close()
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        message_id = sole.append_message("s1", "assistant", content="flushed after teardown")
    assert isinstance(message_id, int) and message_id > 0
    rows = sole.get_messages("s1")
    assert [(row["role"], row["content"]) for row in rows] == [("user", "hello"),
                                                              ("assistant", "flushed after teardown")]
    assert any("reopening" in record.message for record in caplog.records)


def test_the_store_opened_again_takes_more_writes_and_closes_again(sole):
    sole.close()
    sole.append_message("s1", "user", content="a")
    sole.append_message("s1", "assistant", content="b")
    assert len(sole.get_messages("s1")) == 2
    sole.close()
    sole.close()


def test_a_read_after_close_opens_the_store_again(sole):
    sole.append_message("s1", "user", content="hello")
    sole.close()
    assert len(sole.get_messages("s1")) == 1


def test_closes_racing_a_flush_lose_no_write(sole):
    start = threading.Event()
    errors = []

    def flush():
        start.wait()
        for i in range(40):
            try:
                sole.append_message("s1", "tool", content=f"result {i}")
            except Exception as exc:
                errors.append(exc)

    worker = threading.Thread(target=flush)
    worker.start()
    start.set()
    sole.close()
    sole.close()
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert errors == []
    assert len(sole.get_messages("s1")) == 40


def test_a_read_only_door_refuses_after_close(sole, tmp_path):
    reader = hermes_state.SessionDB(db_path=tmp_path / "state.db", read_only=True)
    reader.close()
    with pytest.raises(Exception, match="closed"):
        reader.get_messages("s1")


def test_a_quarantined_store_is_not_opened_again_after_close(sole, tmp_path):
    from persistence.sqlite._state.health import mark_storage_corrupt, reset_storage_state

    sole.close()
    mark_storage_corrupt(tmp_path / "state.db", "seeded")
    try:
        with pytest.raises(persistence.StoreQuarantined):
            sole.append_message("s1", "user", content="lost")
        with pytest.raises(persistence.StoreQuarantined):
            sole.get_session("s1")
    finally:
        reset_storage_state(tmp_path / "state.db")


@pytest.mark.platforms("posix")
def test_a_replaced_file_is_not_opened_again_after_close(sole, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    with open_persistence(profile_scope(home=other), eager=True) as handle:
        seed_session(handle, "o1", "cli")
    sole.close()
    os.replace(other / "state.db", tmp_path / "state.db")
    with pytest.raises(persistence.StoreReplaced):
        sole.append_message("s1", "user", content="lost")
    with open_persistence(profile_scope(home=tmp_path)) as handle:
        assert handle.sessions.get("o1") is not None and handle.messages.read_transcript("o1") == ()


def _agent_on(session_db):
    """An ``AIAgent`` given ``session_db`` and no storage interface handle, its model answering ``done``."""
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, session_db=session_db, session_id="turn-1",
                        platform="cli")
    message = SimpleNamespace(content="done", tool_calls=None)
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")], model="test/model",
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18))
    return agent


def test_an_agent_given_only_the_stand_in_saves_its_turn(sole):
    agent = _agent_on(sole)
    assert agent.run_conversation("hello there")["final_response"] == "done"
    rows = sole.get_messages("turn-1")
    assert [(row["role"], row["content"]) for row in rows] == [("user", "hello there"), ("assistant", "done")]


def test_an_agent_given_only_the_stand_in_saves_its_token_counts(sole):
    agent = _agent_on(sole)
    agent.run_conversation("hello there")
    row = sole.get_session("turn-1")
    assert (row["input_tokens"], row["output_tokens"]) == (11, 7)
