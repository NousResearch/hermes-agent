"""Committed maintenance sources remain durable across transcript changes."""

from hermes_maintenance_source import mint_local_input
from hermes_maintenance_source import (
    read_committed_source, pending_sources, committed_source_for_receipt,
    local_input, revoke_committed_source,
    revise_original_input, delivery_sources, begin_host_question_delivery,
    close_host_question_delivery, host_question_delivery_for_session,
    record_host_question_delivery,
)
from hermes_state import SessionDB
import sqlite3
from contextlib import closing
import pytest


def test_local_receipt_has_host_minted_occurrence(tmp_path):
    first = mint_local_input("hello", home=tmp_path, session_id="s1")
    second = mint_local_input("hello", home=tmp_path, session_id="s1")
    assert first.occurrence != second.occurrence
    assert first.event_id is None


def _store(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    db = SessionDB(db_path=home / "state.db")
    db.create_session("s1", "cli", session_key="cli:test")
    return home, db


def test_delivery_distinguishes_missing_queue_from_locked_queue(tmp_path):
    home, db = _store(tmp_path)
    assert delivery_sources(home) == []  # Existing databases may predate the outbox.
    receipt = mint_local_input("deliver me", home=home, session_id="s1")
    db.append_message("s1", "user", receipt.raw, _maintenance_source=receipt)
    source_id = pending_sources(home)[0]
    with closing(sqlite3.connect(home / "state.db", timeout=0)) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            delivery_sources(home)
        blocker.rollback()
    assert delivery_sources(home) == [source_id]


def test_single_and_batch_commit_source_and_outbox(tmp_path):
    home, db = _store(tmp_path)
    first = mint_local_input("first", home=home, session_id="s1")
    row_id = db.append_message("s1", "user", "first", _maintenance_source=first)
    source = committed_source_for_receipt(home, "s1", first)
    assert source["message_row_id"] == row_id
    second = mint_local_input("second", home=home, session_id="s1")
    db.append_messages_batch("s1", [{"role": "user", "content": "second", "_maintenance_source": second}])
    assert committed_source_for_receipt(home, "s1", second)
    assert len(pending_sources(home)) == 2
    db.append_message("s1", "user", "first", _maintenance_source=first)
    assert len(pending_sources(home)) == 2
    with closing(sqlite3.connect(home / "state.db")) as conn:
        conn.execute("UPDATE messages SET active=0 WHERE id=?", (row_id,))
        conn.commit()
    assert read_committed_source(home, source["source_id"])
    with closing(sqlite3.connect(home / "state.db")) as conn:
        conn.execute("UPDATE maintenance_sources_v1 SET raw_text='tampered' WHERE source_id=?", (source["source_id"],))
        conn.commit()
    assert read_committed_source(home, source["source_id"]) is None


def test_failed_transaction_rolls_back_source_and_outbox(tmp_path, monkeypatch):
    import hermes_maintenance_source as source_module
    home, db = _store(tmp_path)
    original = source_module.bind_committed_source
    def fail_after_insert(*args):
        original(*args)
        raise RuntimeError("rollback")
    monkeypatch.setattr(source_module, "bind_committed_source", fail_after_insert)
    with pytest.raises(RuntimeError, match="rollback"):
        db.append_message("s1", "user", "first", _maintenance_source=mint_local_input("first", home=home, session_id="s1"))
    with closing(sqlite3.connect(home / "state.db")) as conn:
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name IN ('maintenance_sources_v1','maintenance_outbox_v1')").fetchone()[0] == 0


def test_plain_dict_and_wrong_home_do_not_bind(tmp_path):
    home, db = _store(tmp_path)
    db.append_message("s1", "user", "first", _maintenance_source={"authority": "cli-local"})
    assert pending_sources(home) == []
    assert read_committed_source(tmp_path / "wrong", "0" * 64) is None


def test_one_local_receipt_cannot_bind_across_homes(tmp_path):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    first_home, first_db = _store(tmp_path / 'a')
    second_home, second_db = _store(tmp_path / 'b')
    receipt = mint_local_input('same local occurrence', home=first_home, session_id="s1")
    first_db.append_message('s1', 'user', receipt.raw, _maintenance_source=receipt)
    second_db.append_message('s1', 'user', receipt.raw, _maintenance_source=receipt)
    assert len(pending_sources(first_home)) == 1
    assert pending_sources(second_home) == []


def test_local_receipt_is_scoped_to_receiving_session_before_first_commit(tmp_path):
    home, db = _store(tmp_path)
    db.create_session("s2", "cli", session_key="cli:other")
    receipt = mint_local_input("same original", home=home, session_id="s1")
    db.append_message("s2", "user", receipt.raw, _maintenance_source=receipt)
    assert pending_sources(home) == []
    db.append_message("s1", "user", receipt.raw, _maintenance_source=receipt)
    source_id = pending_sources(home)[0]
    db.append_message("s2", "user", receipt.raw, _maintenance_source=receipt)
    assert pending_sources(home) == [source_id]
    assert read_committed_source(home, source_id)["session_id"] == "s1"


def test_cli_stage_carries_original_input_to_real_batch_commit(tmp_path):
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    from agent.session_persistence import _db_flush_row
    from types import SimpleNamespace
    home, db = _store(tmp_path)
    cli = CLIChatTurnMixin()
    cli.conversation_history = []
    agent = SimpleNamespace(_session_messages=[], _session_persist_lock=None)
    receipt = mint_local_input("original @file:note", home=home, session_id="s1")
    with local_input(receipt):
        cli._chat_stage_user_message(agent, "expanded text")
    staged = agent._pending_cli_user_message
    assert staged["_maintenance_source"].raw == "original @file:note"
    row = _db_flush_row(agent, staged, True)
    db.append_messages_batch("s1", [row])
    proof = committed_source_for_receipt(home, "s1", receipt)
    assert proof["raw_text"] == "original @file:note"
    assert proof["extraction_text"] == "expanded text"
    assert proof["message_row_id"] == row["_row_id"]


def test_explicit_revocation_is_separate_from_inactive_transcript(tmp_path):
    home, db = _store(tmp_path)
    receipt = mint_local_input("remember this", home=home, session_id="s1")
    row_id = db.append_message("s1", "user", receipt.raw, _maintenance_source=receipt)
    source = committed_source_for_receipt(home, "s1", receipt)
    with closing(sqlite3.connect(home / "state.db")) as conn:
        conn.execute("UPDATE messages SET active=0 WHERE id=?", (row_id,))
        conn.commit()
    assert read_committed_source(home, source["source_id"])
    revoke_committed_source(home, source["source_id"])
    assert read_committed_source(home, source["source_id"]) is None


def test_transcript_clone_reuses_source_without_new_outbox(tmp_path):
    home, db = _store(tmp_path)
    receipt = mint_local_input("same original", home=home, session_id="s1")
    first = db.append_message("s1", "user", receipt.raw, _maintenance_source=receipt)
    source_id = pending_sources(home)[0]
    db.create_session("s2", "cli", session_key="cli:clone")
    db.append_message("s2", "user", receipt.raw, _maintenance_source=receipt)
    assert pending_sources(home) == [source_id]
    assert read_committed_source(home, source_id)["message_row_id"] == first


def test_edited_original_creates_distinct_revision(tmp_path):
    home, db = _store(tmp_path)
    original = mint_local_input("old", home=home, session_id="s1")
    db.append_message("s1", "user", "old", _maintenance_source=original)
    edited = revise_original_input(original, "new")
    assert edited.occurrence == original.occurrence and edited.revision == "2"
    db.append_message("s1", "user", "new", _maintenance_source=edited)
    assert len(pending_sources(home)) == 2
    assert committed_source_for_receipt(home, "s1", original)["raw_text"] == "old"


def test_interactive_cli_input_boundary_mints_and_commits(tmp_path, monkeypatch):
    from hermes_cli.cli_tui_runtime_mixin import CLITuiRuntimeMixin
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    from hermes_maintenance_source import current_admitted_source
    from agent.session_persistence import _db_flush_row
    from types import SimpleNamespace
    home, db = _store(tmp_path)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    agent = SimpleNamespace(_session_messages=[], _session_persist_lock=None)

    class LocalCLI(CLITuiRuntimeMixin, CLIChatTurnMixin):
        session_id = "s1"
        _pending_resume_sessions = None
        _app = SimpleNamespace(invalidate=lambda: None)
        conversation_history = []
        def _tui_unwrap_input(self, value): return value, False, False
        def _typed_voice_stop(self, value): return False
        def handle_bang_shell(self, value): return False
        def _print_user_message_preview(self, value): pass
        def _turn_summary_begin(self): pass
        def _tui_after_turn(self): pass
        def chat(self, message, **kwargs):
            assert current_admitted_source().raw == message
            self._chat_stage_user_message(agent, message)
            db.append_messages_batch("s1", [_db_flush_row(agent, agent._pending_cli_user_message, True)])

    LocalCLI()._tui_process_one_input("I prefer SQLite")
    source = read_committed_source(home, pending_sources(home)[0])
    assert source["authority"] == "cli-local" and source["raw_text"] == "I prefer SQLite"


def test_host_question_delivery_is_persistent_and_fails_closed_on_ambiguity(tmp_path):
    import time
    home, db = _store(tmp_path)
    key = "cli:host-question"
    binding = {"task_id": "task-1", "item_id": "item-1", "proposal_revision": "rev-1"}
    expires = time.time() + 3600
    begin_host_question_delivery(home, "Q-1", "owner", key, "Which?", None,
                                 binding, expires_at=expires)
    assert host_question_delivery_for_session(home, key) == (None, 0)
    record_host_question_delivery(home, "Q-1", channel="cli", session_key=key,
                                  message_ref="local-1")
    row, count = host_question_delivery_for_session(home, key)
    assert count == 1 and row["prompt_id"] == "Q-1"
    assert record_host_question_delivery(home, "Q-1", channel="cli", session_key=key,
                                         message_ref="local-1")["message_ref"] == "local-1"

    begin_host_question_delivery(home, "Q-2", "owner", key, "Which?", None,
                                 binding, expires_at=expires)
    record_host_question_delivery(home, "Q-2", channel="cli", session_key=key,
                                  message_ref="local-2")
    assert host_question_delivery_for_session(home, key) == (None, 2)
    close_host_question_delivery(home, "Q-2", "CANCELLED")
    row, count = host_question_delivery_for_session(home, key)
    assert count == 1 and row["prompt_id"] == "Q-1"
    close_host_question_delivery(home, "Q-1", "RESOLVED")
    assert host_question_delivery_for_session(home, key) == (None, 0)
    db.close()
