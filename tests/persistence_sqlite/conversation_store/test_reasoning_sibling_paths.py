"""Regression for #125273 on storage-refactor siblings: coverage and reasoning-changing prunes."""

import copy

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.coverage import _replays_as
from persistence.sqlite.conversation._store.messages import append_message, get_messages_as_conversation
from persistence.sqlite.conversation._store.prune_rewrite import rewrite_pruned_rows, rewritten_held_row_ids
from persistence.sqlite.conversation._store.sessions import create_session
from persistence.sqlite.conversation._store.transcript_repair import transcript_row_snapshot

TEXT = "compare both branches"
NEW = "a shorter thought"


@pytest.fixture
def db(tmp_path):
    with StoreHolder(tmp_path / "state.db") as holder:
        create_session(holder, "s", source="cli")
        append_message(holder, "s", role="user", content="ask")
        append_message(holder, "s", role="assistant", content="answer", reasoning=TEXT, reasoning_content=TEXT)
        yield holder


def test_coverage_compares_the_replayed_pair_not_a_folded_null(db):
    held = get_messages_as_conversation(db, "s")[-1]
    db._execute_write(lambda conn: conn.execute("UPDATE messages SET reasoning = ? WHERE role = 'assistant'", (NEW,)))
    with db._read_ctx() as conn:
        row = conn.execute("SELECT * FROM messages WHERE role = 'assistant'").fetchone()
        assert not _replays_as(db, row, held, True)
        held["reasoning"] = NEW
        assert _replays_as(db, row, held, True)


@pytest.mark.parametrize("old_pair", [False, True])
def test_a_prune_changing_shared_text_writes_the_pair_together(db, old_pair):
    if old_pair:
        db._execute_write(lambda conn: conn.execute("UPDATE messages SET reasoning = reasoning_content WHERE role = 'assistant'"))
    original = get_messages_as_conversation(db, "s", repair_alternation=True, include_row_ids=True)[-1]
    replacement = {**copy.deepcopy(original), "reasoning": NEW, "reasoning_content": NEW}
    assert rewrite_pruned_rows(db, "s", [(original, replacement)]) == 1
    with db._read_ctx() as conn:
        row = conn.execute("SELECT * FROM messages WHERE role = 'assistant' AND active = 1").fetchone()
        assert (row["reasoning"], row["reasoning_content"]) == (None, NEW)
    assert get_messages_as_conversation(db, "s")[-1]["reasoning"] == NEW


def test_prune_and_coverage_versions_ignore_only_the_shared_representation(db):
    held = get_messages_as_conversation(db, "s", repair_alternation=True, include_row_ids=True)
    with db._read_ctx() as conn:
        row = conn.execute("SELECT * FROM messages WHERE role = 'assistant'").fetchone()
        version = transcript_row_snapshot(row)
    db._execute_write(lambda conn: conn.execute("UPDATE messages SET reasoning = reasoning_content WHERE role = 'assistant'"))
    assert rewritten_held_row_ids(db, "s", held) == []
    original = held[-1]
    replacement = {**copy.deepcopy(original), "content": "short answer"}
    assert rewrite_pruned_rows(db, "s", [(original, replacement)]) == 1
    assert rewritten_held_row_ids(db, "s", held) == [original["_row_id"]]
    assert replacement["_db_row_snapshot"] != version
