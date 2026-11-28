"""Replacing or deleting a session's prompt reaps only the prompts it can orphan.

``_delete_unreferenced_system_prompts`` used to run the whole-table ``NOT EXISTS`` form from every
call site, so each prompt write and each session delete read a leaf page per ``system_prompts`` row
under the write lock. The call sites now read the hashes the session references before the change
and pass them in, so the reap probes those rows by primary key. The behaviour is unchanged: a
prompt nothing else references is deleted, a shared prompt is kept.

The last test pins the reason for the change: the statement the reap issues must be a primary-key
search, not a scan of the whole table.
"""
import hashlib

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.sessions import (
    create_session,
    update_session_tool_names,
    update_system_prompt,
)


def _hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _prompt_rows(db) -> set:
    return {row[0] for row in db._conn.execute("SELECT hash FROM system_prompts").fetchall()}


def _session_prompt_hash(db, session_id: str):
    row = db._conn.execute(
        "SELECT system_prompt_hash FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    return row[0] if row else None


def test_a_replaced_prompt_no_other_session_uses_is_deleted(tmp_path):
    db = StoreHolder(db_path=tmp_path / "state.db")
    try:
        create_session(db, "s", source="cli")
        update_system_prompt(db, "s", "first prompt")
        assert _hash("first prompt") in _prompt_rows(db)

        update_system_prompt(db, "s", "second prompt")

        rows = _prompt_rows(db)
        assert _hash("second prompt") in rows
        assert _hash("first prompt") not in rows, "the replaced prompt had no other referrer"
        assert _session_prompt_hash(db, "s") == _hash("second prompt")
    finally:
        db.close()


def test_a_prompt_another_session_still_uses_is_kept(tmp_path):
    db = StoreHolder(db_path=tmp_path / "state.db")
    try:
        create_session(db, "a", source="cli")
        create_session(db, "b", source="cli")
        update_system_prompt(db, "a", "shared prompt")
        update_system_prompt(db, "b", "shared prompt")

        update_system_prompt(db, "a", "a's own prompt")

        assert _hash("shared prompt") in _prompt_rows(db), "b still references it"
        assert _hash("a's own prompt") in _prompt_rows(db)
    finally:
        db.close()


def test_a_tool_names_pin_a_deleted_session_held_is_reaped(tmp_path):
    db = StoreHolder(db_path=tmp_path / "state.db")
    try:
        create_session(db, "a", source="cli")
        create_session(db, "b", source="cli")
        update_session_tool_names(db, "a", ["read_file"])
        update_session_tool_names(db, "b", ["read_file"])
        pin = _hash('["read_file"]')
        assert pin in _prompt_rows(db)

        from persistence.sqlite.conversation._store.sessions import delete_session

        delete_session(db, "a", media_root=str(tmp_path / "media"))

        assert pin in _prompt_rows(db), "b still references the tools pin"
    finally:
        db.close()


def test_the_reap_searches_the_prompt_table_by_key(tmp_path):
    """The reap's DELETE must be a primary-key search, not a whole-table scan."""
    db = StoreHolder(db_path=tmp_path / "state.db")
    statements = []
    try:
        create_session(db, "s", source="cli")
        update_system_prompt(db, "s", "first prompt")
        db._conn.set_trace_callback(statements.append)
        update_system_prompt(db, "s", "second prompt")
        db._conn.set_trace_callback(None)
    finally:
        db.close()

    deletes = [s for s in statements if "system_prompts" in s and s.lstrip().upper().startswith("DELETE")]
    assert deletes, "the reap issued no DELETE against system_prompts"
    for statement in deletes:
        plan = " | ".join(
            row[3] for row in db_plan(statement)
        )
        assert "SCAN system_prompts" not in plan, (
            f"the reap scanned the whole prompt table: {plan}"
        )
        assert "SEARCH system_prompts" in plan, (
            f"the reap did not probe the prompt table by key: {plan}"
        )


def db_plan(statement: str):
    """EXPLAIN QUERY PLAN for *statement* on a throwaway in-memory store."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(
            "CREATE TABLE system_prompts (hash TEXT PRIMARY KEY, prompt TEXT NOT NULL);"
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, system_prompt_hash TEXT, tool_names TEXT);"
        )
        return conn.execute("EXPLAIN QUERY PLAN " + statement).fetchall()
    finally:
        conn.close()
