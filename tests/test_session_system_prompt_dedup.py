"""Behavior coverage for content-addressed session system prompts."""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _prompt_count(db: SessionDB) -> int:
    return int(
        db._conn.execute("SELECT COUNT(*) FROM system_prompts").fetchone()[0]
    )


def test_prompt_snapshots_are_deduplicated_and_hydrated_for_readers(db):
    prompt = "You are Hermes.\n" + ("Follow the profile policy.\n" * 5)
    db.create_session(
        "s1",
        "telegram",
        session_key="agent:main:telegram:dm:c1",
        chat_id="c1",
        chat_type="dm",
        system_prompt=prompt,
    )
    db.create_session("s2", "cli", system_prompt=prompt)
    db.request_handoff("s1", "telegram")

    stored = db._conn.execute(
        "SELECT hash, prompt FROM system_prompts"
    ).fetchall()
    assert len(stored) == 1
    assert stored[0]["prompt"] == prompt
    raw_sessions = db._conn.execute(
        "SELECT system_prompt, system_prompt_hash FROM sessions ORDER BY id"
    ).fetchall()
    assert [row["system_prompt"] for row in raw_sessions] == [None, None]
    assert {row["system_prompt_hash"] for row in raw_sessions} == {
        stored[0]["hash"]
    }

    assert db.get_session("s1")["system_prompt"] == prompt
    assert db.list_sessions_rich()[0]["system_prompt"] == prompt
    assert db.search_sessions()[0]["system_prompt"] == prompt
    assert db.export_session("s1")["system_prompt"] == prompt
    assert db.list_gateway_sessions()[0]["system_prompt"] == prompt
    assert db.list_pending_handoffs()[0]["system_prompt"] == prompt


def test_prompt_replacement_and_route_changes_collect_only_orphans(db):
    shared_prompt = "Model: x-ai/grok-4.5\nProvider: nous"
    db.create_session(
        "s1",
        "hermes_browser",
        model="x-ai/grok-4.5",
        model_config={"_branched_from": "parent"},
        system_prompt=shared_prompt,
    )
    db.create_session("s2", "cli", system_prompt=shared_prompt)

    db.update_session_runtime_lock(
        "s1",
        model="anthropic/claude-opus-4.8",
        provider="anthropic",
        confirmed=True,
    )
    s1 = db.get_session("s1")
    assert s1["system_prompt"] is None
    assert json.loads(s1["model_config"])["_branched_from"] == "parent"
    assert db.get_session("s2")["system_prompt"] == shared_prompt
    assert _prompt_count(db) == 1

    db.update_session_billing_route(
        "s2",
        provider="openrouter",
        base_url="https://example.test/v1",
    )
    assert db.get_session("s2")["system_prompt"] is None
    assert _prompt_count(db) == 0

    db.update_system_prompt("s2", "replacement")
    assert db.get_session("s2")["system_prompt"] == "replacement"
    db.update_system_prompt("s2", None)
    assert _prompt_count(db) == 0


def test_existing_session_enrichment_does_not_leak_unused_prompt(db):
    db.create_session("s1", "cli", system_prompt="original prompt")
    db.create_session("s1", "cli", system_prompt="unused prompt")

    prompts = [
        row["prompt"]
        for row in db._conn.execute("SELECT prompt FROM system_prompts")
    ]
    assert prompts == ["original prompt"]
    assert db.get_session("s1")["system_prompt"] == "original prompt"


def test_every_session_deletion_path_reclaims_final_prompt_reference(db):
    def seed(session_id: str, *, source: str = "cli") -> None:
        db.create_session(
            session_id,
            source,
            system_prompt=f"unique prompt for {session_id}",
        )
        assert _prompt_count(db) == 1

    seed("single-empty")
    assert db.delete_session_if_empty("single-empty") is True
    assert _prompt_count(db) == 0

    seed("bulk")
    assert db.delete_sessions(["bulk"]) == 1
    assert _prompt_count(db) == 0

    seed("ended-empty")
    db.end_session("ended-empty", "user_exit")
    assert db.delete_empty_sessions() == 1
    assert _prompt_count(db) == 0

    seed("pruned")
    db.end_session("pruned", "user_exit")
    assert db.prune_sessions(
        older_than_days=None,
        started_before=time.time() + 1,
    ) == 1
    assert _prompt_count(db) == 0

    seed("ghost", source="tui")
    db.end_session("ghost", "user_exit")
    db._conn.execute("UPDATE sessions SET started_at = 0 WHERE id = 'ghost'")
    db._conn.commit()
    assert db.prune_empty_ghost_sessions() == 1
    assert _prompt_count(db) == 0


def test_deleting_one_shared_session_preserves_prompt_until_final_reference(db):
    prompt = "shared deletion prompt"
    db.create_session("s1", "cli", system_prompt=prompt)
    db.create_session("s2", "cli", system_prompt=prompt)

    assert db.delete_session("s1") is True
    assert _prompt_count(db) == 1
    assert db.get_session("s2")["system_prompt"] == prompt

    assert db.delete_session("s2") is True
    assert _prompt_count(db) == 0


def test_compression_child_uses_content_addressed_prompt(db):
    prompt = "compressed child prompt"
    db.create_session("parent", "webui")
    db.append_message("parent", "user", "original")
    assert db.try_acquire_compression_lock("parent", "holder", ttl_seconds=60)

    db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="child",
        source="webui",
        system_prompt=prompt,
        messages=[{"role": "user", "content": "summary"}],
        compression_lock_holder="holder",
    )

    raw = db._conn.execute(
        "SELECT system_prompt, system_prompt_hash FROM sessions WHERE id = 'child'"
    ).fetchone()
    assert raw["system_prompt"] is None
    assert raw["system_prompt_hash"] is not None
    assert db.get_session("child")["system_prompt"] == prompt
    assert _prompt_count(db) == 1


def test_imported_prompts_are_deduplicated(tmp_path):
    prompt = "shared imported prompt"
    source = SessionDB(db_path=tmp_path / "source.db")
    try:
        source.create_session("s1", "cli", system_prompt=prompt)
        source.create_session("s2", "telegram", system_prompt=prompt)
        exported = [source.export_session("s1"), source.export_session("s2")]
    finally:
        source.close()

    target = SessionDB(db_path=tmp_path / "target.db")
    try:
        result = target.import_sessions(exported)
        assert result["ok"] is True
        assert result["imported"] == 2
        assert _prompt_count(target) == 1
        raw = target._conn.execute(
            "SELECT system_prompt, system_prompt_hash FROM sessions ORDER BY id"
        ).fetchall()
        assert [row["system_prompt"] for row in raw] == [None, None]
        assert len({row["system_prompt_hash"] for row in raw}) == 1
        assert target.get_session("s1")["system_prompt"] == prompt
        assert target.get_session("s2")["system_prompt"] == prompt
    finally:
        target.close()


def test_v24_inline_prompts_migrate_once_to_content_addressed_storage(tmp_path):
    db_path = tmp_path / "legacy-prompts.db"
    legacy_prompt = "Legacy system prompt\n" + ("same policy\n" * 20)

    db = SessionDB(db_path=db_path)
    db.create_session("s1", "cli")
    db.create_session("s2", "telegram")
    db._conn.execute(
        "UPDATE sessions SET system_prompt = ?, system_prompt_hash = NULL",
        (legacy_prompt,),
    )
    db._conn.execute("UPDATE schema_version SET version = 24")
    db._conn.commit()
    db.close()

    migrated = SessionDB(db_path=db_path)
    try:
        assert migrated.get_session("s1")["system_prompt"] == legacy_prompt
        assert migrated.get_session("s2")["system_prompt"] == legacy_prompt
        assert _prompt_count(migrated) == 1
        raw_sessions = migrated._conn.execute(
            "SELECT system_prompt, system_prompt_hash FROM sessions ORDER BY id"
        ).fetchall()
        assert [row["system_prompt"] for row in raw_sessions] == [None, None]
        assert len({row["system_prompt_hash"] for row in raw_sessions}) == 1
        assert migrated._conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()[0] == SCHEMA_VERSION
    finally:
        migrated.close()


def test_compact_rows_omit_hash_and_never_read_prompt_blob(db):
    db.create_session("s1", "cli", system_prompt="never materialize me")

    def deny_prompt_reads(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_READ and table == "system_prompts":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    db._conn.set_authorizer(deny_prompt_reads)
    try:
        rows = db.list_sessions_rich(
            compact_rows=True,
            order_by_last_active=True,
        )
        rich = db._get_session_rich_row("s1", compact_rows=True)
    finally:
        db._conn.set_authorizer(None)

    assert rows[0]["id"] == "s1"
    assert rich["id"] == "s1"
    assert "system_prompt" not in rows[0]
    assert "system_prompt_hash" not in rows[0]
    assert "system_prompt" not in rich
    assert "system_prompt_hash" not in rich
