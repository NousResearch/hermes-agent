"""Content-addressed session system prompts on the SQLite backend: one stored copy per distinct prompt, hydrated for every reader and reclaimed with its last reference."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager

import pytest

from hermes_cli.session_export import to_legacy_export_dict
from persistence.housekeeping.maintenance import PruneCriteria
from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence.sqlite.conversation._store.gateway import list_gateway_sessions
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.portability import _get_session_rich_row
from persistence.sqlite.conversation._store.sessions import (
    create_session,
    end_session,
    get_session,
    list_sessions_rich,
    search_sessions,
    update_session_runtime_lock,
    update_session_tool_names,
    update_system_prompt,
)
from persistence.sqlite.conversation._store.usage import update_session_billing_route
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder, schema_version

MEDIA_ROOT = "/nonexistent/native-inputs"


def _open(home):
    home.mkdir(parents=True, exist_ok=True)
    return open_persistence(profile_scope(home=home))


@pytest.fixture()
def store(tmp_path):
    with _open(tmp_path) as handle:
        yield handle


@pytest.fixture()
def db(store):
    return raw_holder(store)


def _prompt_count(db) -> int:
    return int(db._read_one("SELECT COUNT(*) FROM system_prompts")[0])


def test_prompt_snapshots_are_deduplicated_and_hydrated_for_readers(store, db):
    prompt = "You are Hermes.\n" + ("Follow the profile policy.\n" * 5)
    create_session(
        db,
        "s1",
        "telegram",
        session_key="agent:main:telegram:dm:c1",
        chat_id="c1",
        chat_type="dm",
        system_prompt=prompt,
    )
    create_session(db, "s2", "cli", system_prompt=prompt)
    store.coordination.request_handoff(session_id="s1", platform="telegram")

    stored = db._read_all("SELECT hash, prompt FROM system_prompts")
    assert len(stored) == 1
    assert stored[0]["prompt"] == prompt
    raw_sessions = db._read_all(
        "SELECT system_prompt, system_prompt_hash FROM sessions ORDER BY id"
    )
    assert [row["system_prompt"] for row in raw_sessions] == [None, None]
    assert {row["system_prompt_hash"] for row in raw_sessions} == {
        stored[0]["hash"]
    }

    assert get_session(db, "s1")["system_prompt"] == prompt
    assert list_sessions_rich(db)[0]["system_prompt"] == prompt
    assert search_sessions(db)[0]["system_prompt"] == prompt
    assert store.maintenance.export_session("s1").session.system_prompt == prompt
    assert list_gateway_sessions(db)[0]["system_prompt"] == prompt
    # A pending handoff carries what the handoff watcher reads (session, platform, title), not the prompt.
    assert [handoff.session_id for handoff in store.coordination.list_pending_handoffs()] == ["s1"]


def test_route_changes_keep_the_prompt_and_replacement_collects_orphans(db):
    shared_prompt = "Model: x-ai/grok-4.5\nProvider: nous"
    create_session(
        db,
        "s1",
        "hermes_browser",
        model="x-ai/grok-4.5",
        model_config={"_branched_from": "parent"},
        system_prompt=shared_prompt,
    )
    create_session(db, "s2", "cli", system_prompt=shared_prompt)

    # A route change keeps the stored snapshot: it is the session's cache prefix, and the next
    # turn's runtime-identity check rebuilds only when the footer it embeds is actually stale.
    update_session_runtime_lock(
        db,
        "s1",
        model="anthropic/claude-opus-4.8",
        provider="anthropic",
        confirmed=True,
    )
    s1 = get_session(db, "s1")
    assert s1["system_prompt"] == shared_prompt
    assert json.loads(s1["model_config"])["_branched_from"] == "parent"
    assert get_session(db, "s2")["system_prompt"] == shared_prompt
    assert _prompt_count(db) == 1

    update_session_billing_route(
        db,
        "s2",
        provider="openrouter",
        base_url="https://example.test/v1",
    )
    assert get_session(db, "s2")["system_prompt"] == shared_prompt
    assert _prompt_count(db) == 1

    # Replacing the snapshot (a rebuild) is what orphans the previous bytes.
    update_system_prompt(db, "s1", "replacement")
    assert get_session(db, "s1")["system_prompt"] == "replacement"
    assert _prompt_count(db) == 2  # "replacement" + the still-referenced shared prompt
    update_system_prompt(db, "s2", None)
    assert get_session(db, "s2")["system_prompt"] is None
    assert _prompt_count(db) == 1


def test_existing_session_enrichment_does_not_leak_unused_prompt(db):
    create_session(db, "s1", "cli", system_prompt="original prompt")
    create_session(db, "s1", "cli", system_prompt="unused prompt")

    prompts = [row["prompt"] for row in db._read_all("SELECT prompt FROM system_prompts")]
    assert prompts == ["original prompt"]
    assert get_session(db, "s1")["system_prompt"] == "original prompt"


def test_every_session_deletion_path_reclaims_final_prompt_reference(store, db):
    def seed(session_id: str, *, source: str = "cli") -> None:
        create_session(
            db,
            session_id,
            source,
            system_prompt=f"unique prompt for {session_id}",
        )
        assert _prompt_count(db) == 1

    seed("single-empty")
    assert store.sessions.delete_if_empty("single-empty", media_root=MEDIA_ROOT) == ("single-empty",)
    assert _prompt_count(db) == 0

    seed("bulk")
    assert store.sessions.delete_many(["bulk"], media_root=MEDIA_ROOT).deleted == 1
    assert _prompt_count(db) == 0

    seed("ended-empty")
    end_session(db, "ended-empty", "user_exit")
    assert store.sessions.delete_empty(media_root=MEDIA_ROOT) == ("ended-empty",)
    assert _prompt_count(db) == 0

    seed("pruned")
    end_session(db, "pruned", "user_exit")
    assert store.maintenance.prune_sessions(
        older_than_days=None,
        criteria=PruneCriteria(started_before=time.time() + 1),
        media_root=MEDIA_ROOT,
    ).count == 1
    assert _prompt_count(db) == 0

    seed("ghost", source="tui")
    end_session(db, "ghost", "user_exit")
    db._execute_write(lambda conn: conn.execute("UPDATE sessions SET started_at = 0 WHERE id = 'ghost'"))
    assert store.maintenance.prune_empty_ghost_sessions(media_root=MEDIA_ROOT) == ("ghost",)
    assert _prompt_count(db) == 0


def test_deleting_one_shared_session_preserves_prompt_until_final_reference(store, db):
    prompt = "shared deletion prompt"
    create_session(db, "s1", "cli", system_prompt=prompt)
    create_session(db, "s2", "cli", system_prompt=prompt)

    assert store.sessions.delete("s1", media_root=MEDIA_ROOT) == ("s1",)
    assert _prompt_count(db) == 1
    assert get_session(db, "s2")["system_prompt"] == prompt

    assert store.sessions.delete("s2", media_root=MEDIA_ROOT) == ("s2",)
    assert _prompt_count(db) == 0


def test_compression_child_uses_content_addressed_prompt(store, db):
    prompt = "compressed child prompt"
    create_session(db, "parent", "webui")
    append_message(db, "parent", "user", "original")
    assert store.coordination.try_acquire_compression_lock(session_id="parent", holder="holder", ttl_seconds=60)

    publish_compression_child(
        db,
        parent_session_id="parent",
        child_session_id="child",
        source="webui",
        system_prompt=prompt,
        messages=[{"role": "user", "content": "summary"}],
        compression_lock_holder="holder",
    )

    raw = db._read_one(
        "SELECT system_prompt, system_prompt_hash FROM sessions WHERE id = 'child'"
    )
    assert raw["system_prompt"] is None
    assert raw["system_prompt_hash"] is not None
    assert get_session(db, "child")["system_prompt"] == prompt
    assert _prompt_count(db) == 1


def test_compression_child_continues_the_parents_tool_pin(store, db):
    pin = {"version": "sha", "tools": [{"type": "function", "function": {"name": "read_file"}}]}
    create_session(db, "parent", "telegram")
    update_session_tool_names(db, "parent", pin)
    append_message(db, "parent", "user", "original")
    assert store.coordination.try_acquire_compression_lock(session_id="parent", holder="holder", ttl_seconds=60)

    publish_compression_child(
        db, parent_session_id="parent", child_session_id="child", source="telegram", system_prompt="p",
        messages=[{"role": "user", "content": "summary"}], compression_lock_holder="holder")

    assert json.loads(get_session(db, "child")["tool_names"]) == pin


def test_imported_prompts_are_deduplicated(tmp_path):
    prompt = "shared imported prompt"
    with _open(tmp_path / "source") as source:
        create_session(raw_holder(source), "s1", "cli", system_prompt=prompt)
        create_session(raw_holder(source), "s2", "telegram", system_prompt=prompt)
        exported = [to_legacy_export_dict(source.maintenance.export_session(sid)) for sid in ("s1", "s2")]

    with _open(tmp_path / "target") as target:
        result = target.maintenance.import_sessions(exported)
        db = raw_holder(target)
        assert result.ok is True
        assert result.imported == 2
        assert _prompt_count(db) == 1
        raw = db._read_all(
            "SELECT system_prompt, system_prompt_hash FROM sessions ORDER BY id"
        )
        assert [row["system_prompt"] for row in raw] == [None, None]
        assert len({row["system_prompt_hash"] for row in raw}) == 1
        assert get_session(db, "s1")["system_prompt"] == prompt
        assert get_session(db, "s2")["system_prompt"] == prompt


def test_v24_inline_prompts_migrate_once_to_content_addressed_storage(tmp_path):
    home = tmp_path / "legacy-prompts"
    legacy_prompt = "Legacy system prompt\n" + ("same policy\n" * 20)

    with _open(home) as store:
        db = raw_holder(store)
        create_session(db, "s1", "cli")
        create_session(db, "s2", "telegram")

        def inline_v24(conn):
            conn.execute(
                "UPDATE sessions SET system_prompt = ?, system_prompt_hash = NULL",
                (legacy_prompt,),
            )
            conn.execute("UPDATE schema_version SET version = 24")
        db._execute_write(inline_v24)

    with _open(home) as migrated:
        db = raw_holder(migrated)
        assert get_session(db, "s1")["system_prompt"] == legacy_prompt
        assert get_session(db, "s2")["system_prompt"] == legacy_prompt
        assert _prompt_count(db) == 1
        raw_sessions = db._read_all(
            "SELECT system_prompt, system_prompt_hash FROM sessions ORDER BY id"
        )
        assert [row["system_prompt"] for row in raw_sessions] == [None, None]
        assert len({row["system_prompt_hash"] for row in raw_sessions}) == 1
        assert db._read_one(
            "SELECT version FROM schema_version LIMIT 1"
        )[0] == schema_version()


def test_compact_rows_omit_hash_and_never_read_prompt_blob(db, monkeypatch):
    create_session(db, "s1", "cli", system_prompt="never materialize me")

    def deny_prompt_reads(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_READ and table == "system_prompts":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    read_ctx = db._read_ctx

    @contextmanager
    def denying_read_ctx(*args, **kwargs):
        with read_ctx(*args, **kwargs) as conn:
            conn.set_authorizer(deny_prompt_reads)
            try:
                yield conn
            finally:
                conn.set_authorizer(None)

    monkeypatch.setattr(db, "_read_ctx", denying_read_ctx)
    db._conn.set_authorizer(deny_prompt_reads)
    try:
        rows = list_sessions_rich(
            db,
            compact_rows=True,
            order_by_last_active=True,
        )
        rich = _get_session_rich_row(db, "s1", compact_rows=True)
    finally:
        db._conn.set_authorizer(None)

    assert rows[0]["id"] == "s1"
    assert rich["id"] == "s1"
    assert "system_prompt" not in rows[0]
    assert "system_prompt_hash" not in rows[0]
    assert "system_prompt" not in rich
    assert "system_prompt_hash" not in rich
