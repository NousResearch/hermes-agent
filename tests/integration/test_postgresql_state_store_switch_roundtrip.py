"""SQLite↔PostgreSQL backend switch round-trip semantics (real PG18 evidence).

Covers the full switch cycle against one profile home:

    SQLite era → fail-closed switch → sanctioned offline import →
    PostgreSQL era (no SQLite fallback) → switch back to SQLite →
    both backends still own their own semantics.

Every step exercises the real selection seam (``resolve_state_store_config``
and ``open_state_store``) and the real disposable PG18 target from
``postgresql_test_target``; no store is mocked or substituted.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import state_store
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from postgresql_state_store_sqlite_import import (
    SQLitePostgreSQLImportError,
    SQLitePostgreSQLSandboxImporter,
)
from state_store import (
    MessageRecord,
    StateStoreConfigurationError,
    open_state_store,
    resolve_state_store_config,
)
from state_store_postgresql import PostgreSQLStateStore
from state_store_runtime_readiness import trap_state_db_opens
from tests.integration.postgresql_test_target import TEST_DSN, OwnedPostgreSQLTestTarget

_DSN_ENV = "HERMES_STATE_STORE_ROUNDTRIP_DSN"
_SQLITE_CONFIG: dict[str, Any] = {"state_store": {"backend": "sqlite"}}
_PG_SETTINGS = {"dsn_env": _DSN_ENV, "connect_timeout_seconds": 5, "pool_max_size": 2}


def _pg_config() -> dict[str, Any]:
    return {"state_store": {"backend": "postgresql", "postgresql": dict(_PG_SETTINGS)}}


pytestmark = pytest.mark.integration


@pytest.fixture
def switch_home(tmp_path, monkeypatch, postgresql_test_target: OwnedPostgreSQLTestTarget):
    home = tmp_path / "roundtrip-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "state_store:\n  backend: sqlite\n", encoding="utf-8"
    )
    monkeypatch.setenv(_DSN_ENV, TEST_DSN)
    # Route the production tenant-schema resolution to this test's owned,
    # marker-verified disposable schema, exactly as a profile home would.
    monkeypatch.setattr(
        state_store, "postgresql_tenant_schema", lambda *_a, **_k: postgresql_test_target.schema
    )
    token = set_hermes_home_override(str(home))
    opened: list[Any] = []
    try:
        yield home, postgresql_test_target, opened
    finally:
        for store in opened:
            try:
                store.close()
            except Exception:
                pass
        reset_hermes_home_override(token)


def _open(config: dict[str, Any], opened: list[Any], *, db_path: Path | None = None):
    store = open_state_store(config, db_path=db_path)
    opened.append(store)
    return store


def _record_projection(records: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    """Backend-independent transcript semantics: identity + payload + plumbing."""
    fields = (
        "id", "role", "content", "tool_call_id", "tool_calls", "tool_name",
        "platform_message_id", "finish_reason", "observed",
    )
    return [tuple(row.get(field) for field in fields) for row in records]


def _session_projection(session: dict[str, Any]) -> dict[str, Any]:
    stable = (
        "id", "source", "title", "title_source", "system_prompt", "ended_at",
        "end_reason", "hidden", "archived", "pinned",
    )
    return {key: session.get(key) for key in stable}


def _populate_sqlite_era(store: Any, session_id: str) -> None:
    store.ensure_session(session_id, "cli", metadata={"model": "roundtrip-model"})
    store.set_system_prompt(session_id, "stable system prompt")
    store.set_session_title(session_id, "Round-trip title")
    store.append_message_record(session_id, MessageRecord(
        role="user",
        content="remember switch semantics",
        platform_message_id="single-identity",
        timestamp=1_789_000_000,
        display_metadata={"origin": "single"},
    ))
    store.append_message(session_id, role="assistant", content="persisted answer")
    store.append_message_record(session_id, MessageRecord(
        role="tool", content=None, tool_call_id="null-content", tool_name="probe",
    ))
    store.update_token_counts(
        session_id, input_tokens=3, output_tokens=5, model="roundtrip-model",
        api_call_count=1, source="roundtrip-fixture",
    )
    assert store.flush_token_counts(timeout=5.0) is True
    # Structured-content scratch session: real SQLite rows whose content is
    # stored with the internal NUL-prefixed JSON marker encoding.
    scratch = session_id + "-structured"
    store.ensure_session(scratch, "cli")
    store.append_message(scratch, role="user", content={"text": "structured"})
    assert store.flush_token_counts(timeout=5.0) is True


def _tenant_usage_rows(target: OwnedPostgreSQLTestTarget) -> list[tuple[Any, ...]]:
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            f'SELECT model, api_call_count, input_tokens, output_tokens FROM "{target.schema}".session_model_usage ORDER BY model'
        )
        return list(cursor.fetchall())


def test_sqlite_to_pg_to_sqlite_round_trip_preserves_and_isolates_semantics(
    switch_home, tmp_path
):
    home, target, opened = switch_home
    session_id = "20260921_000001_roundtrip"

    # ---- Phase A: SQLite era under the profile home --------------------
    sqlite_store = _open(_SQLITE_CONFIG, opened, db_path=home / "state.db")
    assert isinstance(sqlite_store, state_store.SqliteStateStore)
    _populate_sqlite_era(sqlite_store, session_id)
    sqlite_records = _record_projection(sqlite_store.get_message_records(session_id))
    sqlite_session = _session_projection(sqlite_store.get_session(session_id) or {})
    assert len(sqlite_records) == 3
    sqlite_store.close()
    assert (home / "state.db").is_file()
    state_db_bytes = (home / "state.db").read_bytes()

    # ---- Phase B: switch selection is fail-closed, never silently SQLite
    with pytest.raises(StateStoreConfigurationError, match=_DSN_ENV):
        resolve_state_store_config(_pg_config(), secret_lookup=lambda _name: None)
    resolved = resolve_state_store_config(_pg_config())
    assert resolved.backend == "postgresql" and resolved.postgresql is not None

    # ---- Phase C: sanctioned offline migration SQLite → owned PG tenant.
    # The importer refuses the *active* profile state.db by design; the switch
    # flow supplies a disposable snapshot copy, exactly like production.
    disposable_source = tmp_path / "disposable-roundtrip-source.db"
    shutil.copy2(home / "state.db", disposable_source)
    settings = state_store.PostgreSQLStateStoreConfig(**_PG_SETTINGS)
    importer = SQLitePostgreSQLSandboxImporter(
        settings, TEST_DSN, schema=target.schema, owned_target=target
    )
    # Real-fixture evidence: the production SQLite schema currently carries
    # objects beyond the importer's approved map (session_topics).  The switch
    # must fail closed on them rather than silently dropping or mis-mapping.
    with pytest.raises(SQLitePostgreSQLImportError, match="without an approved map"):
        importer.import_source(disposable_source, snapshot_root=tmp_path)
    with sqlite3.connect(disposable_source) as connection:
        unmapped = sorted(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            if row[0] == "session_topics"
        )
    assert unmapped == ["session_topics"], (
        "if this grows, the approved import map changed; update the round-trip contract"
    )
    with sqlite3.connect(disposable_source) as connection:
        connection.execute("DROP TABLE session_topics")
    # Second real boundary: SQLite bookkeeping rows (schema_version/state_meta)
    # that every production state.db populates at init are also rejected —
    # populated non-migrated objects never ride along silently.
    with pytest.raises(
        SQLitePostgreSQLImportError, match="populated non-migrated objects"
    ):
        importer.import_source(disposable_source, snapshot_root=tmp_path)
    with sqlite3.connect(disposable_source) as connection:
        connection.execute("DELETE FROM schema_version")
        connection.execute("DELETE FROM state_meta")
    # Third real boundary: sessions written by the real SQLite path carry
    # populated denormalized fields (message_count/tool_call_count) that are
    # outside the importer's bounded slice; those are rejected too.
    with pytest.raises(
        SQLitePostgreSQLImportError, match="unimplemented canonical field"
    ):
        importer.import_source(disposable_source, snapshot_root=tmp_path)
    from postgresql_state_store_sqlite_import import (
        _UNSUPPORTED_MESSAGE_COLUMNS,
        _UNSUPPORTED_SESSION_COLUMNS,
    )

    with sqlite3.connect(disposable_source) as connection:
        for column, default in _UNSUPPORTED_SESSION_COLUMNS.items():
            connection.execute(
                f'UPDATE sessions SET "{column}" = ?', (default,)
            )
        for column, default in _UNSUPPORTED_MESSAGE_COLUMNS.items():
            connection.execute(
                f'UPDATE messages SET "{column}" = ?', (default,)
            )
    # Fourth real boundary: genuine structured content rows (SQLite's internal
    # NUL-prefixed JSON encoding) can never live in PostgreSQL text fields.
    with pytest.raises(SQLitePostgreSQLImportError, match="NUL"):
        importer.import_source(disposable_source, snapshot_root=tmp_path)
    # Drop the scratch session and its rows; the bounded slice carries on.
    with sqlite3.connect(disposable_source) as connection:
        connection.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id + "-structured",)
        )
        connection.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id + "-structured",)
        )
    # The rejected attempts left failure manifests bound to earlier source
    # fingerprints; resume across a mutated source is refused by design, so
    # take a fresh owned target for the definitive migration.
    target.reset()
    evidence_path = tmp_path / "roundtrip-import-evidence.json"
    result = importer.import_source(
        disposable_source, snapshot_root=tmp_path, evidence_path=evidence_path
    )
    assert result.status == "complete"
    assert result.object_counts["sessions"] == 1
    assert result.object_counts["messages"] == 3
    assert json.loads(evidence_path.read_text())["status"] == "complete"

    # ---- Phase D: PostgreSQL era — full semantics parity, zero SQLite --
    with trap_state_db_opens(home) as pg_phase_opens:
        pg_store = _open(_pg_config(), opened)
        try:
            assert isinstance(pg_store, PostgreSQLStateStore)
            assert _record_projection(pg_store.get_message_records(session_id)) == sqlite_records
            assert _session_projection(pg_store.get_session(session_id) or {}) == sqlite_session
            assert pg_store.get_system_prompt(session_id) == "stable system prompt"
            # Divergence on the PG side of the switch.
            pg_store.append_message(session_id, role="user", content="pg-era only")
            pg_store.update_token_counts(
                session_id, input_tokens=7, model="roundtrip-model",
                api_call_count=1, source="pg-era",
            )
        finally:
            pg_store.close()
    assert pg_phase_opens == [], "selected PostgreSQL must never open the profile state.db"

    usage_after_pg_era = _tenant_usage_rows(target)
    assert usage_after_pg_era == [("roundtrip-model", 2, 10, 5)]

    # ---- Phase E: switch back to SQLite — isolation in both directions --
    # (No trap here by design: the SQLite selection legitimately opens state.db.)
    back_store = _open(_SQLITE_CONFIG, opened, db_path=home / "state.db")
    assert isinstance(back_store, state_store.SqliteStateStore)
    assert _record_projection(back_store.get_message_records(session_id)) == sqlite_records
    assert "pg-era only" not in [
        row.get("content") for row in back_store.get_message_records(session_id)
    ]
    back_store.append_message(session_id, role="user", content="sqlite-era only")
    back_store.close()
    # PG tenant is untouched by the SQLite-era write.
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            f'SELECT content FROM "{target.schema}".messages WHERE session_id = %s ORDER BY id',
            (session_id,),
        )
        pg_contents = [row[0] for row in cursor.fetchall()]
    assert "sqlite-era only" not in pg_contents
    assert "pg-era only" in pg_contents
    assert _tenant_usage_rows(target) == usage_after_pg_era

    # ---- Phase F: PG tenant still readable through the selection seam ---
    pg_again = _open(_pg_config(), opened)
    try:
        final = _record_projection(pg_again.get_message_records(session_id))
        assert final[:3] == sqlite_records
        assert final[3][1] == "user" and final[3][2] == "pg-era only"
    finally:
        pg_again.close()

    # The SQLite file itself was only legitimately re-opened by the SQLite era;
    # its pre-switch prefix is still present byte-for-byte at the head of the
    # current file (appends never rewrote history).
    current = (home / "state.db").read_bytes()
    assert state_db_bytes[:4096] == current[:4096]


def test_pg_selection_never_creates_or_opens_state_db(switch_home):
    home, _target, opened = switch_home
    import os

    assert not (home / "state.db").exists()
    # Missing secret: fail closed before any file or connection is touched.
    with pytest.raises(StateStoreConfigurationError, match=_DSN_ENV):
        open_state_store(_pg_config(), secret_lookup=lambda _name: None)
    assert not (home / "state.db").exists()
    # Unreachable server: the store fails with a real connection error rather
    # than degrading to SQLite (the PG pool connects eagerly on open).
    os.environ[_DSN_ENV] = "postgresql://hermes_state_store_test@127.0.0.1:1/unreachable"
    try:
        with trap_state_db_opens(home) as opens:
            with pytest.raises(Exception, match="refused|timed out|[Cc]onnection"):
                store = open_state_store(_pg_config())
                store.ensure_session("never-created", "cli")
        assert opens == []
    finally:
        os.environ[_DSN_ENV] = TEST_DSN
    assert not (home / "state.db").exists()


def test_sqlite_selection_ignores_postgresql_section(switch_home):
    """Switching back to SQLite must not validate (or leak) the stale PG block."""
    home, _target, opened = switch_home
    config = {
        "state_store": {
            "backend": "sqlite",
            "postgresql": {"dsn_env": "ROUNDTRIP_NEVER_SET_ENV"},
        }
    }
    resolved = resolve_state_store_config(config)
    assert resolved.backend == "sqlite"
    store = _open(config, opened, db_path=home / "state.db")
    try:
        store.ensure_session("plain-sqlite", "cli")
        store.append_message("plain-sqlite", role="user", content="plain")
        assert len(store.get_messages("plain-sqlite")) == 1
        # The SQLite file is a real SQLite database.
        with sqlite3.connect(home / "state.db") as connection:
            assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
    finally:
        store.close()
