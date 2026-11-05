"""The files a store's schema setup reads beside it (the column-parse cache, the v18 ``sessions.json`` backfill)
come from the opened profile's folder, never from the ambient home; so do the maintenance tools' column probes.
``scripts/ci/check_backend_imports.py`` holds the list of the backend's remaining reads of the process or task home."""

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from hermes_constants import get_hermes_home
from persistence.sqlite import _registry
from persistence.sqlite._schema import apply_declared_schema, schema_read_probe_statements
from persistence.sqlite._schema_ddl import SCHEMA_SQL
from tests.persistence_sqlite.conftest import store_for

_SCHEMA_HASH = hashlib.sha256(SCHEMA_SQL.encode("utf-8")).hexdigest()


def _plant_cache(folder: Path, probe_column: str) -> Path:
    """A valid cache hit in *folder* whose state_meta declares one extra column, so whoever reads it adds it."""
    cache = folder / "cache" / "schema_columns.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    tables = {"state_meta": {"key": "TEXT", "value": "TEXT", probe_column: "TEXT"}}
    cache.write_text(json.dumps({"schema_hash": _SCHEMA_HASH, "tables": tables}), encoding="utf-8")
    return cache


def _columns(db: Path, table: str) -> set:
    with closing(sqlite3.connect(db)) as conn:
        return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _write_sessions_json(folder: Path, display_name: str) -> None:
    (folder / "sessions").mkdir(parents=True, exist_ok=True)
    (folder / "sessions" / "sessions.json").write_text(json.dumps({
        "agent:main:telegram:dm:42": {"session_id": "tg-1", "display_name": display_name},
    }), encoding="utf-8")


def test_each_store_reads_and_writes_the_column_cache_in_its_own_folder(backend, scope_a, scope_b):
    ambient = _plant_cache(get_hermes_home(), "ambient_probe")
    ambient_bytes = ambient.read_bytes()
    _plant_cache(store_for(scope_a).parent, "profile_a_probe")
    for scope in (scope_a, scope_b):
        backend.open(scope).ready()

    assert {"profile_a_probe", "ambient_probe"} & _columns(store_for(scope_a), "state_meta") == {"profile_a_probe"}
    assert {"profile_a_probe", "ambient_probe"} & _columns(store_for(scope_b), "state_meta") == set()
    written = json.loads((store_for(scope_b).parent / "cache" / "schema_columns.json").read_text(encoding="utf-8"))
    assert written["schema_hash"] == _SCHEMA_HASH
    assert set(written["tables"]["state_meta"]) == _columns(store_for(scope_b), "state_meta")
    assert ambient.read_bytes() == ambient_bytes


def test_the_v18_backfill_reads_each_profiles_sessions_json(backend, scope_a, scope_b):
    scopes = (scope_a, scope_b)
    for scope in scopes:
        backend.open(scope).ready()
    backend.reopen()
    _registry.close_all()
    for scope in scopes:
        with closing(sqlite3.connect(store_for(scope), isolation_level=None)) as conn:
            conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('tg-1', 'telegram', 1.0)")
            conn.execute("UPDATE schema_version SET version = 17")
        _write_sessions_json(store_for(scope).parent, f"{scope.name} name")
    _write_sessions_json(get_hermes_home(), "ambient name")

    for scope in scopes:
        backend.open(scope).ready()

    for scope in scopes:
        with closing(sqlite3.connect(store_for(scope))) as conn:
            row = conn.execute("SELECT display_name, session_key FROM sessions WHERE id = 'tg-1'").fetchone()
        assert row == (f"{scope.name} name", "agent:main:telegram:dm:42")


def test_the_raw_connection_reconciler_caches_in_the_folder_it_is_given(tmp_path):
    ambient = _plant_cache(get_hermes_home(), "ambient_probe")
    ambient_bytes = ambient.read_bytes()
    folder = tmp_path / "store"
    _plant_cache(folder, "store_probe")
    with closing(sqlite3.connect(folder / "state.db")) as conn:
        apply_declared_schema(conn.cursor(), folder)
        conn.commit()
    assert {"store_probe", "ambient_probe"} & _columns(folder / "state.db", "state_meta") == {"store_probe"}
    assert ambient.read_bytes() == ambient_bytes
    assert any("state_meta" in statement for statement in schema_read_probe_statements(folder))
