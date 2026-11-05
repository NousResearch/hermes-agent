"""The files a store's schema setup reads beside it (the column-parse cache, the v18 ``sessions.json`` backfill, the
CJK tokenizer library) come from the opened profile's folder, never from the ambient home; so do the maintenance
tools' column probes. No backend module reads the process or task home except the listed ones, and the home reads
left in main's per-process modules the backend imports are listed by function."""

import ast
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from hermes_constants import get_hermes_home
from persistence.sqlite import _registry
from persistence.sqlite._schema import apply_declared_schema, schema_read_probe_statements
from persistence.sqlite._schema_ddl import SCHEMA_SQL, fts5_cjk_so_path
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


def test_the_cjk_tokenizer_library_is_looked_up_in_the_store_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_FTS5_CJK_SO", raising=False)
    assert fts5_cjk_so_path(tmp_path) == tmp_path / "lib" / "libfts5_cjk.so"
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "elsewhere.so"))
    assert fts5_cjk_so_path(tmp_path) == tmp_path / "elsewhere.so"


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


# The reads of the process or task home left in the backend, by file and the name read.
_HOME_READERS = {"get_hermes_home", "get_hermes_home_override", "get_process_hermes_home", "profile_cli_selector",
                 "display_hermes_home", "get_default_hermes_root"}
_REMAINING_HOME_READS = {
    # The owner name the store stamps: default for the install root, the profile name under <root>/profiles.
    ("conversation/_store/sessions.py", "get_default_hermes_root"),
    # The repair command the reconciler prints targets the install the operator's shell resolves.
    ("admin/_reconcile.py", "get_default_hermes_root"),
}

# The per-process modules the backend and the legacy store share.
_STATE_MODULES = {f"persistence.sqlite._state.{name}" for name in (
    "errors", "common", "repair", "holders", "keys", "dbfile", "health", "wal", "readpool", "pidns", "lockowners",
    "lockguard", "guard")}
# Their home reads, by module, function and the name read: the writability preflight's home and the ``-p`` guidance
# for the process's active profile.
_MAINS_HOME_READS = {
    ("persistence.sqlite._state.holders", "held_store_refusal", "profile_cli_selector"),
    ("persistence.sqlite._state.repair", "_persistent_repair_exhausted_error", "profile_cli_selector"),
    ("persistence.sqlite._state.repair", "preflight_db_writability", "get_hermes_home"),
}


def _backend_files():
    backend = Path(_registry.__file__).parent
    return backend, sorted(backend.rglob("*.py"))


def _outside_state(backend, files):
    return [path for path in files if "_state" not in path.relative_to(backend).parts]


def test_no_backend_module_reads_the_process_or_task_home_but_the_listed_ones():
    backend, files = _backend_files()
    found = set()
    for path in _outside_state(backend, files):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "hermes_constants":
                names = {alias.name for alias in node.names}
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "hermes_constants":
                names = {node.attr}
            else:
                continue
            found |= {(str(path.relative_to(backend)), name) for name in names & _HOME_READERS}
    assert found == _REMAINING_HOME_READS


def test_the_home_reads_left_in_mains_modules_the_backend_imports_are_the_listed_ones():
    _, files = _backend_files()
    imported = set()
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
    # No backend module imports the legacy store.
    assert {name for name in imported if name.startswith("hermes_state")} == set()

    readers = _HOME_READERS | {"load_config_readonly"}
    root = Path(_registry.__file__).parents[2]
    found = set()
    for module in _STATE_MODULES:
        for top in ast.parse((root / f"{module.replace('.', '/')}.py").read_text(encoding="utf-8")).body:
            if not isinstance(top, (ast.FunctionDef, ast.ClassDef)):
                continue
            for node in ast.walk(top):
                if isinstance(node, ast.ImportFrom):
                    names = {alias.name for alias in node.names}
                elif isinstance(node, ast.Name):
                    names = {node.id}
                elif isinstance(node, ast.Attribute):
                    names = {node.attr}
                else:
                    continue
                found |= {(module, top.name, name) for name in names & readers}
    assert found == _MAINS_HOME_READS
