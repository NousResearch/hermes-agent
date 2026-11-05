"""The compact listing projection reads its column list from the folder that holds the store, never from the
ambient home."""

import hashlib
import json

from hermes_constants import get_hermes_home
from persistence.conversation.sessions import NewSession
from persistence.sqlite._schema_ddl import SCHEMA_SQL
from persistence.sqlite.conversation._store import portability


def test_compact_rows_take_the_columns_from_the_stores_own_folder(backend, scope_a, scope_b, monkeypatch):
    ambient = get_hermes_home() / "cache" / "schema_columns.json"
    ambient.parent.mkdir(parents=True, exist_ok=True)
    tables = {"sessions": {"id": "TEXT", "ambient_probe": "TEXT"}}
    ambient.write_text(json.dumps({"schema_hash": hashlib.sha256(SCHEMA_SQL.encode("utf-8")).hexdigest(),
                                   "tables": tables}), encoding="utf-8")
    ambient_bytes = ambient.read_bytes()
    monkeypatch.setattr(portability, "_session_compact_cols_sql", None)

    for scope in (scope_a, scope_b):
        store = backend.open(scope)
        store.sessions.create(NewSession(f"{scope.name}-1", "cli"))
        entry = store.sessions.summary(f"{scope.name}-1", compact_rows=True)
        assert entry is not None and entry.session.session_id == f"{scope.name}-1"

    assert "ambient_probe" not in portability._session_compact_cols_sql
    assert "s.started_at" in portability._session_compact_cols_sql
    assert ambient.read_bytes() == ambient_bytes
