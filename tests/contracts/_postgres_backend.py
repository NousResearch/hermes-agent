"""The Postgres provider plugin as a contract-suite backend, registered as ``postgres`` when ``HERMES_PG_TEST_DSN`` names
a database and ``HERMES_PG_PLUGIN_DIR`` names the plugin's source folder (never the real home's ``plugins/`` link: the
suites refuse file I/O under the real Hermes home).

Each factory call serves its profiles from fresh schemas under a prefix of its own and drops them on exit, so every test
starts from empty stores; a profile a suite names beyond the two is served on first open, as the SQLite backend does.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from persistence import ProfileScope


def plugin_module(name: str) -> Any:
    """The loaded plugin's module ``name`` (``conversation._transcript``), for a suite that patches a backend seam."""
    import sys

    suffix = f".postgres.{name}"
    return next(module for key, module in sys.modules.items() if key.startswith("_hermes_user_persistence") and key.endswith(suffix))


def _plugin_dir() -> Optional[Path]:
    configured = os.environ.get("HERMES_PG_PLUGIN_DIR")
    if not configured:
        return None
    path = Path(configured).expanduser().resolve()
    return path if (path / "__init__.py").is_file() else None


class _SessionCells:
    """One stored session row's raw cells, the reference's row-dict seam: ``row[column] = value`` writes the cell
    bypassing every typed write path. A value the typed column cannot hold (a NULL counter, text in a double, a
    non-object config) skips the case: that damaged state is unrepresentable in this schema."""

    def __init__(self, store: Any, session_id: str) -> None:
        self._store, self._session_id = store, session_id

    @staticmethod
    def _cell(column: str, value: Any) -> Any:
        import json
        from datetime import datetime

        from psycopg.types.json import Jsonb

        if isinstance(value, datetime):
            return value.timestamp()
        if column == "model_config" and value is not None:
            decoded = json.loads(value) if isinstance(value, str) else value
            if not isinstance(decoded, dict):
                raise ValueError("model_config cell is not a JSON object")
            return Jsonb(decoded)
        return value

    def __setitem__(self, column: str, value: Any) -> None:
        self.update({column: value})

    def update(self, cells: Any = (), **more: Any) -> None:
        import pytest
        from psycopg import sql

        from persistence import PersistenceError

        cells = {**dict(cells), **more}
        try:
            values = {column: self._cell(column, value) for column, value in cells.items()}
            with self._store.transaction() as conn:
                conn.execute(sql.SQL("UPDATE sessions SET {} WHERE id = %s").format(sql.SQL(", ").join(
                    sql.SQL("{} = %s").format(sql.Identifier(column)) for column in values)),
                    (*values.values(), self._session_id))
        except (PersistenceError, ValueError) as exc:
            pytest.skip(f"QUIRK: postgres types refuse the planted cell {cells!r}: {exc}")

    def __getitem__(self, column: str) -> Any:
        from psycopg import sql

        with self._store.transaction() as conn:
            found = conn.execute(sql.SQL("SELECT {} FROM sessions WHERE id = %s").format(sql.Identifier(column)),
                                 (self._session_id,)).fetchone()
        if found is None:
            raise KeyError(self._session_id)
        return found[0]


class _SessionsTable:
    def __init__(self, store: Any) -> None:
        self._store = store

    def __getitem__(self, session_id: str) -> _SessionCells:
        return _SessionCells(self._store, session_id)


class _SessionDb:
    """The suites' raw read seam over one profile's schema: SQLite-style ``?`` placeholders, rows as tuples."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _read_all(self, query: str, params: Any = ()) -> list[tuple]:
        with self._store.transaction() as conn:
            return conn.execute(query.replace("?", "%s"), tuple(params)).fetchall()

    def _read_one(self, query: str, params: Any = ()) -> Optional[tuple]:
        found = self._read_all(query, params)
        return found[0] if found else None


def _contract_backend_class(backend_module: Any) -> type:
    class ContractPostgresBackend(backend_module.PostgresBackend):
        """The plugin's backend serving any profile the suites open, with the suites' epoch and raw-row seams."""

        def session_db(self, scope: ProfileScope) -> _SessionDb:
            return _SessionDb(self.open(scope))

        def scope_store(self, scope: ProfileScope) -> dict[str, Any]:
            """The reference's namespace shape over ``scope``'s schema: ``["sessions"][id]`` writes raw cells."""
            return {"sessions": _SessionsTable(self.open(scope))}

        def open(self, scope: ProfileScope):
            from_schema = backend_module._schema.schema_name
            self._schemas.setdefault(scope.hermes_home_key, from_schema(self.settings.schema_prefix, scope.hermes_home_key))
            return super().open(scope)

        def begin_epoch(self, scope: ProfileScope, instance_id: str) -> int:
            store = self.open(scope)
            runtime = getattr(store, "runtime", None)
            if runtime is None:
                raise NotImplementedError("postgres: the runtime repository is not ported yet")
            return int(runtime.begin_epoch(instance_id=instance_id))

        def close_store_under(self, scope: ProfileScope) -> None:
            """Force-close ``scope``'s store under its open handles, as a profile delete's sweep does."""
            self.open(scope)
            self.close_stores_under(Path(scope.hermes_home_key))

        def replace_store(self, scope: ProfileScope) -> None:
            """Replace ``scope``'s store under its open handles: the store takes a new incarnation."""
            from psycopg import sql

            schema = self.open(scope).schema
            with self.pool().connection() as conn:
                conn.execute(sql.SQL("UPDATE {}.store_identity SET incarnation = md5(random()::text)").format(sql.Identifier(schema)))

    return ContractPostgresBackend


@contextmanager
def _open_postgres_backend(dsn: str, plugin_dir: Path, scopes: Sequence[ProfileScope]) -> Iterator[Any]:
    """The plugin's backend over the test scopes in schemas prefixed ``hermes_ct_<random>_``, dropped on exit."""
    import sys

    import psycopg
    from psycopg import sql

    from plugins.persistence_providers import _load_provider_from_dir

    provider = _load_provider_from_dir(plugin_dir)
    package = type(provider).__module__.rpartition(".")[0]
    backend_module = sys.modules[f"{package}._backend"]
    settings_module = sys.modules[f"{package}._settings"]
    prefix = f"hermes_ct_{uuid.uuid4().hex[:8]}_"
    settings = settings_module.PostgresSettings(dsn, prefix, allow_incomplete=True)
    stores = {scope.hermes_home_key: scope.profile_dir / "state.db" for scope in scopes}
    from tests.contracts._scopes import CONTRACT_ROOT

    backend = _contract_backend_class(backend_module)(stores, settings, owner_root=CONTRACT_ROOT)
    try:
        yield backend
    finally:
        backend.close()
        with psycopg.connect(dsn, autocommit=True) as conn:
            for (schema,) in conn.execute("SELECT nspname FROM pg_namespace WHERE starts_with(nspname, %s)",
                                          (prefix,)).fetchall():
                conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def register(register_backend: Callable[..., None], scopes: Sequence[ProfileScope]) -> None:
    """Register ``postgres`` over ``scopes`` when a test database and the plugin are both available."""
    dsn = os.environ.get("HERMES_PG_TEST_DSN")
    plugin_dir = _plugin_dir()
    if dsn and plugin_dir is not None:
        register_backend("postgres", partial(_open_postgres_backend, dsn, plugin_dir, tuple(scopes)))
