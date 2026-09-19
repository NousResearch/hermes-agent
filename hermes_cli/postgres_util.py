"""Pooled PostgreSQL connections for the small, explicitly routed Hermes stores.

Store owners supply native SQL and a stable namespace. Only parameter binding is
adapted here; SQLite statements and schema migrations remain with their owners.
"""
from __future__ import annotations

import atexit
import hashlib
import re
import threading
from contextlib import contextmanager

_POOLS = {}
_POOL_LOCK = threading.RLock()


def store_schema(base: str, namespace: str) -> str:
    readable = re.sub(r'[^a-z0-9_]+', '_', namespace.lower()).strip('_') or 'store'
    suffix = '_' + readable[:20] + '_' + hashlib.sha256(namespace.encode()).hexdigest()[:10]
    # Include the complete base in the digest so long profile schemas stay distinct.
    digest = hashlib.sha256((base + ':' + namespace).encode()).hexdigest()[:12]
    return base[:63 - len(suffix) - 13] + suffix + '_' + digest


class StoreConnection:
    backend = 'postgres'

    def __init__(self, pool, schema):
        self._pool = pool
        self._conn = pool.getconn()
        self.schema = schema
        self._closed = False

    def execute(self, statement, params=()):
        from hermes_state_postgres import _bind_parameters
        return self._conn.execute(_bind_parameters(statement), params)

    def executemany(self, statement, params):
        from hermes_state_postgres import _bind_parameters
        with self._conn.cursor() as cursor:
            cursor.executemany(_bind_parameters(statement), params)

    def executescript(self, statement):
        self._conn.execute(statement)

    @property
    def in_transaction(self):
        from psycopg.pq import TransactionStatus
        return self._conn.info.transaction_status != TransactionStatus.IDLE

    def begin_write(self):
        self._conn.execute('BEGIN')
        self.execute("SELECT set_config('lock_timeout', '30000', true)")
        self.execute('SELECT pg_advisory_xact_lock(hashtextextended(?, 0))', (self.schema,))

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self.rollback()
            finally:
                self._pool.putconn(self._conn)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.rollback() if kind else self.commit()

    @contextmanager
    def write(self):
        self.begin_write()
        try:
            yield self
            self.commit()
        except BaseException:
            self.rollback()
            raise


def connect(settings, namespace, *, initialize=None):
    """Open a namespaced store; PostgreSQL failures never fall back to local files."""
    from psycopg import sql
    from psycopg_pool import ConnectionPool
    from hermes_state_postgres import _row_factory

    schema = store_schema(settings.schema, namespace)
    key = settings.database_url, schema
    with _POOL_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            def configure(connection):
                if connection.info.server_version < 170000:
                    raise ValueError('PostgreSQL storage requires PostgreSQL 17 or later')
                connection.execute(sql.SQL('SET search_path TO {}, pg_catalog').format(sql.Identifier(schema)))
            pool = ConnectionPool(settings.database_url, min_size=0, max_size=8, timeout=5, open=True,
                max_idle=60, kwargs={'autocommit': True, 'row_factory': _row_factory,
                'connect_timeout': 5, 'application_name': 'hermes-agent-stores'}, configure=configure)
            connection = StoreConnection(pool, schema)
            try:
                with connection.write():
                    connection._conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(schema)))
                    if initialize:
                        initialize(connection)
            except BaseException:
                connection.close()
                pool.close()
                raise
            connection.close()
            _POOLS[key] = pool
    return StoreConnection(pool, schema)


def table_columns(connection, table):
    return {row[0] for row in connection.execute(
        'SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=?',
        (table,))}


def close_pools():
    with _POOL_LOCK:
        for pool in _POOLS.values():
            pool.close()
        _POOLS.clear()


atexit.register(close_pools)
