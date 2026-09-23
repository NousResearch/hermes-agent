"""PostgreSQL connections and transactions for Hermes's existing SessionDB API."""

from __future__ import annotations

import atexit
import base64
import json
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg_pool import ConnectionPool

from hermes_state import SessionDB, _default_db_path
from hermes_state_backend import resolve_database_settings
from hermes_state_common import SCHEMA_VERSION
from hermes_state_postgres_schema import (
    DISPLAY_SQL, FUNCTION_SQL, POSTGRES_SCHEMA_VERSION, POSTGRES_SCHEMA_SQL, SEARCH_SQL, TELEGRAM_SCHEMA_SQL,
)
from hermes_state_postgres_search import SessionPostgresSearchMixin
from hermes_state_postgres_maintenance import SessionPostgresMaintenanceMixin

_BINDING_TOKEN = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|--[^\n]*|/\*.*?\*/|::|:[A-Za-z_]\w*|\?|%", re.S)


@lru_cache(maxsize=256)
def _bind_parameters(statement: str) -> str:
    def replace(match):
        token = match.group()
        if token == '?':
            return '%s'
        if token.startswith(':') and token != '::':
            return f'%({token[1:]})s'
        return token.replace('%', '%%')
    return _BINDING_TOKEN.sub(replace, statement)


class PostgresRow(dict):
    """Keep SQLite Row's name/index access and value iteration for shared readers."""

    def __init__(self, columns, values):
        super().__init__(zip(columns, values))
        self._values = values

    def __getitem__(self, key):
        return self._values[key] if isinstance(key, (int, slice)) else super().__getitem__(key)

    def __iter__(self):
        return iter(self._values)


def _row_factory(cursor):
    columns = [column.name for column in cursor.description] if cursor.description else []
    return lambda values: PostgresRow(columns, values)


class PostgresConnection:
    """The DB-API operations used by shared session methods, over a borrowed connection."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, statement, params=()):
        return self._conn.execute(_bind_parameters(statement), params)

    def executemany(self, statement, params):
        with self._conn.cursor() as cursor:
            cursor.executemany(_bind_parameters(statement), params)

    def executescript(self, statement):
        self._conn.execute(statement)


class PostgresSessionDB(SessionPostgresSearchMixin, SessionPostgresMaintenanceMixin, SessionDB):
    backend = 'postgres'
    database_errors = (psycopg.Error,)
    database_operational_errors = (psycopg.OperationalError, psycopg.ProgrammingError)
    _unlimited_sql_limit = None
    _messages_session_index_hint = ""
    _CONTENT_JSON_PREFIX = "\x01json:"
    _CONTENT_BYTES_PREFIX = "\x01bytes:"

    @classmethod
    def _encode_content(cls, content):
        if isinstance(content, bytes):
            return cls._CONTENT_BYTES_PREFIX + base64.b64encode(content).decode("ascii")
        if isinstance(content, str) and ("\x00" in content or content.startswith((cls._CONTENT_JSON_PREFIX, cls._CONTENT_BYTES_PREFIX))):
            return cls._CONTENT_JSON_PREFIX + json.dumps(content)
        return super()._encode_content(content)

    @classmethod
    def _decode_content(cls, content):
        if isinstance(content, str) and content.startswith(cls._CONTENT_BYTES_PREFIX):
            return base64.b64decode(content[len(cls._CONTENT_BYTES_PREFIX):])
        return super()._decode_content(content)

    def _holder_process_is_dead(self, holder):
        return False

    def __init__(self, db_path: Path = None, read_only: bool = False, *, database_settings=None):
        self.db_path = Path(db_path or _default_db_path())
        self.read_only = read_only
        self._database_settings = database_settings or getattr(self, '_database_settings', None) or resolve_database_settings(self.db_path)
        settings = self._database_settings
        if settings.backend != 'postgres':
            raise ValueError('PostgresSessionDB requires database.backend: postgres')
        self.schema = settings.schema
        self._closed = False
        self._shared_registry_owned = False
        self._lock = threading.RLock()
        self._token_queue = deque()
        self._token_queue_cond = threading.Condition(threading.Lock())
        self._token_writer_thread = None
        self._token_writer_stop = self._token_writer_busy = False
        self._token_atexit_hook = None
        self._pool = None
        self._fts_enabled = True
        self._fts_stale = self._fts_cjk_available = self._trigram_available = False
        self._write_count = 0
        kwargs = dict(autocommit=True, row_factory=_row_factory, connect_timeout=5,
                      application_name='hermes-agent')
        schema = self.schema

        def configure(conn):
            conn.execute(sql.SQL('SET search_path TO {}, pg_catalog').format(sql.Identifier(schema)))
            if read_only:
                conn.execute('SET default_transaction_read_only = on')

        try:
            with psycopg.connect(settings.database_url, **kwargs) as conn:
                if conn.info.server_version < 170000:
                    raise ValueError('PostgreSQL session storage requires PostgreSQL 17 or later')
                configure(conn)
                self._initialize_schema(conn)
            self._pool = ConnectionPool(settings.database_url, kwargs=kwargs, configure=configure,
                                        min_size=0, max_size=4, timeout=5, open=True)
        except psycopg.Error as exc:
            self._closed = True
            from hermes_state_guard import _set_last_init_error
            message = f'PostgreSQL session database could not be opened ({type(exc).__name__})'
            _set_last_init_error(message)
            raise RuntimeError(message) from None
        except Exception:
            self._closed = True
            raise

    def _initialize_schema(self, conn):
        if self.read_only:
            version = conn.execute('SELECT version FROM postgres_schema_version').fetchone()[0]
            if version != POSTGRES_SCHEMA_VERSION:
                raise RuntimeError('PostgreSQL schema requires a writable upgrade')
            return
        with conn.transaction():
            conn.execute("SELECT set_config('lock_timeout', '5000', true)")
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (self.schema,))
            conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(self.schema)))
            exists = conn.execute("SELECT to_regclass('postgres_schema_version')").fetchone()[0]
            if exists is not None:
                version = conn.execute('SELECT version FROM postgres_schema_version').fetchone()[0]
                if version != POSTGRES_SCHEMA_VERSION:
                    raise RuntimeError('Unsupported PostgreSQL schema version')
                return
            conn.execute(POSTGRES_SCHEMA_SQL)
            conn.execute(FUNCTION_SQL)
            conn.execute(DISPLAY_SQL)
            conn.execute(SEARCH_SQL)
            conn.execute('INSERT INTO schema_version(version) VALUES (%s)', (SCHEMA_VERSION,))
            conn.execute('CREATE TABLE postgres_schema_version (version BIGINT PRIMARY KEY)')
            conn.execute('INSERT INTO postgres_schema_version(version) VALUES (%s)', (POSTGRES_SCHEMA_VERSION,))

    @contextmanager
    def _read_ctx(self):
        if self._closed:
            raise RuntimeError('SessionDB connection is closed')
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute('SET TRANSACTION READ ONLY')
                yield PostgresConnection(conn)

    def _read_retrying_ioerr(self, fn):
        with self._read_ctx() as conn:
            return fn(conn)

    def _read_with_timeout(self, query, params, timeout_seconds):
        try:
            with self._read_ctx() as conn:
                conn.execute("SELECT set_config('statement_timeout', ?, true)",
                             (str(max(1, int(timeout_seconds * 1000))),))
                return conn.execute(query, params).fetchall()
        except psycopg.errors.QueryCanceled as exc:
            raise TimeoutError(f"recent-session browse exceeded {timeout_seconds:g}s deadline") from exc

    def _execute_write(self, fn, patience_s=None):
        if self.read_only:
            raise RuntimeError('SessionDB is read-only')
        if self._closed:
            raise RuntimeError('SessionDB connection is closed')
        patience = self._WRITE_PATIENCE_S if patience_s is None else patience_s
        deadline = time.monotonic() + patience
        with self._pool.connection() as conn:
            with conn.transaction():
                remaining = max(1, int((deadline - time.monotonic()) * 1000))
                conn.execute("SELECT set_config('lock_timeout', %s, true)", (str(remaining),))
                conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))', (self.schema,))
                result = fn(PostgresConnection(conn))
            self._write_count += 1
            return result

    def _message_column_names(self, conn):
        if not hasattr(self, '_message_columns_cache'):
            self._message_columns_cache = [row[0] for row in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() "
                "AND table_name = 'messages' AND column_name <> 'search_vector' ORDER BY ordinal_position")]
        return self._message_columns_cache

    def get_change_revision(self):
        return self._read_one('SELECT revision FROM state_change_revision WHERE id = 1')[0]

    def get_meta(self, key):
        row = self._read_one('SELECT value FROM state_meta WHERE key = ?', (key,))
        return None if row is None else row[0]

    def apply_telegram_topic_migration(self):
        def migrate(conn):
            conn.executescript(TELEGRAM_SCHEMA_SQL)
            self.set_meta('telegram_dm_topic_schema_version', '3', cursor=conn)
        self._execute_write(migrate)

    def close(self):
        if self._shared_registry_owned:
            from hermes_state_registry import release
            release(self)
            return
        if self._closed:
            return
        self._stop_token_writer(join_timeout=self._TRANSCRIPT_WRITE_PATIENCE_S)
        if self._token_writer_busy:
            raise TimeoutError('SessionDB token writer is still draining')
        if self._token_atexit_hook is not None:
            atexit.unregister(self._token_atexit_hook)
            self._token_atexit_hook = None
        self._closed = True
        self._pool.close()

    def __del__(self):
        pool = getattr(self, '_pool', None)
        if pool is not None:
            pool.close()
