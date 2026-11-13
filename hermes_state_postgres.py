"""PostgreSQL connections and transactions for Hermes's existing SessionDB API.

``PostgresSessionDB`` implements the SessionDB API over a psycopg connection pool. The shared
mixin code is dialect-agnostic through the ``_read_ctx`` / ``_execute_write`` seams: it binds
``?`` placeholders, reads ``rowcount``/``total_changes``/``in_transaction``, and calls a small
set of SQLite introspection statements (``sqlite_master`` existence checks, ``PRAGMA table_info``,
``PRAGMA main.data_version``, ``GLOB``, ``substr(``, ``IS NOT``, ``INSERT OR IGNORE/REPLACE``). :func:`_convert_statement`
maps those onto their PostgreSQL equivalents so the shared SQL text stays byte-identical; any
SQLite-ism without a mapping raises :class:`UnsupportedSQLError` instead of failing silently.
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar

import psycopg
import psycopg.errors
from psycopg import sql
from psycopg_pool import ConnectionPool

from hermes_state import SessionDB, _default_db_path
from hermes_state_backend import resolve_database_settings
from hermes_state_common import SCHEMA_VERSION
from hermes_state_errors import SessionCompressionInProgressError
from hermes_state_postgres_maintenance import SessionPostgresMaintenanceMixin
from hermes_state_postgres_schema import (
    DISPLAY_SQL,
    FUNCTION_SQL,
    POSTGRES_SCHEMA_SQL,
    POSTGRES_SCHEMA_VERSION,
    SEARCH_SQL,
    TELEGRAM_SCHEMA_SQL,
)
from hermes_state_postgres_search import SessionPostgresSearchMixin

logger = logging.getLogger(__name__)

T = TypeVar("T")


class UnsupportedSQLError(RuntimeError):
    """A SQLite-ism the PostgreSQL adapter has no mapping for; fail loud, never wrong."""


_BINDING_TOKEN = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|--[^\n]*|/\*.*?\*/|::|:[A-Za-z_]\w*|\?\d+|\?|%")
_INSERT_OR_IGNORE_RE = re.compile(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", re.IGNORECASE)
_INSERT_OR_REPLACE_RE = re.compile(r"\bINSERT\s+OR\s+REPLACE\s+INTO\b", re.IGNORECASE)
_PRAGMA_DATA_VERSION_RE = re.compile(r"\bPRAGMA\s+main\.data_version\b", re.IGNORECASE)
_PRAGMA_TABLE_INFO_RE = re.compile(r"\bPRAGMA\s+table_info\(\s*['\"]?(\w+)['\"]?\s*\)", re.IGNORECASE)
_SQLITE_MASTER_RE = re.compile(r"\bsqlite_master\b", re.IGNORECASE)
_SM_TABLE_IN_RE = re.compile(
    r"SELECT\s+name\s+FROM\s+sqlite_master\s+WHERE\s+type\s*=\s*'table'\s+AND\s+name\s+IN\s*\(([^)]*)\)", re.IGNORECASE)
_SM_TABLE_ONE_RE = re.compile(
    r"SELECT\s+1\s+FROM\s+sqlite_master\s+WHERE\s+type\s*=\s*'table'\s+AND\s+name\s*=\s*(\?|'[^']*')", re.IGNORECASE)
_SM_TABLE_ALL_RE = re.compile(r"SELECT\s+name\s+FROM\s+sqlite_master\s+WHERE\s+type\s*=\s*'table'\s*$", re.IGNORECASE)
_SM_INDEX_ONE_RE = re.compile(
    r"SELECT\s+1\s+FROM\s+sqlite_master\s+WHERE\s+type\s*=\s*'index'\s+AND\s+name\s*=\s*\?", re.IGNORECASE)
_PRAGMA_ANY_RE = re.compile(r"\bPRAGMA\b", re.IGNORECASE)
_GLOB_WORD_RE = re.compile(r"[A-Za-z_]\w*")
# Table hints are SQLite planner syntax; PostgreSQL has no equivalent (its planner uses the
# same indexes on its own). Both forms are dropped.
_INDEXED_BY_RE = re.compile(r"\s+INDEXED\s+BY\s+[A-Za-z_][\w.]*", re.IGNORECASE)
_NOT_INDEXED_RE = re.compile(r"\s+NOT\s+INDEXED\b", re.IGNORECASE)
# ``CROSS JOIN t ON cond`` is SQLite's way of pinning the driving side (an inner join that
# keeps the left table outer); PostgreSQL rejects ON on a CROSS JOIN, so it becomes a plain
# inner join there. The optional table alias is kept (the ON clause references it). Bare
# ``CROSS JOIN t`` (no ON) is unchanged.
_CROSS_JOIN_ON_RE = re.compile(r"\bCROSS\s+JOIN\s+(\w+)(\s+[A-Za-z_]\w*)?\s+ON\b", re.IGNORECASE)


def _rewrite_insert_or(statement: str) -> str:
    """``INSERT OR IGNORE`` has a direct PostgreSQL form; ``INSERT OR REPLACE`` is handled in
    :meth:`PostgresConnection.execute` (it needs the table's primary key from the catalog)."""
    if _INSERT_OR_REPLACE_RE.search(statement):
        return statement
    if _INSERT_OR_IGNORE_RE.search(statement):
        return _INSERT_OR_IGNORE_RE.sub("INSERT INTO", statement, count=1) + " ON CONFLICT DO NOTHING"
    return statement


def _rewrite_pragma(statement: str) -> str:
    if _PRAGMA_DATA_VERSION_RE.search(statement):
        return _PRAGMA_DATA_VERSION_RE.sub("SELECT txid_current()", statement, count=1)
    match = _PRAGMA_TABLE_INFO_RE.search(statement)
    if match:
        table = match.group(1)
        replacement = (
            "SELECT 0 AS cid, column_name AS name, udt_name AS type, 0 AS notnull, "
            "NULL AS dflt_value, 0 AS pk FROM information_schema.columns "
            f"WHERE table_schema = current_schema() AND table_name = '{table}' "
            "ORDER BY ordinal_position"
        )
        return statement[: match.start()] + replacement + statement[match.end():]
    if _PRAGMA_ANY_RE.search(statement):
        raise UnsupportedSQLError(f"unsupported PRAGMA in: {statement[:80]}")
    return statement


def _rewrite_sqlite_master(statement: str) -> str:
    if not _SQLITE_MASTER_RE.search(statement):
        return statement
    match = _SM_TABLE_IN_RE.search(statement)
    if match:
        inner = match.group(1)
        replacement = f"SELECT n FROM UNNEST(ARRAY[{inner}]) AS n WHERE to_regclass(n) IS NOT NULL"
        return statement[: match.start()] + replacement + statement[match.end():]
    match = _SM_TABLE_ONE_RE.search(statement)
    if match:
        name = match.group(1)
        replacement = f"SELECT 1 WHERE to_regclass({name}) IS NOT NULL"
        return statement[: match.start()] + replacement + statement[match.end():]
    match = _SM_TABLE_ALL_RE.search(statement)
    if match:
        replacement = (
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = current_schema() AND c.relkind = 'r'"
        )
        return statement[: match.start()] + replacement + statement[match.end():]
    match = _SM_INDEX_ONE_RE.search(statement)
    if match:
        replacement = (
            "SELECT 1 WHERE EXISTS (SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = current_schema() "
            "AND c.relname = ?)"
        )
        return statement[: match.start()] + replacement + statement[match.end():]
    raise UnsupportedSQLError(f"unsupported sqlite_master query: {statement[:80]}")


def _glob_lhs_start(prefix: str) -> Optional[int]:
    """Start index of the operand ending at the end of *prefix* (a bare/qualified name or a
    call with balanced parentheses); None when the prefix does not end in a GLOB operand."""
    i = len(prefix) - 1
    while i >= 0 and prefix[i] in " \t":
        i -= 1
    if i < 0:
        return None
    if prefix[i] == ")":
        depth = 0
        j = i
        while j >= 0:
            if prefix[j] == ")":
                depth += 1
            elif prefix[j] == "(":
                depth -= 1
                if depth == 0:
                    break
            j -= 1
        if j < 0:
            return None
        k = j - 1
        while k >= 0 and (prefix[k].isalnum() or prefix[k] == "_"):
            k -= 1
        return k + 1
    j = i
    while j >= 0 and (prefix[j].isalnum() or prefix[j] in "._"):
        j -= 1
    return j + 1


def _rewrite_glob(statement: str) -> str:
    """``X GLOB P`` (SQLite operator) -> ``glob(P, X)`` (the SQL function from FUNCTION_SQL).

    Literal-aware: tokens (strings, comments, ``::``, ``:name``, ``?``, ``%``) are copied
    verbatim; the keyword is rewritten only when it sits between two real operands.
    """
    out: list[str] = []
    i, n = 0, len(statement)
    while i < n:
        token = _BINDING_TOKEN.match(statement, i)
        if token:
            out.append(token.group())
            i = token.end()
            continue
        word = _GLOB_WORD_RE.match(statement, i)
        if word:
            text = "".join(out)
            if word.group().upper() == "GLOB":
                j = word.end()
                while j < n and statement[j] in " \t":
                    j += 1
                rhs = _BINDING_TOKEN.match(statement, j)
                lhs_start = _glob_lhs_start(text)
                if lhs_start is None or rhs is None:
                    raise UnsupportedSQLError(f"unsupported GLOB operand in: {statement[:80]}")
                # The LHS is already in ``out``: drop it (token boundaries line up at lhs_start,
                # a separator character index), then append the function form.
                keep: list[str] = []
                pos = 0
                for token in out:
                    if pos + len(token) <= lhs_start:
                        keep.append(token)
                        pos += len(token)
                    else:
                        break
                out[:] = keep
                out.append(f"glob({rhs.group()}, {text[lhs_start:].rstrip()})")
                i = rhs.end()
                continue
            out.append(word.group())
            i = word.end()
            continue
        out.append(statement[i])
        i += 1
    return "".join(out)


_SUBSTR_WORD_RE = re.compile(r"substr", re.IGNORECASE)


def _rewrite_substr(statement: str) -> str:
    """``substr(X, Y[, Z])`` (SQLite) -> ``substring(X FROM Y [FOR Z])``: canonical
    PostgreSQL form (the FROM/FOR keywords instead of positional start/length arguments).

    Literal-aware like :func:`_rewrite_glob`: binding tokens and string literals are copied
    verbatim, and the keyword is rewritten only as a function call with 2-3 real arguments."""
    out: list[str] = []
    i, n = 0, len(statement)
    while i < n:
        token = _BINDING_TOKEN.match(statement, i)
        if token:
            out.append(token.group())
            i = token.end()
            continue
        prev = statement[i - 1] if i else ""
        if _SUBSTR_WORD_RE.fullmatch(statement[i:i + 6]) and not (prev.isalnum() or prev in "_.$"):
            j = i + 6
            while j < n and statement[j] in " \t":
                j += 1
            if j < n and statement[j] == "(":
                depth, k, start = 1, j + 1, j + 1
                args: list[str] = []
                while k < n and depth:
                    c = statement[k]
                    if c == "'":
                        k += 1  # string literal: skip whole (quotes may hide parens/commas)
                        while k < n and statement[k] != "'":
                            k += 1
                    elif c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                        if not depth:
                            break
                    elif c == "," and depth == 1:
                        args.append(statement[start:k].strip())
                        start = k + 1
                    k += 1
                if depth:
                    raise UnsupportedSQLError(f"unbalanced substr( in: {statement[:80]}")
                args.append(statement[start:k].strip())
                if not 1 <= len(args) <= 3 or any(a == "" for a in args):
                    raise UnsupportedSQLError(f"unsupported substr( arity in: {statement[:80]}")
                if len(args) == 1:
                    out.append(f"substring({args[0]} FROM 1)")
                elif len(args) == 2:
                    out.append(f"substring({args[0]} FROM {args[1]})")
                else:
                    out.append(f"substring({args[0]} FROM {args[1]} FOR {args[2]})")
                i = k + 1
                continue
        out.append(statement[i])
        i += 1
    return "".join(out)


# The model-only display marker no longer has a dedicated converter rule: the shared
# type-aware expression (hermes_state_common._sql_model_only_truthy) is built from
# json_type()/json_extract()/json_valid(), which the PG shim registers as-is, so it
# passes through the converter unchanged.
# ``COALESCE(json_extract((CASE WHEN json_valid(X) <> 0 THEN X ELSE 'null' END), 'P'), 0) = 1``
# (the queued-prompt marker): SQLite's json_extract returns JSON true as 1, so the shared SQL
# tests the boolean marker numerically; the TEXT shim returns 'true'. Test the scalar text
# instead (absent/invalid extraction is NULL and drops the row, as COALESCE(..., 0) = 1 did).
_JSON_BOOL_MARKER_RE = re.compile(
    r"COALESCE\(json_extract\(\(CASE WHEN json_valid\(([A-Za-z_][\w.]*)\) <> 0 THEN \1 ELSE 'null' END\), "
    r"('[^']*')\), 0\) = 1")
# Remaining numeric-style ``json_valid(X) <> 0`` tests -> the bare boolean predicate.
_JSON_VALID_NUM_RE = re.compile(r"json_valid\(\s*([A-Za-z_][\w.]*)\s*\)\s*<>\s*0\b")
# ``json_valid(X) AND json_extract(X, 'P')`` (SQLite text-as-truthy, end of predicate) ->
# explicit non-empty test; the lookahead keeps comparison uses (``... < ?``) untouched.
_JSON_TEXT_TRUTHY_RE = re.compile(
    r"json_valid\(([A-Za-z_][\w.]*)\)\s+AND\s+json_extract\(\1, ('[^']+')\)(?=\s*(?:$|AND\b|OR\b|\)))")
# ROUTE_RETIRED_SQL: SQLite's mixed-type CASE (boolean THEN branch, integer ELSE 0) and the
# text-as-truthy ``expires AND retired_at < ?``. The CASE is kept (its json_valid guard
# protects the throwing json_extract shim; AND evaluation order is not guaranteed); only the
# types are fixed and the JSON-number extraction compares as numeric (the shim returns TEXT).
_RETIRED_OWN_CASE_RE = re.compile(
    r"CASE WHEN json_valid\(([^)]+)\)\s+THEN json_extract\(\1, ('[^']+')\) IS NOT NULL ELSE 0 END")
_ROUTE_FENCE_CASE_RE = re.compile(
    r"NOT CASE WHEN json_valid\(([^)]+)\)\s+THEN coalesce\("
    r"json_extract\(\1, ('[^']+')\)\s+AND\s+json_extract\(\1, ('[^']+')\) < ([\w:]+) - ([\w:]+), 0\)\s+ELSE 0 END")
# ``a IS NOT CASE WHEN json_valid(X) THEN json_extract(X, 'P') END`` (moved_receipts): SQLite's
# null-safe inequality; PostgreSQL's IS NOT is a boolean-indicator test, and its null-safe
# inequality is IS DISTINCT FROM (identical NULL semantics on both sides of the expression).
_IS_NOT_GUARDED_EXTRACT_RE = re.compile(
    r"IS NOT (CASE WHEN json_valid\(([^)]+)\) THEN json_extract\(\2, ('[^']+')\) END)")
# ``OR json_extract(X, 'P') THEN`` (state_meta upsert guard): the bare extraction is a
# SQLite text-truthy predicate operand; the extraction result is never a value here.
_JSON_OR_BARE_TRUTHY_RE = re.compile(
    r"OR\s+json_extract\(([A-Za-z_][\w.]*), ('[^']+')\)(?=\s+THEN\b)")
# ``AND json_extract(X, 'P')`` at a predicate end (no json_valid prefix, e.g. the expiry
# sweep DELETE): same SQLite text-truthy contract.
_JSON_AND_BARE_TRUTHY_RE = re.compile(
    r"AND\s+json_extract\(([A-Za-z_][\w.]*), ('[^']+')\)(?=\s*(?:$|AND\b|OR\b|WHERE\b|\)))")
# JSON paths whose values are numeric epochs/sequences in the shared SQL; the TEXT
# extraction result needs a numeric cast where the shared SQL compares it to numbers.
_NUMERIC_JSON_PATHS = tuple(re.escape(path) for path in ("'$.retired_at'", "'$.result.revision'"))
_JSON_NUM_EXTRACT_CMP_RE = re.compile(
    r"json_extract\(([A-Za-z_][\w.]*), (" + "|".join(_NUMERIC_JSON_PATHS) + r")\)(\s*[<>]=?)")


@lru_cache(maxsize=1024)
def _convert_statement(statement: str) -> str:
    """SQLite -> PostgreSQL statement translation (placeholders, sqlite-isms). Cached: shared
    SQL text is stable per call site."""
    out = _rewrite_glob(statement)
    out = _rewrite_substr(out)
    out = _rewrite_insert_or(out)
    out = _rewrite_pragma(out)
    out = _rewrite_sqlite_master(out)
    out = _INDEXED_BY_RE.sub("", out)
    out = _NOT_INDEXED_RE.sub("", out)
    out = _CROSS_JOIN_ON_RE.sub(r"JOIN \1\2 ON", out)

    # ``_sql_json_extract`` builds ``(CASE WHEN json_valid(X) <> 0 THEN X ELSE 'null' END)``;
    # PostgreSQL's jsonb cast on NULL is a no-op and the boolean shim shadows the native
    # integer-returning json_valid, so both <> 0 tests collapse to the bare predicate.
    out = _JSON_BOOL_MARKER_RE.sub(
        r"json_extract((CASE WHEN json_valid(\1) <> 0 THEN \1 ELSE 'null' END), \2) = 'true'", out)
    out = _RETIRED_OWN_CASE_RE.sub(
        r"CASE WHEN json_valid(\1) THEN json_extract(\1, \2) IS NOT NULL ELSE FALSE END", out)
    out = _ROUTE_FENCE_CASE_RE.sub(
        r"NOT CASE WHEN json_valid(\1) "
        r"THEN (json_extract(\1, \2) IS NOT NULL AND json_extract(\1, \2) <> '' "
        r"AND json_extract(\1, \3)::numeric < \4 - \5) ELSE FALSE END", out)
    out = _IS_NOT_GUARDED_EXTRACT_RE.sub(
        r"IS DISTINCT FROM (CASE WHEN json_valid(\2) THEN json_extract(\2, \3) END)", out)
    out = _JSON_NUM_EXTRACT_CMP_RE.sub(r"json_extract(\1, \2)::numeric\3", out)
    out = _JSON_TEXT_TRUTHY_RE.sub(
        r"json_valid(\1) AND json_extract(\1, \2) IS NOT NULL AND json_extract(\1, \2) <> ''", out)
    out = _JSON_OR_BARE_TRUTHY_RE.sub(
        r"OR (json_extract(\1, \2) IS NOT NULL AND json_extract(\1, \2) <> '')", out)
    out = _JSON_AND_BARE_TRUTHY_RE.sub(
        r"AND (json_extract(\1, \2) IS NOT NULL AND json_extract(\1, \2) <> '')", out)
    out = _JSON_VALID_NUM_RE.sub(r"json_valid(\1)", out)

    # SQLite positional bindings become named ordinal placeholders so a repeated ordinal
    # (?1 ... ?1) collapses to one binding and anonymous ? takes the next ordinal, exactly as
    # SQLite numbers them. Named (:name) bindings keep their names.
    ordinals = {"next": 0}

    def replace(match):
        token = match.group()
        if token == "?":
            ordinals["next"] += 1
            return f"%(o{ordinals['next']})s"
        if token.startswith("?"):
            number = token[1:]
            ordinals["next"] = max(ordinals["next"], int(number))
            return f"%(o{number})s"
        if token.startswith(":") and token != "::":
            return f"%({token[1:]})s"
        return token.replace("%", "%%")

    return _BINDING_TOKEN.sub(replace, out)


_ORDINAL_PLACEHOLDER_RE = re.compile(r"%\(o\d+\)s")


def _ordinalize_params(params):
    """Re-key a positional sequence for the %(oN)s placeholders the converter emits."""
    if params is None:
        return None
    if isinstance(params, dict):
        raise UnsupportedSQLError(
            "ordinal (?N) positional placeholders cannot share a named (:name) parameter dict")
    return {f"o{i}": value for i, value in enumerate(params, start=1)}


class PostgresRow(dict):
    """Keep SQLite Row's name/index access and value iteration for shared readers."""

    def __init__(self, columns, values):
        super().__init__(zip(columns, values))
        self._values = values

    def __getitem__(self, key):
        return self._values[key] if isinstance(key, (int, slice)) else super().__getitem__(key)

    def __iter__(self):
        return iter(self._values)


def _normalize_scalar(value):
    """Map psycopg ``numeric`` (returned as ``Decimal``) to the SQLite-equivalent scalar the
    shared readers expect: ``int`` for whole values, ``float`` otherwise. PostgreSQL's
    aggregate/numeric types otherwise leak ``Decimal`` into rows (e.g. ``SUM(int8)`` is
    ``numeric``), which the JSON encoders and int-typed readers assume are plain scalars."""
    if isinstance(value, Decimal):
        whole = value.to_integral_value()
        if whole == value:
            return int(whole)
        return float(value)
    return value


def _row_factory(cursor):
    columns = [column.name for column in cursor.description] if cursor.description else []
    return lambda values: PostgresRow(columns, tuple(_normalize_scalar(v) for v in values))


class PostgresCursor:
    """The DB-API surface shared readers use, over a psycopg cursor.

    ``lastrowid`` synthesizes SQLite's identity-sequence value for ``INSERT`` statements
    (psycopg reports no lastrowid); ``total_changes``/``in_transaction`` are connection-level
    counters the shared code reads through the wrapper.
    """

    def __init__(self, wrapper: PostgresConnection, cursor):
        self._wrapper = wrapper
        self._cursor = cursor

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def fetchmany(self, size=None):
        return self._cursor.fetchmany(size)

    def __iter__(self):
        return iter(self._cursor)

    def __next__(self):
        return next(self._cursor)

    def close(self):
        self._cursor.close()

    @property
    def rowcount(self):
        return self._cursor.rowcount

    @property
    def description(self):
        return self._cursor.description

    @property
    def lastrowid(self):
        table = self._wrapper._last_insert_table
        if table is None:
            return None
        sequence = self._wrapper._identity_sequence(table)
        if sequence is None:
            return None
        row = self._wrapper._conn.execute(
            sql.SQL("SELECT last_value FROM {}").format(sql.Identifier(*sequence))
        ).fetchone()
        return None if row is None else row[0]


class PostgresConnection:
    """The DB-API operations used by shared session methods, over a borrowed connection."""

    def __init__(self, conn):
        self._conn = conn
        self._total_changes = 0
        self._last_insert_table: Optional[str] = None
        self._sequence_cache: dict[str, Optional[tuple[str, str]]] = {}
        self._pk_cache: dict[str, tuple[str, ...]] = {}

    def _identity_sequence(self, table: str) -> Optional[tuple[str, str]]:
        """(schema, name) of the identity sequence backing ``table.id``; None when absent."""
        if table not in self._sequence_cache:
            try:
                row = self._conn.execute(
                    "SELECT n.nspname, c.relname FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE c.oid = pg_get_serial_sequence(%s, 'id')::regclass", (table,)
                ).fetchone()
            except psycopg.Error:
                row = None  # no identity sequence on id: lastrowid stays None
            self._sequence_cache[table] = None if row is None else (row[0], row[1])
        return self._sequence_cache[table]

    def _primary_key_columns(self, table: str) -> tuple[str, ...]:
        """Primary-key columns of *table* in the current schema (empty tuple when none)."""
        if table not in self._pk_cache:
            rows = self._conn.execute(
                "SELECT a.attname FROM pg_index i "
                "JOIN pg_class t ON t.oid = i.indrelid "
                "JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(i.indkey) "
                "JOIN pg_namespace n ON n.oid = t.relnamespace "
                "WHERE n.nspname = current_schema() AND t.relname = %s AND i.indisprimary "
                "ORDER BY array_position(i.indkey, a.attnum)", (table,)
            ).fetchall()
            self._pk_cache[table] = tuple(row[0] for row in rows)
        return self._pk_cache[table]

    def _statement(self, statement: str) -> str:
        converted = _convert_statement(statement)
        if not _INSERT_OR_REPLACE_RE.search(converted):
            return converted
        match = re.match(r"\s*INSERT\s+OR\s+REPLACE\s+INTO\s+[\"']?(\w+)", converted, re.IGNORECASE)
        if not match:
            raise UnsupportedSQLError(f"cannot rewrite INSERT OR REPLACE: {converted[:80]}")
        table = match.group(1)
        cols = re.match(r"\s*INSERT\s+OR\s+REPLACE\s+INTO\s+[\"']?\w+[\"']?\s*\(([^)]*)\)", converted, re.IGNORECASE)
        if cols:
            columns = [c.strip().strip('"') for c in cols.group(1).split(",")]
        else:
            rows = self._conn.execute(
                "SELECT a.attname FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = current_schema() "
                "AND c.relname = %s AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum", (table,)
            ).fetchall()
            columns = [row[0] for row in rows]
        if not columns:
            raise UnsupportedSQLError(f"cannot rewrite INSERT OR REPLACE into {table}: no columns")
        primary = self._primary_key_columns(table)
        if not primary:
            raise UnsupportedSQLError(f"cannot rewrite INSERT OR REPLACE into {table}: no primary key")
        non_pk = [c for c in columns if c not in primary]
        if not non_pk:
            raise UnsupportedSQLError(f"cannot rewrite INSERT OR REPLACE into {table}: no non-key columns")
        on_conflict = " ON CONFLICT ({}) DO UPDATE SET ".format(", ".join(primary)) + ", ".join(
            f"{c} = excluded.{c}" for c in non_pk)
        return re.sub(r"INSERT\s+OR\s+REPLACE\s+INTO", "INSERT INTO", converted, count=1) + on_conflict

    def execute(self, statement: str, params=()):
        converted = self._statement(statement)
        if _ORDINAL_PLACEHOLDER_RE.search(converted):
            params = _ordinalize_params(params)
        cursor = self._conn.execute(converted, params)
        table_match = re.match(r"\s*INSERT\s+INTO\s+[\"']?(\w+)", converted, re.IGNORECASE)
        self._last_insert_table = table_match.group(1) if table_match else None
        if cursor.rowcount is not None and cursor.rowcount >= 0:
            self._total_changes += cursor.rowcount
        return PostgresCursor(self, cursor)

    def executemany(self, statement: str, params):
        converted = self._statement(statement)
        if _ORDINAL_PLACEHOLDER_RE.search(converted):
            params = [_ordinalize_params(row) for row in params]
        with self._conn.cursor() as cursor:
            cursor.executemany(converted, params)
        if cursor.rowcount is not None and cursor.rowcount >= 0:
            self._total_changes += cursor.rowcount

    def executescript(self, statement: str):
        """Multi-statement DDL (telegram topic tables); statements are executed as-is."""
        self._conn.execute(statement)

    def cursor(self):
        return self._conn.cursor()

    @property
    def in_transaction(self):
        # psycopg3 exposes the state via the connection info, not a DB-API attribute:
        # a read holds the explicit ``conn.transaction()`` block (INTRANS) or its error
        # continuation (INERROR), matching SQLite's ``in_transaction`` inside the read ctx.
        return self._conn.info.transaction_status in (
            psycopg.pq.TransactionStatus.INTRANS, psycopg.pq.TransactionStatus.INERROR)

    @property
    def total_changes(self):
        return self._total_changes


class PostgresSessionDB(SessionPostgresSearchMixin, SessionPostgresMaintenanceMixin, SessionDB):
    """SessionDB over PostgreSQL: same API, server-side storage.

    SQLite-only lifecycle machinery (file identity, WAL generation, read pool, corruption
    quarantine) does not apply; the attributes those paths read are set to their neutral
    values so inherited guards stay inert.
    """

    backend = "postgres"
    database_errors = (psycopg.Error,)
    database_operational_errors = (psycopg.OperationalError, psycopg.ProgrammingError)
    _unlimited_sql_limit = None  # PostgreSQL: LIMIT NULL means no limit
    _messages_session_index_hint = ""  # no INDEXED BY in PostgreSQL
    _CONTENT_BYTES_PREFIX = "\x01bytes:"
    # PostgreSQL TEXT cannot store NUL: the JSON dialect's marker uses \x01 instead, and any
    # string that would carry a raw NUL (or collide with a marker) is JSON-string-wrapped.
    _CONTENT_JSON_PREFIX = "\x01json:"

    @classmethod
    def _encode_content(cls, content: Any) -> Any:
        """Bytes cannot ride a TEXT column: base64 them behind a reserved marker. Strings with
        raw NUL (or already carrying a reserved marker) are JSON-string-wrapped so the stored
        form is NUL-free and the read path returns them verbatim. Everything else (str, None,
        scalars, the JSON dialect for list/dict) is the super's dialect unchanged."""
        if isinstance(content, bytes):
            return cls._CONTENT_BYTES_PREFIX + base64.b64encode(content).decode("ascii")
        if isinstance(content, str) and ("\x00" in content
                                         or content.startswith((cls._CONTENT_JSON_PREFIX,
                                                                cls._CONTENT_BYTES_PREFIX))):
            return cls._CONTENT_JSON_PREFIX + json.dumps(content)
        return super()._encode_content(content)

    @classmethod
    def _decode_content(cls, content: Any) -> Any:
        if isinstance(content, str) and content.startswith(cls._CONTENT_BYTES_PREFIX):
            return base64.b64decode(content[len(cls._CONTENT_BYTES_PREFIX):])
        return super()._decode_content(content)

    def _holder_process_is_dead(self, holder):
        # Holders live on other hosts than this process; a local liveness probe would read
        # every live foreign holder as dead and race its lock reclamation.
        return False

    def __init__(self, db_path: Path | None = None, read_only: bool = False, *, database_settings=None):
        self.db_path = Path(db_path or _default_db_path())
        self.read_only = read_only
        self._database_settings = (database_settings or getattr(self, "_database_settings", None)
                                   or resolve_database_settings(self.db_path))
        settings = self._database_settings
        if settings.backend != "postgres":
            raise ValueError("PostgresSessionDB requires database.backend: postgres")
        self.schema = settings.schema
        self._creation_site = "postgres"
        self._closed = False
        self._shared_registry_owned = False
        self._lock = threading.RLock()
        self._conn = None
        self._db_file_identity = None
        self._db_file_application_id = 0
        self._db_sidecar_identity = {}
        self._db_replaced = self._db_wal_generation_lost = False
        self._wal_active = False
        self._db_corrupt, self._db_corrupt_reason = False, ""
        self._fts_enabled = self._fts_stale = False
        self._fts_cjk_loaded = self._fts_cjk_available = self._trigram_available = False
        self._fts_unavailable_warned = False
        self._fts_usermerge_floor_applied = False
        self._write_count = 0
        self._read_conns_closed = False
        # Async token accounting (hermes_state_usage): same queue/thread contract as SQLite.
        self._token_queue: deque = deque()
        self._token_queue_cond = threading.Condition(threading.Lock())
        self._token_writer_thread: Optional[threading.Thread] = None
        self._token_writer_stop = self._token_writer_busy = False
        self._token_atexit_hook: Optional[Callable[[], None]] = None
        self._pool: Optional[ConnectionPool] = None

        kwargs = dict(autocommit=True, row_factory=_row_factory, connect_timeout=5,
                      application_name="hermes-agent")
        schema = self.schema

        def configure(conn):
            conn.execute(sql.SQL("SET search_path TO {}, pg_catalog").format(sql.Identifier(schema)))
            if read_only:
                conn.execute("SET default_transaction_read_only = on")

        try:
            with psycopg.connect(settings.database_url, **kwargs) as conn:
                if conn.info.server_version < 170000:
                    raise ValueError("PostgreSQL session storage requires PostgreSQL 17 or later")
                configure(conn)
                self._initialize_schema(conn)
            self._pool = ConnectionPool(settings.database_url, kwargs=kwargs, configure=configure,
                                        min_size=0, max_size=4, timeout=5, open=True)
        except psycopg.Error as exc:
            self._closed = True
            from hermes_state_guard import _set_last_init_error
            message = f"PostgreSQL session database could not be opened ({type(exc).__name__})"
            _set_last_init_error(message)
            raise RuntimeError(message) from None
        except Exception:
            self._closed = True
            raise

    def _initialize_schema(self, conn) -> None:
        """Create the profile schema once (advisory-locked); verify the version on re-open.

        All DDL runs in one transaction, so a crash mid-init leaves nothing partial: the
        version table's presence is the single completeness marker.
        """
        if self.read_only:
            version = conn.execute("SELECT version FROM postgres_schema_version").fetchone()[0]
            if version != POSTGRES_SCHEMA_VERSION:
                raise RuntimeError("PostgreSQL schema requires a writable upgrade")
            return
        with conn.transaction():
            conn.execute("SELECT set_config('lock_timeout', '5000', true)")
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (self.schema,))
            conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
            exists = conn.execute("SELECT to_regclass('postgres_schema_version')").fetchone()[0]
            if exists is not None:
                version = conn.execute("SELECT version FROM postgres_schema_version").fetchone()[0]
                if version != POSTGRES_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Unsupported PostgreSQL schema version {version} "
                        f"(this build supports {POSTGRES_SCHEMA_VERSION}); restore a backup "
                        "or run the migration tool against a fresh database")
                return
            conn.execute(POSTGRES_SCHEMA_SQL)
            conn.execute(FUNCTION_SQL)
            conn.execute(DISPLAY_SQL)
            conn.execute(SEARCH_SQL)
            conn.execute("INSERT INTO schema_version(version) VALUES (%s)", (SCHEMA_VERSION,))
            conn.execute("CREATE TABLE postgres_schema_version (version BIGINT PRIMARY KEY)")
            conn.execute("INSERT INTO postgres_schema_version(version) VALUES (%s)",
                         (POSTGRES_SCHEMA_VERSION,))

    @contextmanager
    def _read_ctx(self, *, reopen: bool = True) -> Iterator[PostgresConnection]:
        if self._closed:
            raise RuntimeError(f"PostgreSQL connection is closed for {self.db_path}")
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute("SET TRANSACTION READ ONLY")
                yield PostgresConnection(conn)

    def _read_retrying_ioerr(self, fn: Callable[[PostgresConnection], T]) -> T:
        # The SQLite IOERR retry covers a WAL transition window that has no PostgreSQL
        # equivalent; a failed read propagates.
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

    def _execute_write(self, fn: Callable[[PostgresConnection], T], patience_s: Optional[float] = None) -> T:
        """One write transaction under the profile's advisory lock.

        The lock serializes this profile's writers (the advisory lock IS the write lock);
        lock_timeout caps a stuck peer at the patience budget, after which the not-yet-run
        callback is retried with jitter exactly like the SQLite lock path.
        """
        if self.read_only:
            raise RuntimeError("SessionDB is read-only")
        if self._closed:
            raise RuntimeError("SessionDB connection is closed")
        patience = self._WRITE_PATIENCE_S if patience_s is None else patience_s
        deadline = time.monotonic() + patience
        compression_deadline: Optional[float] = None
        while True:
            fn_started = False
            try:
                with self._pool.connection() as conn:
                    with conn.transaction():
                        remaining = max(1, int((deadline - time.monotonic()) * 1000))
                        conn.execute("SELECT set_config('lock_timeout', %s, true)", (str(remaining),))
                        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                                     (self.schema,))
                        fn_started = True
                        result = fn(PostgresConnection(conn))
                self._write_count += 1
                return result
            except SessionCompressionInProgressError:
                # A steer landing mid-compression must not abort: the compressor publishes in
                # seconds. Same short budget as the SQLite path; a still-locked-out writer is
                # refused (the lease is a correctness boundary, not a busy signal).
                if compression_deadline is None:
                    compression_deadline = min(time.monotonic() + self._COMPRESSION_BUSY_WAIT_S, deadline)
                if self._sleep_before_write_retry(compression_deadline, self._COMPRESSION_BUSY_WAIT_S):
                    continue
                raise
            except psycopg.errors.LockNotAvailable:
                if fn_started:
                    raise  # settlement unknown: propagate
                if self._sleep_before_write_retry(deadline, patience):
                    continue
                raise
            except psycopg.OperationalError:
                if fn_started:
                    raise  # a mid-write connection failure is not replay-safe
                if self._sleep_before_write_retry(deadline, patience):
                    continue
                raise

    def _message_column_names(self, conn) -> list[str]:
        """Messages columns a plain INSERT may bind: identity-sequence ``id`` and generated
        ``search_vector`` excluded (the shared twin-copy builds its column list from this)."""
        if not hasattr(self, "_message_columns_cache"):
            rows = conn.execute(
                "SELECT a.attname FROM pg_attribute a "
                "JOIN pg_class c ON c.oid = a.attrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = ? AND c.relname = 'messages' "
                "AND a.attnum > 0 AND NOT a.attisdropped AND a.attgenerated = '' "
                "ORDER BY a.attnum", (self.schema,)).fetchall()
            self._message_columns_cache = [row[0] for row in rows]
        return self._message_columns_cache

    def get_meta(self, key: str) -> Optional[str]:
        """Read state_meta[key] (the SQLite path reads on the writer connection; PostgreSQL
        reads through the read seam)."""
        row = self._read_one("SELECT value FROM state_meta WHERE key = ?", (key,))
        return None if row is None else row[0]

    def get_change_revision(self) -> Optional[int]:
        """Cross-process write signal: the schema's bump triggers advance this on every
        sessions/messages statement, so TUI/web readers poll it instead of file mtimes."""
        row = self._read_one("SELECT revision FROM state_change_revision WHERE id = 1")
        return None if row is None else row[0]

    def _legacy_display_page(self, session_id: str, *, active_clause: str, limit: Optional[int],
                             offset: int, latest: bool) -> list:
        """PostgreSQL persists display identities on every write (trigger + backfill), so the
        fixed-width legacy projection has nothing extra to protect: reuse the deduplicated
        display projection."""
        with self._read_ctx() as conn:
            return self._display_rows_from_conn(conn, session_id, limit=limit, offset=offset,
                                                latest=latest)

    def apply_telegram_topic_migration(self) -> None:
        def migrate(conn):
            conn.executescript(TELEGRAM_SCHEMA_SQL)
            self.set_meta("telegram_dm_topic_schema_version", "3", cursor=conn)

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
            raise TimeoutError("SessionDB token writer is still draining")
        if self._token_atexit_hook is not None:
            atexit.unregister(self._token_atexit_hook)
            self._token_atexit_hook = None
        self._closed = True
        if self._pool is not None:
            self._pool.close()

    def __del__(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.close()
