"""Unit tests for plugins.memory.mem0._entity_sidecar.

All tests mock _entity_sidecar._connect so no real PostgreSQL connection
is made. Tests cover the bug fixes:
  - upsert() and _write() close the connection in a try/finally
  - init_db() failure is logged at WARNING (not debug)
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch, call

import pytest

import plugins.memory.mem0._entity_sidecar as sidecar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conn(cursor_raises=None):
    """Return a (mock_conn, mock_cursor) pair.

    If cursor_raises is set, mock_cursor.execute() raises that exception.
    The cursor is returned from conn.cursor() as a context manager.
    """
    mock_cur = MagicMock()
    if cursor_raises is not None:
        mock_cur.execute.side_effect = cursor_raises
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    # Support `with conn:` context manager
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    return mock_conn, mock_cur


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_lowercases(self):
        assert sidecar.normalize("HELLO") == "hello"

    def test_strips_whitespace(self):
        assert sidecar.normalize("  foo  ") == "foo"

    def test_lowercases_and_strips(self):
        assert sidecar.normalize("  Hello World  ") == "hello world"

    def test_empty_string(self):
        assert sidecar.normalize("") == ""


# ---------------------------------------------------------------------------
# lookup()
# ---------------------------------------------------------------------------

class TestLookup:

    def test_returns_point_id_when_found(self):
        mock_conn, mock_cur = _make_conn()
        mock_cur.fetchone.return_value = ("point-abc",)

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            result = sidecar.lookup("user1", "entity-key")

        assert result == "point-abc"

    def test_returns_none_when_not_found(self):
        mock_conn, mock_cur = _make_conn()
        mock_cur.fetchone.return_value = None

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            result = sidecar.lookup("user1", "missing-key")

        assert result is None

    def test_executes_correct_sql(self):
        mock_conn, mock_cur = _make_conn()
        mock_cur.fetchone.return_value = None

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            sidecar.lookup("u1", "k1")

        mock_cur.execute.assert_called_once()
        sql, params = mock_cur.execute.call_args[0]
        assert "qdrant_point_id" in sql
        assert "mem0_entity_names" in sql
        assert params == ("u1", "k1")

    def test_returns_none_and_logs_warning_on_exception(self, caplog):
        with patch.object(sidecar, "_connect", side_effect=Exception("db down")):
            with caplog.at_level(logging.WARNING, logger="plugins.memory.mem0._entity_sidecar"):
                result = sidecar.lookup("u1", "k1")

        assert result is None
        assert any("sidecar lookup error" in r.message for r in caplog.records)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_conn_closed_on_success(self):
        mock_conn, mock_cur = _make_conn()
        mock_cur.fetchone.return_value = ("id-1",)

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            sidecar.lookup("u1", "k1")

        mock_conn.close.assert_called_once()

    def test_conn_closed_even_when_cursor_raises(self):
        mock_conn, mock_cur = _make_conn(cursor_raises=RuntimeError("boom"))

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            result = sidecar.lookup("u1", "k1")

        # Exception is caught at the outer level; conn.close still called
        mock_conn.close.assert_called_once()
        assert result is None


# ---------------------------------------------------------------------------
# upsert()
# ---------------------------------------------------------------------------

class TestUpsert:

    def test_executes_insert_sql(self):
        mock_conn, mock_cur = _make_conn()

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            sidecar.upsert("u1", "key1", "point-1")

        mock_cur.execute.assert_called_once()
        sql, params = mock_cur.execute.call_args[0]
        assert "INSERT INTO mem0_entity_names" in sql
        assert "ON CONFLICT" in sql
        assert params == ("u1", "key1", "point-1")

    def test_conn_close_called_on_success(self):
        mock_conn, _ = _make_conn()

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            sidecar.upsert("u1", "key1", "point-1")

        mock_conn.close.assert_called_once()

    def test_conn_close_called_even_when_cursor_raises(self):
        """try/finally fix: conn.close() must be called even if cursor.execute raises."""
        mock_conn, mock_cur = _make_conn(cursor_raises=RuntimeError("cursor failed"))

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            sidecar.upsert("u1", "key1", "point-1")  # should not propagate

        mock_conn.close.assert_called_once()

    def test_logs_warning_on_exception(self, caplog):
        mock_conn, mock_cur = _make_conn(cursor_raises=RuntimeError("insert failed"))

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            with caplog.at_level(logging.WARNING, logger="plugins.memory.mem0._entity_sidecar"):
                sidecar.upsert("u1", "key1", "point-1")

        assert any("sidecar upsert error" in r.message for r in caplog.records)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_connect_failure_logs_warning(self, caplog):
        with patch.object(sidecar, "_connect", side_effect=Exception("no db")):
            with caplog.at_level(logging.WARNING, logger="plugins.memory.mem0._entity_sidecar"):
                sidecar.upsert("u1", "key1", "point-1")

        assert any("sidecar upsert error" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _write()
# ---------------------------------------------------------------------------

class TestWrite:

    def test_conn_close_called_on_success(self):
        mock_conn, _ = _make_conn()

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            sidecar._write("DELETE FROM mem0_entity_names WHERE qdrant_point_id=%s", ("p1",))

        mock_conn.close.assert_called_once()

    def test_conn_close_called_even_when_cursor_raises(self):
        """try/finally fix: conn.close() must be called even if cursor.execute raises."""
        mock_conn, mock_cur = _make_conn(cursor_raises=RuntimeError("write failed"))

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            sidecar._write("DELETE FROM mem0_entity_names WHERE qdrant_point_id=%s", ("p1",))

        mock_conn.close.assert_called_once()

    def test_logs_warning_on_cursor_exception(self, caplog):
        mock_conn, mock_cur = _make_conn(cursor_raises=RuntimeError("write failed"))

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            with caplog.at_level(logging.WARNING, logger="plugins.memory.mem0._entity_sidecar"):
                sidecar._write("DELETE FROM mem0_entity_names WHERE qdrant_point_id=%s", ("p1",))

        assert any("sidecar write error" in r.message for r in caplog.records)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_logs_warning_on_connect_exception(self, caplog):
        with patch.object(sidecar, "_connect", side_effect=Exception("no connection")):
            with caplog.at_level(logging.WARNING, logger="plugins.memory.mem0._entity_sidecar"):
                sidecar._write("DELETE FROM mem0_entity_names WHERE qdrant_point_id=%s", ("p1",))

        assert any("sidecar write error" in r.message for r in caplog.records)

    def test_executes_given_sql_and_params(self):
        mock_conn, mock_cur = _make_conn()
        sql = "DELETE FROM mem0_entity_names WHERE qdrant_point_id=%s"
        params = ("p99",)

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            sidecar._write(sql, params)

        mock_cur.execute.assert_called_once_with(sql, params)


# ---------------------------------------------------------------------------
# init_db()
# ---------------------------------------------------------------------------

class TestInitDb:

    @pytest.fixture(autouse=True)
    def _reset_initialized(self):
        """Reset _INITIALIZED between tests so each test gets a clean slate."""
        original = sidecar._INITIALIZED
        sidecar._INITIALIZED = False
        yield
        sidecar._INITIALIZED = original

    def test_logs_at_warning_on_exception(self, caplog):
        """Bug fix: init_db() must log at WARNING level (not debug) on failure."""
        with patch.object(sidecar, "_connect", side_effect=Exception("pg down")):
            with caplog.at_level(logging.WARNING, logger="plugins.memory.mem0._entity_sidecar"):
                sidecar.init_db()

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, "Expected at least one WARNING log record"
        assert any("sidecar init_db error" in r.message for r in warning_records)

    def test_no_debug_only_on_exception(self, caplog):
        """Ensure the failure is NOT only logged at DEBUG (which would be the old bug)."""
        with patch.object(sidecar, "_connect", side_effect=Exception("pg down")):
            with caplog.at_level(logging.DEBUG, logger="plugins.memory.mem0._entity_sidecar"):
                sidecar.init_db()

        # There should be no record at DEBUG but not WARNING for init_db
        for r in caplog.records:
            if "sidecar init_db error" in r.message:
                assert r.levelno >= logging.WARNING, (
                    f"init_db error logged at {r.levelname}, expected WARNING or above"
                )

    def test_sets_initialized_true_on_success(self):
        mock_conn, _ = _make_conn()

        with patch.object(sidecar, "_connect", return_value=mock_conn):
            sidecar.init_db()

        assert sidecar._INITIALIZED is True

    def test_skips_if_already_initialized(self):
        sidecar._INITIALIZED = True
        connect_mock = MagicMock()

        with patch.object(sidecar, "_connect", connect_mock):
            sidecar.init_db()

        connect_mock.assert_not_called()

    def test_initialized_remains_false_on_failure(self):
        with patch.object(sidecar, "_connect", side_effect=Exception("pg down")):
            sidecar.init_db()

        assert sidecar._INITIALIZED is False
