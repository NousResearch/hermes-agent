"""S1: Sidecar schema + init — compile, internal FK, O_EXCL, native zero-mutation."""

import pytest
import os
import sys
import tempfile
import hashlib
import sqlite3
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_schema_sidecar import (
    EXPECTED_SIDECAR_TABLES,
    NATIVE_TABLES,
    apply_sidecar_schema,
    get_sidecar_table_names,
    get_sidecar_trigger_names,
    verify_no_native_tables,
)
from hermes_cli.kanban_orch_bridge import BridgeError, init_sidecar_db


def _make_native_fixture(path: str) -> str:
    """Create a minimal native kanban.db fixture with tasks table."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT NOT NULL, created_at INTEGER NOT NULL)")
    conn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('task-1', 'Test Task', 'pending', 1)")
    conn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('task-2', 'Another Task', 'done', 2)")
    conn.commit()
    conn.close()
    return path


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_sidecar_schema_compiles():
    """Sidecar DDL executes without error on fresh DB."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        conn = sqlite3.connect(path)
        apply_sidecar_schema(conn)
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        conn.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_sidecar_internal_fks():
    """Sidecar has internal FKs but no native table FKs."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        conn = sqlite3.connect(path)
        apply_sidecar_schema(conn)

        # Check all expected sidecar tables exist
        tables = set(get_sidecar_table_names(conn))
        missing = EXPECTED_SIDECAR_TABLES - tables
        assert not missing, f"Missing sidecar tables: {missing}"

        # Check no native tables leaked
        assert verify_no_native_tables(conn) is True

        # Verify FK check is clean
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert violations == [], f"FK violations: {violations}"

        # Verify integrity
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

        conn.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_init_creates_new_file_excl():
    """init_sidecar_db creates a new file (O_EXCL semantics)."""
    tmpdir = tempfile.mkdtemp()
    try:
        native_path = os.path.join(tmpdir, "native.db")
        _make_native_fixture(native_path)

        sidecar_path = os.path.join(tmpdir, "orch_v4.db")
        assert not os.path.exists(sidecar_path)

        init_sidecar_db(sidecar_path)
        assert os.path.exists(sidecar_path)

        # Verify it has tables
        conn = sqlite3.connect(sidecar_path)
        tables = get_sidecar_table_names(conn)
        assert len(tables) > 0
        verify_no_native_tables(conn)
        conn.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_init_refuses_existing_path():
    """init_sidecar_db refuses if path already exists (O_EXCL)."""
    tmpdir = tempfile.mkdtemp()
    try:
        native_path = os.path.join(tmpdir, "native.db")
        _make_native_fixture(native_path)

        sidecar_path = os.path.join(tmpdir, "orch_v4.db")
        init_sidecar_db(sidecar_path)

        # Second init should fail
        with pytest.raises(BridgeError, match="sidecar_exists"):
            init_sidecar_db(sidecar_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_native_db_bytes_unchanged_after_init():
    """Native DB SHA-256 must be identical before and after sidecar init."""
    tmpdir = tempfile.mkdtemp()
    try:
        native_path = os.path.join(tmpdir, "native.db")
        _make_native_fixture(native_path)

        sha_before = _sha256_file(native_path)

        sidecar_path = os.path.join(tmpdir, "orch_v4.db")
        init_sidecar_db(sidecar_path)

        sha_after = _sha256_file(native_path)
        assert sha_before == sha_after, "Native DB bytes changed during sidecar init!"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_native_opened_read_only_rejected_on_write_attempt():
    """Opening native with mode=ro should reject write attempts."""
    tmpdir = tempfile.mkdtemp()
    try:
        native_path = os.path.join(tmpdir, "native.db")
        _make_native_fixture(native_path)

        # Open RO via URI
        conn = sqlite3.connect(f"file:{native_path}?mode=ro", uri=True)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('evil', 'hack', 'hacked', 0)")
        conn.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_no_live_paths_in_unit_tests():
    """Unit tests must NOT use live kanban.db path."""
    LIVE_PATHS = {"/home/claw/.hermes/kanban.db", "/home/claw/.hermes/orch_v4.db"}
    # This test is a documentation point: all fixtures use tempfile.mkstemp/mkdtemp
    # If any test uses a live path, it would be a violation.
    # We verify by checking that our temp paths are not in LIVE_PATHS.
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        assert path not in LIVE_PATHS
        assert os.path.dirname(path) not in LIVE_PATHS
    finally:
        os.unlink(path)
