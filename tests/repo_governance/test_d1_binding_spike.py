"""D1.1 disposable binding spike for repository mutation governance.

This file is feasibility evidence only. It does not activate governance,
create a live store, enroll a repository, or define the D2 tool seam.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import fcntl
import os
import sqlite3
import stat
import sys
from pathlib import Path

import apsw
import pytest


EXPECTED_SQLITE_VERSION = "3.53.1"
EXPECTED_SQLITE_SOURCE_ID = (
    "2026-05-05 10:34:17 "
    "c88b22011a54b4f6fbd149e9f8e4de77658ce58143a1af0e3785e4e6475127e9"
)
ACL_TYPE_EXTENDED = 0x00000100


def _rows(connection: apsw.Connection, sql: str) -> list[tuple[object, ...]]:
    return list(connection.execute(sql))


def _pragma_scalar(connection: apsw.Connection, name: str) -> object:
    rows = _rows(connection, f"PRAGMA {name}")
    assert len(rows) == 1
    assert len(rows[0]) == 1
    return rows[0][0]


def _darwin_extended_acl_state(fd: int) -> tuple[str, int]:
    """Return the exact Darwin extended-ACL query disposition for *fd*.

    Darwin reports a regular file with no extended ACL as NULL/ENOENT. A
    non-NULL ACL object means entries may exist and must be inspected by a
    future production implementation. Every other failure is indeterminate.
    """

    libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
    acl_get_fd_np = libc.acl_get_fd_np
    acl_get_fd_np.argtypes = [ctypes.c_int, ctypes.c_int]
    acl_get_fd_np.restype = ctypes.c_void_p
    acl_free = libc.acl_free
    acl_free.argtypes = [ctypes.c_void_p]
    acl_free.restype = ctypes.c_int

    ctypes.set_errno(0)
    acl = acl_get_fd_np(fd, ACL_TYPE_EXTENDED)
    query_errno = ctypes.get_errno()
    if acl:
        assert acl_free(acl) == 0
        return "acl-object-present", query_errno
    if query_errno == errno.ENOENT:
        return "no-extended-acl", query_errno
    return "indeterminate", query_errno


def test_stdlib_sqlite_binding_cannot_satisfy_defensive_mode_contract() -> None:
    assert not hasattr(sqlite3.Connection, "setconfig")
    assert not hasattr(sqlite3, "SQLITE_DBCONFIG_DEFENSIVE")


def test_apsw_exact_candidate_exposes_defensive_mode_with_readback() -> None:
    assert apsw.apswversion() == "3.53.1.0"
    assert apsw.sqlitelibversion() == EXPECTED_SQLITE_VERSION
    assert apsw.sqlite3_sourceid() == EXPECTED_SQLITE_SOURCE_ID

    connection = apsw.Connection(":memory:")
    try:
        initial = connection.config(apsw.SQLITE_DBCONFIG_DEFENSIVE, -1)
        assert initial in (0, 1)
        assert connection.config(apsw.SQLITE_DBCONFIG_DEFENSIVE, 1) == 1
        assert connection.config(apsw.SQLITE_DBCONFIG_DEFENSIVE, -1) == 1
    finally:
        connection.close()


def test_stdlib_and_apsw_demonstrate_distinct_evaluator_identities() -> None:
    stdlib_connection = sqlite3.connect(":memory:")
    try:
        stdlib_source_id = stdlib_connection.execute(
            "SELECT sqlite_source_id()"
        ).fetchone()[0]
        stdlib_options = {
            row[0] for row in stdlib_connection.execute("PRAGMA compile_options")
        }
    finally:
        stdlib_connection.close()

    apsw_options = set(apsw.compile_options)
    assert stdlib_source_id == apsw.sqlite3_sourceid() == EXPECTED_SQLITE_SOURCE_ID
    assert stdlib_options != apsw_options
    assert "MAX_ATTACHED=10" in stdlib_options
    assert "MAX_ATTACHED=125" in apsw_options


def test_all_connection_local_settings_precede_begin_immediate(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "ordering.db"
    connection = apsw.Connection(str(database_path))
    try:
        # v0.11 exact API table: these three fields use sqlite3_db_config,
        # not similarly named PRAGMAs.
        assert connection.config(apsw.SQLITE_DBCONFIG_TRUSTED_SCHEMA, 0) == 0
        assert connection.config(apsw.SQLITE_DBCONFIG_TRUSTED_SCHEMA, -1) == 0
        assert connection.config(apsw.SQLITE_DBCONFIG_DEFENSIVE, 1) == 1
        assert connection.config(apsw.SQLITE_DBCONFIG_DEFENSIVE, -1) == 1
        assert (
            connection.config(apsw.SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, 0) == 0
        )
        assert (
            connection.config(apsw.SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, -1) == 0
        )

        # v0.11 exact API table: these three fields use set/readback PRAGMAs.
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA recursive_triggers=OFF")
        assert _pragma_scalar(connection, "synchronous") == 2
        assert _pragma_scalar(connection, "foreign_keys") == 1
        assert _pragma_scalar(connection, "recursive_triggers") == 0

        connection.execute("BEGIN IMMEDIATE")
        assert connection.config(apsw.SQLITE_DBCONFIG_TRUSTED_SCHEMA, -1) == 0
        assert connection.config(apsw.SQLITE_DBCONFIG_DEFENSIVE, -1) == 1
        assert (
            connection.config(apsw.SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, -1) == 0
        )
        assert _pragma_scalar(connection, "synchronous") == 2
        assert _pragma_scalar(connection, "foreign_keys") == 1
        assert _pragma_scalar(connection, "recursive_triggers") == 0

        with pytest.raises(apsw.SQLError, match="Safety level may not be changed"):
            connection.execute("PRAGMA synchronous=NORMAL")

        connection.execute("PRAGMA foreign_keys=OFF")
        assert _pragma_scalar(connection, "foreign_keys") == 1
        connection.execute("ROLLBACK")
    finally:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        connection.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin D1.1 witness spike")
def test_darwin_anchored_witness_capabilities(tmp_path: Path) -> None:
    assert hasattr(os, "O_NOFOLLOW")
    assert hasattr(os, "O_DIRECTORY")
    assert hasattr(fcntl, "F_FULLFSYNC")

    root_path = tmp_path / "governance-root"
    root_path.mkdir(mode=0o700)
    os.chmod(root_path, 0o700)

    root_fd = os.open(
        root_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    witness_fd = -1
    try:
        root_before = os.fstat(root_fd)
        assert stat.S_ISDIR(root_before.st_mode)
        assert stat.S_IMODE(root_before.st_mode) == 0o700

        witness_fd = os.open(
            "witness.log",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        payload = b"repo-governance-d1-witness-probe\x00"
        assert os.write(witness_fd, payload) == len(payload)
        assert fcntl.fcntl(witness_fd, fcntl.F_FULLFSYNC) == 0
        assert os.pread(witness_fd, len(payload), 0) == payload

        witness_stat = os.fstat(witness_fd)
        path_stat = os.stat("witness.log", dir_fd=root_fd, follow_symlinks=False)
        assert stat.S_ISREG(witness_stat.st_mode)
        assert stat.S_IMODE(witness_stat.st_mode) == 0o600
        assert witness_stat.st_nlink == 1
        assert (witness_stat.st_dev, witness_stat.st_ino) == (
            path_stat.st_dev,
            path_stat.st_ino,
        )
        assert witness_stat.st_blocks * 512 >= witness_stat.st_size
        assert getattr(witness_stat, "st_flags", 0) == 0
        assert _darwin_extended_acl_state(witness_fd) == (
            "no-extended-acl",
            errno.ENOENT,
        )

        root_after = os.fstat(root_fd)
        assert (root_before.st_dev, root_before.st_ino) == (
            root_after.st_dev,
            root_after.st_ino,
        )
    finally:
        if witness_fd >= 0:
            os.close(witness_fd)
        os.close(root_fd)


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin D1.1 witness spike")
def test_darwin_nofollow_rejects_witness_symlink(tmp_path: Path) -> None:
    root_path = tmp_path / "governance-root"
    root_path.mkdir(mode=0o700)
    target_path = tmp_path / "outside"
    target_path.write_bytes(b"outside")
    (root_path / "witness.log").symlink_to(target_path)

    root_fd = os.open(
        root_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        with pytest.raises(OSError) as raised:
            os.open(
                "witness.log",
                os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=root_fd,
            )
        assert raised.value.errno == errno.ELOOP
    finally:
        os.close(root_fd)
