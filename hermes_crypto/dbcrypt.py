"""SQLCipher integration for whole-database encryption.

``state.db`` (conversation history) and ``kanban.db`` are SQLite databases.
Encrypting them whole-file with SQLCipher — rather than encrypting individual
column values — keeps the FTS5 full-text indexes working, so ``session_search``
is unaffected.

SQLCipher is provided by the optional ``sqlcipher3-wheels`` wheel. The DEK is
passed as a raw key via ``PRAGMA key = "x'<hex>'"``, which must be the first
statement issued on a connection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .errors import DependencyError, HermesCryptoError

# A plaintext SQLite file always starts with this 16-byte header. A SQLCipher
# database does not (the header is encrypted), which is how migration tells
# an already-encrypted database from one still needing conversion.
_SQLITE_MAGIC = b"SQLite format 3\x00"


def sqlcipher_available() -> bool:
    """Return True when the ``sqlcipher3`` module can be imported."""
    try:
        import sqlcipher3.dbapi2  # noqa: F401
    except ImportError:
        return False
    return True


def _import_sqlcipher():
    try:
        import sqlcipher3.dbapi2 as sqlcipher
    except ImportError as exc:
        raise DependencyError(
            "Database encryption needs the 'sqlcipher3-wheels' package. "
            "Install it with:  pip install 'hermes-agent[encryption]'"
        ) from exc
    return sqlcipher


def _key_pragma(dek: bytes) -> str:
    """Return the ``PRAGMA key`` literal for a raw 32-byte key."""
    if len(dek) != 32:
        raise ValueError("DEK must be 32 bytes")
    return f"\"x'{dek.hex()}'\""


def is_plaintext_sqlite(path: Union[str, Path]) -> bool:
    """Return True when *path* is an unencrypted SQLite database."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with open(path, "rb") as handle:
            return handle.read(16) == _SQLITE_MAGIC
    except OSError:
        return False


def is_not_plaintext_sqlite(path: Union[str, Path]) -> bool:
    """Return True when *path* is a nonempty file without the SQLite header.

    Migration heuristic only — not proof of SQLCipher encryption. Any
    nonempty file whose first 16 bytes are not the standard SQLite magic is
    treated as already converted (SQLCipher encrypts the header). Empty or
    missing files return False.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    return not is_plaintext_sqlite(path)


def connect_encrypted(path: Union[str, Path], dek: bytes, **connect_kwargs):
    """Open a SQLCipher connection to *path*, keyed with the raw DEK.

    ``PRAGMA key`` is issued immediately, before any other statement, as
    SQLCipher requires. The returned connection behaves like a ``sqlite3``
    connection.
    """
    sqlcipher = _import_sqlcipher()
    conn = sqlcipher.connect(str(path), **connect_kwargs)
    try:
        conn.execute(f"PRAGMA key = {_key_pragma(dek)}")
        # Touch the schema so a wrong key fails here, loudly, rather than on
        # the first real query somewhere deep in the agent.
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except Exception as exc:
        conn.close()
        raise HermesCryptoError(
            f"Could not open encrypted database {path} — wrong key or corrupt file."
        ) from exc
    return conn


def encrypt_database(path: Union[str, Path], dek: bytes) -> None:
    """Convert a plaintext SQLite database at *path* into a SQLCipher database.

    Uses ``sqlcipher_export`` into a sibling temp file, then atomically
    replaces the original. A no-op when *path* is already encrypted.

    Caller must hold the concurrent-instance guard
    (``migrate._require_no_concurrent_hermes`` or ``--force``); this function
    does not re-check, since the detector is too expensive for per-call
    invocation.
    """
    path = Path(path)
    if not path.is_file():
        return
    if is_not_plaintext_sqlite(path):
        return
    sqlcipher = _import_sqlcipher()
    tmp = path.with_name(f"{path.name}.enc.tmp")
    tmp.unlink(missing_ok=True)
    success = False
    try:
        conn = sqlcipher.connect(str(path))
        try:
            conn.execute("ATTACH DATABASE ? AS encrypted KEY ''", (str(tmp),))
            conn.execute(f"PRAGMA encrypted.key = {_key_pragma(dek)}")
            conn.execute("SELECT sqlcipher_export('encrypted')")
            conn.execute("DETACH DATABASE encrypted")
        finally:
            conn.close()
        from utils import atomic_replace

        atomic_replace(tmp, path)
        success = True
    finally:
        if not success:
            _unlink_tmp(tmp)
    _drop_sidecars(path)


def decrypt_database(path: Union[str, Path], dek: bytes) -> None:
    """Convert a SQLCipher database at *path* back into a plaintext SQLite file.

    A no-op when *path* is already plaintext.

    Caller must hold the concurrent-instance guard
    (``migrate._require_no_concurrent_hermes`` or ``--force``); this function
    does not re-check, since the detector is too expensive for per-call
    invocation.
    """
    path = Path(path)
    if not path.is_file():
        return
    if is_plaintext_sqlite(path):
        return
    sqlcipher = _import_sqlcipher()
    tmp = path.with_name(f"{path.name}.plain.tmp")
    tmp.unlink(missing_ok=True)
    success = False
    try:
        conn = sqlcipher.connect(str(path))
        try:
            conn.execute(f"PRAGMA key = {_key_pragma(dek)}")
            conn.execute("ATTACH DATABASE ? AS plaintext KEY ''", (str(tmp),))
            conn.execute("SELECT sqlcipher_export('plaintext')")
            conn.execute("DETACH DATABASE plaintext")
        finally:
            conn.close()
        from utils import atomic_replace

        atomic_replace(tmp, path)
        success = True
    finally:
        if not success:
            _unlink_tmp(tmp)
    _drop_sidecars(path)


def rekey_database(path: Union[str, Path], old_dek: bytes, new_dek: bytes) -> None:
    """Re-key a SQLCipher database from *old_dek* to *new_dek* in place.

    A no-op when *path* is missing or still plaintext SQLite.

    Caller must hold the concurrent-instance guard
    (``migrate._require_no_concurrent_hermes`` or ``--force``); this function
    does not re-check, since the detector is too expensive for per-call
    invocation.
    """
    path = Path(path)
    if not path.is_file() or is_plaintext_sqlite(path):
        return
    if len(old_dek) != 32 or len(new_dek) != 32:
        raise ValueError("DEK must be 32 bytes")
    sqlcipher = _import_sqlcipher()
    tmp = path.with_name(f"{path.name}.rekey.tmp")
    tmp.unlink(missing_ok=True)
    success = False
    try:
        conn = sqlcipher.connect(str(path))
        try:
            conn.execute(f"PRAGMA key = {_key_pragma(old_dek)}")
            conn.execute("ATTACH DATABASE ? AS rekeyed KEY ''", (str(tmp),))
            conn.execute(f"PRAGMA rekeyed.key = {_key_pragma(new_dek)}")
            conn.execute("SELECT sqlcipher_export('rekeyed')")
            conn.execute("DETACH DATABASE rekeyed")
        finally:
            conn.close()
        from utils import atomic_replace

        atomic_replace(tmp, path)
        success = True
    finally:
        if not success:
            _unlink_tmp(tmp)
    _drop_sidecars(path)
    connect_encrypted(path, new_dek).close()


def _unlink_tmp(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _drop_sidecars(path: Path) -> None:
    """Remove stale sidecar files after a whole-database conversion.

    The WAL and rollback-journal sidecars hold raw page images from before
    the conversion — after encrypting, a stale plaintext ``-wal``/``-journal``
    would leak exactly the data the conversion was meant to protect.
    """
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = path.with_name(path.name + suffix)
        try:
            sidecar.unlink(missing_ok=True)
        except OSError:
            pass
