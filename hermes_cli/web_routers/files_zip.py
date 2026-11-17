"""A workspace folder packed as one zip for ``GET /api/fs/download?format=zip``.

The route resolves and guards the folder exactly as it does a single-file download; this module
only walks it and writes the archive. The walk mirrors what the Files tree lists: it skips the
noise directories the tree hides (``.git``, ``node_modules``, ...), every entry the route's
*admit* check refuses (credential stores, ``.env``, paths outside a locked files root), and
every symlink, so a link can never pull in a file outside the folder or loop the walk. A live
SQLite database is skipped rather than read (a raw close would drop its locks).

The archive is written to a spooled temp file while it is hashed, so the response can carry its
SHA-256 (``X-Content-SHA256``) and length before the first body byte, and a saved copy can be
checked against it. Memory stays bounded: past ``_SPOOL_BYTES`` the spool moves to disk.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Iterator
from urllib.parse import quote

from fastapi import HTTPException

from hermes_cli.sqlite_safe_read import LiveConnectionError, offline_file_access

#: Files one archive may hold, and their total uncompressed bytes. Over either: 413, nothing sent.
MAX_FILES = 20_000
MAX_BYTES = 1024 * 1024 * 1024
_SPOOL_BYTES = 8 * 1024 * 1024
_CHUNK = 1024 * 1024

#: Header names the route sets; the Desktop save bridge checks the saved bytes against the first.
SHA256_HEADER = "X-Content-SHA256"
FILES_HEADER = "X-Hermes-Zip-Files"
SKIPPED_HEADER = "X-Hermes-Zip-Skipped"


@dataclass(frozen=True)
class PackedZip:
    """A finished archive: rewound *body*, its SHA-256 hex digest and size, and what the walk did."""

    body: IO[bytes]
    sha256: str
    size: int
    files: int
    skipped: int

    def chunks(self) -> Iterator[bytes]:
        """The archive bytes, closing the spool once they are all sent (or the client leaves)."""
        try:
            while chunk := self.body.read(_CHUNK):
                yield chunk
        finally:
            self.body.close()


def _walk(root: Path, hidden: frozenset[str], admit: Callable[[Path], bool]) -> Iterator[tuple[Path, int] | None]:
    """``(file, size)`` for each regular file under *root* the archive takes, ``None`` per skip."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as scan:
                entries = sorted(scan, key=lambda e: e.name)
        except OSError:
            yield None
            continue
        for entry in entries:
            path = Path(entry.path)
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                yield None
                continue
            if stat.S_ISLNK(st.st_mode) or not admit(path):
                yield None
            elif stat.S_ISDIR(st.st_mode):
                if entry.name not in hidden:
                    stack.append(path)
            elif stat.S_ISREG(st.st_mode):
                yield path, st.st_size
            else:
                yield None


def _add_file(archive: zipfile.ZipFile, path: Path, arcname: str) -> bool:
    """Copy *path* into *archive*; False (skipped) when it cannot be read or is a live database."""
    try:
        with offline_file_access(path, what="zip"), path.open("rb") as src, archive.open(arcname, "w") as dst:
            while chunk := src.read(_CHUNK):
                dst.write(chunk)
        return True
    except (LiveConnectionError, OSError):
        return False


class _HashingWriter:
    """File-like that hashes and counts every byte on its way into *sink*."""

    def __init__(self, sink: IO[bytes]):
        self._sink, self.digest, self.size = sink, hashlib.sha256(), 0

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        self.size += len(data)
        return self._sink.write(data)

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        """ZipFile never closes a stream it was handed; the spool's owner closes it."""


def pack_directory(root: Path, *, hidden: frozenset[str], admit: Callable[[Path], bool]) -> PackedZip:
    """*root* (a resolved, already-guarded directory) packed as a zip of its admitted files.

    Entry names are relative to *root*'s parent so the archive unpacks into one folder named like
    *root*. 413 past ``MAX_FILES`` / ``MAX_BYTES``."""
    spool = tempfile.SpooledTemporaryFile(max_size=_SPOOL_BYTES)
    writer = _HashingWriter(spool)
    files = skipped = total = 0
    try:
        # A write-only stream: ZipFile emits data descriptors and never seeks back, so every
        # byte passes through the hash exactly once, in order.
        with zipfile.ZipFile(writer, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for item in _walk(root, hidden, admit):
                if item is None:
                    skipped += 1
                    continue
                path, size = item
                files, total = files + 1, total + size
                if files > MAX_FILES or total > MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Folder is too large to zip (limit {MAX_FILES} files / {MAX_BYTES // 2**20} MiB)",
                    )
                if not _add_file(archive, path, path.relative_to(root.parent).as_posix()):
                    files, total, skipped = files - 1, total - size, skipped + 1
        spool.seek(0)
    except BaseException:
        spool.close()
        raise
    return PackedZip(spool, writer.digest.hexdigest(), writer.size, files, skipped)


def zip_headers(packed: PackedZip, name: str) -> dict[str, str]:
    """Attachment headers for *packed* saved as ``<name>.zip``."""
    return {
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name or 'workspace')}.zip",
        "Content-Length": str(packed.size),
        SHA256_HEADER: packed.sha256,
        FILES_HEADER: str(packed.files),
        SKIPPED_HEADER: str(packed.skipped),
        "Access-Control-Expose-Headers": f"{SHA256_HEADER}, {FILES_HEADER}, {SKIPPED_HEADER}",
        "Cache-Control": "no-store",
    }
