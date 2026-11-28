"""A workspace folder packed as one zip for ``GET /api/fs/download?format=zip``.

The route resolves and guards the folder exactly as it does a single-file download; this module
only walks it and writes the archive. The walk mirrors what the Files tree lists: it skips the
noise directories the tree hides (``.git``, ``node_modules``, ...), every entry the route's
*admit* check refuses (credential stores, ``.env``, paths outside a locked files root), and
every symlink or Windows junction, so a link can never pull in a file outside the folder or loop
the walk. A live SQLite database is skipped rather than read (a raw close would drop its locks);
like the streamed download route, that is a point-in-time admission check, not a held lock.

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

from hermes_cli.sqlite_safe_read import has_live_connection
from pm.filesystem import is_junction

#: Files one archive may hold, and the bytes actually copied into it. Over either: 413, nothing sent.
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


def _walk(root: Path, hidden: frozenset[str], admit: Callable[[Path], bool]) -> Iterator[Path | None]:
    """Each regular file under *root* the archive takes, ``None`` per skip. Streams ``scandir``
    (no per-directory list), so a huge folder costs no memory before the limits stop it."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as scan:
                for entry in scan:
                    path = Path(entry.path)
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        yield None
                        continue
                    # A Windows junction is a folder link that S_ISLNK misses: never descend it.
                    if stat.S_ISLNK(st.st_mode) or is_junction(path) or not admit(path):
                        yield None
                    elif stat.S_ISDIR(st.st_mode):
                        if entry.name not in hidden:
                            stack.append(path)
                    else:
                        yield path if stat.S_ISREG(st.st_mode) else None
        except OSError:
            yield None


class _Budget:
    """The archive's file and byte limits, charged with what is actually copied."""

    def __init__(self) -> None:
        self.files = self.bytes = 0

    def charge(self, *, files: int = 0, size: int = 0) -> None:
        self.files += files
        self.bytes += size
        if self.files > MAX_FILES or self.bytes > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Folder is too large to zip (limit {MAX_FILES} files / {MAX_BYTES // 2**20} MiB)",
            )


def _add_file(archive: zipfile.ZipFile, path: Path, arcname: str, budget: _Budget) -> bool:
    """Copy *path* into *archive*; False (skipped, nothing written) when it is a live database or
    cannot be opened. A failure after the member is started aborts the archive: a half-copied file
    must never ship under a SHA-256 that vouches for it."""
    # Point-in-time admission, as the streamed download route does: holding the process-wide
    # SQLite lock across a whole compression would stall every session request meanwhile.
    if has_live_connection(path):
        return False
    try:
        src = path.open("rb")
    except OSError:
        return False
    with src:
        budget.charge(files=1)
        try:
            with archive.open(arcname, "w") as dst:
                while chunk := src.read(_CHUNK):
                    budget.charge(size=len(chunk))
                    dst.write(chunk)
        except OSError as exc:
            raise HTTPException(status_code=409, detail=f"Could not read {arcname} while zipping") from exc
    return True


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
    budget, skipped = _Budget(), 0
    try:
        # A write-only stream: ZipFile emits data descriptors and never seeks back, so every
        # byte passes through the hash exactly once, in order.
        with zipfile.ZipFile(writer, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in _walk(root, hidden, admit):
                if path is None or not _add_file(archive, path, path.relative_to(root.parent).as_posix(), budget):
                    skipped += 1
        spool.seek(0)
    except BaseException:
        spool.close()
        raise
    return PackedZip(spool, writer.digest.hexdigest(), writer.size, budget.files, skipped)


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
