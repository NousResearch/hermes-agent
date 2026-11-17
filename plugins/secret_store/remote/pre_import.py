"""The import's append-only backups of the files it replaces (contract §6).

Immediately before the importer replaces ``.env`` or ``auth.json`` (the cleanup after a PATCH, the
plane-wins path, any other rewrite) it appends the exact bytes being replaced, as one entry, to that
file's backup beside it (:data:`BACKUPS`: ``.env.pre-import`` / ``auth.pre-import.json``, 0600).
The import also appends the bytes it stages from (unless the last entry already holds them), since
a token refresh rewrites ``auth.json`` while the import runs. Nothing ever rewrites, truncates or
removes a backup; ``hermes doctor`` names them.

One entry is a header line, the bytes, and a newline::

    #hermes-pre-import v1 at=2026-11-17T01:02:03.123456Z reason=cleanup bytes=42 sha256=<hex>
    <the 42 bytes>

An append is one write at the end of the file, then fsync. A crash mid-append leaves a torn last
entry (its length or digest does not match); it is reported as torn and never hides an earlier
entry, and the next append starts after it on a fresh line. Bytes not framed this way (a backup an
earlier version wrote whole) are reported as one ``unframed`` entry and kept as they are.

Used under the ``.env`` / ``auth.json`` write locks: this module imports only the standard library,
nothing lazily, and never logs.
"""
from __future__ import annotations

import datetime
import hashlib
import os
import re
from pathlib import Path
from typing import NamedTuple

#: Each import input and its append-only backup beside it.
BACKUPS = {".env": ".env.pre-import", "auth.json": "auth.pre-import.json"}
MAGIC = b"#hermes-pre-import v1 "
_HEADER = re.compile(rb"#hermes-pre-import v1 at=(\S+) reason=([a-z0-9-]+) bytes=(\d+) sha256=([0-9a-f]{64})")


class Entry(NamedTuple):
    """One backup entry: when and why it was written, the bytes, and whether it is whole
    (``torn``: a crashed append; ``unframed``: bytes outside any entry)."""
    at: str
    reason: str
    data: bytes
    whole: bool


def backup_path(source: Path) -> Path:
    """The backup of import input *source* (``.env`` / ``auth.json``)."""
    return source.with_name(BACKUPS[source.name])


def entries(raw: bytes) -> list[Entry]:
    """The entries of a backup's *raw* bytes, oldest first. A torn or unframed region is an
    entry of its own (``whole`` False) and never hides another."""
    out: list[Entry] = []
    pos = 0
    while pos < len(raw):
        if raw.startswith(MAGIC, pos):
            whole = _whole_entry(raw, pos)
            if whole is not None:
                out.append(whole[0])
                pos = whole[1]
                continue
            end = _next_header(raw, pos + 1)
            out.append(Entry("", "torn", raw[pos:end], False))
        else:
            end = _next_header(raw, pos)
            out.append(Entry("", "unframed", raw[pos:end], False))
        pos = end
    return out


def _whole_entry(raw: bytes, pos: int) -> tuple[Entry, int] | None:
    """The whole entry at *pos* and the offset after it, or None when it is torn."""
    eol = raw.find(b"\n", pos)
    match = _HEADER.fullmatch(raw, pos, eol) if eol >= 0 else None
    if match is None:
        return None
    start = eol + 1
    end = start + int(match.group(3))
    data = raw[start:end]
    if end >= len(raw) or raw[end:end + 1] != b"\n" or hashlib.sha256(data).hexdigest().encode() != match.group(4):
        return None
    return Entry(match.group(1).decode("ascii"), match.group(2).decode("ascii"), data, True), end + 1


def _next_header(raw: bytes, start: int) -> int:
    """The offset of the first header at a line start at or after *start* (or the end)."""
    at = raw.find(b"\n" + MAGIC, max(start - 1, 0))
    return len(raw) if at < 0 else at + 1


def read_entries(path: Path) -> list[Entry]:
    """The entries of the backup at *path* ([] when absent)."""
    try:
        return entries(Path(path).read_bytes())
    except FileNotFoundError:
        return []


def append(path: Path, data: bytes, reason: str, *, unless_last: bool = False) -> None:
    """Append *data* as one entry to the backup at *path* (created 0600; never a symlink's
    target), then fsync it. With *unless_last*, skip it when the last whole entry already holds
    exactly *data*. Raises OSError (the caller's rewrite then does not happen)."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    created = not os.path.lexists(path)
    if not created and unless_last:
        whole = [e for e in read_entries(path) if e.whole]
        if whole and whole[-1].data == data:
            return
    fd = os.open(path, flags, 0o600)
    try:
        size = os.fstat(fd).st_size
        lead = b""
        if size:
            with open(path, "rb") as f:
                f.seek(size - 1)
                lead = b"" if f.read(1) == b"\n" else b"\n"
        at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        header = f"at={at} reason={reason} bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()}\n"
        chunk = memoryview(lead + MAGIC + header.encode("ascii") + data + b"\n")
        while chunk:
            chunk = chunk[os.write(fd, chunk):]
        os.fsync(fd)
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    if created:
        _fsync_dir(Path(path).parent)


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of the directory entry of a new backup (a no-op on Windows)."""
    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def present(home: Path) -> list[Path]:
    """The backups present in *home* (``hermes doctor`` names them)."""
    return [p for p in (Path(home) / n for n in BACKUPS.values()) if os.path.lexists(p)]
