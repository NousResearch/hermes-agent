"""Find corrupt entries in PM's uv cache.

uv unpacks every wheel into ``<cache>/.tmpXXXX`` and renames the directory
into ``archive-v0/<id>``; it never flushes the file data first. When the
machine stops mid-install (a hosted restart, a power cut), the rename can
survive while the files read back as NUL bytes. uv then reuses that entry on
every later sync and fails at install time with "The wheel is invalid", and
retrying cannot help. These helpers recognise that failure and name the
packages to purge. Deletion always goes through ``uv cache clean`` (the caller),
which takes uv's cache lock, so a concurrent sync never loses files mid-read;
the leftover-archive sweep takes that same lock itself.

Stdlib only: the bootstrap runner imports pm.environment on an older system
python.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import sys

from pm.filesystem import is_junction, lock_fd

# uv's install step names the wheel it was linking from the cache:
#   error: Failed to install: lxml-6.1.3-cp314-...whl (lxml==6.1.3)
_FAILED_INSTALL = re.compile(r"failed to install: ([^\s()]+?\.whl)\b", re.IGNORECASE)

# Causes that mean the cached copy itself is unusable. A fetch, resolver or
# disk-full failure never prints these, so it is never mistaken for corruption.
_CORRUPT_CAUSES = (
    "the wheel is invalid",       # WHEEL/METADATA unreadable, .dist-info missing
    "`record` file is invalid",   # RECORD unreadable
)
# A cache pointer naming an archive whose files are gone ("No such file or
# directory (os error 2)"; Windows also says "os error 3" for a missing dir).
_MISSING_ARCHIVE = re.compile(r"archive-v0[\\/][^\s`'\"]+.*\(os error [23]\)", re.IGNORECASE)

# PEP 508 project names only: these become `uv cache clean` arguments.
_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_DIST_INFO = re.compile(r"(?P<name>.+?)-[^-]+\.dist-info")


def normalize(name: str) -> str:
    """PEP 503 normal form, the spelling uv's cache uses."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _valid(name: str) -> str | None:
    return normalize(name) if _NAME.fullmatch(name) else None


def failed_cached_packages(output: str) -> set[str]:
    """Packages uv could not install because their cached copy is corrupt.

    Empty unless uv failed in its install step AND the cause is one of the
    cache-integrity shapes above, so a resolver, network or build failure is
    never retried as corruption.
    """
    text = output or ""
    lowered = text.lower()
    if not (any(cause in lowered for cause in _CORRUPT_CAUSES) or _MISSING_ARCHIVE.search(text)):
        return set()
    names = set()
    for match in _FAILED_INSTALL.finditer(text):
        name = _valid(match.group(1).split("-", 1)[0])
        if name:
            names.add(name)
    return names


_HEAD = 65536
# How long a sweep waits for uv's cache lock before leaving the leftovers for later.
_LOCK_WAIT_SECONDS = 10.0
_WHEEL_VERSION = re.compile(rb"(?im)^wheel-version[ \t]*:")
_FILES = (("WHEEL", _WHEEL_VERSION), ("METADATA", None), ("RECORD", None))
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _NOFOLLOW
# The sweep works only through directory descriptors (rmtree's dir_fd is 3.11+;
# the bootstrap runner may be older). Elsewhere it is skipped: the leftovers are
# harmless, and on Windows uv's pointer files let `uv cache clean` take them anyway.
_FD_SWEEP = (os.name == "posix" and sys.version_info >= (3, 11) and shutil.rmtree.avoids_symlink_attacks
             and os.open in os.supports_dir_fd and os.scandir in os.supports_fd)


def _stream_damaged(stream, header: re.Pattern | None) -> bool:
    """Positive evidence of a torn write: empty, NUL bytes anywhere, or *header* never seen.

    The whole file is scanned in bounded chunks: a torn write can keep the
    first pages and lose later ones.
    """
    seen = header is None
    carry = b""
    empty = True
    while chunk := stream.read(_HEAD):
        empty = False
        if b"\0" in chunk:
            return True
        if not seen:
            window = carry + chunk
            seen = bool(header.search(window))
            carry = window[-64:]
    return empty or not seen


def _damaged(path: Path, header: re.Pattern | None = None) -> bool:
    """True only on positive evidence of a torn write (see _stream_damaged).

    A missing file is damage. A linked file, or one that cannot be read for
    another reason (permissions), proves nothing and counts as healthy: a
    false positive would delete an entry uv can still use.
    """
    try:
        if path.is_symlink():
            return False
        with path.open("rb") as stream:
            return _stream_damaged(stream, header)
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _damaged_at(directory: int, name: str, header: re.Pattern | None) -> bool:
    """_damaged for *name* inside an open directory descriptor; links are not followed."""
    try:
        fd = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        # closefd=False: fdopen itself can fail (a directory in the file's place),
        # and the descriptor must be closed either way.
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return _stream_damaged(stream, header)
    except OSError:
        return False
    finally:
        os.close(fd)


def _dist_name(info: str) -> str | None:
    match = _DIST_INFO.fullmatch(info)
    return _valid(match.group("name")) if match else None


def _archive_root(cache: Path) -> Path | None:
    """``cache/archive-v0`` when it is a real directory inside the cache, else None."""
    archive = cache / "archive-v0"
    if archive.is_symlink() or is_junction(archive) or not archive.is_dir():
        return None
    return archive


def corrupt_cache_entries(cache: Path) -> dict[str, list[str]]:
    """Unpacked cache entries uv cannot install from, as package -> archive entry names.

    Checks the three files uv reads first. An interrupted write leaves them
    missing, empty or NUL-filled, and a torn WHEEL loses its Wheel-Version
    line. Nothing else counts: unknown layouts, linked or junctioned entries,
    linked files and a linked ``archive-v0`` are skipped, never guessed at.
    Read-only.
    """
    found: dict[str, list[str]] = {}
    archive = _archive_root(cache)
    if archive is None:
        return found
    try:
        buckets = list(archive.iterdir())
    except OSError:
        return found
    for bucket in buckets:
        try:
            if bucket.is_symlink() or is_junction(bucket) or not bucket.is_dir():
                continue
            infos = [entry for entry in bucket.iterdir()
                     if entry.name.endswith(".dist-info") and not entry.is_symlink() and entry.is_dir()]
        except OSError:
            continue
        for info in infos:
            name = _dist_name(info.name)
            if name and any(_damaged(info / file, header) for file, header in _FILES):
                found.setdefault(name, []).append(bucket.name)
    return found


def corrupt_cached_packages(cache: Path) -> set[str]:
    """Packages with at least one corrupt unpacked cache entry."""
    return set(corrupt_cache_entries(cache))


def _damaged_packages_at(bucket: int) -> set[str]:
    """corrupt_cache_entries for one bucket, read only through descriptors."""
    with os.scandir(bucket) as entries:
        infos = [entry.name for entry in entries
                 if entry.name.endswith(".dist-info") and entry.is_dir(follow_symlinks=False)]
    found = set()
    for info in infos:
        name = _dist_name(info)
        if not name:
            continue
        try:
            info_fd = os.open(info, _DIRECTORY, dir_fd=bucket)
        except OSError:
            continue
        try:
            if any(_damaged_at(info_fd, file, header) for file, header in _FILES):
                found.add(name)
        finally:
            os.close(info_fd)
    return found


def remove_corrupt_entries(cache: Path, names: set[str]) -> bool:
    """Delete the corrupt archive entries of *names* that ``uv cache clean`` left behind.

    Where uv links a wheel to its archive by symlink (Linux, macOS), clean
    drops the link and leaves the archive, which nothing uses again but
    ``hermes pm doctor`` would keep reporting. ``uv cache prune`` is not an
    option: PM seeds this cache from a bundle copy whose links are real
    directories, and prune treats their archives as unreachable, so the next
    install fails.

    The cache directory is opened once, and ``archive-v0``, the lock, every
    entry and every file are then reached only relative to that descriptor
    without following links, so renaming or replacing a path above or inside
    the cache mid-sweep cannot point a delete anywhere else. The sweep holds
    uv's own cache lock, so no uv process runs while it deletes, and it only
    takes entries whose files prove a torn write. Returns False when the
    sweep did not run (lock busy for _LOCK_WAIT_SECONDS, no descriptor
    support); the leftovers are harmless, and a later sweep or ``hermes pm
    doctor --repair`` takes them.
    """
    if not names:
        return True
    if not _FD_SWEEP:
        return False
    try:
        cache_fd = os.open(cache, os.O_RDONLY | os.O_DIRECTORY)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    archive = lock = None
    try:
        try:
            archive = os.open("archive-v0", _DIRECTORY, dir_fd=cache_fd)
        except FileNotFoundError:
            return True
        lock = os.open(".lock", os.O_RDWR | os.O_CREAT | _NOFOLLOW, 0o666, dir_fd=cache_fd)
        if not lock_fd(lock, wait=True, timeout=_LOCK_WAIT_SECONDS):
            return False
        with os.scandir(archive) as entries:
            buckets = [entry.name for entry in entries if entry.is_dir(follow_symlinks=False)]
        for bucket in buckets:
            try:
                bucket_fd = os.open(bucket, _DIRECTORY, dir_fd=archive)
            except OSError:
                continue
            try:
                doomed = bool(_damaged_packages_at(bucket_fd) & names)
            except OSError:
                doomed = False
            finally:
                os.close(bucket_fd)
            if doomed:
                shutil.rmtree(bucket, dir_fd=archive, ignore_errors=True)
        return True
    except OSError:
        return False
    finally:
        for fd in (lock, archive, cache_fd):
            if fd is not None:
                os.close(fd)
