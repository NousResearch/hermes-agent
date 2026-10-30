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


def _damaged(path: Path, header: re.Pattern | None = None) -> bool:
    """True only on positive evidence of a torn write: missing, empty or NUL bytes.

    A file that cannot be read for another reason (permissions) proves
    nothing and counts as healthy: a false positive would delete an entry uv
    can still use.
    """
    try:
        with path.open("rb") as stream:
            head = stream.read(_HEAD)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if not head or b"\0" in head:
        return True
    return header is not None and not header.search(head)


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
    line. Nothing else counts: unknown layouts, symlinked or junctioned
    entries and a symlinked ``archive-v0`` are skipped, never guessed at.
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
            match = _DIST_INFO.fullmatch(info.name)
            name = match and _valid(match.group("name"))
            if not name:
                continue
            if (_damaged(info / "WHEEL", _WHEEL_VERSION) or _damaged(info / "METADATA")
                    or _damaged(info / "RECORD")):
                found.setdefault(name, []).append(bucket.name)
    return found


def corrupt_cached_packages(cache: Path) -> set[str]:
    """Packages with at least one corrupt unpacked cache entry."""
    return set(corrupt_cache_entries(cache))


def remove_corrupt_entries(cache: Path, names: set[str]) -> bool:
    """Delete the corrupt archive entries of *names* that ``uv cache clean`` left behind.

    Where uv links a wheel to its archive by symlink (Linux), clean drops the
    link and leaves the archive, which nothing uses again but ``hermes pm
    doctor`` would keep reporting. ``uv cache prune`` is not an option: PM
    seeds this cache from a bundle copy whose links are real directories, and
    prune treats their archives as unreachable, so the next install fails.

    The sweep holds uv's own cache lock, so no uv process runs while it
    deletes, and it only takes entries the validation above proves torn.
    Deletion is anchored to a descriptor of the real ``archive-v0`` where the
    platform allows it, so a symlink swapped in above it cannot redirect the
    delete. Returns False when the lock stayed busy (the leftovers are
    harmless and the next sweep or ``hermes pm doctor --repair`` takes them).
    """
    archive = _archive_root(cache)
    if archive is None or not names:
        return True
    try:
        lock = os.open(cache / ".lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o666)
    except OSError:
        return False
    try:
        if not lock_fd(lock, wait=True, timeout=_LOCK_WAIT_SECONDS):
            return False
        doomed = [bucket for name, buckets in corrupt_cache_entries(cache).items()
                  if name in names for bucket in buckets]
        if not doomed:
            return True
        # rmtree's dir_fd is 3.11+; the bootstrap runner may be older.
        if sys.version_info >= (3, 11) and shutil.rmtree.avoids_symlink_attacks and hasattr(os, "O_DIRECTORY"):
            directory = os.open(archive, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
            try:
                for bucket in doomed:
                    shutil.rmtree(bucket, dir_fd=directory, ignore_errors=True)
            finally:
                os.close(directory)
        else:
            for bucket in doomed:
                path = archive / bucket
                if not (path.is_symlink() or is_junction(path)):
                    shutil.rmtree(path, ignore_errors=True)
        return True
    finally:
        os.close(lock)
