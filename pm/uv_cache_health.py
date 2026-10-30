"""Find corrupt entries in PM's uv cache.

uv unpacks every wheel into ``<cache>/.tmpXXXX`` and renames the directory
into ``archive-v0/<id>``; it never flushes the file data first. When the
machine stops mid-install (a hosted restart, a power cut), the rename can
survive while the files read back as NUL bytes. uv then reuses that entry on
every later sync and fails at install time with "The wheel is invalid", and
retrying cannot help. These helpers recognise that failure and name the
packages to purge. Deletion always goes through ``uv cache clean`` (the caller),
which takes uv's cache lock, so a concurrent sync never loses files mid-read.

Stdlib only: the bootstrap runner imports pm.environment on an older system
python.
"""
from __future__ import annotations

from pathlib import Path
import re

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


def _readable(path: Path, marker: bytes, limit: int = 65536) -> bool:
    try:
        with path.open("rb") as stream:
            head = stream.read(limit)
    except OSError:
        return False
    return marker in head and b"\0" not in head


def corrupt_cache_entries(cache: Path) -> dict[str, list[Path]]:
    """Unpacked cache entries uv cannot install from, by package.

    Checks the three files uv reads first (WHEEL, METADATA, RECORD); an
    interrupted write leaves them empty or NUL-filled. Entries without a
    .dist-info (unknown layouts) and symlinked entries are skipped, never
    guessed at. Read-only.
    """
    archive = cache / "archive-v0"
    found: dict[str, list[Path]] = {}
    try:
        buckets = list(archive.iterdir())
    except OSError:
        return found
    for bucket in buckets:
        try:
            if bucket.is_symlink() or not bucket.is_dir():
                continue
            infos = [entry for entry in bucket.iterdir() if entry.name.endswith(".dist-info")]
        except OSError:
            continue
        for info in infos:
            match = _DIST_INFO.fullmatch(info.name)
            name = match and _valid(match.group("name"))
            if not name:
                continue
            if not (_readable(info / "WHEEL", b"Wheel-Version:")
                    and _readable(info / "METADATA", b"Name:")
                    and _readable(info / "RECORD", b",")):
                found.setdefault(name, []).append(bucket)
    return found


def corrupt_cached_packages(cache: Path) -> set[str]:
    """Packages with at least one corrupt unpacked cache entry."""
    return set(corrupt_cache_entries(cache))


def remove_corrupt_entries(cache: Path, names: set[str]) -> None:
    """Delete the corrupt archive entries of *names* that ``uv cache clean`` left behind.

    Where uv links a wheel to its archive by symlink (Linux), clean drops the
    link and leaves the archive, which nothing uses again. ``uv cache prune``
    is not an option: PM seeds this cache from a bundle copy whose links are
    real directories, and prune treats their archives as unreachable, so the
    next install fails. An archive entry is never rewritten once uv renames it
    into place, so deleting one that fails validation cannot take a good copy
    from a concurrent install. rmtree refuses a symlinked root and does not
    follow links inside it.
    """
    import shutil

    for name, buckets in corrupt_cache_entries(cache).items():
        if name in names:
            for bucket in buckets:
                shutil.rmtree(bucket, ignore_errors=True)
