"""Reusable mutation oracle for read-only diagnostic tests."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping


@dataclass(frozen=True)
class PathState:
    kind: str
    mode: int
    inode: tuple[int, int]
    size: int
    mtime_ns: int
    digest: str | None
    target: str | None


@dataclass(frozen=True)
class DiagnosticManifest:
    paths: Mapping[str, PathState]
    resources: Mapping[str, object]
    children: frozenset[tuple[int, float]]


def _path_state(path: Path) -> PathState:
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode)
    inode = (info.st_dev, info.st_ino)
    if path.is_symlink():
        return PathState("symlink", mode, inode, info.st_size, info.st_mtime_ns, None, os.readlink(path))
    if path.is_dir():
        return PathState("directory", mode, inode, info.st_size, info.st_mtime_ns, None, None)
    if path.name.endswith("-shm"):
        # SQLite's shared-memory lock bytes and mtime are observationally volatile. Its durable
        # contract is identity/generation plus size; the database and WAL still get full hashes.
        return PathState("sqlite-shm", mode, inode, info.st_size, 0, None, None)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return PathState("file", mode, inode, info.st_size, info.st_mtime_ns, digest, None)


def _children() -> frozenset[tuple[int, float]]:
    try:
        import psutil

        return frozenset((child.pid, child.create_time()) for child in psutil.Process().children(recursive=True))
    except Exception:
        return frozenset()


def diagnostic_manifest(
    roots: Mapping[str, Path], resources: Mapping[str, Callable[[], object]] | None = None,
) -> DiagnosticManifest:
    """Capture durable paths plus caller-defined live resource generations/counts."""
    paths: dict[str, PathState] = {}
    for label, raw_root in roots.items():
        root = Path(raw_root)
        if not root.exists() and not root.is_symlink():
            continue
        paths[label] = _path_state(root)
        if root.is_dir() and not root.is_symlink():
            for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
                paths[f"{label}/{path.relative_to(root).as_posix()}"] = _path_state(path)
    observed = {name: read() for name, read in (resources or {}).items()}
    return DiagnosticManifest(paths=paths, resources=observed, children=_children())


def _allowed(name: str, allow: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in allow)


def manifest_changes(
    before: DiagnosticManifest, after: DiagnosticManifest, *, allow: tuple[str, ...] = (),
) -> list[str]:
    changes: list[str] = []
    for name in sorted(set(before.paths) | set(after.paths)):
        if not _allowed(name, allow) and before.paths.get(name) != after.paths.get(name):
            changes.append(f"path:{name}")
    for name in sorted(set(before.resources) | set(after.resources)):
        key = f"resource:{name}"
        if not _allowed(key, allow) and before.resources.get(name) != after.resources.get(name):
            changes.append(key)
    for child in sorted(after.children - before.children):
        key = f"process:{child[0]}"
        if not _allowed(key, allow):
            changes.append(key)
    return changes


@contextmanager
def assert_diagnostic_pure(
    roots: Mapping[str, Path], *, allow: tuple[str, ...] = (),
    resources: Mapping[str, Callable[[], object]] | None = None,
) -> Iterator[None]:
    """Fail with every unexpected changed path/resource/process identity."""
    before = diagnostic_manifest(roots, resources)
    yield
    after = diagnostic_manifest(roots, resources)
    changes = manifest_changes(before, after, allow=allow)
    assert not changes, "diagnostic mutated observable state: " + ", ".join(changes)
