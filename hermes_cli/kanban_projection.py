"""Build a .git-free projection from a pinned git tree."""
from __future__ import annotations

import io
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from hermes_cli.kanban_ceiling import Scope, ScopeError


@dataclass(frozen=True)
class ProjectionHandle:
    path: Path
    base_sha: str
    paths: frozenset[str]


def _validate_paths(paths: set[str] | frozenset[str]) -> list[str]:
    clean: list[str] = []
    for raw in paths:
        path = str(raw).replace("\\", "/")
        parsed = PurePosixPath(path)
        if not path or path.startswith("/") or ".." in parsed.parts or "\x00" in path:
            raise ScopeError(f"unsafe projection path: {raw!r}")
        if any(part in {"", "."} for part in parsed.parts):
            raise ScopeError(f"unsafe projection path: {raw!r}")
        # A projection is concrete by contract.  Passing a glob to git archive
        # could materialize more files than the resolved scope permits.
        if any(ch in path for ch in "*?["):
            raise ScopeError(f"projection paths must be concrete: {raw!r}")
        clean.append(path)
    return sorted(set(clean))


def _git_tree_types(repo: Any, base_sha: str, paths: list[str]) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "-z", base_sha, "--", *paths],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise ScopeError(f"cannot inspect git tree: {result.stderr.decode(errors='replace').strip()}")
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        meta, _ = record.split(b"\t", 1)
        mode = meta.split(b" ", 1)[0]
        if mode in {b"120000", b"160000"}:
            raise ScopeError("projection scope contains a symlink or submodule")


def build_projection(base_sha: str, scope: Scope, dest: Path, *, repo: Any = ".") -> ProjectionHandle:
    """Materialize exactly ``scope.paths`` using ``git archive``.

    ``dest`` must not be an existing non-empty directory.  Extraction is done
    with Python's tar reader (rather than a shell pipeline) so the archive and
    destination are independently validated and no shell path interpretation is
    involved.
    """
    if not isinstance(scope, Scope):
        raise ScopeError("scope must be a Scope")
    paths = _validate_paths(scope.paths)
    if not paths:
        raise ScopeError("empty scope — refusing to build a projection")
    _git_tree_types(repo, base_sha, paths)
    destination = Path(dest).expanduser()
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ScopeError(f"projection destination must be a fresh empty directory: {destination}")
    else:
        destination.mkdir(parents=True)

    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", base_sha, "--", *paths],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if archive.returncode:
        raise ScopeError(f"git archive failed: {archive.stderr.decode(errors='replace').strip()}")
    try:
        with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tar:
            members = tar.getmembers()
            for member in members:
                member_path = PurePosixPath(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ScopeError("git archive contained a traversal path")
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise ScopeError("git archive contained an unsafe non-regular entry")
                target = (destination / Path(*member_path.parts)).resolve(strict=False)
                if destination.resolve() not in target.parents and target != destination.resolve():
                    raise ScopeError("git archive escaped projection root")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        raise ScopeError("git archive contained an unreadable file")
                    with target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    target.chmod(member.mode & 0o777)
                else:
                    raise ScopeError("git archive contained an unsupported entry")
    except (tarfile.TarError, OSError) as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise ScopeError(f"could not extract projection: {exc}") from exc
    return ProjectionHandle(destination, base_sha, frozenset(paths))
