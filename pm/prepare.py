"""Build a portable, hash-pinned archive from PM's verified stage-only store entry.

Usage: python -m pm.prepare NAME TARGET --out /path/to/archive.{zip,tar.gz}
The caller owns publishing the resulting JSON pins; this module never edits the lock.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path

from pm import paths
from pm.filesystem import is_junction
from pm.lock import Lockfile
from pm.registry import get_package, walk
from pm.store import ALL_TARGETS, current_target, extract, tree_digest


# Changes to staging or its shared store mechanics invalidate a prepared tree
# even when upstream tarballs are unchanged. Hash sources, not checkout paths.
_STAGING_COMMON_FILES = ("package.py", "store.py", "install.py", "prepare.py", "paths.py", "proxy_env.py")


def _source_fingerprints(sources: dict[str, Path]) -> list[list[str]]:
    return [[module, hashlib.sha256(path.read_bytes()).hexdigest()]
            for module, path in sorted(sources.items())]


def source_identity(lockfile: Lockfile, name: str, target: str) -> str:
    """Digest of the selected package's complete, target-resolved source closure.

    A changed dependency pin (notably Python's DLL for ARM64 ripgrep or Node
    for npm) invalidates prepared bytes even when the package's own URL stays.
    The caller stores this alongside prepared sha256 and tree digest and checks
    it against the current lock before using an archive.
    """
    if target not in ALL_TARGETS:
        raise ValueError(f"unknown target: {target}")
    closure = []
    root = Path(__file__).resolve().parent
    sources = {f"pm.{filename.removesuffix('.py')}": root / filename
               for filename in _STAGING_COMMON_FILES}
    for package in walk([name]):
        version = lockfile.version(package.name)
        artifacts = lockfile.artifacts(package.name, target)
        if version is None or not artifacts:
            raise ValueError(f"{package.name} has no source pin for {target}")
        closure.append([package.name, version, [artifact["sha256"] for artifact in artifacts]])
        module_file = sys.modules[type(package).__module__].__file__
        if module_file is None:
            raise ValueError(f"{package.name} has no staging source file")
        sources[type(package).__module__] = Path(module_file).resolve()
    payload = json.dumps([target, closure, _source_fingerprints(sources)],
                         separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _RawLockfile(Lockfile):
    """Preparing a new archive must never use an old prepared archive as input."""

    def prepared(self, name: str, target: str) -> None:
        return None


_EPOCH = 315532800  # 1980-01-01, also the oldest ZIP timestamp.


def preparation_requirements(name: str, target: str, *, host: str | None = None) -> tuple[str, ...]:
    """Requirements stage_only cannot fulfill automatically on an arbitrary build host.

    npm's bionic wrapper is a pure file transform; other npm targets execute the
    installed target's node. ARM64 ripgrep copies a DLL from installed Python.
    """
    host = current_target() if host is None else host
    if name == "git" and target != host:
        return ("native Windows " + target + " host",)
    if name == "python" and (target.startswith("darwin-") or host.startswith("darwin-")) and target != host:
        return ("native macOS " + target + " host",)
    if name == "npm" and target != "linux-arm64-bionic":
        if target != host:
            return ("native " + target + " host with installed node",)
        return ("installed node for " + target,)
    if name == "ripgrep" and target == "win32-arm64":
        if target != host:
            return ("native win32-arm64 host with installed python",)
        return ("installed python for win32-arm64",)
    return ()


def _members(root: Path) -> list[tuple[Path, str, int]]:
    """Walk without dereferencing links; reject types the stock extractors lose."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"not a staged directory: {root}")
    members = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted([*dirs, *files]):
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if is_junction(path) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                raise ValueError(f"unsupported prepared entry: {path}")
            if stat.S_ISLNK(mode):
                link = os.readlink(path)
                if Path(link).is_absolute() or not (path.parent / link).resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"symlink escapes prepared entry: {path}")
            members.append((path, path.relative_to(root).as_posix(), mode))
    return sorted(members, key=lambda member: member[1])


def _write_zip(members: list[tuple[Path, str, int]], output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9,
                         allowZip64=True) as archive:
        for path, rel, mode in members:
            if stat.S_ISLNK(mode):
                raise ValueError(f"stock Windows ZIP extraction cannot restore symlink: {rel}")
            info = zipfile.ZipInfo(rel + ("/" if stat.S_ISDIR(mode) else ""), (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = ((mode & 0o777) | (stat.S_IFDIR if stat.S_ISDIR(mode) else stat.S_IFREG)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            if stat.S_ISDIR(mode):
                archive.writestr(info, b"")
            else:
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as dest:
                    shutil.copyfileobj(source, dest, length=1024 * 1024)


def _write_tar(members: list[tuple[Path, str, int]], output: Path) -> None:
    with output.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                                  mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w|") as archive:
            for path, rel, mode in members:
                member = tarfile.TarInfo(rel + ("/" if stat.S_ISDIR(mode) else ""))
                member.mode = mode & 0o777
                member.mtime = _EPOCH
                member.uid = member.gid = 0
                if stat.S_ISDIR(mode):
                    member.type = tarfile.DIRTYPE
                    archive.addfile(member)
                elif stat.S_ISLNK(mode):
                    member.type = tarfile.SYMTYPE
                    member.linkname = os.readlink(path)
                    archive.addfile(member)
                else:
                    member.size = path.stat().st_size
                    with path.open("rb") as source:
                        archive.addfile(member, source)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_tree(entry: Path, target: str, output: Path) -> dict[str, str]:
    """Atomically write zip (Windows) or tar.gz (POSIX), verify PM extraction."""
    if target not in ALL_TARGETS:
        raise ValueError(f"unknown target: {target}")
    output = Path(output)
    suffix = ".zip" if target.startswith("win32-") else ".tar.gz"
    if not output.name.endswith(suffix):
        raise ValueError(f"{target} prepared archive must end with {suffix}")
    entry = Path(entry)
    if output.resolve().is_relative_to(entry.resolve()):
        raise ValueError("archive output cannot be inside its source entry")
    members = _members(entry)
    if target.startswith("win32-") and any(stat.S_ISLNK(mode) for _, _, mode in members):
        raise ValueError("stock Windows ZIP extraction cannot restore symlinks")
    digest = tree_digest(entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        archive = Path(temporary) / ("entry" + suffix)
        (_write_zip if suffix == ".zip" else _write_tar)(members, archive)
        if tree_digest(entry) != digest:
            raise ValueError("store entry changed while archiving")
        extracted = Path(temporary) / "extracted"
        extract(archive, extracted)
        if tree_digest(extracted) != digest:
            raise ValueError("prepared archive changed the store entry digest")
        for path, rel, mode in members:
            if stat.S_ISDIR(mode) and not (extracted / rel).is_dir():
                raise ValueError(f"prepared archive lost directory: {rel}")
            if stat.S_ISREG(mode) and not target.startswith("win32-"):
                if bool((extracted / rel).stat().st_mode & 0o111) != bool(mode & 0o111):
                    raise ValueError(f"prepared archive lost executable mode: {rel}")
        pin = {"sha256": _sha256(archive), "digest": digest}
        os.replace(archive, output)
    return pin


def prepare(name: str, target: str, output: Path) -> dict[str, str]:
    """Build from stage_only's pinned upstream pipeline, never from raw unpack."""
    if target not in ALL_TARGETS:
        raise ValueError(f"unknown target: {target}")
    package = get_package(name)
    lock = _RawLockfile(paths.lockfile_path())
    if getattr(package, "pin_only", False) or not lock.artifacts(name, target) or package.missing_reason(target):
        raise ValueError(f"{name} has no stageable artifact on {target}")
    host = current_target()
    requirements = preparation_requirements(name, target, host=host)
    if requirements:
        from pm.install import _installed_location
        for requirement in requirements:
            if requirement.startswith("native"):
                raise ValueError(f"{name} on {target} requires {requirement}; host is {host}")
            dependency = requirement.split(" ")[1]
            if _installed_location(get_package(dependency), lock, target, verify=True) is None:
                raise ValueError(f"{name} on {target} requires {requirement}")
    identity = source_identity(lock, name, target)
    from pm.install import _install, _store

    # stage_only's cache key contains only THIS package's raw pins. Re-stage
    # on every build: a changed dependency (npm's node / ripgrep's Python)
    # can alter bytes without changing the package's own pin.
    entry = _install(package, lock, None, _store(), target, _fresh_copy=True)
    if source_identity(Lockfile(paths.lockfile_path()), name, target) != identity:
        raise ValueError("source lock changed while preparing archive")
    # The stage marker is local bookkeeping, not part of a native install. The
    # reader regenerates it for cross-target stages; native entries stay equal
    # to the raw installer output. Never edit the verified store entry itself.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepared-tree-", dir=output.parent) as temp:
        tree = Path(temp) / "entry"
        shutil.copytree(entry, tree, symlinks=True,
                        ignore=lambda directory, names: {".pm-stage-pin.json"}
                        if Path(directory) == entry and ".pm-stage-pin.json" in names else set())
        result = archive_tree(tree, target, output)
    return {**result, "source": identity}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("target", choices=ALL_TARGETS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.package, args.target, args.out), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
