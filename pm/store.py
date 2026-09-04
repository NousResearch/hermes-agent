"""Machine-wide byte store: download, verify, extract, publish atomically."""

from __future__ import annotations

import os
import platform
import shutil
import stat
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

_UA = {"User-Agent": "hermes-pm"}


ALL_TARGETS = (
    "win32-x64",
    "win32-arm64",
    "linux-x64",
    "linux-arm64",
    "darwin-x64",
    "darwin-arm64",
)


def _native_machine() -> str:
    """The MACHINE's architecture, not the interpreter's. An x64 python on
    Windows-on-ARM reports AMD64 — staging a payload for the wrong target.
    IsWow64Process2 reports the real machine regardless of emulation."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.IsWow64Process2.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ctypes.c_ushort),
                ctypes.POINTER(ctypes.c_ushort),
            ]
            k32.IsWow64Process2.restype = wintypes.BOOL
            process_machine = ctypes.c_ushort()
            native_machine = ctypes.c_ushort()
            if k32.IsWow64Process2(
                k32.GetCurrentProcess(),
                ctypes.byref(process_machine),
                ctypes.byref(native_machine),
            ):
                if native_machine.value == 0xAA64:
                    return "arm64"
                if native_machine.value == 0x8664:
                    return "x86_64"
        except Exception:
            pass
        # Pre-IsWow64Process2 hosts: WOW64 exposes the real machine here.
        wow = os.environ.get("PROCESSOR_ARCHITEW6432", "")
        if wow.upper() == "ARM64":
            return "arm64"
        if wow.upper() == "AMD64":
            return "x86_64"
    return platform.machine().lower()


def current_target() -> str:
    machine = _native_machine()
    if machine in ("arm64", "aarch64"):
        arch = "arm64"
    elif machine in ("x86_64", "amd64", "x64"):
        arch = "x64"
    else:
        raise RuntimeError(f"unsupported architecture: {platform.machine()}")
    if sys.platform.startswith("win"):
        return f"win32-{arch}"
    if sys.platform == "darwin":
        return f"darwin-{arch}"
    return f"linux-{arch}"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_url(url: str) -> str:
    """sha256 of a url's content, streamed. `pm lock` uses this to pin."""
    import hashlib
    import urllib.request

    from pm.downloader import _OPENER

    digest = hashlib.sha256()
    with _OPENER.open(
        urllib.request.Request(url, headers=_UA), timeout=600
    ) as resp:
        for block in iter(lambda: resp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, sha256: str, progress=None) -> Path:
    """Fetch url into dest dir, hash-verified, via the resumable downloader.
    The digest is proven before the caller ever sees the file.
    ``progress(done, total)`` ticks per chunk — a several-hundred-MB engine
    archive on a slow line must never look hung. Partial state lives in
    the downloader's writable partials area (outside scratch, keyed by
    sha256(url)), so an interrupted = failed fetch resumes on the next
    call instead of re-fetching the whole archive. Non-https/non-loopback
    urls are refused by Download itself (ValueError)."""
    from pm.downloader import Download, Source

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / url.rsplit("/", 1)[-1]
    p = (lambda d, t, r: progress(d, t)) if progress is not None else None
    Download([Source(url, archive, sha256)]).run(progress=p)
    return archive


def extract(archive: Path, dest: Path) -> None:
    import tarfile

    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")
    elif name.endswith(".zip"):
        _extract_zip(archive, dest)
    else:
        raise ValueError(f"unsupported archive: {archive.name}")


def _extract_zip(archive: Path, dest: Path) -> None:
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        symlinks: list[tuple] = []
        for info in zf.infolist():
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                symlinks.append((info, zf.read(info).decode("utf-8")))
                continue
            written = Path(zf.extract(info, dest))
            if mode & 0o111 and written.is_file():
                written.chmod(mode & 0o777)
        for info, target in symlinks:
            _zip_symlink(info.filename, target, dest)


def _zip_symlink(member: str, target: str, dest: Path) -> None:
    root = dest.resolve()
    link = (root / member).resolve()
    if not link.is_relative_to(root):
        return
    if Path(target).is_absolute():
        return
    resolved = (link.parent / target).resolve()
    if not resolved.is_relative_to(root):
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except OSError:
        link.write_text(target, encoding="utf-8")


def flatten_single_dir(dest: Path) -> None:
    """Hoist a lone top-level dir's contents unless it IS the layout
    (bin/, cmd/, lib/...). Refuses on name collisions."""
    keep = {"bin", "cmd", "lib", "libexec", "share", "etc", "usr"}
    entries = list(dest.iterdir())
    if len(entries) != 1 or not entries[0].is_dir() or entries[0].name in keep:
        return
    inner = entries[0]
    for item in list(inner.iterdir()):
        target = dest / item.name
        if target.exists():
            return
        item.rename(target)
    inner.rmdir()


def merge_tree(src: Path, dst: Path) -> None:
    """Move src's tree into dst, keeping both layouts. A file present in
    both is unresolvable — two archives disagreeing about one file cannot
    be settled by extraction order, so it fails loudly instead."""
    for item in sorted(src.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        target = dst / rel
        if target.exists():
            raise FileExistsError(f"archives disagree about {rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        item.replace(target)


def tree_digest(root: Path) -> str:
    """Deterministic sha256 over a directory tree: walk every file, sort
    by posix relpath, hash `relpath\\0<content>` per entry. No mtimes, no
    mode bits. Symlinks contribute their LINK TARGET TEXT (os.readlink),
    not the target's bytes — the link is the data. Directory symlinks are
    not followed.

    ``__pycache__`` directories are skipped: CPython writes .pyc caches
    into them the first time the staged interpreter runs (uv venv/uv sync
    in a bundle build; first boot of a shipped app), so they are runtime
    state, not package bytes — the digest is over what pm published."""
    import hashlib

    files: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames) if d != "__pycache__"]
        for fname in filenames:
            path = Path(dirpath) / fname
            files.append((path.relative_to(root).as_posix(), path))
    files.sort(key=lambda item: item[0])

    digest = hashlib.sha256()
    for rel, path in files:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(os.readlink(path).encode("utf-8"))
        else:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


class Store:
    """One directory of immutable published entries plus a scratch area.
    Downloads are entries too, keyed by hash, so rebuilds never re-fetch."""

    def __init__(self, root: Path):
        self.root = root

    def entry(self, name: str) -> Path:
        return self.root / name

    def published(self, name: str) -> bool:
        return self.entry(name).is_dir()

    def fetch(self, url: str, sha256: str, scratch: Path, progress=None) -> Path:
        """Verified archive for url, from the store if already fetched.
        The cache entry is `fetch-<full sha256>/` holding the single file.
        The cached file is RE-HASHED against the requested digest before
        it is returned: the cache lives on a mutable disk, so trust is
        re-proven, not assumed. On mismatch the entry is deleted and the
        archive re-downloaded (the 100MB+ re-hash is install-time only)."""
        entry_name = f"fetch-{sha256}"
        entry = self.entry(entry_name)
        if entry.is_dir():
            files = [p for p in entry.iterdir() if p.is_file()]
            if len(files) == 1 and sha256_file(files[0]) == sha256:
                return files[0]
            shutil.rmtree(entry, ignore_errors=True)
        archive = download(url, scratch, sha256, progress=progress)
        staged = scratch / entry_name
        staged.mkdir(parents=True)
        archive.rename(staged / archive.name)
        published = self.publish(staged, entry_name)
        return published / archive.name

    @contextmanager
    def scratch(self):
        self.root.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def publish(self, staged: Path, name: str) -> Path:
        """Atomic rename into place, retried for Windows file-lock holds
        (Defender, indexers). A concurrent winner's entry is kept."""
        target = self.entry(name)
        delay = 0.5
        for _ in range(5):
            try:
                os.replace(staged, target)
                return target
            except OSError:
                if target.is_dir():
                    return target
                time.sleep(delay)
                delay *= 2
        try:
            os.replace(staged, target)
        except OSError:
            if not target.is_dir():
                raise
        return target

    @contextmanager
    def install_lock(self):
        """Single-flight advisory lock for all installs on this machine.
        Blocks until acquired: installs legitimately hold it for minutes
        (browser downloads), so the Windows rung polls LK_NBLCK instead of
        msvcrt's 10-second LK_LOCK ladder."""
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".install.lock"
        handle = open(lock_path, "a+b")
        try:
            if sys.platform.startswith("win"):
                import msvcrt

                handle.seek(0)
                waited = 0
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if waited == 5:
                            print("waiting for another hermes install to finish...")
                        time.sleep(1)
                        waited += 1
            else:
                import fcntl

                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    print("waiting for another hermes install to finish...")
                    fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            try:
                if sys.platform.startswith("win"):
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()
