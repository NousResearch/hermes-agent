"""Machine-wide byte store: download, verify, extract, publish atomically."""

from __future__ import annotations

import os
import platform
import shutil
import stat
import sys
import threading
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from pm.filesystem import is_junction



ALL_TARGETS = (
    "win32-x64",
    "win32-arm64",
    "linux-x64",
    "linux-arm64",
    "linux-arm64-bionic",
    "darwin-x64",
    "darwin-arm64",
)


def _native_machine() -> str:
    """Native host arch; PM also runs alone before the app is importable."""
    try:
        from hermes_platform.host.facts import native_arch
    except ModuleNotFoundError as exc:
        if exc.name != "hermes_platform":
            raise
    else:
        return native_arch()

    # Bootstrap's independent PM runtime contains only pm/, not the app.
    # Consult the OS for translated processes rather than trusting Python's
    # process architecture (Rosetta and WOW64 both report the wrong target).
    if sys.platform == "darwin" and platform.machine().lower() in ("x86_64", "amd64"):
        try:
            import ctypes

            value = ctypes.c_int()
            size = ctypes.c_size_t(ctypes.sizeof(value))
            if ctypes.CDLL(None).sysctlbyname(b"sysctl.proc_translated", ctypes.byref(value),
                                              ctypes.byref(size), None, 0) == 0 and value.value == 1:
                return "arm64"
        except (AttributeError, OSError):
            pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            kernel.IsWow64Process2.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_ushort),
                                                ctypes.POINTER(ctypes.c_ushort)]
            kernel.IsWow64Process2.restype = wintypes.BOOL
            process, native = ctypes.c_ushort(), ctypes.c_ushort()
            if kernel.IsWow64Process2(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native)):
                if native.value == 0xAA64:
                    return "arm64"
                if native.value == 0x8664:
                    return "x86_64"
        except (AttributeError, OSError):
            pass
        wow = os.environ.get("PROCESSOR_ARCHITEW6432", "").lower()
        if wow in ("arm64", "amd64"):
            return wow
    return platform.machine().lower()


def _is_bionic_libc() -> bool:
    """True on Android/bionic userlands (Termux and friends).

    The libc flavor is part of the target triple, not a platform runtime
    branch: a linux-arm64 glibc artifact cannot exec under bionic and vice
    versa, so the resolver must pick the right row of the lock table. This
    is the ONE place hermes probes for bionic; nothing downstream of
    current_target() needs to know how it was decided.
    """
    if sys.platform == "android":
        return True
    # Termux's pre-3.13 interpreter reports Linux but records its Android
    # build target in sysconfig. This also works in a container without
    # the phone's /system mount or a recognizable string in libc's header.
    import sysconfig

    return bool(sysconfig.get_config_var("ANDROID_API_LEVEL"))


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
    if _is_bionic_libc():
        return f"linux-{arch}-bionic"
    return f"linux-{arch}"


def hash_url(url: str) -> str:
    """sha256 of a url's content, streamed. `pm lock` uses this to pin."""
    import hashlib
    import http.client
    import urllib.request

    from pm.downloader import _OPENER, _UA
    from pm.network import retry_network

    def request():
        digest = hashlib.sha256()
        size = 0
        with _OPENER.open(
            urllib.request.Request(url, headers=_UA), timeout=600
        ) as resp:
            declared = int(resp.headers.get("Content-Length") or 0)
            for block in iter(lambda: resp.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            if size < declared:
                raise http.client.IncompleteRead(b"", declared - size)
        return digest.hexdigest()

    return retry_network(request)


def _tar_filter(member, dest: str):
    """The stdlib 'data' filter, with symlink targets resolved from the link's own
    directory. Bootstrap interpreters (Ubuntu 22.04 ships 3.10) resolve them from
    the archive root and reject python-build-standalone's terminfo links."""
    import tarfile

    if member.issym():
        if os.path.isabs(member.linkname):
            raise tarfile.AbsoluteLinkError(member)
        name = member.name.rstrip("/")
        placed = os.path.realpath(os.path.join(dest, name))
        link_dir = os.path.dirname(name)
        target = os.path.realpath(os.path.join(dest, link_dir, member.linkname))
        for path in (placed, target):
            if os.path.commonpath([path, dest]) != dest:
                raise tarfile.LinkOutsideDestinationError(member, path)
        return member.replace(deep=False, uid=None, gid=None, uname=None, gname=None, mode=None)
    return tarfile.data_filter(member, dest)

def extract_tar(archive: Path | IO[bytes], dest: Path, *, git_msys: bool = False) -> None:
    """Extract a tarball (a path, or an open stream such as a .deb's data.tar)
    with the one containment policy every PM tar consumer shares. Unsafe
    members raise tarfile.FilterError.

    MSYS Git ships dev/fd links and etc/mtab into /proc; those aren't usable
    on Windows. Skip only those known links, never a filter error or failed file write.
    """
    import tarfile

    dest.mkdir(parents=True, exist_ok=True)
    real_dest = os.path.realpath(dest)
    opened = tarfile.open(archive) if isinstance(archive, (str, os.PathLike)) else tarfile.open(fileobj=archive)
    with opened as tf:
        if git_msys:
            members = (m for m in tf if not (m.issym() and (
                (m.name.lstrip("./").startswith("dev/") and m.linkname.startswith("/proc/"))
                or (m.name == "etc/mtab" and m.linkname == "/proc/mounts")
            )))
            for member in members:
                tf.extract(member, dest, filter=lambda item, path: _tar_filter(item, real_dest))
        else:
            tf.extractall(dest, filter=lambda member, path: _tar_filter(member, real_dest))


def extract(archive: Path, dest: Path) -> None:
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2")):
        extract_tar(archive, dest)
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
    not the target's bytes — the link is the data. Directory symlinks and
    junctions are not followed.

    ``__pycache__`` directories are skipped: CPython writes .pyc caches
    into them the first time the staged interpreter runs (uv venv/uv sync
    in a bundle build; first boot of a shipped app), so they are runtime
    state, not package bytes — the digest is over what pm published."""
    import hashlib

    files: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        descend = []
        for name in sorted(dirnames):
            path = Path(dirpath) / name
            if path.is_symlink() or is_junction(path):
                files.append((path.relative_to(root).as_posix(), path))
            elif name != "__pycache__":
                descend.append(name)
        dirnames[:] = descend
        for fname in filenames:
            path = Path(dirpath) / fname
            files.append((path.relative_to(root).as_posix(), path))
    files.sort(key=lambda item: item[0])

    digest = hashlib.sha256()
    for rel, path in files:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        if path.is_symlink() or is_junction(path):
            digest.update(os.readlink(path).encode("utf-8"))
        else:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


class Store:
    """One directory of immutable published entries plus a scratch area.
    Hash-keyed archives survive failed installs. Publication releases them."""

    def __init__(self, root: Path):
        self.root = root

    def entry(self, name: str) -> Path:
        return self.root / name

    def published(self, name: str) -> bool:
        return self.entry(name).is_dir()

    def fetch(self, url: str, sha256: str, scratch: Path, progress=None, *, pause_event: threading.Event | None = None) -> Path:
        tick = (lambda done, total, ranges: progress(done, total)) if progress is not None else None
        return self.fetch_many([{"url": url, "sha256": sha256}], scratch,
                               progress=tick, pause_event=pause_event)[0]

    def fetch_many(self, artifacts: list[dict], scratch: Path, *, progress=None,
                   pause_event: threading.Event | None = None) -> list[Path]:
        """Fetch a pinned plan with one aggregate progress stream.

        Completed archives enter the cache even if a later source pauses.
        The downloader owns partial bytes outside this disposable scratch.
        """
        from pm.downloader import Download, DownloadPaused
        from pm.artifact_mirror import pinned_source

        sources = []
        for artifact in artifacts:
            if pause_event is not None and pause_event.is_set():
                raise DownloadPaused("download paused")
            url, digest = artifact["url"], artifact["sha256"]
            entry_name = f"fetch-{digest}"
            entry = self.entry(entry_name)
            files = list(entry.iterdir()) if entry.is_dir() else []
            if len(files) == 1 and files[0].is_file():
                destination = files[0]
            else:
                if entry.exists():
                    shutil.rmtree(entry)
                destination = scratch / entry_name / url.rsplit("/", 1)[-1]
            sources.append(pinned_source(url, destination, digest))

        urls = {str(source.dest): source.url for source in sources}

        def tick(done, total, ranges):
            if progress is not None:
                progress(done, total, {urls[key]: rows for key, rows in ranges.items()})

        try:
            Download(sources, pause_event=pause_event).run(progress=tick)
        finally:
            # Only finalized, hash-verified files can exist at these paths.
            for source in sources:
                if source.dest.is_relative_to(scratch) and source.dest.is_file():
                    self.publish(source.dest.parent, f"fetch-{source.sha256}")
        return [self.entry(f"fetch-{source.sha256}") / source.dest.name for source in sources]

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
        """Serialize writers using the same advisory lock as runtime publication."""
        from pm.filesystem import lock_fd
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".install.lock"
        fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            # A second `hermes pm install` behind an sdist build otherwise sits
            # silent for minutes; say what it is waiting on.
            if not lock_fd(fd, wait=True, timeout=2):
                print(f"waiting for {lock} (another PM operation holds it)", file=sys.stderr, flush=True)
                lock_fd(fd, wait=True)
            yield
        finally:
            os.close(fd)
