"""Whole-directory transfer for the managed files API: ``GET``/``PUT /api/files/archive``.

A remote Desktop, a peer gateway or a script moves a workspace folder in one request instead of
one ``/api/files/upload`` per file. ``GET`` streams a gzip tar of a directory; ``PUT`` unpacks a
tar or tar.gz body into one. Paths resolve through the same managed-files policy as every other
``/api/files`` route (locked root, ``..`` refusal, credential denylist), and auth is the dashboard's
session middleware, as for those routes.

Upload semantics: the whole archive is checked before anything is written; a refused archive
leaves the target untouched. Then each file is written beside its destination and renamed into
place. Existing files at an entry's path are replaced; nothing the archive does not name is
deleted. Refused entries: absolute or ``..`` paths, hard links, devices, FIFOs, symlinks that
point outside the target directory (or entries that pass through one), credential files, and
anything that would replace a directory with a file or the reverse.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import logging
import os
import posixpath
import queue
import stat
import tarfile
import tempfile
import threading
import urllib.parse
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Iterator, Optional, cast

from agent.memory_provider import spawn_context_thread
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from hermes_cli.sqlite_safe_read import LiveConnectionError, offline_file_access
from hermes_cli.web_routers.files import _is_local_sensitive_path, _is_sensitive_path, _refuse_sensitive_write
from hermes_cli.web_server_files import _managed_response_meta, _resolve_managed_path
from hermes_constants import get_hermes_home

router = APIRouter()
logger = logging.getLogger(__name__)

ARCHIVE_MAX_BYTES = 1 << 30  # total file content, either direction
ARCHIVE_MAX_ENTRIES = 100_000
# The request body may carry tar headers and padding on top of the content cap.
_UPLOAD_BODY_MAX_BYTES = ARCHIVE_MAX_BYTES + 128 * 1024 * 1024
_COPY_CHUNK = 1024 * 1024
_STREAM_CHUNK = 256 * 1024


class _Refused(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _escapes(rel_parent: str, linkname: str) -> bool:
    """True when a symlink at *rel_parent*/x pointing at *linkname* resolves outside the root."""
    if not linkname or linkname.startswith(("/", "\\")) or PurePosixPath(linkname).drive or ":" in linkname[:3]:
        return True
    joined = posixpath.normpath(posixpath.join(rel_parent or ".", linkname))
    return joined == ".." or joined.startswith("../")


def _is_under(root: str, path: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


# ---------------------------------------------------------------- download


@dataclass(frozen=True)
class _Entry:
    path: Path
    arcname: str
    kind: str  # "dir" | "file" | "symlink"
    linkname: str = ""


def _walk_entries(root: Path) -> list[_Entry]:
    """Every entry under *root* the archive will carry, checked against the caps up front so a
    413 arrives before any byte is streamed. Credential files, special files and symlinks that
    point outside *root* are left out; symlinked directories are carried as links, not followed."""
    entries: list[_Entry] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root)
        rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
        keep = []
        for name in sorted(dirnames) + sorted(filenames):
            path = Path(dirpath, name)
            arcname = f"{rel_dir}/{name}" if rel_dir else name
            entry = _entry_for(path, arcname, rel_dir)
            if entry is None:
                continue
            if entry.kind == "dir":
                keep.append(name)
            elif entry.kind == "file":
                total += path.lstat().st_size
            entries.append(entry)
            if len(entries) > ARCHIVE_MAX_ENTRIES or total > ARCHIVE_MAX_BYTES:
                raise HTTPException(status_code=413, detail=(
                    f"Directory is too large to archive (limit {ARCHIVE_MAX_ENTRIES} entries, "
                    f"{ARCHIVE_MAX_BYTES >> 20} MiB)"))
        dirnames[:] = keep
    return entries


def _entry_for(path: Path, arcname: str, rel_dir: str) -> Optional[_Entry]:
    try:
        st = path.lstat()
    except OSError:
        return None
    if _is_local_sensitive_path(path):
        return None
    if stat.S_ISLNK(st.st_mode):
        link = os.readlink(path)
        return None if _escapes(rel_dir, link) else _Entry(path, arcname, "symlink", link)
    if stat.S_ISDIR(st.st_mode):
        return _Entry(path, arcname, "dir")
    if stat.S_ISREG(st.st_mode):
        return _Entry(path, arcname, "file")
    return None  # sockets, FIFOs, devices


class _QueueWriter:
    """File-like sink for ``tarfile`` that hands bounded chunks to the response iterator."""

    def __init__(self, chunks: queue.Queue, cancelled: threading.Event):
        self._chunks, self._cancelled, self._buf = chunks, cancelled, bytearray()

    def write(self, data) -> int:
        self._buf += data
        if len(self._buf) >= _STREAM_CHUNK:
            self.flush()
        return len(data)

    def flush(self) -> None:
        if self._buf:
            self._put(bytes(self._buf))
            self._buf.clear()

    def _put(self, item) -> None:
        while True:
            if self._cancelled.is_set():
                raise BrokenPipeError("archive download cancelled")
            with contextlib.suppress(queue.Full):
                self._chunks.put(item, timeout=0.5)
                return


def _add_entry(tf: tarfile.TarFile, entry: _Entry) -> None:
    if entry.kind == "file":
        _add_file(tf, entry)
        return
    info = tarfile.TarInfo(entry.arcname)
    info.mtime = int(entry.path.lstat().st_mtime)
    if entry.kind == "dir":
        info.type, info.mode = tarfile.DIRTYPE, 0o755
    else:
        info.type, info.linkname, info.mode = tarfile.SYMTYPE, entry.linkname, 0o777
    tf.addfile(info)


def _add_file(tf: tarfile.TarFile, entry: _Entry) -> None:
    try:
        # A live SQLite file is skipped: a raw read's close would cancel this process's locks on it.
        with offline_file_access(entry.path, what="archive"):
            fd = os.open(entry.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
            with os.fdopen(fd, "rb") as handle:
                info = tf.gettarinfo(arcname=entry.arcname, fileobj=handle)
                if not info.isreg():
                    return
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tf.addfile(info, handle)
    except BrokenPipeError:
        raise
    except (LiveConnectionError, OSError) as exc:  # vanished, unreadable or live: leave it out
        logger.info("files archive: skipped %s: %s", entry.path, exc)


def _produce(entries: list[_Entry], chunks: queue.Queue, cancelled: threading.Event) -> None:
    writer = _QueueWriter(chunks, cancelled)
    try:
        with tarfile.open(fileobj=cast(IO[bytes], writer), mode="w|gz", format=tarfile.PAX_FORMAT) as tf:
            for entry in entries:
                _add_entry(tf, entry)
        writer.flush()
        writer._put(None)
    except BrokenPipeError:
        pass
    except Exception as exc:  # the response is already streaming: end it short, never hang it
        logger.warning("files archive: stream aborted: %s", exc, exc_info=True)
        with contextlib.suppress(BrokenPipeError):
            writer._put(exc)


def _stream_archive(entries: list[_Entry]) -> Iterator[bytes]:
    chunks: queue.Queue = queue.Queue(maxsize=16)
    cancelled = threading.Event()
    producer = spawn_context_thread(_produce, args=(entries, chunks, cancelled), name="files-archive")
    producer.start()
    try:
        while True:
            item = chunks.get()
            if item is None or isinstance(item, BaseException):
                return
            yield item
    finally:
        cancelled.set()


def _download_root(path: str, request: Request):
    _policy, target, _display = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    if _is_local_sensitive_path(target):
        raise HTTPException(status_code=403, detail="Access to sensitive files is not allowed")
    return target, _walk_entries(target)


@router.get("/api/files/archive")
async def download_managed_archive(request: Request, path: str):
    """Stream *path* (a directory) as a gzip tar whose member names are relative to it."""
    target, entries = await asyncio.to_thread(_download_root, path, request)
    name = f"{target.name or 'archive'}.tar.gz"
    return StreamingResponse(
        _stream_archive(entries),
        media_type="application/gzip",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{urllib.parse.quote(name)}",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------- upload

_REFUSED_TYPES = (
    (tarfile.TarInfo.islnk, "hard links are not allowed"),
    (tarfile.TarInfo.ischr, "device files are not allowed"),
    (tarfile.TarInfo.isblk, "device files are not allowed"),
    (tarfile.TarInfo.isfifo, "FIFOs are not allowed"),
)


@dataclass(frozen=True)
class _Planned:
    member: tarfile.TarInfo
    rel: PurePosixPath
    kind: str  # "dir" | "file" | "symlink"


def _member_kind(member: tarfile.TarInfo) -> str:
    for check, reason in _REFUSED_TYPES:
        if check(member):
            raise _Refused(400, f"{member.name}: {reason}")
    if member.isdir():
        return "dir"
    if member.isreg():
        return "file"
    if member.issym():
        return "symlink"
    raise _Refused(400, f"{member.name}: unsupported entry type")


def _member_rel(member: tarfile.TarInfo) -> Optional[PurePosixPath]:
    name = member.name
    if "\x00" in name or name.startswith(("/", "\\")) or ":" in name[:3]:
        raise _Refused(400, f"{name}: absolute paths are not allowed")
    if os.name == "nt" and "\\" in name:
        raise _Refused(400, f"{name}: backslashes are not allowed in entry names")
    parts = [part for part in PurePosixPath(name).parts if part != "."]
    if ".." in parts:
        raise _Refused(400, f"{name}: '..' is not allowed in entry names")
    return PurePosixPath(*parts) if parts else None


def _plan_member(member: tarfile.TarInfo, kinds: dict[PurePosixPath, str]) -> Optional[_Planned]:
    kind = _member_kind(member)
    rel = _member_rel(member)
    if rel is None:
        if kind != "dir":
            raise _Refused(400, f"{member.name}: entry names the target directory itself")
        return None
    for ancestor in list(rel.parents)[:-1]:
        if kinds.setdefault(ancestor, "dir") != "dir":  # implied parents count: a later link can't replace one
            raise _Refused(400, f"{rel}: path passes through a non-directory entry ({ancestor})")
    if kinds.get(rel, kind) != kind:
        raise _Refused(400, f"{rel}: listed both as {kinds[rel]} and {kind}")
    if kind == "symlink" and _escapes(str(rel.parent) if str(rel.parent) != "." else "", member.linkname):
        raise _Refused(400, f"{rel}: symlink points outside the target directory")
    if _is_sensitive_path(Path(*rel.parts)):
        raise _Refused(403, f"{rel}: writing sensitive files is not allowed")
    kinds[rel] = kind
    return _Planned(member, rel, kind)


def _plan_archive(tf: tarfile.TarFile) -> list[_Planned]:
    """Check every member before anything is written; caps counted as the headers stream in."""
    plan: list[_Planned] = []
    kinds: dict[PurePosixPath, str] = {}
    total = count = 0
    for member in tf:
        count += 1
        total += member.size if member.isreg() else 0
        if count > ARCHIVE_MAX_ENTRIES or total > ARCHIVE_MAX_BYTES:
            raise _Refused(413, f"Archive is too large (limit {ARCHIVE_MAX_ENTRIES} entries, "
                                f"{ARCHIVE_MAX_BYTES >> 20} MiB)")
        planned = _plan_member(member, kinds)
        if planned is not None:
            plan.append(planned)
    return plan


def _check_destination(root: str, target: Path, item: _Planned) -> None:
    """Refuse an entry the disk would send outside *root* (an existing symlink on the way) or that
    would replace a directory with a non-directory, or the reverse."""
    dest = target.joinpath(*item.rel.parts)
    if not _is_under(root, os.path.realpath(dest.parent)):
        raise _Refused(400, f"{item.rel}: an existing symlink leads outside the target directory")
    if item.kind == "dir":
        if os.path.lexists(dest) and not dest.is_dir():
            raise _Refused(409, f"{item.rel}: a file already exists where the archive has a directory")
        if dest.is_dir() and not _is_under(root, os.path.realpath(dest)):
            raise _Refused(400, f"{item.rel}: an existing symlink leads outside the target directory")
    elif dest.is_dir() and not dest.is_symlink():
        raise _Refused(409, f"{item.rel}: a directory already exists where the archive has a file")
    if _is_local_sensitive_path(dest):
        raise _Refused(403, f"{item.rel}: writing sensitive files is not allowed")


def _write_file(tf: tarfile.TarFile, item: _Planned, dest: Path) -> None:
    mode = (item.member.mode & 0o755) | 0o600
    if not item.member.mode & 0o100:
        mode &= ~0o111
    source = tf.extractfile(item.member)
    assert source is not None  # planned as a regular file
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".archive", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := source.read(_COPY_CHUNK):
                out.write(chunk)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, dest)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    with contextlib.suppress(OSError):
        os.utime(dest, (item.member.mtime, item.member.mtime))


def _apply(tf: tarfile.TarFile, plan: list[_Planned], target: Path) -> dict[str, int]:
    target.mkdir(parents=True, exist_ok=True)
    root = os.path.realpath(target)
    counts = {"files": 0, "directories": 0, "symlinks": 0, "bytes": 0}
    for item in plan:
        dest = target.joinpath(*item.rel.parts)
        _check_destination(root, target, item)  # again at write time: earlier entries changed the disk
        dest.parent.mkdir(parents=True, exist_ok=True)
        if item.kind == "dir":
            dest.mkdir(exist_ok=True)
            counts["directories"] += 1
        elif item.kind == "file":
            _write_file(tf, item, dest)
            counts["files"] += 1
            counts["bytes"] += item.member.size
        else:
            if os.path.lexists(dest):
                dest.unlink()
            os.symlink(item.member.linkname, dest)
            counts["symlinks"] += 1
    return counts


def _plain_tar(spool: Path) -> Path:
    """A gzip body is inflated to a plain tar beside it first: the plan pass and the write pass
    each walk the archive, and a gzip stream can only seek backward by decompressing again.
    Inflation is capped too, so a small gzip bomb cannot fill the disk."""
    with spool.open("rb") as raw:
        if raw.read(2) != b"\x1f\x8b":
            return spool
    plain, total = spool.with_suffix(".plain.tar"), 0
    try:
        with gzip.open(spool, "rb") as source, plain.open("wb") as out:
            while chunk := source.read(_COPY_CHUNK):
                total += len(chunk)
                if total > _UPLOAD_BODY_MAX_BYTES:
                    raise _Refused(413, "Archive is too large once decompressed")
                out.write(chunk)
    except BaseException:
        plain.unlink(missing_ok=True)
        raise
    return plain


def _extract(spool: Path, target: Path) -> dict[str, int]:
    plain = spool
    try:
        plain = _plain_tar(spool)
        with tarfile.open(plain, mode="r:") as tf:
            plan = _plan_archive(tf)
            root = os.path.realpath(target)
            for item in plan:
                _check_destination(root, target, item)
            return _apply(tf, plan, target)
    except _Refused as exc:
        raise HTTPException(status_code=exc.status, detail=f"Archive rejected: {exc.detail}") from exc
    except (tarfile.TarError, EOFError, gzip.BadGzipFile, zlib.error) as exc:
        raise HTTPException(status_code=400, detail=f"Body is not a readable tar or tar.gz archive: {exc}") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Directory is not writable") from exc
    finally:
        if plain != spool:
            plain.unlink(missing_ok=True)


def _upload_root(path: str, request: Request):
    policy, target, display_path = _resolve_managed_path(path, request, for_write=True)
    _refuse_sensitive_write(target, path, policy.locked_root)
    if os.path.lexists(target) and not target.is_dir():
        raise HTTPException(status_code=409, detail="A file already exists at that path")
    return policy, target, display_path


def _new_spool() -> tuple[int, str]:
    spool_dir = get_hermes_home() / "cache" / "file-archives"
    spool_dir.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix="upload-", suffix=".tar", dir=str(spool_dir))


async def _spool_body(request: Request) -> Path:
    """Copy the request body to a private file under the profile's cache, capped as it arrives."""
    fd, name = await asyncio.to_thread(_new_spool)
    spool, total = Path(name), 0
    try:
        with os.fdopen(fd, "wb") as out:
            async for chunk in request.stream():
                total += len(chunk)
                if total > _UPLOAD_BODY_MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Archive is too large")
                await asyncio.to_thread(out.write, chunk)
    except BaseException:
        await asyncio.to_thread(spool.unlink, missing_ok=True)
        raise
    return spool


@router.put("/api/files/archive")
@router.post("/api/files/archive")
async def upload_managed_archive(request: Request, path: str):
    """Unpack a tar or tar.gz request body into *path* (created when missing)."""
    policy, target, display_path = await asyncio.to_thread(_upload_root, path, request)
    spool = await _spool_body(request)
    try:
        counts = await asyncio.to_thread(_extract, spool, target)
    finally:
        await asyncio.to_thread(spool.unlink, missing_ok=True)
    return {"ok": True, "path": display_path, **counts, **_managed_response_meta(policy)}
