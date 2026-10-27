"""Read-only bounded helpers for the stock cron file sources.

This module deliberately knows only the fixed files owned by ``cron``.  It does
not import the cron package: importing that package can initialise an empty
profile.  Callers keep the returned bytes and source-owned JSON values; these
helpers never repair, rewrite, or normalize a file.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any, Iterator


_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_CHUNK = 64 * 1024


class FileCaptureRefused(RuntimeError):
    """A cron-owned file is absent, substituted, malformed, or unbounded."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Stable file identity; timestamps and sizes are intentionally excluded."""

    device: int
    inode: int
    file_type: int
    permissions: int
    owner: int
    group: int
    attributes: int
    reparse: bool

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> FileIdentity:
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        return cls(
            device=int(getattr(metadata, "st_dev", -1)),
            inode=int(getattr(metadata, "st_ino", -1)),
            file_type=stat.S_IFMT(metadata.st_mode),
            permissions=stat.S_IMODE(metadata.st_mode),
            owner=int(getattr(metadata, "st_uid", -1)),
            group=int(getattr(metadata, "st_gid", -1)),
            attributes=attributes,
            reparse=stat.S_ISLNK(metadata.st_mode)
            or bool(attributes & _REPARSE),
        )

    @property
    def stable(self) -> tuple[int, int, int, int, int, int, bool]:
        return (
            self.device,
            self.inode,
            self.file_type,
            self.owner,
            self.group,
            self.attributes,
            self.reparse,
        )


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: Path
    identity: FileIdentity
    size: int
    digest: bytes | None = None


def _metadata(path: Path, *, follow: bool = False) -> os.stat_result:
    try:
        return path.stat() if follow else path.lstat()
    except FileNotFoundError as exc:
        raise FileCaptureRefused("cron_capture_file_missing") from exc
    except OSError as exc:
        raise FileCaptureRefused("cron_capture_file_unavailable") from exc


def identity(path: Path, *, follow: bool = False) -> FileIdentity:
    return FileIdentity.from_stat(_metadata(path, follow=follow))


def snapshot(path: Path, *, regular: bool = True) -> FileSnapshot:
    metadata = _metadata(path)
    current = FileIdentity.from_stat(metadata)
    if current.reparse:
        raise FileCaptureRefused("cron_capture_file_reparse")
    if regular and current.file_type != stat.S_IFREG:
        raise FileCaptureRefused("cron_capture_file_not_regular")
    if not regular and current.file_type != stat.S_IFDIR:
        raise FileCaptureRefused("cron_capture_file_not_directory")
    # Directory length is allocator metadata that changes when children are
    # created or removed.  The custody contract compares stable directory
    # identity/permissions only; only regular-file size is part of this
    # snapshot.
    return FileSnapshot(path, current, int(metadata.st_size) if regular else -1)


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise FileCaptureRefused("cron_capture_deadline_exceeded")


def assert_current(saved: FileSnapshot, *, deadline: float | None = None) -> None:
    _check_deadline(deadline)
    current = snapshot(saved.path, regular=saved.identity.file_type == stat.S_IFREG)
    if current.identity != saved.identity or current.size != saved.size:
        raise FileCaptureRefused("cron_capture_file_changed")
    if saved.digest is not None and _content_digest(saved.path, deadline=deadline) != saved.digest:
        raise FileCaptureRefused("cron_capture_file_changed")


def _open_read(path: Path, *, deadline: float | None = None) -> tuple[int, FileSnapshot]:
    _check_deadline(deadline)
    before = snapshot(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FileCaptureRefused("cron_capture_file_unavailable") from exc
    try:
        _check_deadline(deadline)
        opened = FileIdentity.from_stat(os.fstat(descriptor))
        if opened != before.identity:
            raise FileCaptureRefused("cron_capture_file_changed")
        os.set_inheritable(descriptor, False)
        return descriptor, before
    except BaseException:
        with _suppress_os_error():
            os.close(descriptor)
        raise


def _content_digest(path: Path, *, deadline: float | None = None) -> bytes:
    descriptor, saved = _open_read(path, deadline=deadline)
    digest = hashlib.sha256()
    try:
        while True:
            _check_deadline(deadline)
            chunk = os.read(descriptor, _CHUNK)
            if not chunk:
                break
            digest.update(chunk)
        _check_deadline(deadline)
    except OSError as exc:
        raise FileCaptureRefused("cron_capture_file_unavailable") from exc
    finally:
        with _suppress_os_error():
            os.close(descriptor)
    _check_deadline(deadline)
    current = snapshot(path)
    if current.identity != saved.identity or current.size != saved.size:
        raise FileCaptureRefused("cron_capture_file_changed")
    return digest.digest()


class _suppress_os_error:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is OSError


def read_bytes(
    path: Path,
    *,
    max_bytes: int,
    deadline: float | None = None,
) -> tuple[bytes, FileSnapshot]:
    """Read one regular file through a descriptor and verify its name is stable."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise FileCaptureRefused("cron_capture_budget_invalid")
    descriptor, saved = _open_read(path, deadline=deadline)
    pieces: list[bytes] = []
    digest = hashlib.sha256()
    total = 0
    try:
        while True:
            _check_deadline(deadline)
            chunk = os.read(descriptor, min(_CHUNK, max_bytes - total + 1))
            if not chunk:
                break
            pieces.append(chunk)
            digest.update(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise FileCaptureRefused("cron_capture_byte_budget_exceeded")
        _check_deadline(deadline)
    except OSError as exc:
        raise FileCaptureRefused("cron_capture_file_unavailable") from exc
    finally:
        with _suppress_os_error():
            os.close(descriptor)
    raw = b"".join(pieces)
    assert_current(saved, deadline=deadline)
    return raw, FileSnapshot(saved.path, saved.identity, saved.size, digest.digest())


def parse_json(
    raw: bytes,
    *,
    code: str = "cron_capture_json_malformed",
    deadline: float | None = None,
) -> Any:
    _check_deadline(deadline)
    try:
        text = raw.decode("utf-8-sig")
        _check_deadline(deadline)
        value = json.loads(text, object_pairs_hook=_object_pairs)
        _check_deadline(deadline)
        return value
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise FileCaptureRefused(code) from exc


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FileCaptureRefused("cron_capture_json_duplicate_key")
        result[key] = value
    return result


def json_lines(
    raw: bytes,
    *,
    code: str,
    deadline: float | None = None,
) -> Iterator[tuple[bytes, Any]]:
    """Yield exact JSONL lines and their producer-compatible parsed values."""

    number = 0
    start = 0
    while start < len(raw):
        _check_deadline(deadline)
        end = raw.find(b"\n", start)
        if end < 0:
            end = len(raw)
            line = raw[start:end]
            start = end
        else:
            end += 1
            line = raw[start:end]
            start = end
        number += 1
        content = line.rstrip(b"\r\n")
        if not content.strip():
            continue
        try:
            _check_deadline(deadline)
            value = json.loads(
                content.decode("utf-8-sig" if number == 1 else "utf-8"),
                object_pairs_hook=_object_pairs,
            )
            _check_deadline(deadline)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise FileCaptureRefused(code) from exc
        yield line, value
    _check_deadline(deadline)


def safe_component(value: object) -> bool:
    """Match the producer's single output-directory component boundary."""

    if not isinstance(value, str) or not value or value in {".", ".."}:
        return False
    if Path(value).name != value:
        return False
    return "/" not in value and "\\" not in value and "\x00" not in value


def bounded_children(
    path: Path,
    *,
    max_entries: int | None = None,
    deadline: float | None = None,
) -> tuple[Path, ...]:
    if max_entries is not None and (type(max_entries) is not int or max_entries <= 0):
        raise FileCaptureRefused("cron_capture_record_budget_exceeded")
    _check_deadline(deadline)
    children: list[Path] = []
    try:
        for child in path.iterdir():
            _check_deadline(deadline)
            if max_entries is not None and len(children) >= max_entries:
                raise FileCaptureRefused("cron_capture_record_budget_exceeded")
            children.append(child)
    except FileNotFoundError as exc:
        raise FileCaptureRefused("cron_capture_directory_missing") from exc
    except OSError as exc:
        raise FileCaptureRefused("cron_capture_directory_unavailable") from exc
    _check_deadline(deadline)
    return tuple(sorted(children, key=lambda item: item.name))


def scalar_bytes(value: Any) -> bytes | None:
    """Preserve source scalar bytes where SQLite/file JSON exposes a scalar."""

    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", "surrogatepass")
    return None


def encoded_size(value: Any) -> int:
    """Deterministic budget estimate for a field value without changing its value."""

    try:
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError, UnicodeEncodeError):
        return len(repr(value).encode("utf-8", "replace"))


__all__ = [
    "FileCaptureRefused",
    "FileIdentity",
    "FileSnapshot",
    "assert_current",
    "bounded_children",
    "encoded_size",
    "identity",
    "json_lines",
    "parse_json",
    "read_bytes",
    "safe_component",
    "scalar_bytes",
    "snapshot",
]
