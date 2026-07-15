#!/usr/bin/env python3
"""Owner-only append-only journal for the canary storage boot transaction."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Mapping


DEFAULT_JOURNAL_ROOT = Path(
    "/Users/emillomliev/.hermes/canary-storage-boot-transactions"
)
OWNER_UID = 501
OWNER_GID = 20
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_MAX_ARTIFACT_BYTES = 256 * 1024
_TRANSACTION_ID = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_NAMES = ("intent", "stop", "start", "completion")


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeError(
            "storage boot journal artifact is not canonical JSON"
        ) from exc


class StorageBootJournal:
    """No-replace journal with atomic hard-link publication and fsync readback."""

    def __init__(
        self,
        *,
        _root: Path = DEFAULT_JOURNAL_ROOT,
        _owner_uid: int = OWNER_UID,
        _owner_gid: int = OWNER_GID,
    ) -> None:
        self._root = _root
        self._owner_uid = _owner_uid
        self._owner_gid = _owner_gid

    @classmethod
    def _for_test(cls, root: Path) -> StorageBootJournal:
        return cls(_root=root, _owner_uid=os.geteuid(), _owner_gid=os.getegid())

    @property
    def root(self) -> Path:
        return self._root

    def _require_owner_process(self) -> None:
        if os.geteuid() != self._owner_uid or os.getegid() != self._owner_gid:
            raise PermissionError("storage boot journal owner process is not exact")

    def _validate_directory(self, path: Path) -> None:
        item = os.lstat(path)
        if (
            not stat.S_ISDIR(item.st_mode)
            or stat.S_ISLNK(item.st_mode)
            or item.st_uid != self._owner_uid
            or item.st_gid != self._owner_gid
            or stat.S_IMODE(item.st_mode) != _DIRECTORY_MODE
        ):
            raise PermissionError("storage boot journal directory is not owner-only")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _ensure_root(self) -> None:
        self._require_owner_process()
        parent = self._root.parent
        self._validate_directory(parent)
        try:
            os.mkdir(self._root, _DIRECTORY_MODE)
        except FileExistsError:
            pass
        self._validate_directory(self._root)
        self._fsync_directory(parent)

    def _transaction_root(self, transaction_id: str, *, create: bool) -> Path | None:
        if (
            not isinstance(transaction_id, str)
            or _TRANSACTION_ID.fullmatch(transaction_id) is None
        ):
            raise RuntimeError("storage boot journal transaction id is invalid")
        self._require_owner_process()
        if not os.path.lexists(self._root):
            if not create:
                return None
            self._ensure_root()
        else:
            self._validate_directory(self._root)
        path = self._root / transaction_id
        if not os.path.lexists(path):
            if not create:
                return None
            os.mkdir(path, _DIRECTORY_MODE)
            self._fsync_directory(self._root)
        self._validate_directory(path)
        allowed = {
            *(f"{name}.json" for name in _ARTIFACT_NAMES),
            *(f".{name}.pending" for name in _ARTIFACT_NAMES),
        }
        if not set(os.listdir(path)) <= allowed:
            raise RuntimeError("storage boot journal transaction has extra entries")
        return path

    def _validate_file(self, path: Path, *, allow_link_pair: bool = False) -> bytes:
        before = os.lstat(path)
        allowed_links = {1, 2} if allow_link_pair else {1}
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != self._owner_uid
            or before.st_gid != self._owner_gid
            or stat.S_IMODE(before.st_mode) != _FILE_MODE
            or before.st_nlink not in allowed_links
            or not 0 < before.st_size <= _MAX_ARTIFACT_BYTES
        ):
            raise PermissionError("storage boot journal artifact identity is invalid")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
                opened.st_uid,
                opened.st_gid,
                opened.st_size,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_uid,
                before.st_gid,
                before.st_size,
            ):
                raise RuntimeError("storage boot journal artifact changed before read")
            raw = os.read(descriptor, _MAX_ARTIFACT_BYTES + 1)
            if os.read(descriptor, 1) or len(raw) != before.st_size:
                raise RuntimeError("storage boot journal artifact framing is invalid")
            after = os.fstat(descriptor)
            if (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_uid,
                after.st_gid,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
                opened.st_uid,
                opened.st_gid,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ):
                raise RuntimeError("storage boot journal artifact changed during read")
        finally:
            os.close(descriptor)
        return raw

    @staticmethod
    def _decode(raw: bytes) -> dict[str, object]:
        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            value: dict[str, object] = {}
            for name, item in pairs:
                if name in value:
                    raise ValueError("duplicate key")
                value[name] = item
            return value

        try:
            value = json.loads(
                raw.decode("ascii", errors="strict"),
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda _item: (_ for _ in ()).throw(ValueError()),
            )
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("storage boot journal artifact is invalid JSON") from exc
        if not isinstance(value, dict) or _canonical_bytes(value) != raw:
            raise RuntimeError("storage boot journal artifact is not canonical")
        return value

    def _write_pending(self, path: Path, raw: bytes) -> None:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path, flags, _FILE_MODE)
        try:
            os.fchown(descriptor, self._owner_uid, self._owner_gid)
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise OSError("storage boot journal write made no progress")
                offset += written
            os.fchmod(descriptor, _FILE_MODE)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _recover_pending(self, root: Path, name: str) -> None:
        final = root / f"{name}.json"
        pending = root / f".{name}.pending"
        if not os.path.lexists(pending):
            return
        pending_raw = self._validate_file(pending, allow_link_pair=True)
        if not os.path.lexists(final):
            os.link(pending, final, follow_symlinks=False)
            self._fsync_directory(root)
        final_raw = self._validate_file(final, allow_link_pair=True)
        if final_raw != pending_raw:
            raise RuntimeError("storage boot journal pending artifact diverged")
        os.unlink(pending)
        self._fsync_directory(root)
        if self._validate_file(final) != final_raw:
            raise RuntimeError("storage boot journal publication readback diverged")

    def read(self, transaction_id: str, name: str) -> dict[str, object] | None:
        if name not in _ARTIFACT_NAMES:
            raise RuntimeError("storage boot journal artifact name is invalid")
        root = self._transaction_root(transaction_id, create=False)
        if root is None:
            return None
        self._recover_pending(root, name)
        path = root / f"{name}.json"
        if not os.path.lexists(path):
            return None
        return self._decode(self._validate_file(path))

    def publish(
        self,
        transaction_id: str,
        name: str,
        value: Mapping[str, object],
    ) -> dict[str, object]:
        if name not in _ARTIFACT_NAMES:
            raise RuntimeError("storage boot journal artifact name is invalid")
        raw = _canonical_bytes(value)
        if not raw or len(raw) > _MAX_ARTIFACT_BYTES:
            raise RuntimeError("storage boot journal artifact exceeds its bound")
        root = self._transaction_root(transaction_id, create=True)
        assert root is not None
        self._recover_pending(root, name)
        final = root / f"{name}.json"
        if os.path.lexists(final):
            existing = self._validate_file(final)
            if existing != raw:
                raise RuntimeError("storage boot journal artifact diverged")
            return self._decode(existing)
        pending = root / f".{name}.pending"
        self._write_pending(pending, raw)
        self._fsync_directory(root)
        self._recover_pending(root, name)
        existing = self._validate_file(final)
        if existing != raw:
            raise RuntimeError("storage boot journal publication diverged")
        return self._decode(existing)


__all__ = [
    "DEFAULT_JOURNAL_ROOT",
    "OWNER_GID",
    "OWNER_UID",
    "StorageBootJournal",
]
