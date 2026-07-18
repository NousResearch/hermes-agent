#!/usr/bin/env python3
"""Crash-safe append-only journal for bounded foundation provisioning.

Every transition is an immutable canonical artifact.  Publication uses a
no-replace hard-link protocol with file and directory fsync, so a process
death cannot turn an unrecorded provider call into an apparently clean retry.
The artifact inventory is closed to the nine signed foundation steps.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


DEFAULT_JOURNAL_ROOT = Path(
    "/Users/emillomliev/.hermes/owner-gate-foundation-transactions"
)
OWNER_UID = 501
OWNER_GID = 20
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_MAX_ARTIFACT_BYTES = 1024 * 1024
_TRANSACTION_ID = re.compile(r"^[0-9a-f]{64}$")
_STEP_ARTIFACT = re.compile(
    r"^s([0-8])-(pre|intent|operation|post|preexisting|"
    r"rollback-intent|rollback-operation|rollback-post)$"
)
_FIXED_ARTIFACTS = frozenset({
    "manifest",
    "failure-intent",
    "success",
    "failure",
})


def _artifact_name_valid(name: str) -> bool:
    return name in _FIXED_ARTIFACTS or _STEP_ARTIFACT.fullmatch(name) is not None


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
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
            "owner_gate_foundation_journal_json_invalid"
        ) from None


class FoundationApplyJournal:
    """Owner-only, append-only, fsynced foundation transition store."""

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
        self._active_lease: (
            tuple[str, int, Path, tuple[int, int]] | None
        ) = None

    @property
    def root(self) -> Path:
        return self._root

    def _require_owner_process(self) -> None:
        if (
            not callable(getattr(os, "geteuid", None))
            or not callable(getattr(os, "getegid", None))
            or int(os.geteuid()) != self._owner_uid  # windows-footgun: ok — POSIX owner boundary
            or int(os.getegid()) != self._owner_gid  # windows-footgun: ok — POSIX owner boundary
        ):
            raise PermissionError(
                "owner_gate_foundation_journal_owner_invalid"
            )

    def _validate_directory(self, path: Path) -> None:
        item = os.lstat(path)
        if (
            not stat.S_ISDIR(item.st_mode)
            or stat.S_ISLNK(item.st_mode)
            or item.st_uid != self._owner_uid
            or item.st_gid != self._owner_gid
            or stat.S_IMODE(item.st_mode) != _DIRECTORY_MODE
        ):
            raise PermissionError(
                "owner_gate_foundation_journal_directory_invalid"
            )

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
        self._validate_directory(self._root.parent)
        try:
            os.mkdir(self._root, _DIRECTORY_MODE)
        except FileExistsError:
            pass
        self._validate_directory(self._root)
        self._fsync_directory(self._root.parent)

    @contextmanager
    def _locked_transaction(
        self,
        transaction_id: str,
        *,
        create: bool,
    ) -> Iterator[tuple[Path, tuple[int, int]] | None]:
        if (
            not isinstance(transaction_id, str)
            or _TRANSACTION_ID.fullmatch(transaction_id) is None
        ):
            raise RuntimeError(
                "owner_gate_foundation_journal_transaction_invalid"
            )
        self._require_owner_process()
        active = self._active_lease
        if active is not None:
            (
                active_transaction,
                owner_thread,
                active_path,
                active_identity,
            ) = active
            if (
                active_transaction != transaction_id
                or owner_thread != threading.get_ident()
            ):
                raise RuntimeError(
                    "owner_gate_foundation_journal_lease_conflict"
                )
            self._validate_directory(active_path)
            descriptor = os.open(
                active_path,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                active_opened = os.fstat(descriptor)
                active_state = os.lstat(active_path)
            finally:
                os.close(descriptor)
            if (
                (active_state.st_dev, active_state.st_ino)
                != active_identity
                or (active_opened.st_dev, active_opened.st_ino)
                != active_identity
            ):
                raise RuntimeError(
                    "owner_gate_foundation_journal_directory_changed"
                )
            self._validate_inventory(active_path)
            yield active_path, active_identity
            return
        if not os.path.lexists(self._root):
            if not create:
                yield None
                return
            self._ensure_root()
        else:
            self._validate_directory(self._root)
        path = self._root / transaction_id
        if not os.path.lexists(path):
            if not create:
                yield None
                return
            try:
                os.mkdir(path, _DIRECTORY_MODE)
            except FileExistsError:
                # A concurrent process may have created the same canonical
                # transaction directory after the lexists check.  Its exact
                # identity is validated below before the shared OS lease is
                # acquired; do not turn this benign race into a lock bypass.
                pass
            self._fsync_directory(self._root)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            opened = os.fstat(descriptor)
            current = os.lstat(path)
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_uid,
                opened.st_gid,
            ) != (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_uid,
                current.st_gid,
            ):
                raise RuntimeError(
                    "owner_gate_foundation_journal_directory_changed"
                )
            self._validate_directory(path)
            self._validate_inventory(path)
            yield path, (opened.st_dev, opened.st_ino)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @contextmanager
    def transaction_lease(
        self,
        transaction_id: str,
    ) -> Iterator[None]:
        """Hold one OS-released exclusive lease for the whole state machine."""

        if self._active_lease is not None:
            raise RuntimeError("owner_gate_foundation_journal_lease_conflict")
        with self._locked_transaction(transaction_id, create=True) as locked:
            assert locked is not None
            root, identity = locked
            self._active_lease = (
                transaction_id,
                threading.get_ident(),
                root,
                identity,
            )
            try:
                yield
            finally:
                self._active_lease = None

    @staticmethod
    def _entry_name(entry: str) -> str | None:
        if entry.endswith(".json"):
            return entry[:-5]
        if entry.startswith(".") and entry.endswith(".pending"):
            return entry[1:-8]
        return None

    def _validate_inventory(self, root: Path) -> None:
        entries = os.listdir(root)
        names = [self._entry_name(entry) for entry in entries]
        if (
            any(name is None or not _artifact_name_valid(name) for name in names)
            or len(entries) > 2 * (9 * 8 + len(_FIXED_ARTIFACTS))
        ):
            raise RuntimeError(
                "owner_gate_foundation_journal_inventory_invalid"
            )

    def _validate_file(
        self,
        path: Path,
        *,
        allow_link_pair: bool = False,
    ) -> bytes:
        before = os.lstat(path)
        links = {1, 2} if allow_link_pair else {1}
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != self._owner_uid
            or before.st_gid != self._owner_gid
            or stat.S_IMODE(before.st_mode) != _FILE_MODE
            or before.st_nlink not in links
            or not 0 < before.st_size <= _MAX_ARTIFACT_BYTES
        ):
            raise PermissionError(
                "owner_gate_foundation_journal_artifact_invalid"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_uid,
                before.st_gid,
                before.st_size,
            )
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
                opened.st_uid,
                opened.st_gid,
                opened.st_size,
            ) != identity:
                raise RuntimeError(
                    "owner_gate_foundation_journal_artifact_changed"
                )
            raw = os.read(descriptor, _MAX_ARTIFACT_BYTES + 1)
            if os.read(descriptor, 1) or len(raw) != before.st_size:
                raise RuntimeError(
                    "owner_gate_foundation_journal_artifact_framing_invalid"
                )
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
                raise RuntimeError(
                    "owner_gate_foundation_journal_artifact_changed"
                )
        finally:
            os.close(descriptor)
        return raw

    @staticmethod
    def _decode(raw: bytes) -> dict[str, Any]:
        def reject_duplicates(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for name, item in pairs:
                if name in value:
                    raise ValueError("duplicate key")
                value[name] = item
            return value

        try:
            value = json.loads(
                raw.decode("ascii", errors="strict"),
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda _item: (_ for _ in ()).throw(
                    ValueError()
                ),
            )
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "owner_gate_foundation_journal_json_invalid"
            ) from None
        if not isinstance(value, dict) or _canonical_bytes(value) != raw:
            raise RuntimeError(
                "owner_gate_foundation_journal_canonical_invalid"
            )
        return value

    def _write_pending(self, path: Path, raw: bytes) -> None:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            _FILE_MODE,
        )
        try:
            os.fchown(descriptor, self._owner_uid, self._owner_gid)
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise OSError(
                        "owner_gate_foundation_journal_write_stalled"
                    )
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
        self._decode(pending_raw)
        if not os.path.lexists(final):
            os.link(pending, final, follow_symlinks=False)
            self._fsync_directory(root)
        final_raw = self._validate_file(final, allow_link_pair=True)
        if final_raw != pending_raw:
            raise RuntimeError(
                "owner_gate_foundation_journal_pending_diverged"
            )
        os.unlink(pending)
        self._fsync_directory(root)
        if self._validate_file(final) != final_raw:
            raise RuntimeError(
                "owner_gate_foundation_journal_readback_diverged"
            )

    def read(
        self,
        transaction_id: str,
        name: str,
    ) -> dict[str, Any] | None:
        if not _artifact_name_valid(name):
            raise RuntimeError(
                "owner_gate_foundation_journal_artifact_name_invalid"
            )
        with self._locked_transaction(transaction_id, create=False) as locked:
            if locked is None:
                return None
            root, _identity = locked
            self._recover_pending(root, name)
            path = root / f"{name}.json"
            if not os.path.lexists(path):
                return None
            return self._decode(self._validate_file(path))

    def read_strict(
        self,
        transaction_id: str,
        name: str,
    ) -> dict[str, Any] | None:
        """Read one final artifact without recovery or any filesystem write."""

        if not _artifact_name_valid(name):
            raise RuntimeError(
                "owner_gate_foundation_journal_artifact_name_invalid"
            )
        with self._locked_transaction(transaction_id, create=False) as locked:
            if locked is None:
                return None
            root, _identity = locked
            pending = root / f".{name}.pending"
            if os.path.lexists(pending):
                raise RuntimeError(
                    "owner_gate_foundation_journal_pending_requires_recovery"
                )
            path = root / f"{name}.json"
            if not os.path.lexists(path):
                return None
            return self._decode(self._validate_file(path))

    def list(self, transaction_id: str) -> dict[str, dict[str, Any]]:
        with self._locked_transaction(transaction_id, create=False) as locked:
            if locked is None:
                return {}
            root, _identity = locked
            names = sorted({
                name
                for entry in os.listdir(root)
                if (name := self._entry_name(entry)) is not None
            })
            result: dict[str, dict[str, Any]] = {}
            for name in names:
                self._recover_pending(root, name)
                path = root / f"{name}.json"
                if not os.path.lexists(path):
                    raise RuntimeError(
                        "owner_gate_foundation_journal_artifact_disappeared"
                    )
                result[name] = self._decode(self._validate_file(path))
            return result

    def publish(
        self,
        transaction_id: str,
        name: str,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not _artifact_name_valid(name):
            raise RuntimeError(
                "owner_gate_foundation_journal_artifact_name_invalid"
            )
        raw = _canonical_bytes(value)
        if not raw or len(raw) > _MAX_ARTIFACT_BYTES:
            raise RuntimeError(
                "owner_gate_foundation_journal_artifact_too_large"
            )
        with self._locked_transaction(transaction_id, create=True) as locked:
            assert locked is not None
            root, _identity = locked
            self._recover_pending(root, name)
            final = root / f"{name}.json"
            if os.path.lexists(final):
                existing = self._validate_file(final)
                if existing != raw:
                    raise RuntimeError(
                        "owner_gate_foundation_journal_artifact_diverged"
                    )
                return self._decode(existing)
            pending = root / f".{name}.pending"
            self._write_pending(pending, raw)
            self._fsync_directory(root)
            self._recover_pending(root, name)
            existing = self._validate_file(final)
            if existing != raw:
                raise RuntimeError(
                    "owner_gate_foundation_journal_publication_diverged"
                )
            return self._decode(existing)


__all__ = [
    "DEFAULT_JOURNAL_ROOT",
    "FoundationApplyJournal",
]
