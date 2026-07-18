#!/usr/bin/env python3
"""Owner-only append-only journal for foundation author-and-apply."""

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

from scripts.canary import owner_gate_foundation_journal as apply_journal
from scripts.canary import owner_gate_trust_author as trust_author


DEFAULT_ROOT = trust_author.AUTHORITY_PARENT / "owner-gate-foundation-authoring"
OWNER_UID = apply_journal.OWNER_UID
OWNER_GID = apply_journal.OWNER_GID
ARTIFACTS = frozenset({
    "intent",
    "owner-reauth",
    "interpreter-evidence",
    "network-evidence",
    "ancestry-evidence",
    "authority",
    "apply-receipt",
    "apply-failure-receipt",
    "terminal",
})
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_TRANSACTION = re.compile(r"^[0-9a-f]{64}$")
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600
_MAX_BYTES = 4 * 1024 * 1024


class OwnerGateAuthorJournalError(RuntimeError):
    """Stable owner journal failure."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise OwnerGateAuthorJournalError(
            "owner_gate_author_journal_json_invalid"
        ) from None


class OwnerGateAuthorJournal:
    """One release lease plus immutable no-replace transaction artifacts."""

    def __init__(
        self,
        *,
        _root: Path = DEFAULT_ROOT,
        _owner_uid: int = OWNER_UID,
        _owner_gid: int = OWNER_GID,
        _artifacts: frozenset[str] = ARTIFACTS,
        _maximum_bytes: int = _MAX_BYTES,
    ) -> None:
        if (
            not isinstance(_artifacts, frozenset)
            or not _artifacts
            or any(
                not isinstance(name, str)
                or re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name) is None
                for name in _artifacts
            )
            or type(_maximum_bytes) is not int
            or not 1 <= _maximum_bytes <= 256 * 1024 * 1024
        ):
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_configuration_invalid"
            )
        self._root = _root
        self._owner_uid = _owner_uid
        self._owner_gid = _owner_gid
        self._artifacts = _artifacts
        self._maximum_bytes = _maximum_bytes
        self._lease: tuple[str, int, Path, tuple[int, int]] | None = None

    @property
    def root(self) -> Path:
        return self._root

    def _require_owner(self) -> None:
        if (
            os.geteuid() != self._owner_uid  # windows-footgun: ok — POSIX owner boundary
            or os.getegid() != self._owner_gid  # windows-footgun: ok — POSIX owner boundary
        ):
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_owner_invalid"
            )

    def _validate_directory(self, path: Path) -> tuple[int, int]:
        try:
            item = os.lstat(path)
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_directory_invalid"
            ) from None
        if (
            not stat.S_ISDIR(item.st_mode)
            or stat.S_ISLNK(item.st_mode)
            or item.st_uid != self._owner_uid
            or item.st_gid != self._owner_gid
            or stat.S_IMODE(item.st_mode) != _DIRECTORY_MODE
        ):
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_directory_invalid"
            )
        return item.st_dev, item.st_ino

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor: int | None = None
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            os.fsync(descriptor)
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_directory_fsync_failed"
            ) from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _ensure_directory(self, path: Path, *, parent: Path) -> None:
        self._validate_directory(parent)
        try:
            os.mkdir(path, _DIRECTORY_MODE)
        except FileExistsError:
            pass
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_directory_create_failed"
            ) from None
        self._validate_directory(path)
        self._fsync_directory(parent)

    def _ensure_release(self, release_revision: str) -> Path:
        self._require_owner()
        if _REVISION.fullmatch(release_revision or "") is None:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_release_invalid"
            )
        self._validate_directory(self._root.parent)
        self._ensure_directory(self._root, parent=self._root.parent)
        release = self._root / release_revision
        self._ensure_directory(release, parent=self._root)
        return release

    def _validate_release_inventory(self, release: Path) -> None:
        try:
            names = os.listdir(release)
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_inventory_invalid"
            ) from None
        if len(names) > 128 or any(_TRANSACTION.fullmatch(name) is None for name in names):
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_inventory_invalid"
            )
        for name in names:
            self._validate_directory(release / name)

    @contextmanager
    def release_lease(self, release_revision: str) -> Iterator[None]:
        if self._lease is not None:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_lease_conflict"
            )
        release = self._ensure_release(release_revision)
        descriptor = os.open(
            release,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            opened = os.fstat(descriptor)
            identity = self._validate_directory(release)
            if identity != (opened.st_dev, opened.st_ino):
                raise OwnerGateAuthorJournalError(
                    "owner_gate_author_journal_directory_changed"
                )
            self._validate_release_inventory(release)
            self._lease = (
                release_revision,
                threading.get_ident(),
                release,
                identity,
            )
            try:
                yield
            finally:
                self._lease = None
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _release(self, release_revision: str) -> Path:
        active = self._lease
        if (
            active is None
            or active[0] != release_revision
            or active[1] != threading.get_ident()
        ):
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_lease_required"
            )
        if self._validate_directory(active[2]) != active[3]:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_directory_changed"
            )
        return active[2]

    def _transaction(
        self,
        release_revision: str,
        transaction_id: str,
        *,
        create: bool,
    ) -> Path | None:
        if _TRANSACTION.fullmatch(transaction_id or "") is None:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_transaction_invalid"
            )
        release = self._release(release_revision)
        path = release / transaction_id
        if not os.path.lexists(path):
            if not create:
                return None
            self._ensure_directory(path, parent=release)
        self._validate_directory(path)
        self._validate_transaction_inventory(path)
        return path

    @staticmethod
    def _entry_name(name: str) -> str | None:
        if name.endswith(".json"):
            return name[:-5]
        if name.startswith(".") and name.endswith(".pending"):
            return name[1:-8]
        return None

    def _validate_transaction_inventory(self, path: Path) -> None:
        try:
            entries = os.listdir(path)
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_inventory_invalid"
            ) from None
        names = [self._entry_name(entry) for entry in entries]
        if (
            len(entries) > 2 * len(self._artifacts)
            or any(name not in self._artifacts for name in names)
        ):
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_inventory_invalid"
            )

    def _read_file(self, path: Path, *, linked: bool = False) -> bytes:
        try:
            before = os.lstat(path)
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_artifact_invalid"
            ) from None
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != self._owner_uid
            or before.st_gid != self._owner_gid
            or stat.S_IMODE(before.st_mode) != _FILE_MODE
            or before.st_nlink not in ({1, 2} if linked else {1})
            or not 0 < before.st_size <= self._maximum_bytes
        ):
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_artifact_invalid"
            )
        descriptor: int | None = None
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
            ):
                raise OwnerGateAuthorJournalError(
                    "owner_gate_author_journal_artifact_changed"
                )
            raw = os.read(descriptor, self._maximum_bytes + 1)
            if len(raw) != before.st_size or os.read(descriptor, 1):
                raise OwnerGateAuthorJournalError(
                    "owner_gate_author_journal_artifact_changed"
                )
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_artifact_invalid"
            ) from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
        self._decode(raw)
        return raw

    @staticmethod
    def _decode(raw: bytes) -> dict[str, Any]:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate")
                value[key] = item
            return value

        try:
            value = json.loads(
                raw.decode("ascii", errors="strict"),
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda _item: (_ for _ in ()).throw(ValueError()),
            )
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_json_invalid"
            ) from None
        if not isinstance(value, dict) or canonical_bytes(value) != raw:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_canonical_invalid"
            )
        return value

    def _recover(self, path: Path, name: str) -> None:
        pending = path / f".{name}.pending"
        final = path / f"{name}.json"
        if not os.path.lexists(pending):
            return
        try:
            pending_raw = self._read_file(pending, linked=True)
        except OwnerGateAuthorJournalError:
            # A process can die after O_EXCL created the private pending inode
            # but before the canonical payload and fsync completed.  Only that
            # exact unlinked, owner-only inode is disposable.  A committed
            # hardlink, a final path, or suspicious metadata always fails
            # closed instead of being rewritten.
            if self._discard_uncommitted_pending(
                path=path,
                pending=pending,
                final=final,
            ):
                return
            raise
        try:
            if not os.path.lexists(final):
                os.link(pending, final, follow_symlinks=False)
                self._fsync_directory(path)
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_recovery_failed"
            ) from None
        if self._read_file(final, linked=True) != pending_raw:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_pending_diverged"
            )
        try:
            os.unlink(pending)
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_recovery_failed"
            ) from None
        self._fsync_directory(path)
        if self._read_file(final) != pending_raw:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_readback_diverged"
            )

    def _discard_uncommitted_pending(
        self,
        *,
        path: Path,
        pending: Path,
        final: Path,
    ) -> bool:
        if os.path.lexists(final):
            return False
        try:
            before = os.lstat(pending)
        except OSError:
            return False
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != self._owner_uid
            or before.st_gid != self._owner_gid
            or stat.S_IMODE(before.st_mode) != _FILE_MODE
            or before.st_nlink != 1
            or not 0 <= before.st_size <= self._maximum_bytes
        ):
            return False
        descriptor: int | None = None
        valid = False
        close_failed = False
        try:
            descriptor = os.open(
                pending,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            after = os.lstat(pending)
            identity = (before.st_dev, before.st_ino, before.st_size)
            valid = not (
                identity
                != (opened.st_dev, opened.st_ino, opened.st_size)
                or identity != (after.st_dev, after.st_ino, after.st_size)
                or opened.st_nlink != 1
                or after.st_nlink != 1
                or os.path.lexists(final)
            )
        except OSError:
            return False
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    close_failed = True
        if close_failed or not valid:
            return False
        try:
            os.unlink(pending)
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_recovery_failed"
            ) from None
        self._fsync_directory(path)
        return True

    def list_transactions(self, release_revision: str) -> dict[str, dict[str, dict[str, Any]]]:
        release = self._release(release_revision)
        self._validate_release_inventory(release)
        try:
            names = sorted(os.listdir(release))
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_inventory_invalid"
            ) from None
        return {
            name: self.list_artifacts(release_revision, name) for name in names
        }

    def list_artifacts(
        self,
        release_revision: str,
        transaction_id: str,
    ) -> dict[str, dict[str, Any]]:
        path = self._transaction(release_revision, transaction_id, create=False)
        if path is None:
            return {}
        try:
            entries = os.listdir(path)
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_inventory_invalid"
            ) from None
        names = sorted({
            item for entry in entries if (item := self._entry_name(entry)) is not None
        })
        result: dict[str, dict[str, Any]] = {}
        for name in names:
            self._recover(path, name)
            result[name] = self._decode(self._read_file(path / f"{name}.json"))
        return result

    def publish(
        self,
        release_revision: str,
        transaction_id: str,
        name: str,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        if name not in self._artifacts:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_artifact_name_invalid"
            )
        raw = canonical_bytes(value)
        if not raw or len(raw) > self._maximum_bytes:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_artifact_too_large"
            )
        path = self._transaction(release_revision, transaction_id, create=True)
        assert path is not None
        self._recover(path, name)
        final = path / f"{name}.json"
        if os.path.lexists(final):
            existing = self._read_file(final)
            if existing != raw:
                raise OwnerGateAuthorJournalError(
                    "owner_gate_author_journal_artifact_diverged"
                )
            return self._decode(existing)
        pending = path / f".{name}.pending"
        try:
            descriptor = os.open(
                pending,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                _FILE_MODE,
            )
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_write_failed"
            ) from None
        try:
            os.fchown(descriptor, self._owner_uid, self._owner_gid)
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise OSError("stalled")
                offset += written
            os.fchmod(descriptor, _FILE_MODE)
            os.fsync(descriptor)
        except OSError:
            raise OwnerGateAuthorJournalError(
                "owner_gate_author_journal_write_failed"
            ) from None
        finally:
            os.close(descriptor)
        self._fsync_directory(path)
        self._recover(path, name)
        return self._decode(self._read_file(final))


__all__ = [
    "DEFAULT_ROOT",
    "OwnerGateAuthorJournal",
    "OwnerGateAuthorJournalError",
    "canonical_bytes",
]
