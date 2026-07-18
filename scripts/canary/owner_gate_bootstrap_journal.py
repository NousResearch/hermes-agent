#!/usr/bin/env python3
"""Crash-safe append-only journal for the owner-gate bootstrap.

The bootstrap mutates a fresh VM.  A mutable progress document cannot prove
whether a process died immediately before or immediately after a side effect,
so every transition is an immutable canonical artifact.  Publication uses a
same-filesystem, fully-fsynced scratch file and a no-replace hard link.  One
OS ``flock`` lease covers the complete install or rollback state machine.
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


DIRECTORY_MODE = 0o700
ARTIFACT_MODE = 0o600
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
PHASE_COUNT = 7

_PHASE_ARTIFACT = re.compile(r"^p([0-6])-(intent|success)$")
_FIXED_ARTIFACTS = frozenset(
    {
        "manifest",
        "terminal-success",
        "rollback-intent",
        "rollback-success",
        "rollback-terminal",
    }
)
_PENDING_ARTIFACT = re.compile(
    r"^\.(manifest|terminal-success|rollback-intent|rollback-success|"
    r"rollback-terminal|p[0-6]-(?:intent|success))\."
    r"[0-9]+\.[0-9a-f]{32}\.pending$"
)


class BootstrapJournalError(RuntimeError):
    """Stable, secret-free journal failure."""


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise BootstrapJournalError(
            "owner_gate_bootstrap_journal_json_invalid"
        ) from None


def _artifact_name_valid(name: str) -> bool:
    return name in _FIXED_ARTIFACTS or _PHASE_ARTIFACT.fullmatch(name) is not None


class BootstrapInstallJournal:
    """Root-owned immutable transition store for one release transaction."""

    def __init__(
        self,
        transaction_path: Path,
        *,
        owner_uid: int = 0,
        owner_gid: int = 0,
    ) -> None:
        if (
            not transaction_path.is_absolute()
            or transaction_path.name in {"", ".", ".."}
            or ".." in transaction_path.parts
            or transaction_path.suffix != ".json"
        ):
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_path_invalid"
            )
        self._transaction_path = transaction_path
        self._root = transaction_path.with_suffix(".journal")
        self._owner_uid = owner_uid
        self._owner_gid = owner_gid
        self._active: tuple[int, tuple[int, int]] | None = None

    @property
    def root(self) -> Path:
        return self._root

    def _require_owner(self) -> None:
        if (
            int(os.geteuid()) != self._owner_uid  # windows-footgun: ok — Debian root boundary
            or int(os.getegid()) != self._owner_gid  # windows-footgun: ok — Debian root boundary
        ):
            raise PermissionError(
                "owner_gate_bootstrap_journal_owner_invalid"
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

    def _validate_directory(self, path: Path) -> os.stat_result:
        state = os.lstat(path)
        if (
            not stat.S_ISDIR(state.st_mode)
            or stat.S_ISLNK(state.st_mode)
            or state.st_uid != self._owner_uid
            or state.st_gid != self._owner_gid
            or stat.S_IMODE(state.st_mode) != DIRECTORY_MODE
        ):
            raise PermissionError(
                "owner_gate_bootstrap_journal_directory_invalid"
            )
        return state

    def _ensure_root(self) -> None:
        self._require_owner()
        parent = self._root.parent
        if not os.path.lexists(parent):
            os.mkdir(parent, DIRECTORY_MODE)
            os.chown(parent, self._owner_uid, self._owner_gid)
            self._fsync_directory(parent.parent)
        self._validate_directory(parent)
        try:
            os.mkdir(self._root, DIRECTORY_MODE)
            os.chown(self._root, self._owner_uid, self._owner_gid)
            self._fsync_directory(parent)
        except FileExistsError:
            pass
        self._validate_directory(self._root)

    @staticmethod
    def _pending_target(entry: str) -> str | None:
        match = _PENDING_ARTIFACT.fullmatch(entry)
        return None if match is None else match.group(1)

    def _validate_inventory(self) -> None:
        entries = os.listdir(self._root)
        if len(entries) > 2 * (PHASE_COUNT * 2 + len(_FIXED_ARTIFACTS)):
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_inventory_invalid"
            )
        for entry in entries:
            if entry.endswith(".json") and _artifact_name_valid(entry[:-5]):
                continue
            if self._pending_target(entry) is not None:
                continue
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_inventory_invalid"
            )

    def _validate_file(
        self,
        path: Path,
        *,
        link_counts: frozenset[int] = frozenset({1}),
    ) -> tuple[bytes, os.stat_result]:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != self._owner_uid
            or before.st_gid != self._owner_gid
            or stat.S_IMODE(before.st_mode) != ARTIFACT_MODE
            or before.st_nlink not in link_counts
            or not 0 < before.st_size <= MAX_ARTIFACT_BYTES
        ):
            raise PermissionError(
                "owner_gate_bootstrap_journal_artifact_invalid"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
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
                raise BootstrapJournalError(
                    "owner_gate_bootstrap_journal_artifact_changed"
                )
            chunks: list[bytes] = []
            remaining = opened.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    raise BootstrapJournalError(
                        "owner_gate_bootstrap_journal_artifact_framing_invalid"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) != opened.st_size or os.read(descriptor, 1):
                raise BootstrapJournalError(
                    "owner_gate_bootstrap_journal_artifact_framing_invalid"
                )
            after = os.fstat(descriptor)
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ):
                raise BootstrapJournalError(
                    "owner_gate_bootstrap_journal_artifact_changed"
                )
        finally:
            os.close(descriptor)
        self._decode(raw)
        return raw, before

    @staticmethod
    def _decode(raw: bytes) -> dict[str, Any]:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError("duplicate key")
                result[name] = value
            return result

        try:
            value = json.loads(
                raw.decode("ascii", errors="strict"),
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ValueError("non-finite")
                ),
            )
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_json_invalid"
            ) from None
        if not isinstance(value, dict) or canonical_bytes(value) != raw:
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_canonical_invalid"
            )
        return value

    def _recover_pending(self, name: str) -> None:
        pending = [
            self._root / entry
            for entry in os.listdir(self._root)
            if self._pending_target(entry) == name
        ]
        if len(pending) > 1:
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_pending_ambiguous"
            )
        if not pending:
            return
        scratch = pending[0]
        final = self._root / f"{name}.json"
        scratch_lstat = os.lstat(scratch)
        if not os.path.lexists(final) and scratch_lstat.st_nlink != 1:
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_pending_diverged"
            )
        if not os.path.lexists(final) and scratch_lstat.st_nlink == 1:
            try:
                scratch_raw, scratch_state = self._validate_file(scratch)
            except (BootstrapJournalError, PermissionError):
                # A process can die between O_EXCL and the final file fsync.
                # Under the recovered directory flock an unlinked unique
                # scratch is not truth.  Discard only the exact bounded node
                # that this transaction could have created, then let publish
                # recreate it.  Anything else remains a fail-closed conflict.
                current = os.lstat(scratch)
                if (
                    not stat.S_ISREG(current.st_mode)
                    or stat.S_ISLNK(current.st_mode)
                    or current.st_uid != self._owner_uid
                    or current.st_gid != self._owner_gid
                    or stat.S_IMODE(current.st_mode) != ARTIFACT_MODE
                    or current.st_nlink != 1
                    or current.st_size > MAX_ARTIFACT_BYTES
                ):
                    raise
                os.unlink(scratch)
                self._fsync_directory(self._root)
                return
        else:
            scratch_raw, scratch_state = self._validate_file(
                scratch,
                link_counts=frozenset({1, 2}),
            )
        if not os.path.lexists(final):
            os.link(scratch, final, follow_symlinks=False)
            self._fsync_directory(self._root)
        final_raw, final_state = self._validate_file(
            final,
            link_counts=frozenset({1, 2}),
        )
        if (
            final_raw != scratch_raw
            or (final_state.st_dev, final_state.st_ino)
            != (scratch_state.st_dev, scratch_state.st_ino)
        ):
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_pending_diverged"
            )
        os.unlink(scratch)
        self._fsync_directory(self._root)
        readback, state = self._validate_file(final)
        if readback != final_raw or state.st_nlink != 1:
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_readback_diverged"
            )

    def _recover_all_pending(self) -> None:
        names = {
            target
            for entry in os.listdir(self._root)
            if (target := self._pending_target(entry)) is not None
        }
        for name in sorted(names):
            self._recover_pending(name)

    @contextmanager
    def transaction_lease(self, *, create: bool) -> Iterator[None]:
        """Hold the one process-released lock for the whole state machine."""

        self._require_owner()
        if self._active is not None:
            if self._active[0] != threading.get_ident():
                raise BootstrapJournalError(
                    "owner_gate_bootstrap_journal_lease_conflict"
                )
            yield
            return
        if not os.path.lexists(self._root):
            if not create:
                raise BootstrapJournalError(
                    "owner_gate_bootstrap_transaction_missing"
                )
            self._ensure_root()
        else:
            self._validate_directory(self._root.parent)
            self._validate_directory(self._root)
        descriptor = os.open(
            self._root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BootstrapJournalError(
                    "owner_gate_bootstrap_transaction_locked"
                ) from None
            opened = os.fstat(descriptor)
            current = self._validate_directory(self._root)
            identity = (opened.st_dev, opened.st_ino)
            if identity != (current.st_dev, current.st_ino):
                raise BootstrapJournalError(
                    "owner_gate_bootstrap_journal_directory_changed"
                )
            self._active = (threading.get_ident(), identity)
            self._validate_inventory()
            self._recover_all_pending()
            self._validate_inventory()
            yield
        finally:
            self._active = None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _require_lease(self) -> None:
        if self._active is None or self._active[0] != threading.get_ident():
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_lease_required"
            )
        current = self._validate_directory(self._root)
        if (current.st_dev, current.st_ino) != self._active[1]:
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_directory_changed"
            )

    def read(self, name: str) -> dict[str, Any] | None:
        if not _artifact_name_valid(name):
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_artifact_name_invalid"
            )
        self._require_lease()
        self._recover_pending(name)
        path = self._root / f"{name}.json"
        if not os.path.lexists(path):
            return None
        raw, _state = self._validate_file(path)
        return self._decode(raw)

    def publish(
        self,
        name: str,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not _artifact_name_valid(name):
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_artifact_name_invalid"
            )
        self._require_lease()
        raw = canonical_bytes(value)
        if not raw or len(raw) > MAX_ARTIFACT_BYTES:
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_artifact_too_large"
            )
        self._recover_pending(name)
        final = self._root / f"{name}.json"
        if os.path.lexists(final):
            existing, _state = self._validate_file(final)
            if existing != raw:
                raise BootstrapJournalError(
                    "owner_gate_bootstrap_journal_artifact_diverged"
                )
            return self._decode(existing)
        nonce = os.urandom(16).hex()
        scratch = self._root / f".{name}.{os.getpid()}.{nonce}.pending"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                scratch,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                ARTIFACT_MODE,
            )
            os.fchown(descriptor, self._owner_uid, self._owner_gid)
            os.fchmod(descriptor, ARTIFACT_MODE)
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise OSError("journal write stalled")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            self._fsync_directory(self._root)
            self._recover_pending(name)
        finally:
            if descriptor is not None:
                os.close(descriptor)
        existing, _state = self._validate_file(final)
        if existing != raw:
            raise BootstrapJournalError(
                "owner_gate_bootstrap_journal_publication_diverged"
            )
        return self._decode(existing)

    def artifacts(self) -> dict[str, dict[str, Any]]:
        self._require_lease()
        self._recover_all_pending()
        result: dict[str, dict[str, Any]] = {}
        for entry in sorted(os.listdir(self._root)):
            if not entry.endswith(".json"):
                continue
            name = entry[:-5]
            if not _artifact_name_valid(name):
                raise BootstrapJournalError(
                    "owner_gate_bootstrap_journal_inventory_invalid"
                )
            value = self.read(name)
            assert value is not None
            result[name] = value
        return result


__all__ = [
    "BootstrapInstallJournal",
    "BootstrapJournalError",
    "canonical_bytes",
]
