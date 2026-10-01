"""Ares-owned, descriptor-safe Context Governor key lifecycle.

Normal runtime resolution never creates, rotates, or reads a key by pathname.
It holds a shared lifecycle lock and returns inherited descriptors to the Rust
cryptographic owner.  Initialization and rotation are explicit lifecycle
actions; neither is used by activation or certification tests with real keys.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import stat
import subprocess
from dataclasses import dataclass
from typing import Any


class ContextGovernorKeyError(RuntimeError):
    """Typed governed-key failure; the code is stable for callers/tests."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass
class GovernedKeyBinding:
    """Held descriptors valid for exactly one certified child operation."""

    key_id: str
    key_fd: int
    snapshot_fd: int
    retired_key_fds: tuple[tuple[str, int], ...]
    lock_fd: int
    snapshot_digest: str

    def command_args(self) -> list[str]:
        args = [
            "--governed-key-fd",
            str(self.key_fd),
            "--governed-snapshot-fd",
            str(self.snapshot_fd),
        ]
        for key_id, descriptor in self.retired_key_fds:
            args.extend(["--governed-retired-key-fd", f"{key_id}:{descriptor}"])
        return args

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (
            self.key_fd,
            self.snapshot_fd,
            self.lock_fd,
            *(fd for _, fd in self.retired_key_fds),
        )

    def close(self) -> None:
        for descriptor in {
            self.key_fd,
            self.snapshot_fd,
            self.lock_fd,
            *(fd for _, fd in self.retired_key_fds),
        }:
            try:
                os.close(descriptor)
            except OSError:
                pass


class ContextGovernorKeyState:
    """Descriptor-relative governed state under ``HERMES_HOME``."""

    snapshot_schema = "AresContextGovernorKeySnapshotV2"
    current_schema = "AresContextGovernorCurrentKeySnapshotV2"
    _dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    _file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC

    def __init__(self, hermes_home, binary: str) -> None:
        self.hermes_home = os.fspath(hermes_home)
        self.binary = binary

    @staticmethod
    def _digest(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    @staticmethod
    def _canonical(value: dict[str, Any]) -> bytes:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )

    def _open_dir(self, name: str, parent: int | None = None) -> int:
        try:
            descriptor = os.open(name, self._dir_flags, dir_fd=parent)
        except OSError as exc:
            raise ContextGovernorKeyError(
                "KeySymlinkRejected"
                if exc.errno in {getattr(os, "ELOOP", 40), 40}
                else "MissingGovernedKey",
                name,
            ) from exc
        self._validate_fd(descriptor, directory=True, expected_mode=0o700, name=name)
        return descriptor

    def _open_root(self) -> int:
        """Open each governed boundary; no path check is later trusted."""
        home = self._open_dir(self.hermes_home)
        try:
            context = self._open_dir("context-governor", home)
            try:
                return self._open_dir("keys", context)
            finally:
                os.close(context)
        finally:
            os.close(home)

    def _validate_fd(
        self,
        descriptor: int,
        *,
        directory: bool,
        expected_mode: int,
        name: str,
        secret: bool = False,
    ) -> os.stat_result:
        metadata = os.fstat(descriptor)
        valid_type = (
            stat.S_ISDIR(metadata.st_mode)
            if directory
            else stat.S_ISREG(metadata.st_mode)
        )
        if not valid_type:
            raise ContextGovernorKeyError("KeyPathEscape", name)
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise ContextGovernorKeyError("WrongKeyOwner", name)
        if stat.S_IMODE(metadata.st_mode) != expected_mode:
            raise ContextGovernorKeyError("InvalidKeyPermissions", name)
        if secret and metadata.st_nlink != 1:
            raise ContextGovernorKeyError("KeyHardLinkRejected", name)
        return metadata

    def _open_file(self, root: int, name: str, *, secret: bool = False) -> int:
        try:
            descriptor = os.open(name, self._file_flags, dir_fd=root)
        except FileNotFoundError as exc:
            raise ContextGovernorKeyError("MissingGovernedKey", name) from exc
        except OSError as exc:
            raise ContextGovernorKeyError(
                "KeySymlinkRejected"
                if exc.errno in {getattr(os, "ELOOP", 40), 40}
                else "KeyUnreadable",
                name,
            ) from exc
        self._validate_fd(
            descriptor, directory=False, expected_mode=0o600, name=name, secret=secret
        )
        return descriptor

    def _read_fd(self, descriptor: int, name: str, *, maximum: int = 131072) -> bytes:
        before = os.fstat(descriptor)
        if before.st_size > maximum:
            raise ContextGovernorKeyError("InvalidKeyEncoding", name)
        os.lseek(descriptor, 0, os.SEEK_SET)
        value = bytearray()
        while len(value) <= maximum:
            chunk = os.read(descriptor, min(65536, maximum + 1 - len(value)))
            if not chunk:
                break
            value.extend(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise ContextGovernorKeyError("KeyChangedDuringRead", name)
        if len(value) > maximum:
            raise ContextGovernorKeyError("InvalidKeyEncoding", name)
        return bytes(value)

    def _read_json(self, descriptor: int, name: str) -> tuple[dict[str, Any], bytes]:
        raw = self._read_fd(descriptor, name)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContextGovernorKeyError("ActiveKeyMetadataMismatch", name) from exc
        if not isinstance(value, dict):
            raise ContextGovernorKeyError("ActiveKeyMetadataMismatch", name)
        return value, raw

    def _derive_key_id(self, key_fd: int) -> str:
        """Ask the Rust crypto owner to derive an ID from the held descriptor."""
        try:
            completed = subprocess.run(
                [self.binary, "key-id-fd", "--governed-key-fd", str(key_fd)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(key_fd,),
                check=False,
            )
        except OSError as exc:
            raise ContextGovernorKeyError("SecureKeyTransportUnavailable") from exc
        if completed.returncode:
            raise ContextGovernorKeyError(
                "InvalidKeyEncoding", completed.stderr.strip()
            )
        try:
            key_id = json.loads(completed.stdout)["key_id"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ContextGovernorKeyError("InvalidKeyEncoding") from exc
        if (
            not isinstance(key_id, str)
            or len(key_id) != 64
            or set(key_id) - set("0123456789abcdef")
        ):
            raise ContextGovernorKeyError("InvalidKeyEncoding")
        return key_id

    def _lock(self, root: int, exclusive: bool) -> int:
        descriptor = self._open_file(root, "lifecycle.lock")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            return descriptor
        except OSError as exc:
            os.close(descriptor)
            raise ContextGovernorKeyError(
                "SecureKeyTransportUnavailable", "lifecycle lock"
            ) from exc

    def active_binding(self) -> GovernedKeyBinding:
        """Resolve and hold one fully-authenticated governed snapshot."""
        root = self._open_root()
        try:
            lock = self._lock(root, exclusive=False)
            try:
                current_fd = self._open_file(root, "current.json")
                try:
                    current, current_raw = self._read_json(current_fd, "current.json")
                finally:
                    os.close(current_fd)
                if (
                    current.get("schema") != self.current_schema
                    or not isinstance(current.get("snapshot"), str)
                    or not isinstance(current.get("snapshot_sha256"), str)
                ):
                    raise ContextGovernorKeyError(
                        "ActiveKeyMetadataMismatch", "current.json"
                    )
                snapshots = self._open_dir("snapshots", root)
                try:
                    snapshot_fd = self._open_file(snapshots, current["snapshot"])
                finally:
                    os.close(snapshots)
                snapshot, snapshot_raw = self._read_json(
                    snapshot_fd, current["snapshot"]
                )
                if (
                    self._digest(snapshot_raw) != current["snapshot_sha256"]
                    or snapshot.get("schema") != self.snapshot_schema
                ):
                    raise ContextGovernorKeyError(
                        "ActiveKeyMetadataMismatch", "snapshot"
                    )
                active_id = snapshot.get("active_key_id")
                retired_ids = snapshot.get("retired_key_ids")
                compromised = snapshot.get("compromised_key_ids", [])
                if (
                    not isinstance(active_id, str)
                    or not isinstance(retired_ids, list)
                    or active_id in retired_ids
                    or len(set(retired_ids)) != len(retired_ids)
                ):
                    raise ContextGovernorKeyError(
                        "ActiveKeyMetadataMismatch", "snapshot"
                    )
                if active_id in compromised:
                    raise ContextGovernorKeyError("CompromisedKey", active_id)
                by_id = self._open_dir("by-id", root)
                try:
                    key_fd = self._open_file(by_id, f"{active_id}.key", secret=True)
                    actual_id = self._derive_key_id(key_fd)
                    if actual_id != active_id:
                        raise ContextGovernorKeyError(
                            "ActiveKeyMetadataMismatch", active_id
                        )
                    retired_fds: list[tuple[str, int]] = []
                    for retired_id in retired_ids:
                        if not isinstance(retired_id, str):
                            raise ContextGovernorKeyError(
                                "ActiveKeyMetadataMismatch", "retired key id"
                            )
                        descriptor = self._open_file(
                            by_id, f"{retired_id}.key", secret=True
                        )
                        if self._derive_key_id(descriptor) != retired_id:
                            os.close(descriptor)
                            raise ContextGovernorKeyError(
                                "ActiveKeyMetadataMismatch", retired_id
                            )
                        retired_fds.append((retired_id, descriptor))
                finally:
                    os.close(by_id)
                return GovernedKeyBinding(
                    active_id,
                    key_fd,
                    snapshot_fd,
                    tuple(retired_fds),
                    lock,
                    self._digest(snapshot_raw),
                )
            except Exception:
                os.close(lock)
                raise
        finally:
            os.close(root)

    def _ensure_dir(self, parent: int, name: str) -> int:
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
        except FileExistsError:
            pass
        descriptor = self._open_dir(name, parent)
        return descriptor

    def _fsync_dir(self, descriptor: int) -> None:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise ContextGovernorKeyError(
                "KeyRotationCommitFailed", "directory fsync"
            ) from exc

    def _write_published(self, directory: int, name: str, payload: bytes) -> None:
        temporary = f".{name}.{secrets.token_hex(16)}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        try:
            written = 0
            while written < len(payload):
                written += os.write(descriptor, payload[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.rename(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
            self._fsync_dir(directory)
        except Exception:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
            raise

    def _bootstrap_root(self) -> int:
        home = self._open_dir(self.hermes_home)
        try:
            context = self._ensure_dir(home, "context-governor")
            try:
                root = self._ensure_dir(context, "keys")
            finally:
                os.close(context)
        finally:
            os.close(home)
        return root

    def initialize_first_install(self) -> GovernedKeyBinding:
        """Create the first complete snapshot with durable publish ordering."""
        root = self._bootstrap_root()
        try:
            try:
                lock_create = os.open(
                    "lifecycle.lock",
                    os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=root,
                )
                os.close(lock_create)
            except OSError as exc:
                raise ContextGovernorKeyError(
                    "KeyPathEscape", "lifecycle.lock"
                ) from exc
            lock = self._lock(root, exclusive=True)
            try:
                try:
                    probe = self._open_file(root, "current.json")
                except ContextGovernorKeyError as exc:
                    if exc.code != "MissingGovernedKey":
                        raise
                else:
                    os.close(probe)
                    raise ContextGovernorKeyError(
                        "ActiveKeyMetadataMismatch", "state already initialized"
                    )
                by_id = self._ensure_dir(root, "by-id")
                snapshots = self._ensure_dir(root, "snapshots")
                try:
                    temporary = f".new-key.{secrets.token_hex(16)}.tmp"
                    key_writer = os.open(
                        temporary,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                        0o600,
                        dir_fd=by_id,
                    )
                    try:
                        os.write(key_writer, secrets.token_bytes(32))
                        os.fsync(key_writer)
                    finally:
                        os.close(key_writer)
                    key_reader = self._open_file(by_id, temporary, secret=True)
                    try:
                        key_id = self._derive_key_id(key_reader)
                    finally:
                        os.close(key_reader)
                    os.rename(
                        temporary, f"{key_id}.key", src_dir_fd=by_id, dst_dir_fd=by_id
                    )
                    self._fsync_dir(by_id)
                    snapshot = {
                        "schema": self.snapshot_schema,
                        "sequence": 1,
                        "active_key_id": key_id,
                        "retired_key_ids": [],
                        "compromised_key_ids": [],
                        "keys": {key_id: "active"},
                    }
                    snapshot_bytes = self._canonical(snapshot)
                    snapshot_name = f"1-{self._digest(snapshot_bytes)}.json"
                    self._write_published(snapshots, snapshot_name, snapshot_bytes)
                    current = {
                        "schema": self.current_schema,
                        "snapshot": snapshot_name,
                        "snapshot_sha256": self._digest(snapshot_bytes),
                    }
                    self._write_published(
                        root, "current.json", self._canonical(current)
                    )
                finally:
                    os.close(snapshots)
                    os.close(by_id)
            finally:
                os.close(lock)
        finally:
            os.close(root)
        return self.active_binding()

    def rotate(self) -> GovernedKeyBinding:
        """Publish a new immutable snapshot; old key files are never moved."""
        root = self._open_root()
        try:
            lock = self._lock(root, exclusive=True)
            try:
                current_fd = self._open_file(root, "current.json")
                try:
                    current, _ = self._read_json(current_fd, "current.json")
                finally:
                    os.close(current_fd)
                if current.get("schema") != self.current_schema or not isinstance(
                    current.get("snapshot"), str
                ):
                    raise ContextGovernorKeyError(
                        "ActiveKeyMetadataMismatch", "current.json"
                    )
                snapshots = self._open_dir("snapshots", root)
                try:
                    snapshot_fd = self._open_file(snapshots, current["snapshot"])
                    try:
                        snapshot, raw = self._read_json(
                            snapshot_fd, current["snapshot"]
                        )
                    finally:
                        os.close(snapshot_fd)
                    if snapshot.get("schema") != self.snapshot_schema or self._digest(
                        raw
                    ) != current.get("snapshot_sha256"):
                        raise ContextGovernorKeyError(
                            "ActiveKeyMetadataMismatch", "current snapshot"
                        )
                    old_id = snapshot.get("active_key_id")
                    sequence = snapshot.get("sequence")
                    retired = snapshot.get("retired_key_ids")
                    if (
                        not isinstance(old_id, str)
                        or not isinstance(sequence, int)
                        or not isinstance(retired, list)
                        or old_id in retired
                    ):
                        raise ContextGovernorKeyError(
                            "ActiveKeyMetadataMismatch", "current snapshot"
                        )
                    by_id = self._open_dir("by-id", root)
                    try:
                        old_fd = self._open_file(by_id, f"{old_id}.key", secret=True)
                        try:
                            if self._derive_key_id(old_fd) != old_id:
                                raise ContextGovernorKeyError(
                                    "ActiveKeyMetadataMismatch", old_id
                                )
                        finally:
                            os.close(old_fd)
                        temporary = f".new-key.{secrets.token_hex(16)}.tmp"
                        new_writer = os.open(
                            temporary,
                            os.O_WRONLY
                            | os.O_CREAT
                            | os.O_EXCL
                            | os.O_NOFOLLOW
                            | os.O_CLOEXEC,
                            0o600,
                            dir_fd=by_id,
                        )
                        try:
                            os.write(new_writer, secrets.token_bytes(32))
                            os.fsync(new_writer)
                        finally:
                            os.close(new_writer)
                        new_reader = self._open_file(by_id, temporary, secret=True)
                        try:
                            new_id = self._derive_key_id(new_reader)
                        finally:
                            os.close(new_reader)
                        os.rename(
                            temporary,
                            f"{new_id}.key",
                            src_dir_fd=by_id,
                            dst_dir_fd=by_id,
                        )
                        self._fsync_dir(by_id)
                    finally:
                        os.close(by_id)
                    new_retired = sorted(set(retired) | {old_id})
                    keys = {key_id: "retired" for key_id in new_retired}
                    keys[new_id] = "active"
                    new_snapshot = {
                        "schema": self.snapshot_schema,
                        "sequence": sequence + 1,
                        "active_key_id": new_id,
                        "retired_key_ids": new_retired,
                        "compromised_key_ids": snapshot.get("compromised_key_ids", []),
                        "keys": keys,
                    }
                    snapshot_bytes = self._canonical(new_snapshot)
                    snapshot_name = (
                        f"{sequence + 1}-{self._digest(snapshot_bytes)}.json"
                    )
                    self._write_published(snapshots, snapshot_name, snapshot_bytes)
                    # This atomic replacement is the transaction commit point.
                    self._write_published(
                        root,
                        "current.json",
                        self._canonical({
                            "schema": self.current_schema,
                            "snapshot": snapshot_name,
                            "snapshot_sha256": self._digest(snapshot_bytes),
                        }),
                    )
                finally:
                    os.close(snapshots)
            finally:
                os.close(lock)
        finally:
            os.close(root)
        return self.active_binding()
