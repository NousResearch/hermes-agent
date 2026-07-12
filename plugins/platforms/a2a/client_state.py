"""Hardened profile-scoped continuity state for named A2A peers."""

from __future__ import annotations

import json
import math
import os
import secrets
import stat
import threading
import time
import copy
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from . import auth, config

_LOCK = threading.RLock()
_MAX_FILE_BYTES = 256 * 1024
_MAX_PEERS = 1024
_MAX_VALUE = 256
_ALLOWED = {"generation", "revision", "context_id", "task_id"}
_LEASE_SECONDS = 180.0
_MAX_REVISION = 2**63 - 1
_ALLOWED = _ALLOWED | {"revision_epoch", "lease_owner", "lease_expires_at"}


def state_path() -> Path:
    return get_hermes_home() / "a2a" / "client-state.json"


def _lock_path() -> Path:
    return get_hermes_home() / "a2a" / "client-state.lock"


def _owned_regular(info: os.stat_result, label: str) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"A2A client {label} must be a regular file")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise RuntimeError(f"A2A client {label} has an unexpected owner")


@contextmanager
def _state_lock():
    path = state_path()
    with _LOCK:
        get_hermes_home().mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.mkdir(exist_ok=True, mode=0o700)
        try:
            with auth._safe_file_lock(_lock_path()) as directory_fd:
                yield directory_fd
        except auth.CredentialStoreError as exc:
            raise RuntimeError("A2A client state lock is unsafe") from exc


def _empty() -> dict[str, Any]:
    return {"version": 1, "peers": {}}


def _bounded_id(value: Any, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_VALUE:
        raise RuntimeError("A2A client state contains an invalid identifier")
    return value


def _validate(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != {"version", "peers"} or data.get("version") != 1:
        raise RuntimeError("A2A client state is invalid")
    peers = data.get("peers")
    if not isinstance(peers, dict) or len(peers) > _MAX_PEERS:
        raise RuntimeError("A2A client state is invalid")
    for name, entry in peers.items():
        try:
            config.validate_name(name, label="peer")
        except ValueError as exc:
            raise RuntimeError("A2A client state is invalid") from exc
        if not isinstance(entry, dict) or not set(entry).issubset(_ALLOWED):
            raise RuntimeError("A2A client state is invalid")
        _bounded_id(entry.get("generation"), required=True)
        revision = entry.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or not 0 <= revision <= _MAX_REVISION:
            raise RuntimeError("A2A client state is invalid")
        _bounded_id(entry.get("revision_epoch"), required=True)
        lease_owner = _bounded_id(entry.get("lease_owner"))
        lease_expiry = entry.get("lease_expires_at")
        if (lease_owner is None) != (lease_expiry is None):
            raise RuntimeError("A2A client state is invalid")
        if lease_expiry is not None and (
            not isinstance(lease_expiry, (int, float))
            or isinstance(lease_expiry, bool)
            or not math.isfinite(lease_expiry)
            or lease_expiry < 0
            or lease_expiry > time.time() + (_LEASE_SECONDS * 2)
        ):
            raise RuntimeError("A2A client state is invalid")
        _bounded_id(entry.get("context_id"))
        _bounded_id(entry.get("task_id"))
    return data


def _load_unlocked(directory_fd: int) -> dict[str, Any]:
    name = state_path().name
    try:
        expected = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return _empty()
    if stat.S_ISLNK(expected.st_mode):
        raise RuntimeError("A2A client state must be a regular file")
    _owned_regular(expected, "state")
    if expected.st_size > _MAX_FILE_BYTES:
        raise RuntimeError("A2A client state is too large")
    flags = os.O_RDONLY | (getattr(os, "O_NOFOLLOW", 0))
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
        with os.fdopen(fd, "rb") as stream:
            actual = os.fstat(stream.fileno())
            _owned_regular(actual, "state")
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise RuntimeError("A2A client state changed during access")
            os.fchmod(stream.fileno(), 0o600)
            raw = stream.read(_MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise RuntimeError("A2A client state is unreadable") from exc
    if len(raw) > _MAX_FILE_BYTES:
        raise RuntimeError("A2A client state is too large")
    try:
        return _validate(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("A2A client state is invalid") from exc


def _save_unlocked(data: dict[str, Any], directory_fd: int) -> None:
    _validate(data)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > _MAX_FILE_BYTES:
        raise RuntimeError("A2A client state is too large")
    path = state_path()
    temp_name = f".{path.name}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temp_name, flags, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temp_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temp_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def get_peer_state(peer: str) -> dict[str, Any]:
    peer = config.validate_name(peer, label="peer")
    with _state_lock() as directory_fd:
        return dict(_load_unlocked(directory_fd)["peers"].get(peer, {}))


@dataclass(frozen=True)
class LeaseClaim:
    owner: str
    epoch: str
    revision: int
    prior_revision: int
    context_id: str | None


def try_begin_request(
    peer: str,
    generation: str,
    owner: str,
    *,
    new_context: bool,
) -> LeaseClaim | None:
    """Atomically acquire an expired/absent lease, or return None."""
    from . import setup

    peer = config.validate_name(peer, label="peer")
    generation = _bounded_id(generation, required=True)
    owner = _bounded_id(owner, required=True)
    with setup._setup_transaction(), _state_lock() as directory_fd:
        current = config.load_a2a_settings().peers.get(peer)
        if current is None or current.get("generation") != generation:
            raise RuntimeError("A2A peer authority changed")
        data = _load_unlocked(directory_fd)
        entry = data["peers"].get(peer)
        if not isinstance(entry, dict) or entry.get("generation") != generation:
            entry = {
                "generation": generation,
                "revision_epoch": secrets.token_urlsafe(18),
                "revision": 0,
            }
            data["peers"][peer] = entry
        now = time.time()
        if entry.get("lease_owner") and entry.get("lease_expires_at", 0) > now:
            return None
        prior_revision = entry["revision"]
        if prior_revision >= _MAX_REVISION:
            entry["revision_epoch"] = secrets.token_urlsafe(18)
            entry["revision"] = 0
            prior_revision = 0
        prior_context = None if new_context else entry.get("context_id")
        entry["revision"] += 1
        entry["lease_owner"] = owner
        entry["lease_expires_at"] = now + _LEASE_SECONDS
        if new_context:
            entry.pop("context_id", None)
            entry.pop("task_id", None)
        _save_unlocked(data, directory_fd)
        return LeaseClaim(
            owner=owner,
            epoch=entry["revision_epoch"],
            revision=entry["revision"],
            prior_revision=prior_revision,
            context_id=prior_context,
        )


def complete_request(
    peer: str,
    generation: str,
    claim: LeaseClaim,
    *,
    context_id: str,
    task_id: str,
) -> bool:
    from . import setup

    context_id = _bounded_id(context_id, required=True)
    task_id = _bounded_id(task_id, required=True)
    with setup._setup_transaction(), _state_lock() as directory_fd:
        current = config.load_a2a_settings().peers.get(peer)
        if current is None or current.get("generation") != generation:
            return False
        data = _load_unlocked(directory_fd)
        entry = data["peers"].get(peer)
        if (
            not isinstance(entry, dict)
            or entry.get("generation") != generation
            or entry.get("revision_epoch") != claim.epoch
            or entry.get("revision") != claim.revision
            or entry.get("lease_owner") != claim.owner
        ):
            return False
        entry["context_id"] = context_id
        entry["task_id"] = task_id
        entry.pop("lease_owner", None)
        entry.pop("lease_expires_at", None)
        _save_unlocked(data, directory_fd)
        return True


def abort_request(peer: str, generation: str, claim: LeaseClaim) -> None:
    """Release only the caller's lease; preserve the prior successful context."""
    from . import setup

    with setup._setup_transaction(), _state_lock() as directory_fd:
        current = config.load_a2a_settings().peers.get(peer)
        if current is None or current.get("generation") != generation:
            return
        data = _load_unlocked(directory_fd)
        entry = data["peers"].get(peer)
        if (
            not isinstance(entry, dict)
            or entry.get("revision_epoch") != claim.epoch
            or entry.get("revision") != claim.revision
            or entry.get("lease_owner") != claim.owner
        ):
            return
        entry.pop("lease_owner", None)
        entry.pop("lease_expires_at", None)
        _save_unlocked(data, directory_fd)


def _clear_peer_state_unlocked(peer: str) -> None:
    """Caller owns setup transaction; acquire only the state lock."""
    with _state_lock() as directory_fd:
        data = _load_unlocked(directory_fd)
        if data["peers"].pop(peer, None) is not None:
            _save_unlocked(data, directory_fd)


def _snapshot_peer_state_unlocked(peer: str) -> dict[str, Any] | None:
    with _state_lock() as directory_fd:
        entry = _load_unlocked(directory_fd)["peers"].get(peer)
        return copy.deepcopy(entry) if isinstance(entry, dict) else None


def _restore_peer_state_unlocked(peer: str, snapshot: dict[str, Any] | None) -> None:
    with _state_lock() as directory_fd:
        data = _load_unlocked(directory_fd)
        if snapshot is None:
            data["peers"].pop(peer, None)
        else:
            data["peers"][peer] = copy.deepcopy(snapshot)
        _save_unlocked(data, directory_fd)


def clear_peer_state(peer: str) -> None:
    from . import setup

    peer = config.validate_name(peer, label="peer")
    with setup._setup_transaction():
        _clear_peer_state_unlocked(peer)
