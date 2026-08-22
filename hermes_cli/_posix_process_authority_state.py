"""Shared state and contracts for retained POSIX process authority."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

AUTHORITY_MODE = "posix-session-v1"
AUTHORITY_MODE_ENV = "HERMES_DESKTOP_PROCESS_AUTHORITY"
GENERATION_ENV = "HERMES_DESKTOP_PROCESS_GENERATION"
PARENT_PID_ENV = "HERMES_DESKTOP_PARENT_PID"
PARENT_STARTED_AT_ENV = "HERMES_DESKTOP_PARENT_STARTED_AT_MS"
ROLE_ENV = "HERMES_DESKTOP_PROCESS_ROLE"
LIFETIME_ENV = "HERMES_DESKTOP_PROCESS_LIFETIME"
TRANSFER_RECEIPT_ENV = "HERMES_DESKTOP_PROCESS_TRANSFER_RECEIPT"
TRANSFER_TOKEN_ENV = "_HERMES_DESKTOP_PROCESS_TRANSFER_TOKEN"
TRANSFER_RECEIVER_ENV = "_HERMES_DESKTOP_PROCESS_TRANSFER_RECEIVER"
DESCENDANT_GUARD_ENV = "_HERMES_DESKTOP_POSIX_DESCENDANT_GUARD"

LIFETIME_CONTAINED = "contained"
LIFETIME_TRANSFERRED = "transferred"
LIFETIME_FOREIGN = "foreign"
ALLOWED_LIFETIMES = {
    LIFETIME_CONTAINED,
    LIFETIME_TRANSFERRED,
    LIFETIME_FOREIGN,
}
GENERATION_RE = re.compile(r"^[A-Za-z0-9._-]{16,128}$")
RECEIVER_RE = re.compile(r"^[A-Za-z0-9._:/-]{3,128}$")
TRANSFER_PROTOCOL = "desktop-posix-transfer-v1"
TRANSFER_TTL_SECONDS = 30.0
TRANSFER_ACK_SECONDS = 3.0
DEFAULT_GRACE_SECONDS = 5.0
DEFAULT_FORCE_SECONDS = 2.0
SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)
SIGHUP = getattr(signal, "SIGHUP", None)
SIGUSR2 = getattr(signal, "SIGUSR2", None)


class ProcessAuthorityError(RuntimeError):
    """The requested process operation could not preserve authority."""


@dataclass(frozen=True)
class AuthoritySpec:
    generation: str
    parent_pid: int
    parent_started_at_ms: int


@dataclass(frozen=True)
class InstalledPosixAuthority:
    spec: AuthoritySpec
    role: str
    scope_id: int


@dataclass(frozen=True)
class ProcessTransferGrant:
    """One-use permission to hand a child to a named receiving owner."""

    token: str
    receiver: str


@dataclass(frozen=True)
class PendingTransfer:
    receiver: str
    expires_at: float


@dataclass(frozen=True)
class TransferStartNewSession:
    """In-process marker carried by the existing detach-helper kwargs."""

    grant: ProcessTransferGrant

    def __bool__(self) -> bool:
        return True


install_lock = threading.RLock()
transfer_lock = threading.Lock()
installed: InstalledPosixAuthority | None = None
guard_installed = False
pending_transfers: dict[str, PendingTransfer] = {}
original_popen_init = subprocess.Popen.__init__
original_setsid = getattr(os, "setsid", None)
original_setpgid = getattr(os, "setpgid", None)
original_setpgrp = getattr(os, "setpgrp", None)
original_posix_spawn = getattr(os, "posix_spawn", None)
original_posix_spawnp = getattr(os, "posix_spawnp", None)
original_killpg = getattr(os, "killpg", None)
original_detach_helper: Any | None = None


def read_spec(environ: Mapping[str, str]) -> AuthoritySpec | None:
    mode = (environ.get(AUTHORITY_MODE_ENV) or "").strip()
    if not mode:
        return None
    if mode != AUTHORITY_MODE:
        raise ProcessAuthorityError(f"unsupported POSIX process authority mode: {mode!r}")

    generation = (environ.get(GENERATION_ENV) or "").strip()
    if not GENERATION_RE.fullmatch(generation):
        raise ProcessAuthorityError("desktop process generation is missing or malformed")
    try:
        parent_pid = int((environ.get(PARENT_PID_ENV) or "").strip())
        parent_started_at_ms = int((environ.get(PARENT_STARTED_AT_ENV) or "").strip())
    except ValueError as exc:
        raise ProcessAuthorityError("desktop parent identity is malformed") from exc
    if parent_pid <= 0 or parent_started_at_ms <= 0:
        raise ProcessAuthorityError("desktop parent identity must be positive")
    return AuthoritySpec(generation, parent_pid, parent_started_at_ms)


def authority_keys() -> tuple[str, ...]:
    return (
        AUTHORITY_MODE_ENV,
        GENERATION_ENV,
        PARENT_PID_ENV,
        PARENT_STARTED_AT_ENV,
        ROLE_ENV,
    )


def transfer_keys() -> tuple[str, ...]:
    return (
        TRANSFER_TOKEN_ENV,
        TRANSFER_RECEIVER_ENV,
        TRANSFER_RECEIPT_ENV,
    )


def strip_authority_envelope(env: MutableMapping[str, str]) -> None:
    for key in authority_keys():
        env.pop(key, None)


def strip_transfer_envelope(env: MutableMapping[str, str]) -> None:
    for key in transfer_keys():
        env.pop(key, None)


def prune_transfers(now: float | None = None) -> None:
    cutoff = time.monotonic() if now is None else now
    expired = [
        token
        for token, pending in pending_transfers.items()
        if pending.expires_at <= cutoff
    ]
    for token in expired:
        pending_transfers.pop(token, None)
