"""Host-wide singleton rendezvous: one lock + one record per ROLE per OS user.

Multiplex-only (Teknium ruling): exactly ONE ``hermes serve`` and ONE ``hermes gateway run``
per host, each multiplexing every profile. The per-``HERMES_HOME`` gateway lock/PID files
(``gateway.status``) cannot express that — N profiles are N homes, so N processes each take
their own flock and none of them ever sees the others. This module adds the missing layer:

* a **host lock** (flock/``msvcrt``) held for the lifetime of the winning process, and
* a **rendezvous record** the winner publishes so a second invocation can find it, prove it
  is the same live process, and ATTACH instead of binding a second port.

Both live in :func:`gateway.status._get_lock_dir` — the only cross-profile lock root already
in the tree (``$HERMES_GATEWAY_LOCK_DIR`` else ``$XDG_STATE_HOME/hermes/gateway-locks``),
which scopes to the **OS user**. That is the correct granularity: separate OS users have
separate ``$HOME``s, separate ``~/.hermes`` profile roots, separate ports-by-convention and
separate credentials, so "one per host" means "one per host per OS user".

**Staleness is proved, never assumed.** A record carries ``(pid, createTime)``; a record whose
PID is dead, or whose PID is alive with a different process creation time (PID reuse), is
STALE and is ignored — an attaching client must never dial a recycled PID's port.

**Relationship to ``spawn-ledger.json``** (``hermes_cli/process_identity.py``): the ledger stays
the append-only machine roster of every long-lived Hermes process (Desktop's attach ladder reads
it) and is still written unchanged. It cannot be the host record: it has no lock, no
single-writer semantics, no removal on exit, and no place to publish a protocol version or
an authentication handle. The record here is authoritative for "who owns this host role"; the
ledger remains authoritative for "what is running". Both are written, and this module reuses the
ledger's ``(pid, create_time)`` liveness proof rather than inventing a second one.
"""

from __future__ import annotations

import contextlib
import enum
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from utils import atomic_json_write

logger = logging.getLogger(__name__)

#: Bumped when the record's shape or the attach handshake changes incompatibly. A reader that
#: does not recognise the version refuses to attach instead of guessing.
HOST_PROTOCOL_VERSION = 1

#: Token-gated endpoint the owner answers with its own identity. An attaching client dials the
#: RECORDED port and only attaches when the answer is the recorded owner — a record alone proves
#: nothing about who holds that port right now.
HOST_IDENTITY_PATH = "/api/host/identity"

#: Bounded: a closed port, a black-holed one or a foreign listener must cost a second, not a hang.
PROBE_TIMEOUT_S = 2.0

ROLE_GATEWAY = "gateway"
ROLE_SERVE = "serve"
_ROLES = (ROLE_GATEWAY, ROLE_SERVE)

# Open lock handles, keyed by (role, resolved lock path): the OS releases the flock when this
# process dies, which is what makes a crashed owner's host lock re-acquirable without a reaper.
# The PATH is part of the key because the lock dir is env-derived (HERMES_GATEWAY_LOCK_DIR):
# keyed by role alone, a second call after the dir changed returned "already held" without ever
# creating the new lock file, so owns_host_lock() lied and one pytest process leaked the handle
# across tests.
_lock_handles: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class HostRecord:
    """A published host-role owner. ``profiles`` is the SERVED set, not the launch profile."""

    role: str
    pid: int
    create_time: Optional[float]
    host: str
    port: Optional[int]
    protocol_version: int
    token_fingerprint: str
    profiles: tuple[str, ...]
    updated_at: str

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "pid": self.pid,
            "createTime": self.create_time,
            "host": self.host,
            "port": self.port,
            "protocolVersion": self.protocol_version,
            "tokenFingerprint": self.token_fingerprint,
            "profiles": list(self.profiles),
            "updatedAt": self.updated_at,
        }

    @classmethod
    def from_json(cls, payload: Any) -> Optional["HostRecord"]:
        if not isinstance(payload, dict):
            return None
        pid = payload.get("pid")
        role = payload.get("role")
        if not isinstance(pid, int) or pid <= 0 or role not in _ROLES:
            return None
        create = payload.get("createTime")
        port = payload.get("port")
        profiles = payload.get("profiles")
        version = payload.get("protocolVersion")
        return cls(
            role=role,
            pid=pid,
            create_time=float(create) if isinstance(create, (int, float)) else None,
            host=str(payload.get("host") or ""),
            port=int(port) if isinstance(port, int) and 0 < port <= 65535 else None,
            protocol_version=version if isinstance(version, int) else 0,
            token_fingerprint=str(payload.get("tokenFingerprint") or ""),
            profiles=tuple(str(p) for p in profiles if isinstance(p, str)) if isinstance(profiles, list) else (),
            updated_at=str(payload.get("updatedAt") or ""),
        )


def host_state_dir() -> Path:
    """Per-OS-USER rendezvous dir (shared by every profile of this user)."""
    from gateway.status import _get_lock_dir

    return _get_lock_dir()


def _validated_role(role: str) -> str:
    if role not in _ROLES:
        raise ValueError(f"unknown host role: {role!r}")
    return role


def record_path(role: str) -> Path:
    return host_state_dir() / f"host-{_validated_role(role)}.json"


def lock_path(role: str) -> Path:
    return host_state_dir() / f"host-{_validated_role(role)}.lock"


def token_path(role: str) -> Path:
    return host_state_dir() / f"host-{_validated_role(role)}.token"


def token_fingerprint(token: str) -> str:
    """Short, non-reversible handle for a session token (safe to publish in the record)."""
    return hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()[:16] if token else ""


def process_create_time(pid: Optional[int] = None) -> Optional[float]:
    """Creation time of ``pid`` (default: this process); ``None`` when unknowable."""
    from hermes_cli.process_identity import _process_create_time

    return _process_create_time(pid)


def _pid_incarnation_matches(pid: int, create_time: Optional[float]) -> Optional[bool]:
    """Reuse the spawn ledger's proof: True/False when provable, ``None`` when it cannot say."""
    from hermes_cli.process_identity import _pid_alive_matches

    return _pid_alive_matches(pid, create_time)


def record_is_stale(record: Optional[HostRecord]) -> bool:
    """A record nobody may attach to: absent, unknown protocol, dead PID, or PID reuse.

    ``None`` from the liveness probe (no psutil, permission denied, an unexpected psutil error)
    means UNPROVABLE, and an unprovable record is a CANDIDATE, never an owner: it survives this
    predicate only to be handed to :func:`probe_owner`, which dials the recorded port and makes
    the owner prove itself. Treating unprovable as dead would let a second process bind a second
    port; treating it as alive *without the probe* made a record for a long-dead PID a permanent
    silent outage (attach exited 0 forever, with nothing listening).
    """
    if record is None:
        return True
    if record.protocol_version != HOST_PROTOCOL_VERSION:
        return True
    return _pid_incarnation_matches(record.pid, record.create_time) is False


def liveness_is_proven(record: HostRecord) -> bool:
    """True only when the PID+createTime probe positively matched (never on ``None``)."""
    return _pid_incarnation_matches(record.pid, record.create_time) is True


def dial_host(record: HostRecord) -> str:
    """Address to dial for ``record``: a wildcard bind is reached over loopback."""
    host = record.host or "127.0.0.1"
    return "127.0.0.1" if host in ("0.0.0.0", "::", "*", "") else host


def probe_owner(record: HostRecord, *, timeout: float = PROBE_TIMEOUT_S) -> Optional[dict]:
    """Make the recorded endpoint prove it is this record's owner; ``None`` when it does not.

    Two gates, both required: a bounded TCP connect (nothing listening → the owner is gone, even
    though its record and PID may still look alive during a graceful-shutdown window) and an
    identity GET authenticated with the 0600 token (a foreign listener that inherited the port
    answers the connect but cannot answer as PID N of role R).

    Returns the owner's identity payload (``pid``/``role``/``servesSpa``) on success.
    """
    if not record.port:
        return None
    import socket

    host = dial_host(record)
    try:
        with socket.create_connection((host, record.port), timeout=timeout):
            pass
    except OSError:
        logger.debug("host %s owner does not answer at %s:%s", record.role, host, record.port)
        return None

    import urllib.request

    token = read_token(record.role)
    headers = {"X-Hermes-Token": token, "Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(
        f"http://{host}:{record.port}{HOST_IDENTITY_PATH}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 — fixed http scheme
            if response.status != 200:
                return None
            payload = json.loads(response.read(65536).decode("utf-8", "replace"))
    except Exception:
        logger.debug("host %s identity probe failed at %s:%s", record.role, host, record.port, exc_info=True)
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("role") != record.role or payload.get("pid") != record.pid:
        return None
    return payload


def read_record(role: str, *, include_stale: bool = False) -> Optional[HostRecord]:
    """Published record for ``role``; ``None`` when absent, corrupt or (by default) stale."""
    try:
        raw = record_path(role).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        record = HostRecord.from_json(json.loads(raw))
    except (ValueError, TypeError):
        return None
    if record is None:
        return None
    return record if include_stale or not record_is_stale(record) else None


def read_token(role: str) -> str:
    """Owner-written session token for ``role`` (``""`` when absent/unreadable).

    This is the handle an attaching client uses when the backend is auth-gated and therefore
    withholds its token from an unauthenticated ``GET /``. Confidentiality is enforced at WRITE
    time by :func:`_write_private_text` (POSIX ``0o600``; an owner+SYSTEM-only protected DACL on
    Windows, where mode bits set no ACLs at all). Reading it is therefore evidence of, not proof
    of, same-OS-user authority — the authority boundary the host lock is scoped to.
    """
    try:
        return token_path(role).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def record_token_is_consistent(record: HostRecord) -> bool:
    """Does the on-disk token still hash to the record's fingerprint?

    A record published without a token (the gateway) has an empty fingerprint and is consistent
    by definition. A mismatch means the record and the token file come from different
    incarnations (a torn restart) — discovery must not attach with a token the owner rejects.
    """
    if not record.token_fingerprint:
        return True
    return token_fingerprint(read_token(record.role)) == record.token_fingerprint


def _write_private_text(path: Path, text: str) -> None:
    """Create/replace ``path`` with owner-only content.

    POSIX: ``0o600`` via tmp + atomic replace. Windows: mode bits set NO ACLs, so the same
    ``os.open`` would leave a live session token readable by whatever the inherited DACL grants;
    the SSH runtime's protected owner+SYSTEM DACL writer is the repo's primitive for exactly this
    credential class and is reused here. It also replaces in place, because ``os.replace`` onto a
    token file another process still holds open fails on Windows.
    """
    if sys.platform == "win32":
        from hermes_cli.windows_ssh_runtime import write_private_file

        write_private_file(path, text.encode("utf-8"))
        return
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            with contextlib.suppress(OSError):
                os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(str(tmp))
        raise
    os.replace(str(tmp), str(path))


class HostLockOutcome(enum.Enum):
    """Why a host-lock claim ended the way it did.

    ``COULD_NOT_OPEN`` is NOT contention: a read-only/undeletable lock dir (EROFS, EACCES,
    ENOSPC) collapsed into the same "another owner holds it" log and sent operators hunting a
    second gateway that never existed.
    """

    ACQUIRED = "acquired"
    HELD_BY_OTHER = "held-by-other"
    COULD_NOT_OPEN = "could-not-open"


def _lock_key(role: str) -> tuple[str, str]:
    path = lock_path(role)
    with contextlib.suppress(OSError):
        return (role, os.path.abspath(str(path)))
    return (role, str(path))


def claim_host_lock(role: str) -> tuple[HostLockOutcome, Optional[OSError]]:
    """Take the host-wide lock for ``role``. Idempotent per (role, lock path).

    Returns the outcome and, for ``COULD_NOT_OPEN``, the OSError that explains it.
    """
    role = _validated_role(role)
    key = _lock_key(role)
    if _lock_handles.get(key) is not None:
        return (HostLockOutcome.ACQUIRED, None)
    from gateway.status import _try_acquire_file_lock

    path = Path(key[1])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+", encoding="utf-8")
    except OSError as exc:
        logger.debug("host %s lock could not be opened at %s", role, path, exc_info=True)
        return (HostLockOutcome.COULD_NOT_OPEN, exc)
    if not _try_acquire_file_lock(handle):
        with contextlib.suppress(OSError):
            handle.close()
        return (HostLockOutcome.HELD_BY_OTHER, None)
    _lock_handles[key] = handle
    return (HostLockOutcome.ACQUIRED, None)


def release_host_lock(role: str) -> None:
    """Release the host lock for ``role`` when this process holds it."""
    handle = _lock_handles.pop(_lock_key(_validated_role(role)), None)
    if handle is None:
        return
    from gateway.status import _release_file_lock

    _release_file_lock(handle)
    with contextlib.suppress(OSError):
        handle.close()


def owns_host_lock(role: str) -> bool:
    """True when THIS process holds the host lock for ``role`` (re-probing our own flock lies)."""
    return _lock_handles.get(_lock_key(_validated_role(role))) is not None


def publish_record(
    role: str,
    *,
    host: str = "",
    port: Optional[int] = None,
    profiles: Sequence[str] = (),
    token: Optional[str] = None,
) -> Optional[HostRecord]:
    """Publish this process as the host owner of ``role``. ``None`` when the write failed.

    ``token`` (serve) is persisted 0600 next to the record and only its fingerprint is published.
    """
    role = _validated_role(role)
    record = HostRecord(
        role=role,
        pid=os.getpid(),
        create_time=process_create_time(),
        host=str(host or ""),
        port=int(port) if isinstance(port, int) and port > 0 else None,
        protocol_version=HOST_PROTOCOL_VERSION,
        token_fingerprint=token_fingerprint(token or ""),
        profiles=tuple(str(p) for p in profiles),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        record_path(role).parent.mkdir(parents=True, exist_ok=True)
        if token:
            # No record without its token: publishing one an attaching client cannot
            # authenticate against would degrade to a silent "attach refused forever".
            _write_private_text(token_path(role), token)
        atomic_json_write(record_path(role), record.to_json(), mode=0o600)
    except OSError:
        logger.warning("host %s record could not be published; discovery will not find it", role, exc_info=True)
        return None
    return record


def clear_record(role: str) -> None:
    """Remove this process's record + token on exit (never another owner's)."""
    role = _validated_role(role)
    existing = read_record(role, include_stale=True)
    if existing is not None and existing.pid != os.getpid():
        return
    for path in (record_path(role), token_path(role)):
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


# Roles this process must clean up on the way out, and the signal handlers we prepended.
_cleanup_roles: list[str] = []
_prev_signal_handlers: dict[int, Any] = {}


def _cleanup_role(role: str) -> None:
    with contextlib.suppress(Exception):
        clear_record(role)
    with contextlib.suppress(Exception):
        release_host_lock(role)


def _handle_terminating_signal(signum, frame) -> None:
    """Drop the record + token, then hand off to the handler we prepended to."""
    import signal as _signal

    for role in tuple(_cleanup_roles):
        _cleanup_role(role)
    prev = _prev_signal_handlers.get(signum)
    if callable(prev):
        prev(signum, frame)
        return
    if prev is _signal.SIG_IGN:
        return
    with contextlib.suppress(Exception):
        _signal.signal(signum, _signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    raise SystemExit(128 + int(signum))


def cleanup_on_exit(role: str) -> None:
    """Clear ``role``'s record + token and release its lock on exit — SIGTERM included.

    ``atexit`` alone is false advertising for the NORMAL stop: systemd stop, ``docker stop`` and
    the update relaunch all send SIGTERM, and every terminating SIGTERM path here ends in the
    default disposition (uvicorn's ``capture_signals`` re-raises it after its graceful shutdown),
    which kills the process without running ``atexit``. The record then outlived its process and
    the 0600 token kept a LIVE session token on disk indefinitely.

    The handler only PREPENDS cleanup: whatever handler was installed before (uvicorn's graceful
    shutdown, the exit-flush chain, a supervisor's) still runs, so the shutdown sequence is
    unchanged.
    """
    import atexit
    import signal as _signal
    import threading

    role = _validated_role(role)
    if role not in _cleanup_roles:
        _cleanup_roles.append(role)
        atexit.register(_cleanup_role, role)
    if threading.current_thread() is not threading.main_thread():
        return
    for name in ("SIGTERM", "SIGBREAK"):
        signum = getattr(_signal, name, None)
        if signum is None or signum in _prev_signal_handlers:
            continue
        with contextlib.suppress(ValueError, OSError, RuntimeError):
            prev = _signal.getsignal(signum)
            _signal.signal(signum, _handle_terminating_signal)
            _prev_signal_handlers[signum] = prev


def served_profiles() -> tuple[str, ...]:
    """Profiles this process multiplexes; ``()`` when the roster cannot be read."""
    try:
        from hermes_cli.profiles import profiles_to_serve

        return tuple(name for name, _ in profiles_to_serve(multiplex=True))
    except Exception:
        logger.debug("served profile roster unavailable", exc_info=True)
        return ()


def describe(record: HostRecord) -> str:
    """One-line human description used by attach messages and conflict logs."""
    where = f"{record.host or '127.0.0.1'}:{record.port}" if record.port else "no bound port"
    profiles = ", ".join(record.profiles) if record.profiles else "unknown"
    return f"PID {record.pid} ({where}; profiles: {profiles})"
