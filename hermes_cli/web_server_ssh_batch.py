"""Server-owned liveness leases for one Desktop SSH spawn batch.

A Desktop SSH connection can publish several detached ``serve --isolated``
backends.  Each backend owns its WebSocket/turn state, but sibling backends
must not self-retire while one of them is actively serving the same Desktop
connection.  This module communicates only that bounded server-side fact; it
never treats a client artifact or a process environment variable as liveness.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import stat
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

_log = logging.getLogger(__name__)

_BATCH_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_MEMBER_NONCE_RE = re.compile(r"[0-9a-f]{16}\Z")
_MAX_BATCH_MEMBERS = 64


def _default_runtime_root() -> Path:
    if os.name == "nt":
        # The one-shot token reader uses the machine-rooted, DACL-protected
        # Windows runtime directory rather than profile-relative ~/.hermes.
        from hermes_cli.windows_ssh_runtime import _root

        return _root()
    return Path.home() / ".hermes" / "desktop-ssh"


class SshSpawnBatchLease:
    """Atomic, expiring liveness record for one isolated server process.

    Leases live below the same per-user ``desktop-ssh`` runtime root that
    carries the one-shot bootstrap token.  Each member writes only its own
    nonce-named record, so sibling writers do not need a shared lock.  A stale
    record is ignored rather than deleted: a process may be paused or killed at
    any point, and this process has no authority to remove another member's
    record.
    """

    def __init__(
        self,
        batch_id: str,
        owner_nonce: str,
        *,
        root: Optional[Path] = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        if not _BATCH_ID_RE.fullmatch(str(batch_id)):
            raise ValueError("SSH spawn batch ID must be 32 lowercase hex characters")
        if not _MEMBER_NONCE_RE.fullmatch(str(owner_nonce)):
            raise ValueError("SSH spawn member nonce must be 16 lowercase hex characters")

        self._batch_id = batch_id
        self._owner_nonce = owner_nonce
        self._root = root or _default_runtime_root()
        # An injected root is a deterministic unit-test seam. Production uses
        # the native runtime root, where every component gets the Windows
        # no-reparse-point/DACL validation below.
        self._secure_windows_runtime = root is None
        self._now = now
        self._failure_logged = False

    @property
    def _lease_name(self) -> str:
        return f"{self._owner_nonce}.json"

    def _check_directory(self, directory: Path, *, create: bool) -> None:
        if os.name == "nt" and self._secure_windows_runtime:
            if not create and not directory.exists():
                raise OSError("SSH batch runtime root is not accessible")
            # Reuse the native runtime's no-reparse-point + protected-DACL
            # checks instead of trusting Path operations on a Windows remote.
            from hermes_cli.windows_ssh_runtime import _ensure_directory

            _ensure_directory(directory)
            return
        if create:
            directory.mkdir(mode=0o700, exist_ok=True)

        entry = directory.lstat()
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
            raise OSError("SSH batch runtime path is not a real directory")

        if os.name != "nt":
            if entry.st_uid != os.getuid():
                raise OSError("SSH batch runtime path has the wrong owner")
            if create and (entry.st_mode & 0o077):
                os.chmod(directory, 0o700)

    def _batch_directory(self) -> Path:
        # The token-file parser already required this root to exist and belong
        # to the current account.  Do not create a new root from a server flag.
        self._check_directory(self._root, create=False)
        batches = self._root / "batches"
        self._check_directory(batches, create=True)
        batch = batches / self._batch_id
        self._check_directory(batch, create=True)
        return batch

    def _log_failure_once(self) -> None:
        if not self._failure_logged:
            self._failure_logged = True
            _log.warning("SSH spawn-batch lease unavailable; idle exit will fail closed", exc_info=True)

    def publish(self, *, active: bool) -> bool:
        """Publish this process's current liveness without exposing credentials."""
        try:
            batch = self._batch_directory()
            payload = json.dumps(
                {"active": bool(active), "updated_at": self._now()},
                separators=(",", ":"),
            ).encode("utf-8")
            descriptor, temporary = tempfile.mkstemp(
                dir=batch,
                prefix=f".{self._owner_nonce}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    descriptor = -1
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                if os.name != "nt":
                    os.chmod(temporary, 0o600)
                os.replace(temporary, batch / self._lease_name)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            return True
        except OSError:
            self._log_failure_once()
            return False

    def _read_peer(self, path: Path, now: float, ttl_s: float) -> Optional[bool]:
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError:
            return None

        try:
            entry = os.fstat(descriptor)
            if not stat.S_ISREG(entry.st_mode):
                return None
            if os.name != "nt" and (entry.st_uid != os.getuid() or entry.st_mode & 0o077):
                return None
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                descriptor = -1
                payload = json.load(stream)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        finally:
            if descriptor >= 0:
                os.close(descriptor)

        if not isinstance(payload, dict) or not isinstance(payload.get("active"), bool):
            return None
        updated_at = payload.get("updated_at")
        if not isinstance(updated_at, (int, float)) or not math.isfinite(updated_at):
            return None
        if updated_at > now + ttl_s:
            return None
        if updated_at < now - ttl_s:
            return False
        return payload["active"]

    def has_active_peer(self, *, ttl_s: float) -> Optional[bool]:
        """Return peer activity, or ``None`` when the shared state is unsafe.

        ``None`` is deliberately distinct from ``False``: callers fail closed
        rather than retiring when they cannot prove that every fresh sibling is
        inactive.
        """
        if not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("SSH spawn-batch lease TTL must be positive")

        try:
            batch = self._batch_directory()
            names = [path for path in batch.iterdir() if _MEMBER_NONCE_RE.fullmatch(path.stem) and path.suffix == ".json"]
        except OSError:
            self._log_failure_once()
            return None

        if len(names) > _MAX_BATCH_MEMBERS:
            return None

        now = self._now()
        for path in names:
            if path.name == self._lease_name:
                continue
            active = self._read_peer(path, now, ttl_s)
            if active is None:
                return None
            if active:
                return True
        return False

    def close(self) -> None:
        """Remove only this member's record during graceful server shutdown."""
        try:
            (self._batch_directory() / self._lease_name).unlink(missing_ok=True)
        except OSError:
            self._log_failure_once()
