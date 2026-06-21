"""Encrypting log rotation for Hermes agent/gateway/error logs.

When ``security.encryption.encrypt_logs`` is true, rotated log segments
(``agent.log.1``, ``.2``, …) are encrypted with the DEK at rotation time.
The live log file stays plaintext so ``tail -f`` continues to work.

Upstream Hermes logging setup should use :func:`build_rotating_handler` in
place of a plain :class:`logging.handlers.RotatingFileHandler` for files
under ``~/.hermes/logs/``.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

from .detect import is_encrypted

PathLike = Union[str, Path]

__all__ = [
    "EncryptingRotatingFileHandler",
    "build_rotating_handler",
    "read_log_bytes",
    "read_log_text",
]


def build_rotating_handler(
    path: PathLike,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    encoding: str = "utf-8",
) -> Optional[logging.Handler]:
    """Return a rotating file handler, encrypting rolled segments when configured.

    When ``security.encryption.encrypt_logs`` is false, this is a standard
    :class:`RotatingFileHandler`. When true, backups are HRMSENC-encrypted
    after each rollover. Returns ``None`` if the log directory cannot be
    created.
    """
    log_path = Path(path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    # harden the log directory to 0o700 (or 0o770 in managed
    # mode so the gateway group can tail) so rolled-over plaintext log segments
    # and the live log file aren't world-readable on multi-user POSIX hosts.
    # Best-effort — Windows is a silent no-op.
    try:
        os.chmod(log_path.parent, 0o770 if _check_managed_mode() else 0o700)
    except OSError:
        pass

    handler_cls = (
        EncryptingRotatingFileHandler
        if _logs_encryption_active()
        else RotatingFileHandler
    )
    return handler_cls(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding=encoding,
    )


def _logs_encryption_active() -> bool:
    from . import logs_encryption_active

    return logs_encryption_active()


def _check_managed_mode() -> bool:
    """Return True when running under the managed (NixOS / systemd) marker.

    Lazy import: ``hermes_cli.config`` is upstream-only and may not be
    importable in every fork. Honors the import-safety invariant in
    ``hermes_crypto/__init__.py`` (AGENTS.md §3.1) — never let a missing
    helper or an unrelated import error break logging setup.
    """
    try:
        from hermes_cli.config import is_managed

        return bool(is_managed())
    except Exception:
        return False


class EncryptingRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that AES-GCM-encrypts each rolled-over segment.

    In managed mode (NixOS / systemd with the ``HERMES_MANAGED`` marker),
    the live log file and rotated segments are chmodded to ``0o660`` after
    each open and after each rollover so the gateway and interactive users
    can share access — mirroring the upstream ``_ManagedRotatingFileHandler``
    contract that this class replaces on the encrypt-logs code path.
    """

    def __init__(self, *args, **kwargs):
        # Lazy managed-mode check — see ``_check_managed_mode`` for the
        # rationale (AGENTS.md §3.1 import safety).
        self._managed = _check_managed_mode()
        super().__init__(*args, **kwargs)

    def _chmod_after_open(self) -> None:
        # 0o660 in managed mode so the gateway group can
        # tail the live log; 0o600 otherwise so a multi-user POSIX host
        # doesn't leak the live log to other local users until rotation
        # encrypts the rolled segment. Best-effort — Windows is a no-op.
        target_mode = 0o660 if self._managed else 0o600
        try:
            os.chmod(self.baseFilename, target_mode)
        except OSError:
            pass

    def _open(self):
        stream = super()._open()
        self._chmod_after_open()
        return stream

    def doRollover(self) -> None:
        super().doRollover()
        self._chmod_after_open()
        _encrypt_log_segment_if_needed(f"{self.baseFilename}.1")


def _encrypt_log_segment_if_needed(path: str) -> None:
    """Best-effort: encrypt *path* in place when it is plaintext."""
    segment = Path(path)
    if not segment.is_file():
        return
    try:
        raw = segment.read_bytes()
    except OSError:
        return
    if not raw or is_encrypted(raw):
        return
    try:
        from . import envelope, get_data_key
        from .fileio import atomic_write_private

        encrypted = envelope.encrypt(raw, get_data_key())
        atomic_write_private(segment, encrypted)
    except Exception as exc:
        # Logging must not crash because encryption failed, but the segment
        # is now plaintext on disk while the operator believes logs are
        # encrypted — make the failure visible via the audit trail.
        from . import audit

        audit.log_event(
            audit.LOG_ENCRYPT_FAILED,
            audit.FAILURE,
            path=str(segment),
            reason=type(exc).__name__,
        )


def read_log_bytes(path: PathLike) -> bytes:
    """Return cleartext log bytes from *path* (decrypting HRMSENC segments)."""
    raw = Path(path).read_bytes()
    if not is_encrypted(raw):
        return raw
    from . import envelope, get_data_key

    return envelope.decrypt(raw, get_data_key())


def read_log_text(path: PathLike, *, encoding: str = "utf-8") -> str:
    """Return cleartext log text from *path*."""
    return read_log_bytes(path).decode(encoding, errors="replace")
