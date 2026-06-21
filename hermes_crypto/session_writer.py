"""Encrypting JSONL writer for Hermes gateway session transcripts.

Gateway adapters append one JSON object per turn to
``~/.hermes/sessions/<session-id>.jsonl``. Without encryption-at-rest those
files leak full prompt/response history. With ``security.encryption.encrypt_logs``
turned on, :class:`EncryptingSessionWriter` writes plaintext while the session
is active (so ``tail -f`` works during debugging) and AES-GCM-encrypts the
whole file on :meth:`close`.

A per-session **lockfile** lets the next :func:`hermes_crypto.migrate.sweep_sessions`
run identify abandoned plaintext sessions left behind by a crashed writer —
the kernel releases the advisory lock automatically on process death, so the
sweep can re-acquire it and finish the encryption.

The live-file-plaintext / archive-encrypted split mirrors
:class:`hermes_crypto.log_handler.EncryptingRotatingFileHandler`. The crash
window (between hard kill and next sweep) is the documented residual risk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Union

from ._lockfile import _acquire_exclusive, _open_lockfile, _release_exclusive
from .detect import is_encrypted
from .errors import HermesCryptoError
from .fileio import harden_dir, open_private_append

PathLike = Union[str, Path]

__all__ = [
    "EncryptingSessionWriter",
    "build_session_writer",
    "read_session_bytes",
    "read_session_text",
]

_LOCK_SUFFIX = ".lock"


# ─── Lockfile primitives ────────────────────────────────────────────────────
#
# The generic advisory-lock primitives live in :mod:`._lockfile` so
# :mod:`.audit` can reuse them for rotation-lock recovery. The names are
# re-exported here (``from ._lockfile import …``) so existing test imports
# of ``session_writer._open_lockfile`` keep working.


def _try_lock_session(jsonl_path: Path) -> Optional[int]:
    """Try to take the session's lockfile.

    Returns the open lock fd on success, ``None`` when another live writer
    holds the lock. Caller must release via :func:`_drop_session_lock`.
    """
    lock_path = jsonl_path.with_name(jsonl_path.name + _LOCK_SUFFIX)
    fd: Optional[int] = None
    try:
        fd = _open_lockfile(lock_path)
        _acquire_exclusive(fd)
    except OSError:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        return None
    return fd


def _probe_session_lock_held(jsonl_path: Path) -> bool:
    """Non-destructive: return True when *jsonl_path*'s lockfile is held by a live writer.

    Used by status reporters that must not have side effects. The sweep
    path uses :func:`_try_lock_session` + :func:`_drop_session_lock`
    instead, which take ownership of an acquirable lockfile.
    """
    lock_path = jsonl_path.with_name(jsonl_path.name + _LOCK_SUFFIX)
    if not lock_path.is_file():
        return False
    fd: Optional[int] = None
    try:
        fd = _open_lockfile(lock_path)
        try:
            _acquire_exclusive(fd)
        except OSError:
            return True
        _release_exclusive(fd)
        return False
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _drop_session_lock(fd: int, jsonl_path: Path) -> None:
    """Release the lock held on *fd* and delete the lockfile. Best-effort."""
    _release_exclusive(fd)
    try:
        os.close(fd)
    except OSError:
        pass
    lock_path = jsonl_path.with_name(jsonl_path.name + _LOCK_SUFFIX)
    try:
        lock_path.unlink()
    except OSError:
        pass


# ─── Encrypt-on-close ───────────────────────────────────────────────────────


def _encrypt_session_if_needed(path: Path) -> None:
    """Encrypt the closed session file *path* in place when it is plaintext.

    Idempotent — re-encrypting an HRMSENC file is skipped via magic detection.
    Empty files are skipped too (no point sealing nothing). On any failure
    emit ``audit.SESSION_ENCRYPT_FAILED`` (critical severity) and return:
    the operator-paging signal is the audit event, not an exception.
    """
    if not path.is_file():
        return
    try:
        raw = path.read_bytes()
    except OSError:
        return
    if not raw or is_encrypted(raw):
        return
    try:
        from . import envelope, get_data_key
        from .fileio import atomic_write_private

        encrypted = envelope.encrypt(raw, get_data_key())
        atomic_write_private(path, encrypted)
    except Exception as exc:
        from . import audit

        audit.log_event(
            audit.SESSION_ENCRYPT_FAILED,
            audit.FAILURE,
            path=str(path),
            reason=type(exc).__name__,
        )


# ─── Public writer API ──────────────────────────────────────────────────────


class EncryptingSessionWriter:
    """JSONL append writer that AES-GCM-encrypts the file on :meth:`close`.

    Acquires an exclusive advisory lock on ``<path>.lock`` on init. The lock
    is released and the lockfile removed by :meth:`close`; on hard process
    death the kernel releases the lock automatically so the next sweep can
    finish the work.

    Raises :class:`HermesCryptoError` when another live writer already holds
    the session's lockfile — pick a different session id rather than racing.
    """

    def __init__(self, path: PathLike, *, encoding: str = "utf-8") -> None:
        self._path = Path(path)
        self._encoding = encoding
        self._lock_fd: Optional[int] = None
        self._fh = None
        self._closed = False

        # harden the session directory (0o700) before the
        # lockfile mkdir or the live JSONL open — otherwise the live transcript
        # is world-readable on a multi-user POSIX host until close() seals it.
        harden_dir(self._path.parent)

        fd = _try_lock_session(self._path)
        if fd is None:
            raise HermesCryptoError(
                f"Session {self._path.name} is already being written by another "
                "process (lockfile is held). Use a different session id."
            )
        self._lock_fd = fd

        try:
            # Refuse to append plaintext onto an already-encrypted file —
            # that would corrupt the envelope. The caller should re-key or
            # decrypt-then-reopen.
            if self._path.is_file():
                try:
                    head = self._path.read_bytes()[:32]
                except OSError:
                    head = b""
                if is_encrypted(head):
                    raise HermesCryptoError(
                        f"Refusing to append to encrypted session "
                        f"{self._path.name} — already sealed."
                    )
            # O_CREAT|0o600 instead of bare open() so the
            # live JSONL is never momentarily world-readable.
            self._fh = open_private_append(self._path, encoding=self._encoding)
        except Exception:
            if self._lock_fd is not None:
                _drop_session_lock(self._lock_fd, self._path)
                self._lock_fd = None
            raise

    @property
    def path(self) -> Path:
        return self._path

    @property
    def name(self) -> str:
        return self._path.name

    def write(self, text: str) -> int:
        """Append *text* to the live JSONL file. Returns char count written.

        durability note: ``write`` lands in Python's
        userspace buffer, not on stable storage. Callers that want crash
        durability for the just-written turn must follow up with
        :meth:`flush` (which empties Python's buffer to the OS) and accept
        that an OS-level cache flush still isn't issued. A hard kill or
        power loss between ``write`` and :meth:`close` may drop the
        buffered tail; the rest of the session has already been sealed by
        the next sweep so confidentiality is preserved, but the lost-tail
        content is not recoverable. The contract is "live file plaintext,
        sealed on close" — not "every write is durable on disk."
        """
        return self._fh.write(text)

    def writelines(self, lines: Iterable[str]) -> None:
        self._fh.writelines(lines)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._fh is not None:
                # Isolate flush so a failing flush() (disk full, I/O error)
                # never skips the encrypt-on-close below — what's already on
                # disk must still get sealed. _encrypt_session_if_needed
                # internally catches Exception and emits SESSION_ENCRYPT_FAILED
                # so callers still get a paging signal if encryption itself
                # fails for an unrelated reason.
                try:
                    self._fh.flush()
                except Exception:
                    pass
                try:
                    self._fh.close()
                except Exception:
                    pass
            _encrypt_session_if_needed(self._path)
        finally:
            if self._lock_fd is not None:
                _drop_session_lock(self._lock_fd, self._path)
                self._lock_fd = None

    def __enter__(self) -> "EncryptingSessionWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class _PlainSessionWriter:
    """Drop-in stand-in for :class:`EncryptingSessionWriter` when encryption is off.

    No lockfile, no encrypt-on-close — just a thin wrapper over an append-mode
    file handle so callers can use the same context-manager API.
    """

    def __init__(self, path: PathLike, *, encoding: str = "utf-8") -> None:
        self._path = Path(path)
        # same dir/file hardening as EncryptingSessionWriter
        # so the off-encryption code path doesn't ship the world-readable bug.
        harden_dir(self._path.parent)
        self._fh = open_private_append(self._path, encoding=encoding)
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def name(self) -> str:
        return self._path.name

    def write(self, text: str) -> int:
        return self._fh.write(text)

    def writelines(self, lines: Iterable[str]) -> None:
        self._fh.writelines(lines)

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "_PlainSessionWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


SessionWriter = Union[EncryptingSessionWriter, _PlainSessionWriter]


def build_session_writer(
    path: PathLike, *, encoding: str = "utf-8"
) -> SessionWriter:
    """Return a session writer suited to the current config.

    When ``security.encryption.encrypt_logs`` is active, returns an
    :class:`EncryptingSessionWriter` that encrypts on close. Otherwise returns
    a plain append writer with the same context-manager API.
    """
    if _logs_encryption_active():
        return EncryptingSessionWriter(path, encoding=encoding)
    return _PlainSessionWriter(path, encoding=encoding)


def _logs_encryption_active() -> bool:
    from . import logs_encryption_active

    return logs_encryption_active()


# ─── Read-back helpers (mirror log_handler) ─────────────────────────────────


def read_session_bytes(path: PathLike) -> bytes:
    """Return cleartext session bytes from *path* (decrypting HRMSENC sessions)."""
    raw = Path(path).read_bytes()
    if not is_encrypted(raw):
        return raw
    from . import envelope, get_data_key

    return envelope.decrypt(raw, get_data_key())


def read_session_text(path: PathLike, *, encoding: str = "utf-8") -> str:
    """Return cleartext session text from *path*."""
    return read_session_bytes(path).decode(encoding, errors="replace")
