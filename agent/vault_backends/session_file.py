"""Secure, bounded reads of an opt-in externally maintained Bitwarden session file."""

from __future__ import annotations

import errno
import hashlib
import os
import secrets
import stat
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

_Fingerprint = tuple[int, int, int, int, bytes]
_MAX_BYTES = 4096


@dataclass(frozen=True, slots=True)
class SessionFileRead:
    token: str = field(repr=False)
    fingerprint: _Fingerprint = field(repr=False)


class SessionFileUnavailable(Exception):
    def __init__(self, detail: str):
        super().__init__(f"Bitwarden session file unavailable: {detail}")
        self.detail = detail


class _VerificationOps(NamedTuple):
    digest: Callable[[bytes], bytes]
    is_verified: Callable[[str, _Fingerprint], bool]
    mark_verified: Callable[[str, _Fingerprint], None]
    forget: Callable[[str], None]


def _verification_store() -> _VerificationOps:
    # Keep the digest key and validation records out of module-state dumps, like interactive tokens.
    key = secrets.token_bytes(32)
    verified: dict[tuple[str, str], _Fingerprint] = {}
    guard = threading.Lock()

    def cache_key(path: str) -> tuple[str, str]:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home()), path

    def digest(data: bytes) -> bytes:
        return hashlib.blake2b(data, key=key, digest_size=16).digest()

    def is_verified(path: str, fingerprint: _Fingerprint) -> bool:
        with guard:
            return verified.get(cache_key(path)) == fingerprint

    def mark_verified(path: str, fingerprint: _Fingerprint) -> None:
        with guard:
            verified[cache_key(path)] = fingerprint

    def forget(path: str) -> None:
        with guard:
            verified.pop(cache_key(path), None)

    return _VerificationOps(digest, is_verified, mark_verified, forget)


_verification = _verification_store()
is_verified = _verification.is_verified
mark_verified = _verification.mark_verified
forget = _verification.forget


def _check_stat(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise SessionFileUnavailable("not_regular_file")
    if info.st_uid != os.geteuid():  # windows-footgun: ok — only reached after read_session_file's POSIX gate
        raise SessionFileUnavailable("wrong_owner")
    if info.st_mode & 0o077:
        raise SessionFileUnavailable("insecure_permissions")
    if info.st_size > _MAX_BYTES:
        raise SessionFileUnavailable("too_large")


def _decode_token(data: bytes) -> str:
    # Raise outside the handler: even with ``from None`` the UnicodeDecodeError stays on __context__,
    # and its ``.object`` is the whole payload.
    try:
        token = data.decode("utf-8")
    except UnicodeDecodeError:
        token = None
    if token is None:
        raise SessionFileUnavailable("malformed")
    token = token.removesuffix("\r\n") if token.endswith("\r\n") else token.removesuffix("\n")
    if not token:
        raise SessionFileUnavailable("empty")
    if any(unicodedata.category(char) == "Cc" for char in token):
        raise SessionFileUnavailable("malformed")
    return token


def read_session_file(path: str) -> SessionFileRead:
    """Validate and read the same fd: a concurrent path replacement cannot swap the checked file."""
    if os.name != "posix":
        raise SessionFileUnavailable("unsupported_platform")
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        raise SessionFileUnavailable("not_absolute")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as exc:
        detail = {errno.ELOOP: "symlink", errno.ENOENT: "missing"}.get(exc.errno, "unreadable")
        raise SessionFileUnavailable(detail) from None
    except ValueError:  # an embedded NUL in the configured path
        raise SessionFileUnavailable("invalid_path") from None
    try:
        info = os.fstat(fd)
        _check_stat(info)
        data = bytearray()
        # Bound actual reads too: the file may grow after fstat, or os.read may return a short chunk.
        while len(data) <= _MAX_BYTES:
            chunk = os.read(fd, _MAX_BYTES + 1 - len(data))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > _MAX_BYTES:
            raise SessionFileUnavailable("too_large")
    except OSError:
        raise SessionFileUnavailable("unreadable") from None
    finally:
        os.close(fd)
    payload = bytes(data)
    token = _decode_token(payload)
    fingerprint = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, _verification.digest(payload))
    return SessionFileRead(token, fingerprint)
