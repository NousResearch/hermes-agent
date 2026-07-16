"""Fail-closed access to the current POSIX process identity.

The ``os.get*id`` functions do not exist on Windows.  Keeping the attribute
lookup inside these call-time helpers lets POSIX-only gateway modules remain
importable there without weakening any Linux identity check.
"""

from __future__ import annotations

import os as _os
from types import ModuleType


class PosixIdentityUnavailable(RuntimeError):
    """Raised when the host cannot provide an exact POSIX process identity."""


_OS: ModuleType = _os


def _required_identity(getter_name: str) -> int:
    getter = getattr(_OS, getter_name, None)
    if not callable(getter):
        raise PosixIdentityUnavailable("POSIX process identity is unavailable")
    try:
        value = getter()
    except (AttributeError, OSError) as exc:
        raise PosixIdentityUnavailable(
            "POSIX process identity is unavailable"
        ) from exc
    if type(value) is not int or value < 0:
        raise PosixIdentityUnavailable("POSIX process identity is invalid")
    return value


def real_uid() -> int:
    """Return the real user ID, or fail closed when it is unavailable."""

    return _required_identity("getuid")


def real_gid() -> int:
    """Return the real group ID, or fail closed when it is unavailable."""

    return _required_identity("getgid")


def effective_uid() -> int:
    """Return the effective user ID, or fail closed when it is unavailable."""

    return _required_identity("geteuid")


def effective_gid() -> int:
    """Return the effective group ID, or fail closed when it is unavailable."""

    return _required_identity("getegid")
