"""Windows must not treat POSIX root prefixes as local system paths (#131141).

``base._MEDIA_DELIVERY_DENIED_PREFIXES`` holds POSIX system paths ("/etc",
"/dev", ...) and is shared with the sandbox check in ``gateway.media_fetch``,
which compares ``PurePosixPath`` values of *remote* paths — there the POSIX list
is correct even when the host is Windows.

The LOCAL consumer ``base._media_delivery_denied_paths()`` must not use those
prefixes on Windows: ``Path("/dev").resolve()`` yields ``<SystemDrive>:\\dev``
(e.g. ``C:\\DEV``), and the case-insensitive ``WindowsPath`` comparison then
treats an entire developer tree as a denied system path, so media files below it
are dropped from delivery.

RED on Windows before the fix: the local denylist contains the resolved
``<drive>:\\dev`` entry, and the shared tuple is emptied (which also silently
disabled the sandbox protection in ``gateway.media_fetch``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gateway.platforms import base

POSIX_PREFIXES = (
    "/etc", "/proc", "/sys", "/dev", "/root", "/boot", "/var/log", "/var/lib", "/var/run",
)


def test_shared_prefix_tuple_stays_complete():
    """The shared tuple keeps the POSIX paths — the media_fetch sandbox check relies on it."""
    assert tuple(base._MEDIA_DELIVERY_DENIED_PREFIXES) == POSIX_PREFIXES


def test_sandbox_prefix_list_keeps_posix_paths():
    """``media_fetch`` converts the same tuple to PurePosixPath; it must not be empty on Windows."""
    from gateway import media_fetch

    assert tuple(str(p) for p in media_fetch._DENIED_PREFIXES) == POSIX_PREFIXES


@pytest.mark.skipif(os.name != "nt", reason="POSIX prefixes only resolve onto a drive root on Windows")
def test_local_denylist_has_no_resolved_posix_prefix():
    """No POSIX prefix may be resolved onto the local Windows drive root."""
    denied = [Path(p) for p in base._media_delivery_denied_paths()]
    resolved = {prefix: Path(prefix).resolve() for prefix in POSIX_PREFIXES}

    for prefix, resolved_path in resolved.items():
        assert resolved_path not in denied, (
            f"{prefix!r} resolved to {resolved_path} and denies that whole tree on Windows"
        )
