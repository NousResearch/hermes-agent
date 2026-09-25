"""Compare stable and canary versions accepted by release feeds.

Canary identity uses SemVer build metadata (``X.Y.Z+canary.<stamp>``) and
compares equal to its stable. Channel records, not SemVer precedence, decide
which canary is newer.
"""
from __future__ import annotations

import re

from hermes_cli.update_channel import STABLE_TAG_RE

_CANARY_VERSION_RE = re.compile(r"^[0-9.]+[+]canary[.]20\d{6}T\d{6}Z$")


def is_canary_version(version: str) -> bool:
    """True for the canonical build-metadata canary identity."""
    if not isinstance(version, str) or not _CANARY_VERSION_RE.fullmatch(version):
        return False
    return bool(STABLE_TAG_RE.fullmatch("v" + version.split("+", 1)[0]))


def is_stable_version(version: str) -> bool:
    """True for a final stable version."""
    return isinstance(version, str) and bool(STABLE_TAG_RE.fullmatch("v" + version))


def is_valid_version(version: str) -> bool:
    """True for a stable version or canonical canary identity."""
    return is_stable_version(version) or is_canary_version(version)


def is_release_version(version: str) -> bool:
    """True for a stable version or canonical canary identity."""
    return is_valid_version(version)


def _core(version: str) -> list[int]:
    return [int(part) for part in version.split("+", 1)[0].split(".")]


def compare(a: str, b: str) -> int:
    """Compare release versions while ignoring build metadata."""
    if not is_release_version(a) or not is_release_version(b):
        raise ValueError(f"invalid release version(s): {a!r}, {b!r}")
    ka, kb = _core(a), _core(b)
    return (ka > kb) - (ka < kb)
