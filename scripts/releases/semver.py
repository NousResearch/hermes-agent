"""Compare the stable and canary versions accepted by release feeds."""
from __future__ import annotations

import re
from hermes_cli.update_channel import _CANARY_TAG_RE

STABLE_TAG = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def is_valid_version(version: str) -> bool:
    """semver.valid restricted to our release grammar (no build metadata,
    no alphanumeric prerelease identifiers — those never appear in
    generated tags)."""
    if not isinstance(version, str):
        return False
    core, sep, tail = version.partition("-")
    if sep and not STABLE_TAG.fullmatch("v" + core):
        return False
    if not sep:
        return bool(STABLE_TAG.fullmatch("v" + version))
    if not _CANARY_TAG_RE.fullmatch("v" + version):
        return False
    # Canary timestamp is a fixed-length 14-digit numeric stamp.
    stamp = tail.split(".", 1)[1]
    return len(stamp) == 14


def _prerelease_key(tail: str) -> list[int]:
    """Numeric sort key for a canary suffix — 'canary.<14 digits>'."""
    return [int(tail.split(".", 1)[1])]


def compare(a: str, b: str) -> int:
    """semver.compare for our grammar. Both sides must be valid release
    versions (ValueError otherwise — feed publication fails loudly).
    Semver ordering: stable 0.28.0 > any 0.28.0-canary.<stamp>; stamps
    compare numerically."""
    if not is_valid_version(a) or not is_valid_version(b):
        raise ValueError(f"invalid release version(s): {a!r}, {b!r}")
    a_core, _, a_tail = a.partition("-")
    b_core, _, b_tail = b.partition("-")
    if a_core != b_core:
        ka = [int(p) for p in a_core.split(".")]
        kb = [int(p) for p in b_core.split(".")]
        return -1 if ka < kb else 1
    # Same core: a prerelease (canary) sorts BEFORE the stable release.
    if a_tail and b_tail:
        ka, kb = _prerelease_key(a_tail), _prerelease_key(b_tail)
        return -1 if ka < kb else (1 if ka > kb else 0)
    if a_tail:
        return -1
    if b_tail:
        return 1
    return 0
