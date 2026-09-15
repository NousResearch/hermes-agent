#!/usr/bin/env python3
"""Derive a Debian package version (or channel) from a hermes-agent release tag.

Pure function; imported by scripts/termux/build_deb.sh and unit-tested by
tests/test_termux_deb_version.py (Task 4 of .hermes/plans/2026-08-31_termux-deb.md).

Mapping:
    v1.2.3                     -> 1.2.3-1
    v1.2.3-canary.2026083112  -> 1.2.3~canary.2026083112-1

The ``~`` ranks the nightly below the corresponding stable in dpkg's version
ordering. The major version is capped at 3 digits (CalVer-style cap): a tag
with a 4+ digit major is rejected as malformed.

``--channel`` derives the release channel from the SAME tag regex: a tag with
a nightly timestamp is ``nightly``, everything else is ``stable``. This is the
single source of truth for the channel; workflows and other tooling must call
this instead of re-typing a case statement.
"""

from __future__ import annotations

import re
import sys

# The canary timestamp shape MUST match the canonical _CANARY_TAG_RE in
# hermes_cli/update_channel.py (exactly 8 or 14 digits, 20-prefixed) and
# channel_for_tag in scripts/releases/r2.py. Cross-referenced by
# tests/test_termux_deb_version.py::test_canary_tag_shape_matches_canonical.
_TAG_RE = re.compile(
    r"^v(?P<major>0|[1-9]\d{0,2})\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-canary\.(?P<ts>20\d{6}(?:\d{6})?))?$"
)


def _match_tag(tag: str) -> re.Match[str]:
    m = _TAG_RE.match(tag)
    if m is None:
        raise ValueError(
            f"malformed release tag {tag!r}: expected v<MAJOR>.<MINOR>.<PATCH> "
            "or v<MAJOR>.<MINOR>.<PATCH>-canary.<timestamp>"
        )
    return m


def deb_version_for_tag(tag: str) -> str:
    """Map a release tag to its Debian version. Raises ValueError on malformed tags."""
    m = _match_tag(tag)
    base = f"{m.group('major')}.{m.group('minor')}.{m.group('patch')}"
    ts = m.group("ts")
    if ts is None:
        return f"{base}-1"
    return f"{base}~canary.{ts}-1"


def channel_for_tag(tag: str) -> str:
    """Map a release tag to its channel: 'canary' or 'stable'.

    Derived from the same _TAG_RE as deb_version_for_tag, so the two can never
    drift: a tag that yields a '~canary' deb version is canary, and the
    malformed-tag rejection is identical.
    """
    m = _match_tag(tag)
    return "canary" if m.group("ts") is not None else "stable"


def main(argv: list[str]) -> int:
    args = argv[1:]
    channel_mode = False
    if args and args[0] == "--channel":
        channel_mode = True
        args = args[1:]
    if len(args) != 1:
        mode = "deb_version.py --channel <tag>" if channel_mode else "deb_version.py <tag>"
        print(f"usage: {mode}", file=sys.stderr)
        return 2
    try:
        print(channel_for_tag(args[0]) if channel_mode else deb_version_for_tag(args[0]))
    except ValueError as exc:
        print(f"deb_version: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
