#!/usr/bin/env python3
"""Derive a Debian package version (or channel) from a hermes-agent release tag.

Pure function; imported by scripts/termux/build_deb.sh and unit-tested by
tests/scripts/test_termux_deb_version.py (Task 4 of .hermes/plans/2026-08-31_termux-deb.md).

Mapping:
    v1.2.3                              -> 1.2.3-1
    canary.20260831T120000Z-v1.2.3      -> 1.2.3~canary.20260831T120000Z-1

The ``~`` ranks the canary below the corresponding stable in dpkg's version
ordering.

``--channel`` derives the release channel from the SAME tag regex: a tag with
a nightly timestamp is ``nightly``, everything else is ``stable``. This is the
single source of truth for the channel; workflows and other tooling must call
this instead of re-typing a case statement.
"""

from __future__ import annotations

import re
import sys

_TAG_RE = re.compile(
    r"^(?:(?P<kind>canary|commit)\.(?P<ts>20\d{6}T\d{6}Z)-)?"
    r"v(?P<major>0|[1-9]\d{0,2})\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)


def _match_tag(tag: str) -> re.Match[str]:
    m = _TAG_RE.match(tag)
    if m is None:
        raise ValueError(
            f"malformed release tag {tag!r}: expected v<MAJOR>.<MINOR>.<PATCH> "
            "or canary|commit.<UTC timestamp>-v<MAJOR>.<MINOR>.<PATCH>"
        )
    return m


def deb_version_for_tag(tag: str) -> str:
    """Map a release tag to its Debian version. Raises ValueError on malformed tags."""
    m = _match_tag(tag)
    base = f"{m.group('major')}.{m.group('minor')}.{m.group('patch')}"
    ts = m.group("ts")
    if ts is None:
        return f"{base}-1"
    return f"{base}~{m.group('kind')}.{ts}-1"


def release_version_for_tag(tag: str) -> str:
    """The version a release tag carries: ``X.Y.Z``, ``X.Y.Z+canary.<stamp>`` or ``X.Y.Z+commit.<stamp>``.

    The same value as hermes_cli.update_channel.release_version; build_deb.sh stamps it as the
    package's base version and cannot import the application package.
    """
    m = _match_tag(tag)
    base = f"{m.group('major')}.{m.group('minor')}.{m.group('patch')}"
    return f"{base}+{m.group('kind')}.{m.group('ts')}" if m.group("ts") else base


def channel_for_tag(tag: str) -> str:
    """Map a release tag to its channel: 'stable', 'canary' or 'commit'.

    Derived from the same _TAG_RE as deb_version_for_tag, so the two can never
    drift: a tag that yields a '~canary' deb version is canary, and the
    malformed-tag rejection is identical. A commit build's 'commit' channel names
    only its archived APT suite; it is never published to a feed.
    """
    m = _match_tag(tag)
    return m.group("kind") or "stable"


def main(argv: list[str]) -> int:
    args = argv[1:]
    mode = deb_version_for_tag
    if args and args[0] in ("--channel", "--release-version"):
        mode = channel_for_tag if args[0] == "--channel" else release_version_for_tag
        args = args[1:]
    if len(args) != 1:
        print("usage: deb_version.py [--channel | --release-version] <tag>", file=sys.stderr)
        return 2
    try:
        print(mode(args[0]))
    except ValueError as exc:
        print(f"deb_version: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
