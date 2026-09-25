"""Ref name to version, and the next version from the published stable head.

A version is spent only by publication. Attempt refs (``rc.<N>-vX.Y.Z``) and
their abandon markers number attempts within one version; they never move the
line. CalVer tags, canary identities and receipt namespaces are not versions.
"""
from __future__ import annotations

import json
import re
import subprocess
from functools import cmp_to_key

from hermes_cli.update_channel import STABLE_TAG_RE
from scripts.releases.semver import compare

SEED = "0.21.4"
BUMPS = ("major", "minor", "patch")


def published_channel_identity(repository: str, channel: str, *, base_url: str | None = None,
                               reader_type=None) -> tuple[str, str] | None:
    """Resolve one validated protected channel's payload version and commit."""
    from hermes_cli.release_channels import ChannelNotFound, ChannelReader
    from scripts.releases.r2 import public_base_url

    if channel not in {"stable", "canary"}:
        raise ValueError("Expected a protected release channel")
    reader_type = reader_type or ChannelReader
    try:
        resolved = reader_type(base_url or public_base_url(), repository=repository).resolve(channel)
    except ChannelNotFound:
        return None
    if resolved.manifest is None:
        return None
    if resolved.terminal.get("policy") != f"{channel}-release":
        raise ValueError(f"{channel.title()} channel has the wrong publication policy")
    request = resolved.manifest["request"]
    version, commit = request.get("version"), request.get("commit")
    if not isinstance(version, str) or not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError(f"{channel.title()} channel has an invalid published identity")
    return version, commit


def published_release_identity(repository: str, run=None) -> tuple[str, str] | None:
    """The newest published stable GitHub release and its commit, or None.

    Publication makes a release public only after its final tag and draft edits
    read back, so this covers every published release. That includes a release
    that skipped bundles, which never moves the protected channel. A bare
    ``vX.Y.Z`` tag without a published release does not count.
    """
    run = run or _gh
    pages = json.loads(run(["gh", "api", "--paginate", "--slurp",
                            f"repos/{repository}/releases?per_page=100"]))
    versions = [version for page in pages for row in page
                if isinstance(row, dict) and row.get("draft") is False
                and row.get("prerelease") is False
                and (version := version_from_tag(row.get("tag_name")))]
    if not versions:
        return None
    newest = max(versions, key=cmp_to_key(compare))
    commit = run(["gh", "api", f"repos/{repository}/commits/v{newest}", "--jq", ".sha"]).strip()
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError(f"v{newest} has no valid release commit")
    return newest, commit


def published_stable_identity(repository: str, *, base_url: str | None = None,
                              reader_type=None, run=None) -> tuple[str, str | None]:
    """The newest published stable ``(version, commit)``, or the seed before one exists.

    The protected channel names the newest release that shipped bundles. The
    GitHub releases also name one that skipped them. The newer of the two wins.
    """
    found = published_channel_identity(
        repository, "stable", base_url=base_url, reader_type=reader_type,
    )
    if found is not None and version_from_tag("v" + found[0]) is None:
        raise ValueError("Stable channel has an invalid source version")
    candidates = [identity for identity in (found, published_release_identity(repository, run))
                  if identity is not None]
    if not candidates:
        return SEED, None
    return max(candidates, key=cmp_to_key(lambda a, b: compare(a[0], b[0])))


def published_stable_version(repository: str, *, base_url: str | None = None, reader_type=None,
                             run=None) -> str:
    return published_stable_identity(repository, base_url=base_url, reader_type=reader_type,
                                     run=run)[0]


def _gh(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True, encoding="utf-8")


def version_from_tag(ref: str) -> str | None:
    """The version a final release tag names, or None for anything else.

    ``-rc`` claims, build-metadata identities and non-``v`` receipt namespaces
    are not final tags, and a 4-digit major is a CalVer label.
    """
    if not isinstance(ref, str) or not STABLE_TAG_RE.fullmatch(ref):
        return None
    return ref[1:]


# The ref grammar only anchors; ``version_from_tag`` owns the version shape.
_ATTEMPT_REF_RE = re.compile(r"rc\.(?P<attempt>[1-9][0-9]*)-(?P<tag>v[^/]+)")
_MARKER_PREFIX = "abandoned-"


def parse_attempt_ref(ref: str) -> tuple[str, int] | None:
    """``rc.<N>-vX.Y.Z`` to ``(version, N)``, or None for anything else.

    The attempt comes first so the ref can never read as a SemVer prerelease
    of the version it claims.
    """
    match = _ATTEMPT_REF_RE.fullmatch(ref) if isinstance(ref, str) else None
    if match is None:
        return None
    version = version_from_tag(match.group("tag"))
    return (version, int(match.group("attempt"))) if version else None


def parse_marker_ref(ref: str) -> tuple[str, int] | None:
    """``abandoned-rc.<N>-vX.Y.Z``, the record that clears one attempt."""
    if not isinstance(ref, str) or not ref.startswith(_MARKER_PREFIX):
        return None
    return parse_attempt_ref(ref[len(_MARKER_PREFIX):])


def attempt_ref(version: str, attempt: int) -> str:
    ref = f"rc.{attempt}-v{version}"
    if parse_attempt_ref(ref) != (version, attempt):
        raise ValueError(f"invalid attempt ref: {version!r} attempt {attempt!r}")
    return ref


def marker_ref(version: str, attempt: int) -> str:
    return _MARKER_PREFIX + attempt_ref(version, attempt)


def _bump(version: str, bump: str) -> str:
    if bump not in BUMPS:
        raise ValueError(f"unknown bump {bump!r}")
    major, minor, patch = (int(part) for part in version.split("."))
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def derive_next_version(*, published: str | None, bump: str) -> str:
    """The next version from the published head, or the seed before one exists.

    Attempts do not move the line. A version is spent by publication, so an
    abandoned attempt leaves its version free for the next cut.
    """
    return _bump(published or SEED, bump)


def next_attempt(version: str, refs: list[str]) -> int:
    """One past the highest attempt ref for ``version``.

    Marker refs do not count and do not free a number: an abandoned attempt
    keeps its ref, so its number is never cut again.
    """
    attempts = [parsed[1] for parsed in map(parse_attempt_ref, refs)
                if parsed and parsed[0] == version]
    return max(attempts, default=0) + 1


def outstanding_attempts(refs: list[str], is_published) -> list[tuple[str, int, str]]:
    """The one definition of an outstanding attempt.

    An attempt ref is outstanding when its abandon marker ref does not exist
    and its version has no final tag. ``is_published(version)`` answers whether
    the final tag ``v{version}`` exists, so callers keep their own remote
    lookup. Both the release entrypoint and the sequencer refuse more than one
    of these, across all versions.
    """
    cleared = {parsed for parsed in map(parse_marker_ref, refs) if parsed}
    published: dict[str, bool] = {}
    outstanding: list[tuple[str, int, str]] = []
    for ref in refs:
        parsed = parse_attempt_ref(ref)
        if parsed is None or parsed in cleared:
            continue
        version, attempt = parsed
        if version not in published:
            published[version] = bool(is_published(version))
        if not published[version]:
            outstanding.append((version, attempt, ref))
    return outstanding
