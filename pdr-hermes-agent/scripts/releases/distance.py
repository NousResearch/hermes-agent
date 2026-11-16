"""Dev-install version distance, counted from a release tag rather than a claim.

``git describe`` with two tags on one commit is ambiguous: which name it
returns depends on tagger dates, so the selection here never asks it to
choose. It lists the reachable tags and keeps the highest final release.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.releases.versioning import version_from_tag


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()


class NoReleaseTag(ValueError):
    """HEAD reaches no final release tag, so there is no base to count from."""


def dev_version(repo: Path) -> str:
    """``<release>+<distance>.g<short>`` from the highest reachable release tag."""
    tags = [line for line in _git(repo, "tag", "--merged", "HEAD", "--list", "v[0-9]*").splitlines() if line]
    releases = [version for version in (version_from_tag(tag) for tag in tags) if version]
    if not releases:
        raise NoReleaseTag("no reachable release tag")
    version = max(releases, key=lambda value: [int(part) for part in value.split(".")])
    distance = _git(repo, "rev-list", "--count", f"v{version}..HEAD")
    if distance == "0":
        return version
    return f"{version}+{distance}.g{_git(repo, 'rev-parse', '--short=7', 'HEAD')}"


if __name__ == "__main__":
    # Packagers stamp a commit-only identity when no release is reachable.
    try:
        print(dev_version(Path.cwd()))
    except NoReleaseTag:
        pass
