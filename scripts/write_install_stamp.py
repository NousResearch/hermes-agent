"""Generate canonical install-stamp.json for packaged Hermes builds.

All packagers (Docker, Nix, desktop) call this script to produce the same
``install-stamp.json`` file. Runtime surfaces (CLI, TUI, desktop) read the
stamp through ``hermes_cli.version_info`` — no env vars, no separate
docker/nix code paths.

Usage::

    # From a repo root with .git available (dev/CI builds):
    python scripts/write_install_stamp.py --output /path/to/install-stamp.json

    # Override provenance for reproducible/packaged builds:
    python scripts/write_install_stamp.py --output ... \\
        --commit <sha> --branch <name> --dirty \\
        --base-version 0.19.0 --distance 42 --source nix --distribution nix

    # Docker (no .git, commit known from build arg):
    python scripts/write_install_stamp.py --output /opt/hermes/.hermes_build_info.json \\
        --commit ${HERMES_GIT_SHA} --source ci --distribution docker
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

STAMP_SCHEMA_VERSION = 2
_REPO_ROOT = Path(__file__).parent.parent.resolve()

# Hermes's historical tags use a four-digit calendar year as their major
# component (for example v2026.7.20). Restrict release majors to three digits
# so these date tags cannot masquerade as the v0.x.y SemVer boundaries.
_SEMVER_TAG_RE = re.compile(r"^v(0|[1-9]\d{0,2})\.(\d+)\.(\d+)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_LEGACY_CALVER_TAG_RE = re.compile(r"^v20\d{2}\.\d+\.\d+(?:\.\d+)?$")

FALLBACK_COMMIT = "0" * 40


def _run_git(*args: str, cwd: str | Path = _REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, cwd=str(cwd)
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (result.stdout or "").strip()
    return value if result.returncode == 0 and value else None


def _parse_release_metadata() -> tuple[str | None, str | None]:
    """Read __version__ and __release_date__ from hermes_cli/__init__.py."""
    try:
        text = (_REPO_ROOT / "hermes_cli" / "__init__.py").read_text(encoding="utf-8")
    except OSError:
        return None, None
    version = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    date = re.search(r'__release_date__\s*=\s*["\']([^"\']+)["\']', text)
    return (version.group(1) if version else None, date.group(1) if date else None)


def _resolve_commit_from_env() -> str | None:
    """CI builds pass the commit via $GITHUB_SHA."""
    return os.environ.get("GITHUB_SHA") or None


def _resolve_commit_from_git() -> str | None:
    return _run_git("rev-parse", "HEAD")


def _resolve_branch_from_env() -> str | None:
    return os.environ.get("GITHUB_REF_NAME") or os.environ.get("GITHUB_HEAD_REF") or None


def _resolve_branch_from_git() -> str | None:
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    return branch if branch and branch != "HEAD" else None


def _resolve_commit_date_from_git() -> int | None:
    """Return the commit timestamp (Unix epoch seconds) of HEAD, or None."""
    raw = _run_git("log", "-1", "--format=%ct", "HEAD")
    if raw and raw.isdigit():
        return int(raw)
    return None


def _resolve_dirty_from_git() -> bool:
    status = _run_git("status", "--porcelain", "-uno")
    return status is not None and len(status) > 0


def _compute_distance(base_version: str | None, release_date: str | None) -> int | None:
    """Count commits since the release tag, trying SemVer then CalVer fallback."""
    if not base_version:
        return None

    # Try SemVer tag first, then legacy CalVer tag.
    for tag in (f"v{base_version}", f"v{release_date}" if release_date else None):
        if not tag:
            continue
        raw = _run_git("rev-list", "--count", f"{tag}..HEAD")
        if raw is None:
            continue
        try:
            count = int(raw)
        except ValueError:
            continue
        if count >= 0:
            return count
    return None


def build_stamp(
    *,
    commit: str | None = None,
    branch: str | None = None,
    dirty: bool | None = None,
    base_version: str | None = None,
    distance: int | None = None,
    commit_date: int | None = None,
    source: str = "local",
    distribution: str | None = None,
) -> dict:
    """Build a stamp dict from explicit args, filling gaps from git/env.

    Args override detection — if you pass ``commit``, it's used directly.
    ``source`` identifies where the stamp came from (``ci``, ``local``,
    ``docker``, ``nix``, ``fallback``).
    """
    _base_version, _release_date = _parse_release_metadata()
    if base_version is None:
        base_version = _base_version

    # Commit: explicit > CI env > git
    if commit is None:
        commit = _resolve_commit_from_env()
        source = "ci" if commit else source
    if commit is None:
        commit = _resolve_commit_from_git()
        source = "local" if commit else source
    if not commit:
        commit = FALLBACK_COMMIT
        source = "fallback"

    # Branch: explicit > CI env > git
    if branch is None:
        branch = _resolve_branch_from_env()
    if branch is None:
        branch = _resolve_branch_from_git()

    # Dirty: explicit > git
    if dirty is None:
        dirty = _resolve_dirty_from_git()

    # Distance: explicit > computed from git
    if distance is None:
        distance = _compute_distance(base_version, _release_date)

    # Commit date: explicit > git
    if commit_date is None:
        commit_date = _resolve_commit_date_from_git()

    # Display version
    display_version = base_version or ""
    if distance is not None and distance > 0:
        display_version = f"{display_version}+{distance}"
    elif dirty and distance is None:
        display_version = f"{display_version}+?"

    return {
        "schemaVersion": STAMP_SCHEMA_VERSION,
        "commit": commit,
        "commitDate": commit_date,
        "branch": branch,
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "dirty": dirty,
        "source": source,
        "distribution": distribution,
        "baseVersion": base_version,
        "displayVersion": display_version,
        "distance": distance,
    }


def write_stamp(output: str | Path, **kwargs) -> dict:
    """Build and write an install-stamp.json to ``output``. Returns the stamp."""
    stamp = build_stamp(**kwargs)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return stamp


def main() -> int:
    parser = argparse.ArgumentParser(description="Write install-stamp.json")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--commit", default=None, help="Override commit SHA")
    parser.add_argument("--branch", default=None, help="Override branch name")
    parser.add_argument("--dirty", action="store_true", default=None, help="Mark as dirty")
    parser.add_argument("--base-version", default=None, help="Override base version")
    parser.add_argument("--distance", type=int, default=None, help="Override commit distance")
    parser.add_argument("--commit-date", type=int, default=None, help="Override commit timestamp (Unix epoch seconds)")
    parser.add_argument("--source", default="local", help="Stamp source label")
    parser.add_argument("--distribution", choices=("docker", "nix"), help="Package distribution")
    args = parser.parse_args()

    stamp = write_stamp(
        args.output,
        commit=args.commit,
        branch=args.branch,
        dirty=args.dirty,
        base_version=args.base_version,
        distance=args.distance,
        commit_date=args.commit_date,
        source=args.source,
        distribution=args.distribution,
    )

    commit_short = stamp["commit"][:12]
    branch_str = f" ({stamp['branch']})" if stamp["branch"] else ""
    dirty_str = " [DIRTY]" if stamp["dirty"] else ""
    fallback_str = " [FALLBACK]" if stamp["source"] == "fallback" else ""
    print(f"[write_install_stamp] wrote {args.output} -> {commit_short}{branch_str}{dirty_str}{fallback_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
