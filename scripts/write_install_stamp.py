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

    # Docker (no .git, commit known from CI):
    python scripts/write_install_stamp.py --output install-stamp.json \\
        --source ci --distribution docker
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

# Bootstrap the repo root onto sys.path so the canary tag shape can come
# from hermes_cli.update_channel — the single authority — instead of a
# re-typed regex (hermes_cli/__init__.py is import-light).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_cli import update_channel  # noqa: E402

STAMP_SCHEMA_VERSION = 2
_REPO_ROOT = Path(__file__).parent.parent.resolve()

# Who applies the next update to the tree this stamp describes. REQUIRED —
# stamp readers hard-fail a stamp without it.
#   self             — `hermes update` owns the tree (installer-created
#                      source checkouts).
#   electron-updater — the in-app updater replaces the artifact (NSIS,
#                      mac .app, AppImage).
#   external         — something else replaces the artifact wholesale
#                      (nix, docker, MSIX / app stores).
UPDATE_MECHANISMS = ("self", "electron-updater", "external")

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
        text = (_REPO_ROOT / "hermes_cli" / "__init__.py").read_text(encoding="utf-8-sig")
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
    update_mechanism: str,
    commit: str | None = None,
    branch: str | None = None,
    dirty: bool | None = None,
    base_version: str | None = None,
    distance: int | None = None,
    commit_date: int | None = None,
    source: str = "local",
    distribution: str | None = None,
    runtime_dir: str | None = None,
) -> dict:
    """Build a stamp dict from explicit args, filling gaps from git/env.

    Args override detection — an explicit ``commit`` is used directly.
    ``source`` identifies where the stamp came from (``ci``, ``local``,
    ``docker``, ``nix``, ``fallback``). ``update_mechanism`` is required:
    every stamp names who applies the next update (see UPDATE_MECHANISMS).

    ``runtime_dir`` is a path RELATIVE to the stamp file's directory, naming
    where the pm store (the managed tool bytes) lives. A sealed desktop-app
    payload stages its runtime dir as a sibling of ``repo/`` (the install
    root), so the value is ``..``. The Python boot path reads this instead
    of deriving ``<install_root>/.hermes-runtime``, which is wrong for that
    layout: the runtime dir is the payload dir, not a child of the install
    root. ``..`` keeps it relocatable — an absolute path would break under
    MSIX translocation.
    """
    if update_mechanism not in UPDATE_MECHANISMS:
        raise SystemExit(
            f"write_install_stamp: invalid --update-mechanism {update_mechanism!r} "
            f"(expected one of {', '.join(UPDATE_MECHANISMS)})"
        )
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

    # The desktop artifact kind, from the one build-time selector
    # HERMES_DESKTOP_VARIANT. Every stamp carries it:
    #   bootstrap — no runtime in the artifact; first launch bootstraps a
    #               local install. The default (variable unset/empty; also
    #               the value for non-desktop stamps, where it is inert).
    #   bundled   — the agent runtime ships inside the artifact resources.
    #   light     — NO runtime at all, remote connections only. A Python
    #               process must never read a light stamp: the artifact
    #               contains no Python (the stamp readers raise on it).
    #   store     — a Store-submission build: the SAME bundled payload, but a
    #               different MSIX packaging identity. Stamps as 'bundled'
    #               so the bundled shape logic (shared userData, steward-
    #               owned updates, no in-app updater) holds for it too.
    # bundled, light and store all pin a release tag: electron-updater keys on
    # it, so a tagless artifact of either kind cannot update itself.
    variant = os.environ.get("HERMES_DESKTOP_VARIANT", "").strip()
    if variant not in ("", "bootstrap", "bundled", "light", "store"):
        raise SystemExit(
            f"write_install_stamp: unknown HERMES_DESKTOP_VARIANT {variant!r} "
            "(expected unset, 'bootstrap', 'bundled', 'light', or 'store')"
        )
    payload = "bundled" if variant == "store" else (variant or "bootstrap")
    tag = os.environ.get("HERMES_PAYLOAD_TAG") or None
    # Stable (vX.Y.Z) and canary (vX.Y.<any patch>-canary.<YYYYMMDDHHMMSS>,
    # or the legacy date-only shape) tags are both release-feed keys; anything
    # else cannot update itself and refuses. The canary shape is validated by
    # hermes_cli.update_channel — the single authority — so it accepts the
    # patch+1 form scripts/release.py's canary_tag_for_date produces.
    _stable_tag = re.compile(r"^v(0|[1-9]\d{0,2})\.\d+\.\d+$")
    if payload != "bootstrap" and not (
        tag and (_stable_tag.match(tag) or update_channel.is_canary_tag(tag))
    ):
        raise SystemExit(
            f"write_install_stamp: HERMES_DESKTOP_VARIANT={payload} requires "
            f"HERMES_PAYLOAD_TAG=vX.Y.Z or vX.Y.<n>-canary.YYYYMMDDHHMMSS (got {tag!r})"
        )

    stamp = {
        "schemaVersion": STAMP_SCHEMA_VERSION,
        "commit": commit,
        "commitDate": commit_date,
        "branch": branch,
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "dirty": dirty,
        "source": source,
        "distribution": distribution,
        "updateMechanism": update_mechanism,
        "baseVersion": base_version,
        "displayVersion": display_version,
        "distance": distance,
        "payload": payload,
        "tag": tag if payload != "bootstrap" else None,
    }
    if runtime_dir is not None:
        stamp["runtimeDir"] = runtime_dir
    return stamp


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
    parser.add_argument(
        "--distribution",
        choices=("docker", "nix", "desktop-app"),
        help="Package distribution (the steward that replaces this tree)",
    )
    parser.add_argument(
        "--update-mechanism",
        required=True,
        choices=UPDATE_MECHANISMS,
        help="Who applies the next update: 'self' (hermes update), "
        "'electron-updater' (in-app updater), 'external' (nix/docker/store)",
    )
    parser.add_argument(
        "--runtime-dir",
        default=None,
        help="Path RELATIVE to the stamp file's directory, naming where the "
        "pm store (the managed tool bytes) lives. A sealed desktop-app "
        "payload passes '..' (the payload dir is the install root's "
        "parent). Omitted for source/docker/nix stamps.",
    )
    args = parser.parse_args()

    stamp = write_stamp(
        args.output,
        update_mechanism=args.update_mechanism,
        commit=args.commit,
        branch=args.branch,
        dirty=args.dirty,
        base_version=args.base_version,
        distance=args.distance,
        commit_date=args.commit_date,
        source=args.source,
        distribution=args.distribution,
        runtime_dir=args.runtime_dir,
    )

    commit_short = stamp["commit"][:12]
    branch_str = f" ({stamp['branch']})" if stamp["branch"] else ""
    dirty_str = " [DIRTY]" if stamp["dirty"] else ""
    fallback_str = " [FALLBACK]" if stamp["source"] == "fallback" else ""
    print(f"[write_install_stamp] wrote {args.output} -> {commit_short}{branch_str}{dirty_str}{fallback_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
