"""Generate canonical install-stamp.json for packaged Hermes builds.

All packagers (Docker, Nix, desktop) call this script to produce the same
``install-stamp.json`` file. Runtime surfaces (CLI, TUI, desktop) read the
stamp through ``hermes_cli.version_info`` — no env vars, no separate
docker/nix code paths.

Usage::

    # From a repo root with .git available (provenance is detected, version is not):
    python scripts/write_install_stamp.py --output /path/to/install-stamp.json \
        --base-version 0.19.0 --distance 0 --update-mechanism self

    # Override provenance for reproducible/packaged builds:
    python scripts/write_install_stamp.py --output ... \\
        --commit <sha> --branch <name> --dirty \\
        --base-version 0.19.0 --distance 42 --source nix --distribution nix

    # Docker (identity admitted by the workflow; with no reachable release the
    # commit alone is the identity and the runtime reports base "unknown"):
    python scripts/write_install_stamp.py --output install-stamp.json \\
        --commit <sha> [--base-version 0.19.0 --distance 0] \\
        --source ci --distribution docker --update-mechanism external
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
from hermes_cli.steward import UPDATE_MECHANISMS  # noqa: E402

STAMP_SCHEMA_VERSION = 2
_REPO_ROOT = Path(__file__).parent.parent.resolve()

# Who applies the next update to the tree this stamp describes. REQUIRED —
# stamp readers hard-fail a stamp without it.
#   self             — `hermes update` owns the tree (installer-created
#                      source checkouts).
#   electron-updater — the in-app updater replaces the artifact (NSIS,
#                      mac .app, AppImage).
#   app-installer    — the app hands the update to Windows App Installer.
#   external         — a package manager or app store owns updates.

# Hermes's historical tags use a four-digit calendar year as their major
# component (for example v2026.7.20). Restrict release majors to three digits
# so these date tags cannot masquerade as the v0.x.y SemVer boundaries.
_SEMVER_TAG_RE = re.compile(r"^v(0|[1-9]\d{0,2})\.(\d+)\.(\d+)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_LEGACY_CALVER_TAG_RE = re.compile(r"^v20\d{2}\.\d+\.\d+(?:\.\d+)?$")

FALLBACK_COMMIT = "0" * 40


def _run_git(*args: str, cwd: str | Path = _REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5, cwd=str(cwd),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (result.stdout or "").strip()
    return value if result.returncode == 0 and value else None



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


def build_stamp(
    *,
    update_mechanism: str,
    commit: str | None = None,
    branch: str | None = None,
    dirty: bool | None = None,
    base_version: str | None = None,
    display_version: str | None = None,
    distance: int | None = None,
    commit_date: int | None = None,
    source: str = "local",
    distribution: str | None = None,
    runtime_dir: str | None = None,
    channel_request: dict | None = None,
) -> dict:
    """Build a stamp dict from explicit args, filling gaps from git/env.

    Args override detection except for commit-build checkout verification.
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
    # Only a caller-supplied commit may stand in for a missing release version.
    commit_admitted = commit is not None
    if channel_request is not None:
        from scripts.bundles.desktop_prepare import git, require_source, validate_channel_request
        channel_request = validate_channel_request(channel_request)
        if os.environ.get("HERMES_BUILD_COMMIT") or os.environ.get("HERMES_PAYLOAD_TAG"):
            raise ValueError("channel request conflicts with commit-build or tag identity")
        if os.environ.get("HERMES_DESKTOP_VARIANT") != "bundled" or update_mechanism not in {"electron-updater", "app-installer"}:
            raise ValueError("channel builds require a bundled native update owner")
        if _run_git("rev-parse", "HEAD", cwd=_REPO_ROOT) != channel_request["commit"] or commit not in (None, channel_request["commit"]):
            raise ValueError("channel build identity does not match checkout HEAD")
        require_source(_REPO_ROOT, channel_request["commit"])
        commit, source, base_version = channel_request["commit"], "channel-build", channel_request["sourceVersion"]
        dirty, distance = False, 0
        commit_date = int(git(_REPO_ROOT, "log", "-1", "--format=%ct", "HEAD"))

    commit_build = os.environ.get("HERMES_BUILD_COMMIT")
    if commit_build:
        from scripts.releases.commit_build import require_commit

        require_commit(commit_build)
        if os.environ.get("HERMES_PAYLOAD_TAG"):
            raise ValueError("Commit builds cannot also select a release tag")
        if _resolve_commit_from_git() != commit_build or commit not in (None, commit_build):
            raise ValueError("Commit build identity does not match the checkout HEAD")
        commit, source, update_mechanism = commit_build, "commit-build", "external"

    # A dispatch SHA describes workflow code, not an admitted build checkout.
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
    if commit_build or channel_request is not None:
        branch = None
    elif branch is None:
        branch = _resolve_branch_from_env()
    if branch is None and not commit_build and channel_request is None:
        branch = _resolve_branch_from_git()

    # Dirty: explicit > git
    if dirty is None:
        dirty = _resolve_dirty_from_git()

    if base_version is None and not commit_admitted:
        raise ValueError(
            "install stamps require an explicit base version or commit; leave local development "
            "trees unstamped so runtime identity can come from Git"
        )
    if base_version is None and display_version is None:
        # No reachable release tag: record the commit, the way a tagless checkout reads from Git.
        display_version = f"git.{commit[:7]}{'.dirty' if dirty else ''}"

    # Commit date: explicit > git
    if commit_date is None:
        commit_date = _resolve_commit_date_from_git()

    # Display version
    if display_version is None:
        display_version = base_version
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
    #   runtime   — a self-contained CLI runtime with NO Electron app around
    #               it (the Termux .deb). It is sealed like 'bundled' but
    #               must not stamp as one: 'bundled' readers locate an
    #               enclosing desktop app (bundled_app.resolve_bundle_layout)
    #               and a tree without one is damage to them.
    # Release artifacts pin a tag. Commit builds never enter an update channel.
    variant = os.environ.get("HERMES_DESKTOP_VARIANT", "").strip()
    if variant not in ("", "bootstrap", "bundled", "light", "store", "runtime"):
        raise SystemExit(
            f"write_install_stamp: unknown HERMES_DESKTOP_VARIANT {variant!r} "
            "(expected unset, 'bootstrap', 'bundled', 'light', 'store', or 'runtime')"
        )
    payload = "bundled" if variant == "store" else (variant or "bootstrap")
    tag = os.environ.get("HERMES_PAYLOAD_TAG") or None

    _stable_tag = re.compile(r"^v(0|[1-9]\d{0,2})\.\d+\.\d+$")
    if payload != "bootstrap" and not commit_build and channel_request is None and not (
        tag and (_stable_tag.match(tag) or update_channel.is_canary_tag(tag))
    ):
        raise SystemExit(
            f"write_install_stamp: HERMES_DESKTOP_VARIANT={payload} requires "
            f"HERMES_PAYLOAD_TAG=vX.Y.Z or vX.Y.Z+canary.YYYYMMDDTHHMMSSZ (got {tag!r})"
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
    if channel_request is not None:
        stamp["channelBuild"] = channel_request
        stamp["displayVersion"] = f'{base_version} ({channel_request["channel"]} #{channel_request["sequence"]}, {commit[:7]})'
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
    parser.add_argument("--channel-request", type=Path, help="Immutable admitted channel request JSON")
    parser.add_argument("--commit", default=None, help="Override commit SHA")
    parser.add_argument("--branch", default=None, help="Override branch name")
    parser.add_argument("--dirty", action="store_true", default=None, help="Mark as dirty")
    parser.add_argument("--base-version", default=None, help="Override base version")
    parser.add_argument("--display-version", default=None, help="Exact user-facing version")
    parser.add_argument("--distance", type=int, default=None, help="Override commit distance")
    parser.add_argument("--commit-date", type=int, default=None, help="Override commit timestamp (Unix epoch seconds)")
    parser.add_argument("--source", default="local", help="Stamp source label")
    parser.add_argument(
        "--distribution",
        choices=("docker", "nix", "desktop-app", "apt-termux"),
        help="Package distribution (the steward that replaces this tree)",
    )
    parser.add_argument(
        "--update-mechanism",
        required=True,
        choices=UPDATE_MECHANISMS,
        help="Who applies the next update: 'self' (hermes update), "
        "'app-installer' (Windows App Installer), 'electron-updater' (in-app updater), "
        "'external' (nix/docker/store)",
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
        display_version=args.display_version,
        distance=args.distance,
        commit_date=args.commit_date,
        source=args.source,
        distribution=args.distribution,
        runtime_dir=args.runtime_dir,
        channel_request=json.loads(args.channel_request.read_text(encoding="utf-8-sig")) if args.channel_request else None,
    )

    commit_short = stamp["commit"][:12]
    branch_str = f" ({stamp['branch']})" if stamp["branch"] else ""
    dirty_str = " [DIRTY]" if stamp["dirty"] else ""
    fallback_str = " [FALLBACK]" if stamp["source"] == "fallback" else ""
    print(f"[write_install_stamp] wrote {args.output} -> {commit_short}{branch_str}{dirty_str}{fallback_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
