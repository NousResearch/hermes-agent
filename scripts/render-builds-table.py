#!/usr/bin/env python3
"""Render the release download tables into <!-- HERMES_BUILDS_TABLE -->.

Runs as the LAST job of desktop-bundled-release.yml, after every matrix
leg has uploaded, and edits the GitHub release body in place. The tables
are built from the bucket's ACTUAL object names (scripts/r2-release.mjs
list --prefix releases/tag/<tag>/), filtered to the tag's exact version —
a missing artifact shows up as a missing row, never a dead link. The
GitHub release carries the notes only; the binaries live in the R2 bucket
under releases/tag/<tag>/, and the download links point at the R2 public
URL (CLOUDFLARE_R2_PUBLIC_URL / --r2-base-url).

Tables: Hermes Desktop (bundled) and Hermes Light, one row per (OS,
arch). Feed manifests (latest*/light*/canary*.yml), blockmaps and mac .zip
(an electron-updater delta target, not a user download) stay out of the
tables on purpose; they still live in the bucket for the updater to
consume.

With --pending-run-url, renders a "builds in progress" link to the
workflow run instead of the tables. The builds-pending job runs this
mode as the first job of the run, so the draft body points at the live
run while the matrix builds. The link block keeps the marker wrapper,
so the final render replaces it.

Usage: render-builds-table.py --tag vX.Y.Z [--repo owner/repo] [--r2-base-url URL] [--dry-run]
Idempotent: re-running replaces the previously rendered block (the
marker is kept as an HTML comment wrapper around the tables).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

MARKER = "<!-- HERMES_BUILDS_TABLE -->"
END_MARKER = "<!-- /HERMES_BUILDS_TABLE -->"

# Asset name shapes (electron-builder artifactName in
# apps/desktop/electron-builder.config.cjs):
#   Hermes-0.28.0-mac-arm64.dmg        (bundled)
#   HermesBundled-0.28.0-win-x64.msix  (bundled)
_ASSET_RE = re.compile(
    r"^(?P<app>HermesBundled|HermesLight)-(?P<version>[^-]+(?:-canary\.\d{8}(?:\d{6})?)?)"
    r"-(?P<os>mac|win|linux)-(?P<arch>x64|arm64)\.(?P<ext>dmg|msix|AppImage)$"
)

_OS_LABEL = {"mac": "macOS", "win": "Windows", "linux": "Linux"}
_ARCH_LABEL = {"x64": "x64 (Intel/AMD)", "arm64": "arm64 (Apple Silicon/ARM)"}
_KIND_LABEL = {"dmg": "DMG", "msix": "MSIX", "AppImage": "AppImage"}
_ROW_ORDER = [("mac", "arm64"), ("mac", "x64"), ("win", "x64"), ("win", "arm64"),
              ("linux", "x64"), ("linux", "arm64")]


def parse_assets(names: list[str]) -> dict[str, dict[tuple[str, str], tuple[str, str]]]:
    """{app: {(os, arch): (full_key, ext)}} for table-shaped assets only.

    Object keys are releases/tag/<tag>/<filename>; the shape match runs on
    the basename, but the stored name keeps the full key so the download
    link points at the object's real location.
    """
    out: dict[str, dict[tuple[str, str], tuple[str, str]]] = {"HermesBundled": {}, "HermesLight": {}}
    for name in names:
        base = name.rsplit("/", 1)[-1]
        m = _ASSET_RE.match(base)
        if m:
            out[m.group("app")][(m.group("os"), m.group("arch"))] = (name, m.group("ext"))
    return out


def render_tables(assets_by_app: dict, base_url: str) -> str:
    """The replacement block: marker + tables + end marker."""
    base = base_url.rstrip("/")
    sections = []
    for app, title in (("HermesBundled", "Hermes Desktop"), ("HermesLight", "Hermes Light (remote-only client)")):
        rows = []
        for key in _ROW_ORDER:
            entry = assets_by_app.get(app, {}).get(key)
            if not entry:
                continue
            name, ext = entry
            os_name, arch = key
            rows.append(
                f"| {_OS_LABEL[os_name]} | {_ARCH_LABEL[arch]} "
                f"| [{_KIND_LABEL[ext]}]({base}/{name}) |"
            )
        if rows:
            sections.append(
                f"### {title}\n\n| OS | Architecture | Download |\n|---|---|---|\n"
                + "\n".join(rows)
            )
    if not sections:
        return ""
    return MARKER + "\n## Downloads\n\n" + "\n\n".join(sections) + "\n" + END_MARKER


def render_pending(run_url: str) -> str:
    """The placeholder block: a link to the run, in the same marker wrapper."""
    return (
        MARKER
        + f"\n> 🚧 [Builds in progress]({run_url}) — the download links"
        + " appear here when the build matrix finishes.\n"
        + END_MARKER
    )


def filter_names_for_version(names: list[str], version: str) -> list[str]:
    """Table-shaped names whose embedded version equals `version` (exact, not prefix).

    Matches on the basename (keys carry the releases/tag/<tag>/ prefix) and
    returns the full keys.
    """
    out = []
    for name in names:
        base = name.rsplit("/", 1)[-1]
        m = _ASSET_RE.match(base)
        if m and m.group("version") == version:
            out.append(name)
    return out


def r2_object_names(tag: str) -> list[str]:
    """Object keys in the R2 staging dir for `tag`, under releases/tag/<tag>/.

    Exact version match, never prefix: 'v0.28.0' must not pick up
    '0.28.0-canary.20260818...' objects (they live under their own tag
    directory, and the basename filter would reject them anyway). The list
    call shells out to scripts/r2-release.mjs, which reads the R2 env vars
    and needs only node (no npm ci in this job).
    """
    run = subprocess.run(
        ["node", "scripts/r2-release.mjs", "list", "--prefix", f"releases/tag/{tag}/"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if run.returncode != 0:
        print(f"::error::r2 list failed: {run.stderr.strip() or run.stdout.strip()}")
        raise SystemExit(1)
    keys = [line.strip() for line in run.stdout.splitlines() if line.strip()]
    return filter_names_for_version(keys, tag.lstrip("v"))


def splice(body: str, block: str) -> str:
    """Replace the marker (or a previously rendered block) with `block`."""
    if END_MARKER in body:
        pattern = re.compile(re.escape(MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
        return pattern.sub(lambda _m: block, body, count=1)
    return body.replace(MARKER, block, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo", default="NousResearch/hermes-agent")
    parser.add_argument("--r2-base-url", default=os.environ.get("CLOUDFLARE_R2_PUBLIC_URL"),
                        help="Public base URL of the R2 bucket (default: $CLOUDFLARE_R2_PUBLIC_URL)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the spliced body instead of editing the release")
    parser.add_argument("--pending-run-url", default=None,
                        help="Render a 'builds in progress' link to this workflow run "
                             "instead of the tables")
    args = parser.parse_args()

    view = subprocess.run(
        ["gh", "release", "view", args.tag, "--repo", args.repo,
         "--json", "body"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if view.returncode != 0:
        print(f"::error::gh release view failed: {view.stderr.strip()}")
        return 1
    release = json.loads(view.stdout)
    body = release.get("body") or ""

    if args.pending_run_url:
        block = render_pending(args.pending_run_url)
        names: list[str] = []
    else:
        if not args.r2_base_url:
            print("::error::--r2-base-url (or CLOUDFLARE_R2_PUBLIC_URL) is required to render the tables")
            return 1
        names = r2_object_names(args.tag)
        block = render_tables(parse_assets(names), args.r2_base_url)
        if not block:
            print("::warning::no table-shaped assets for this tag in the bucket; leaving the body unchanged")
            return 0
    if MARKER not in body:
        print("::warning::release body has no HERMES_BUILDS_TABLE marker; leaving it unchanged")
        return 0

    new_body = splice(body, block)
    if args.dry_run:
        print(new_body)
        return 0

    edit = subprocess.run(
        ["gh", "release", "edit", args.tag, "--repo", args.repo,
         "--notes-file", "-"],
        input=new_body, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    if edit.returncode != 0:
        print(f"::error::gh release edit failed: {edit.stderr.strip()}")
        return 1
    what = "Builds-in-progress link" if args.pending_run_url else "Builds table"
    print(f"✓ {what} rendered into {args.tag} ({len(names)} assets scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
