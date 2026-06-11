#!/usr/bin/env python3
"""Prepare portable tool archives for the Tauri bootstrap installer bundle.

The installer can use archives placed under
`apps/bootstrap-installer/src-tauri/bootstrap-tools` before it falls back to
network downloads. This helper is intended for release workflows, not for the
runtime installer path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "apps" / "bootstrap-installer" / "src-tauri" / "bootstrap-tools"
NODE_MAJOR = 22
USER_AGENT = "Hermes-Setup"
NODE_INDEX_URL = f"https://nodejs.org/dist/latest-v{NODE_MAJOR}.x/"
RIPGREP_VERSION = "15.1.0"
GIT_TAG = "v2.54.0.windows.1"
GIT_VERSION = "2.54.0"
MANIFEST_NAME = "bootstrap-tools-manifest.json"

UV_ARCHIVE_NAMES = {
    "x64": "uv-x86_64-pc-windows-msvc.zip",
    "arm64": "uv-aarch64-pc-windows-msvc.zip",
    "x86": "uv-i686-pc-windows-msvc.zip",
}

UNIX_UV_ARCHIVE_NAMES = {
    ("linux", "x64"): "uv-x86_64-unknown-linux-gnu.tar.gz",
    ("linux", "arm64"): "uv-aarch64-unknown-linux-gnu.tar.gz",
    ("macos", "x64"): "uv-x86_64-apple-darwin.tar.gz",
    ("macos", "arm64"): "uv-aarch64-apple-darwin.tar.gz",
}

GIT_ARCHIVE_NAMES = {
    "x64": "PortableGit-2.54.0-64-bit.7z.exe",
    "arm64": "PortableGit-2.54.0-arm64.7z.exe",
    "x86": "MinGit-2.54.0-32-bit.zip",
}

RIPGREP_ARCHIVE_NAMES = {
    "x64": f"ripgrep-{RIPGREP_VERSION}-x86_64-pc-windows-msvc.zip",
    "arm64": f"ripgrep-{RIPGREP_VERSION}-aarch64-pc-windows-msvc.zip",
    "x86": f"ripgrep-{RIPGREP_VERSION}-i686-pc-windows-msvc.zip",
}


@dataclass(frozen=True)
class ArchiveSpec:
    """One release archive that should be copied into the Tauri resource dir."""

    name: str
    url: str


@dataclass(frozen=True)
class PreparedArchive:
    """One downloaded archive with audit metadata for the release manifest."""

    arch: str
    name: str
    url: str
    path: Path
    size_bytes: int
    sha256: str


def select_latest_node_archive(index_html: str, arch: str, major: int = NODE_MAJOR) -> str:
    """Return the newest Node.js Windows archive name for one architecture."""

    pattern = re.compile(rf"node-v({major})\.(\d+)\.(\d+)-win-{re.escape(arch)}\.zip")
    matches: list[tuple[tuple[int, int, int], str]] = []
    for match in pattern.finditer(index_html):
        version = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        matches.append((version, match.group(0)))
    if not matches:
        raise ValueError(f"Node.js v{major} Windows {arch} archive not found")
    return max(matches, key=lambda item: item[0])[1]


def select_latest_unix_node_archive(
    index_html: str,
    node_os: str,
    arch: str,
    major: int = NODE_MAJOR,
) -> str:
    """Return the newest Node.js Unix tarball name, preferring xz over gzip."""

    for extension in ("tar.xz", "tar.gz"):
        pattern = re.compile(
            rf"node-v({major})\.(\d+)\.(\d+)-{re.escape(node_os)}-{re.escape(arch)}\.{extension}"
        )
        matches: list[tuple[tuple[int, int, int], str]] = []
        for match in pattern.finditer(index_html):
            version = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            matches.append((version, match.group(0)))
        if matches:
            return max(matches, key=lambda item: item[0])[1]
    raise ValueError(f"Node.js v{major} {node_os}-{arch} archive not found")


def archive_specs_for_arch(arch: str, node_archive_name: str) -> list[ArchiveSpec]:
    """Build the archive list that matches the Rust installer runtime matrix."""

    if arch not in UV_ARCHIVE_NAMES or arch not in GIT_ARCHIVE_NAMES or arch not in RIPGREP_ARCHIVE_NAMES:
        raise ValueError(f"unsupported Windows architecture: {arch}")
    return [
        ArchiveSpec(
            name=node_archive_name,
            url=f"{NODE_INDEX_URL}{node_archive_name}",
        ),
        ArchiveSpec(
            name=UV_ARCHIVE_NAMES[arch],
            url=f"https://github.com/astral-sh/uv/releases/latest/download/{UV_ARCHIVE_NAMES[arch]}",
        ),
        ArchiveSpec(
            name=RIPGREP_ARCHIVE_NAMES[arch],
            url=(
                "https://github.com/BurntSushi/ripgrep/releases/download/"
                f"{RIPGREP_VERSION}/{RIPGREP_ARCHIVE_NAMES[arch]}"
            ),
        ),
        ArchiveSpec(
            name=GIT_ARCHIVE_NAMES[arch],
            url=f"https://github.com/git-for-windows/git/releases/download/{GIT_TAG}/{GIT_ARCHIVE_NAMES[arch]}",
        ),
    ]


def archive_specs_for_target(
    platform: str,
    arch: str,
    node_archive_name: str | None = None,
) -> list[ArchiveSpec]:
    """Build archive specs for one release platform and architecture."""

    normalized_platform = "macos" if platform == "darwin" else platform
    if normalized_platform == "windows":
        if node_archive_name is None:
            raise ValueError("Windows bootstrap tools require a Node.js archive name")
        return archive_specs_for_arch(arch, node_archive_name)

    archive_name = UNIX_UV_ARCHIVE_NAMES.get((normalized_platform, arch))
    if archive_name is None:
        raise ValueError(f"unsupported Unix uv platform: {normalized_platform}-{arch}")
    if node_archive_name is None:
        raise ValueError("Unix bootstrap tools require a Node.js archive name")
    return [
        ArchiveSpec(
            name=node_archive_name,
            url=f"{NODE_INDEX_URL}{node_archive_name}",
        ),
        ArchiveSpec(
            name=archive_name,
            url=f"https://github.com/astral-sh/uv/releases/latest/download/{archive_name}",
        )
    ]


def fetch_text(url: str) -> str:
    """Fetch a small text resource using the release helper user agent."""

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def download_archive(spec: ArchiveSpec, output_dir: Path, force: bool) -> Path:
    """Download one archive atomically unless a non-empty file already exists."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / spec.name
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        print(f"[bootstrap-tools] keep {dest}")
        return dest

    tmp = dest.with_name(f"{dest.name}.tmp")
    tmp.unlink(missing_ok=True)
    print(f"[bootstrap-tools] download {spec.url}")
    request = urllib.request.Request(spec.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=600) as response:
        with tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded empty archive: {spec.url}")
    os.replace(tmp, dest)
    return dest


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest for one archive file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepared_archive_record(arch: str, spec: ArchiveSpec, path: Path) -> PreparedArchive:
    """Build manifest metadata for one downloaded archive."""

    return PreparedArchive(
        arch=arch,
        name=spec.name,
        url=spec.url,
        path=path,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def write_manifest(output_dir: Path, archives: list[PreparedArchive]) -> Path:
    """Write the bundled tool archive manifest consumed by release reviewers."""

    payload = {
        "schemaVersion": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "archives": [
            {
                "arch": archive.arch,
                "name": archive.name,
                "url": archive.url,
                "sizeBytes": archive.size_bytes,
                "sha256": archive.sha256,
            }
            for archive in archives
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def prepare_archives(
    output_dir: Path,
    arches: list[str],
    force: bool,
    dry_run: bool,
    platform: str = "windows",
) -> list[PreparedArchive]:
    """Resolve and optionally download all archives for the requested architectures."""

    normalized_platform = "macos" if platform == "darwin" else platform
    index_html = fetch_text(NODE_INDEX_URL)
    downloaded: list[PreparedArchive] = []
    for arch in arches:
        if normalized_platform == "windows":
            node_archive = select_latest_node_archive(index_html, arch)
        else:
            node_os = "darwin" if normalized_platform == "macos" else normalized_platform
            node_archive = select_latest_unix_node_archive(index_html, node_os, arch)
        for spec in archive_specs_for_target(normalized_platform, arch, node_archive):
            if dry_run:
                print(f"[bootstrap-tools] would download {spec.name} <- {spec.url}")
            else:
                path = download_archive(spec, output_dir, force)
                manifest_arch = arch if normalized_platform == "windows" else f"{normalized_platform}-{arch}"
                downloaded.append(prepared_archive_record(manifest_arch, spec, path))
    if downloaded:
        manifest_path = write_manifest(output_dir, downloaded)
        print(f"[bootstrap-tools] wrote manifest {manifest_path}")
    return downloaded


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line options for release automation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("windows", "linux", "macos", "darwin"),
        default="windows",
        help="Release platform to bundle tools for. Defaults to windows.",
    )
    parser.add_argument(
        "--arch",
        action="append",
        choices=sorted(set(UV_ARCHIVE_NAMES) | {"x64", "arm64"}),
        default=None,
        help="Architecture to bundle. Can be passed more than once. Defaults to x64.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory copied by tauri.conf.json bundle.resources.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download archives that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned archive URLs without downloading.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the archive preparation helper."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    arches = args.arch or ["x64"]
    try:
        prepared = prepare_archives(
            args.output_dir,
            arches,
            args.force,
            args.dry_run,
            args.platform,
        )
    except Exception as exc:
        print(f"[bootstrap-tools] error: {exc}", file=sys.stderr)
        return 1
    print(f"[bootstrap-tools] prepared {len(prepared)} archive(s) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
