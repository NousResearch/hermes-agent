#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_android.linux_assets import (
    ANDROID_LINUX_ASSET_ROOT,
    ANDROID_TO_TERMUX_ARCH,
    asset_manifest_path,
    asset_prefix_dir,
    load_data_tar_bytes_from_deb,
    normalize_text_shebang,
    open_data_tar,
    parse_packages_index,
    resolve_dependency_closure,
    serializable_manifest,
    TERMUX_PACKAGES_INDEX_TEMPLATE,
    verify_sha256,
    write_manifest,
)


def download_bytes(url: str, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - exercised by live smoke checks
            last_error = exc
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def _termux_root(extracted_root: Path) -> Path:
    return extracted_root / "data" / "data" / "com.termux" / "files" / "usr"


def _normalize_link_target(source: Path, target: str, termux_root: Path) -> str | None:
    target_path = Path(target)
    if target_path.is_absolute():
        if str(target_path).startswith(str(termux_root)):
            return str(target_path.relative_to(termux_root))
        return None
    resolved = (source.parent / target_path).resolve(strict=False)
    try:
        return str(resolved.relative_to(termux_root))
    except ValueError:
        return None


def mirror_extracted_tree(extracted_root: Path, staging_prefix: Path) -> list[dict]:
    termux_root = _termux_root(extracted_root)
    if not termux_root.exists():
        return []

    inode_first_paths: dict[tuple[int, int], str] = {}
    links: list[dict] = []

    for source in termux_root.rglob("*"):
        relative = source.relative_to(termux_root)
        destination = staging_prefix / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)

        if source.is_symlink():
            normalized_target = _normalize_link_target(source, os.readlink(source), termux_root)
            if normalized_target:
                links.append({"path": relative.as_posix(), "target": normalized_target})
            continue

        stat = source.stat()
        inode_key = (stat.st_dev, stat.st_ino)
        first_path = inode_first_paths.get(inode_key)
        if first_path is not None:
            links.append({"path": relative.as_posix(), "target": first_path})
            continue
        inode_first_paths[inode_key] = relative.as_posix()

        payload = normalize_text_shebang(source.read_bytes())
        destination.write_bytes(payload)
        if relative.as_posix().startswith(("bin/", "libexec/")) or os.access(source, os.X_OK):
            destination.chmod(0o755)

    return links


def prune_staging_prefix(prefix_dir: Path) -> None:
    removable = [
        prefix_dir / "include",
        prefix_dir / "lib" / "pkgconfig",
        prefix_dir / "share" / "doc",
        prefix_dir / "share" / "info",
        prefix_dir / "share" / "man",
        prefix_dir / "share" / "zsh",
        prefix_dir / "share" / "LICENSES",
        prefix_dir / "var" / "cache",
    ]
    for path in removable:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def prepare_assets(output_dir: Path) -> None:
    for android_abi, termux_arch in ANDROID_TO_TERMUX_ARCH.items():
        index_url = TERMUX_PACKAGES_INDEX_TEMPLATE.format(termux_arch=termux_arch)
        records = parse_packages_index(download_bytes(index_url).decode("utf-8", "ignore"))
        packages = resolve_dependency_closure(records)

        prefix_dir = asset_prefix_dir(output_dir, android_abi)
        if prefix_dir.exists():
            shutil.rmtree(prefix_dir)
        prefix_dir.mkdir(parents=True, exist_ok=True)

        links: list[dict] = []
        for package in packages:
            payload = download_bytes(package.download_url)
            verify_sha256(payload, package.sha256)
            data_bytes, data_name = load_data_tar_bytes_from_deb(payload)
            with tempfile.TemporaryDirectory() as extracted_dir:
                with open_data_tar(data_bytes, data_name) as tar:
                    tar.extractall(extracted_dir)
                links.extend(mirror_extracted_tree(Path(extracted_dir), prefix_dir))

        prune_staging_prefix(prefix_dir)
        for extra_dir in [prefix_dir / "home", prefix_dir / "tmp"]:
            extra_dir.mkdir(parents=True, exist_ok=True)

        write_manifest(asset_manifest_path(output_dir, android_abi), serializable_manifest(android_abi, packages, links=links))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Android Linux CLI assets for Hermes Android builds")
    parser.add_argument("--output-dir", required=True, help="Directory where generated assets should be written")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    asset_root = output_dir / ANDROID_LINUX_ASSET_ROOT
    if asset_root.exists():
        shutil.rmtree(asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)
    prepare_assets(output_dir)


if __name__ == "__main__":
    main()
