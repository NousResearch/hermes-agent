#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import posixpath
import shutil
import sys
import tarfile
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
    ROOT_PACKAGES,
    serializable_manifest,
    strip_termux_prefix,
    TERMUX_MAIN_BASE_URL,
    verify_sha256,
    write_manifest,
    TermuxPackageRecord,
)

ANDROID_SPAWN_NEEDED = b"libandroid-spawn.so\0"
BIONIC_LIBC_NEEDED = b"libc.so\0"
BIONIC_LLAMA_SERVER_NAME = "llama-server-bionic"
DEFAULT_LOCK_FILE = REPO_ROOT / "hermes_android" / "termux_linux_assets.lock.json"
LOCK_FILE_VERSION = 1
TERMUX_MAIN_FALLBACK_BASE_URLS = (
    "https://termux.librehat.com/apt/termux-main",
    "https://mirror.rinarin.dev/termux/termux-main",
    TERMUX_MAIN_BASE_URL,
    "https://packages-cf.termux.dev/apt/termux-main",
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


def configured_termux_main_base_urls() -> list[str]:
    configured = []
    for key in ("HERMES_TERMUX_MAIN_BASE_URLS", "HERMES_TERMUX_MAIN_BASE_URL"):
        raw = os.environ.get(key, "")
        configured.extend(item.strip() for item in raw.replace(";", ",").split(",") if item.strip())
    configured.extend(TERMUX_MAIN_FALLBACK_BASE_URLS)
    return list(dict.fromkeys(url.rstrip("/") for url in configured if url.strip()))


def _termux_main_url(base_url: str, relative_path: str) -> str:
    return f"{base_url.rstrip('/')}/{relative_path.lstrip('/')}"


def _packages_index_path(termux_arch: str) -> str:
    return f"dists/stable/main/binary-{termux_arch}/Packages"


def download_termux_main_path(relative_path: str) -> bytes:
    errors: list[str] = []
    for base_url in configured_termux_main_base_urls():
        url = _termux_main_url(base_url, relative_path)
        try:
            return download_bytes(url)
        except Exception as exc:  # pragma: no cover - exercised by live release builds
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"Failed to download Termux path {relative_path}: {'; '.join(errors)}")


def read_packages_index(termux_arch: str) -> dict[str, TermuxPackageRecord]:
    return parse_packages_index(download_termux_main_path(_packages_index_path(termux_arch)).decode("utf-8", "ignore"))


def _package_record_to_json(record: TermuxPackageRecord) -> dict:
    return {
        "name": record.name,
        "version": record.version,
        "filename": record.filename,
        "sha256": record.sha256,
        "depends": list(record.depends),
    }


def _package_record_from_json(payload: dict) -> TermuxPackageRecord:
    return TermuxPackageRecord(
        name=str(payload["name"]),
        version=str(payload["version"]),
        filename=str(payload["filename"]),
        sha256=str(payload["sha256"]),
        depends=tuple(str(item) for item in payload.get("depends", [])),
    )


def build_lock_payload() -> dict:
    architectures = {}
    for android_abi, termux_arch in ANDROID_TO_TERMUX_ARCH.items():
        packages = resolve_dependency_closure(read_packages_index(termux_arch))
        architectures[android_abi] = {
            "termux_arch": termux_arch,
            "packages": [_package_record_to_json(package) for package in packages],
        }
    return {
        "version": LOCK_FILE_VERSION,
        "termux_main_base_url": TERMUX_MAIN_BASE_URL,
        "root_packages": list(ROOT_PACKAGES),
        "architectures": architectures,
    }


def write_lock_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def load_lock_file(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != LOCK_FILE_VERSION:
        raise ValueError(f"Unsupported Termux asset lock file version in {path}")
    return payload


def locked_packages(lock_payload: dict, android_abi: str, termux_arch: str) -> list[TermuxPackageRecord]:
    arch_payload = lock_payload.get("architectures", {}).get(android_abi)
    if not arch_payload:
        raise KeyError(f"Termux asset lock file does not contain Android ABI {android_abi}")
    if arch_payload.get("termux_arch") != termux_arch:
        raise ValueError(f"Termux asset lock file maps {android_abi} to {arch_payload.get('termux_arch')}, not {termux_arch}")
    return [_package_record_from_json(item) for item in arch_payload.get("packages", [])]


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
        return resolved.relative_to(termux_root).as_posix()
    except ValueError:
        return None


def mirror_extracted_tree(extracted_root: Path, staging_prefix: Path) -> list[dict]:
    termux_root = _termux_root(extracted_root)
    if not termux_root.exists():
        return []

    inode_first_paths: dict[tuple[int, int], str] = {}
    links: list[dict] = []

    for source in sorted(termux_root.rglob("*"), key=lambda item: item.relative_to(termux_root).as_posix()):
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


def _archive_termux_relative(path: str) -> str | None:
    normalized = posixpath.normpath(str(path).replace("\\", "/"))
    relative = strip_termux_prefix(normalized)
    if relative is None:
        return None
    relative = posixpath.normpath(relative)
    if relative == ".":
        return ""
    if relative.startswith("../"):
        return None
    return relative


def _staging_destination(staging_prefix: Path, relative: str) -> Path:
    parts = [part for part in relative.split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        raise ValueError(f"Unsafe archive member path: {relative!r}")
    destination = staging_prefix.joinpath(*parts)
    destination.resolve(strict=False).relative_to(staging_prefix.resolve(strict=False))
    return destination


def _normalize_archive_link_target(source_relative: str, target: str) -> str | None:
    direct = _archive_termux_relative(target)
    if direct:
        return direct
    if direct == "":
        return None

    normalized = str(target).replace("\\", "/")
    if normalized.startswith("/"):
        return None
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source_relative), normalized))
    if resolved == "." or resolved.startswith("../"):
        return None
    return resolved


def _normalize_archive_hardlink_target(source_relative: str, target: str) -> str | None:
    direct = _archive_termux_relative(target)
    if direct:
        return direct
    if direct == "":
        return None

    normalized = posixpath.normpath(str(target).replace("\\", "/").lstrip("./"))
    if normalized and normalized != "." and not normalized.startswith("../"):
        return normalized
    return _normalize_archive_link_target(source_relative, target)


def mirror_data_tar(data_tar: tarfile.TarFile, staging_prefix: Path) -> list[dict]:
    links: list[dict] = []
    staging_prefix.mkdir(parents=True, exist_ok=True)

    for member in sorted(data_tar.getmembers(), key=lambda item: item.name):
        relative = _archive_termux_relative(member.name)
        if relative is None:
            continue
        if relative == "":
            staging_prefix.mkdir(parents=True, exist_ok=True)
            continue

        destination = _staging_destination(staging_prefix, relative)

        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True)
            continue

        if member.issym():
            target = _normalize_archive_link_target(relative, member.linkname)
            if target:
                links.append({"path": relative, "target": target})
            continue

        if member.islnk():
            target = _normalize_archive_hardlink_target(relative, member.linkname)
            if target:
                links.append({"path": relative, "target": target})
            continue

        if not member.isfile():
            continue

        file_obj = data_tar.extractfile(member)
        if file_obj is None:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = normalize_text_shebang(file_obj.read())
        destination.write_bytes(payload)
        if relative.startswith(("bin/", "libexec/")) or member.mode & 0o111:
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


def patch_android_spawn_needed_to_libc(path: Path) -> bool:
    payload = path.read_bytes()
    if ANDROID_SPAWN_NEEDED not in payload:
        return False
    replacement = BIONIC_LIBC_NEEDED + (b"\0" * (len(ANDROID_SPAWN_NEEDED) - len(BIONIC_LIBC_NEEDED)))
    path.write_bytes(payload.replace(ANDROID_SPAWN_NEEDED, replacement))
    return True


def create_bionic_llama_server_launcher(prefix_dir: Path) -> None:
    source = prefix_dir / "bin" / "llama-server"
    if not source.is_file():
        return
    destination = prefix_dir / "bin" / BIONIC_LLAMA_SERVER_NAME
    shutil.copy2(source, destination)
    if patch_android_spawn_needed_to_libc(destination):
        destination.chmod(0o755)
    else:
        destination.unlink(missing_ok=True)


def prepare_assets(output_dir: Path, lock_file: Path | None = DEFAULT_LOCK_FILE, refresh_lock_file: bool = False) -> None:
    lock_payload = None
    if lock_file is not None:
        if refresh_lock_file:
            lock_payload = build_lock_payload()
            write_lock_file(lock_file, lock_payload)
        else:
            lock_payload = load_lock_file(lock_file)

    for android_abi, termux_arch in ANDROID_TO_TERMUX_ARCH.items():
        if lock_payload:
            packages = locked_packages(lock_payload, android_abi, termux_arch)
        else:
            records = read_packages_index(termux_arch)
            packages = resolve_dependency_closure(records)

        prefix_dir = asset_prefix_dir(output_dir, android_abi)
        if prefix_dir.exists():
            shutil.rmtree(prefix_dir)
        prefix_dir.mkdir(parents=True, exist_ok=True)

        links: list[dict] = []
        for package in packages:
            payload = download_termux_main_path(package.filename)
            verify_sha256(payload, package.sha256)
            data_bytes, data_name = load_data_tar_bytes_from_deb(payload)
            with open_data_tar(data_bytes, data_name) as tar:
                links.extend(mirror_data_tar(tar, prefix_dir))

        prune_staging_prefix(prefix_dir)
        create_bionic_llama_server_launcher(prefix_dir)
        for extra_dir in [prefix_dir / "home", prefix_dir / "tmp"]:
            extra_dir.mkdir(parents=True, exist_ok=True)

        write_manifest(asset_manifest_path(output_dir, android_abi), serializable_manifest(android_abi, packages, links=links))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Android Linux CLI assets for Hermes Android builds")
    parser.add_argument("--output-dir", required=True, help="Directory where generated assets should be written")
    parser.add_argument(
        "--lock-file",
        default=str(DEFAULT_LOCK_FILE),
        help="Pinned Termux package lock file used for reproducible release builds",
    )
    parser.add_argument("--refresh-lock-file", action="store_true", help="Refresh the pinned Termux package lock file")
    parser.add_argument("--lock-only", action="store_true", help="Only refresh the lock file, without extracting assets")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    lock_file = Path(args.lock_file).expanduser().resolve() if args.lock_file else None
    if args.lock_only:
        if lock_file is None:
            raise ValueError("--lock-only requires --lock-file")
        write_lock_file(lock_file, build_lock_payload())
        return
    asset_root = output_dir / ANDROID_LINUX_ASSET_ROOT
    if asset_root.exists():
        shutil.rmtree(asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)
    prepare_assets(output_dir, lock_file=lock_file, refresh_lock_file=args.refresh_lock_file)


if __name__ == "__main__":
    main()
