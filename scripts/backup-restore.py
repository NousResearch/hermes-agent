#!/usr/bin/env python3
"""Disposable backup restore proof.

Selects the newest archive whose archive SHA-256 matches its manifest,
extracts it only to a temporary directory, and verifies the extracted file set
and each manifest-recorded byte size. It never restores over live state.

The current backup-producer manifest records archive SHA-256 plus per-file
path and size (not per-file digests), so extracted-file verification is bounded
to the fields the manifest actually contains.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

BACKUP_ROOT = Path(os.path.expanduser("~/backups/daily"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for(archive: Path) -> Path:
    if not archive.name.endswith(".tar.gz"):
        raise ValueError(f"Unsupported archive name: {archive.name}")
    return archive.with_name(f"{archive.name[:-len('.tar.gz')]}.manifest.json")


def load_verified_latest() -> tuple[Path, dict]:
    for archive in sorted(BACKUP_ROOT.glob("kensei-*.tar.gz"), reverse=True):
        manifest_path = manifest_path_for(archive)
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"SKIP {archive.name}: unreadable manifest ({exc})", file=sys.stderr)
            continue
        expected = manifest.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            print(f"SKIP {archive.name}: manifest missing valid archive SHA-256", file=sys.stderr)
            continue
        if sha256_file(archive) != expected:
            print(f"SKIP {archive.name}: archive SHA-256 mismatch", file=sys.stderr)
            continue
        return archive, manifest
    raise RuntimeError(f"No SHA-verified backup archive found in {BACKUP_ROOT}")


def safe_extract(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination_root):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Unsupported archive link: {member.name}")
        tar.extractall(destination, members=members, filter="data")


def verify_extracted(destination: Path, manifest: dict) -> int:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Manifest files list is absent or empty")
    if manifest.get("file_count") != len(files):
        raise RuntimeError(
            f"Manifest file_count mismatch: declared {manifest.get('file_count')}, listed {len(files)}"
        )
    expected: dict[str, int] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("size"), int):
            raise RuntimeError("Manifest contains an invalid file entry")
        if item["path"] in expected:
            raise RuntimeError(f"Manifest contains duplicate path: {item['path']}")
        expected[item["path"]] = item["size"]

    actual: dict[str, int] = {}
    for root, _, names in os.walk(destination):
        for name in names:
            path = Path(root) / name
            relative = path.relative_to(destination).as_posix()
            actual[relative] = path.stat().st_size

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(path for path in set(expected) & set(actual) if expected[path] != actual[path])
    if missing or extra or mismatched:
        lines = []
        if missing:
            lines.append(f"missing={', '.join(missing)}")
        if extra:
            lines.append(f"extra={', '.join(extra)}")
        if mismatched:
            lines.append(f"size-mismatch={', '.join(mismatched)}")
        raise RuntimeError("Extracted files do not match manifest: " + "; ".join(lines))
    return len(actual)


def main() -> int:
    try:
        archive, manifest = load_verified_latest()
        with tempfile.TemporaryDirectory(prefix="kensei-backup-restore-") as temp_dir:
            destination = Path(temp_dir)
            safe_extract(archive, destination)
            count = verify_extracted(destination, manifest)
        print(f"RESTORE OK: {archive.name}; SHA-256 verified; {count} manifest files extracted and verified in a disposable directory")
        return 0
    except (OSError, RuntimeError, tarfile.TarError, ValueError) as exc:
        print(f"RESTORE FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
