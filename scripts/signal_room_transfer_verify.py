#!/usr/bin/env python3
"""Verify a transferred Signal Room handoff bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_file(path: Path, expected: dict[str, Any], label: str, errors: list[str]) -> dict[str, Any]:
    actual: dict[str, Any] = {"path": str(path), "exists": path.exists(), "passed": False}
    if not path.exists():
        errors.append(f"missing {label}: {path}")
        return actual

    actual["size_bytes"] = path.stat().st_size
    actual["sha256"] = sha256_file(path)
    if expected.get("size_bytes") is not None and actual["size_bytes"] != expected["size_bytes"]:
        errors.append(f"{label} size mismatch: {path}")
    if expected.get("sha256") and actual["sha256"] != expected["sha256"]:
        errors.append(f"{label} hash mismatch: {path}")
    actual["passed"] = not any(error.startswith(f"{label} ") for error in errors)
    return actual


def _archive_member_names(bundle_path: Path) -> set[str]:
    with tarfile.open(bundle_path, "r:gz") as archive:
        return set(archive.getnames())


def _resolve_bundle_path(checksum_path: Path, recorded_path: str) -> Path:
    path = Path(recorded_path)
    if path.exists():
        return path
    sibling = checksum_path.parent / path.name
    if sibling.exists():
        return sibling
    return path


def verify_transfer(checksum_path: Path, package_dir: Path | None = None) -> dict[str, Any]:
    checksums = read_json(checksum_path)
    errors: list[str] = []
    bundle_info = checksums.get("bundle", {})
    bundle_path = _resolve_bundle_path(checksum_path, str(bundle_info.get("path", "")))
    bundle = _check_file(bundle_path, bundle_info, "bundle", errors)

    artifacts = checksums.get("artifacts", {})
    archive_errors: list[str] = []
    archive = {
        "passed": False,
        "checked": False,
        "missing": [],
    }
    if bundle_path.exists():
        try:
            archive["checked"] = True
            member_names = _archive_member_names(bundle_path)
            expected_names = [f"signal-room-review/{rel_path}" for rel_path in artifacts]
            missing = [name for name in expected_names if name not in member_names]
            archive["missing"] = missing
            archive_errors.extend(f"missing archive member: {name}" for name in missing)
        except tarfile.TarError as exc:
            archive_errors.append(f"archive unreadable: {exc}")
    errors.extend(archive_errors)
    archive["passed"] = archive["checked"] and not archive_errors

    artifact_errors: list[str] = []
    checked_artifacts = 0
    artifact_results: dict[str, Any] = {}
    if package_dir is not None:
        for rel_path, expected in artifacts.items():
            checked_artifacts += 1
            path = package_dir / str(rel_path)
            result = _check_file(path, expected, "artifact", artifact_errors)
            artifact_results[str(rel_path)] = result
            if result.get("exists") and result.get("sha256") != expected.get("sha256"):
                artifact_errors.append(f"artifact hash mismatch: {rel_path}")
            if result.get("exists") and result.get("size_bytes") != expected.get("size_bytes"):
                artifact_errors.append(f"artifact size mismatch: {rel_path}")
    errors.extend(artifact_errors)

    return {
        "passed": not errors,
        "errors": errors,
        "checksum_path": str(checksum_path),
        "artifact_count": len(artifacts),
        "objective_evidence": checksums.get("objective_evidence", {}),
        "bundle": bundle,
        "archive": archive,
        "artifacts": {
            "passed": not artifact_errors,
            "checked": package_dir is not None,
            "checked_count": checked_artifacts,
            "items": artifact_results,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checksums", type=Path)
    parser.add_argument("--package-dir", type=Path)
    args = parser.parse_args()
    result = verify_transfer(args.checksums, package_dir=args.package_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
