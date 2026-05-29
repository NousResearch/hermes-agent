#!/usr/bin/env python3
"""Create SHA-256 checksums for a Signal Room handoff package."""
from __future__ import annotations

import argparse
import hashlib
import json
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


def _file_info(path: Path) -> dict[str, Any]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_checksum_manifest(package_dir: Path) -> dict[str, Any]:
    manifest = read_json(package_dir / "handoff_manifest.json")
    errors: list[str] = []
    artifacts: dict[str, Any] = {}

    for rel_path in manifest.get("artifacts", []):
        path = package_dir / str(rel_path)
        if not path.exists():
            errors.append(f"missing artifact: {rel_path}")
            continue
        artifacts[str(rel_path)] = _file_info(path)

    bundle_path = Path(str(manifest.get("bundle_path", "")))
    bundle: dict[str, Any] = {"path": str(bundle_path)}
    if not bundle_path.exists():
        errors.append(f"missing bundle: {bundle_path}")
    else:
        bundle.update(_file_info(bundle_path))

    return {
        "passed": not errors,
        "errors": errors,
        "package_dir": str(package_dir),
        "bundle": bundle,
        "objective_evidence": manifest.get("objective_evidence", {}),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def write_checksum_manifest(package_dir: Path, out: Path) -> dict[str, Any]:
    result = build_checksum_manifest(package_dir)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = write_checksum_manifest(args.package_dir, args.out)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
