from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

SCREEN_ONLY_STATUS = "SCREEN-ONLY · NOT EXECUTABLE"
SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact_arcname(label: str, original_path: str) -> str:
    if not SAFE_LABEL.match(label) or label in {"", ".", ".."}:
        raise ValueError(f"unsafe artifact label: {label!r}")
    suffix = Path(original_path).suffix
    return f"artifacts/{label}{suffix}"


def verify_evidence_bundle(bundle_path: str | Path) -> dict[str, Any]:
    errors: list[str] = []
    missing_artifacts: list[str] = []
    hash_mismatches: list[str] = []
    orphan_members: list[str] = []
    verified = 0
    try:
        with zipfile.ZipFile(bundle_path) as zf:
            names = set(zf.namelist())
            if "evidence_manifest.json" not in names:
                return {
                    "ok": False,
                    "run_id": None,
                    "evidence_status": None,
                    "verified_artifacts": 0,
                    "missing_artifacts": [],
                    "hash_mismatches": [],
                    "orphan_members": [],
                    "errors": ["missing evidence_manifest.json"],
                }
            manifest = json.loads(zf.read("evidence_manifest.json"))
            status = manifest.get("evidence_status")
            if status != SCREEN_ONLY_STATUS:
                errors.append("missing SCREEN-ONLY · NOT EXECUTABLE evidence status")
            if "evidence_index.md" not in names:
                errors.append("missing evidence_index.md")
            expected_names = {"evidence_manifest.json", "evidence_index.md"}
            for artifact in manifest.get("artifacts", []):
                label = str(artifact.get("label", ""))
                try:
                    arcname = _artifact_arcname(label, str(artifact.get("path", "")))
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                if arcname in expected_names:
                    errors.append(f"duplicate artifact archive path: {arcname}")
                    continue
                expected_names.add(arcname)
                if arcname not in names:
                    missing_artifacts.append(label)
                    continue
                data = zf.read(arcname)
                if int(artifact.get("bytes") or 0) != len(data):
                    hash_mismatches.append(label)
                    continue
                if str(artifact.get("sha256", "")) != _sha256(data):
                    hash_mismatches.append(label)
                    continue
                verified += 1
            orphan_members = sorted(name for name in names if name not in expected_names and not name.endswith("/"))
            if orphan_members:
                errors.append("orphan ZIP members present")
    except zipfile.BadZipFile:
        return {
            "ok": False,
            "run_id": None,
            "evidence_status": None,
            "verified_artifacts": 0,
            "missing_artifacts": [],
            "hash_mismatches": [],
            "orphan_members": [],
            "errors": ["invalid zip archive"],
        }

    ok = not errors and not missing_artifacts and not hash_mismatches
    return {
        "ok": ok,
        "run_id": manifest.get("run_id"),
        "evidence_status": manifest.get("evidence_status"),
        "verified_artifacts": verified,
        "missing_artifacts": missing_artifacts,
        "hash_mismatches": hash_mismatches,
        "orphan_members": orphan_members,
        "errors": errors,
    }
