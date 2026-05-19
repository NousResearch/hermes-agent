from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

SCREEN_ONLY_STATUS = "SCREEN-ONLY · NOT EXECUTABLE"
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_file(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    info = zipfile.ZipInfo(arcname)
    info.date_time = FIXED_ZIP_TIME
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    zf.writestr(info, path.read_bytes())


def _safe_label(label: str) -> str:
    if not SAFE_LABEL.match(label) or label in {"", ".", ".."}:
        raise ValueError(f"Unsafe evidence artifact label: {label!r}")
    return label


def build_evidence_bundle(
    bundle_path: str | Path,
    *,
    run_id: str,
    manifest_json: str | Path,
    manifest_markdown: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_json)
    index_path = Path(manifest_markdown)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evidence_status") != SCREEN_ONLY_STATUS:
        raise ValueError("Evidence bundle requires SCREEN-ONLY · NOT EXECUTABLE status")

    target = Path(bundle_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    seen_arcnames = {"evidence_manifest.json", "evidence_index.md"}
    with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_STORED) as zf:
        _write_file(zf, manifest_path, "evidence_manifest.json")
        file_count += 1
        _write_file(zf, index_path, "evidence_index.md")
        file_count += 1
        for artifact in sorted(manifest.get("artifacts", []), key=lambda item: str(item.get("label") or item.get("path") or "")):
            artifact_path = Path(str(artifact.get("path", "")))
            if not artifact_path.exists():
                continue
            label = _safe_label(str(artifact.get("label") or artifact_path.stem))
            suffix = artifact_path.suffix
            arcname = f"artifacts/{label}{suffix}"
            if arcname in seen_arcnames:
                raise ValueError(f"Duplicate evidence bundle artifact archive path: {arcname}")
            seen_arcnames.add(arcname)
            _write_file(zf, artifact_path, arcname)
            file_count += 1
    return {
        "run_id": run_id,
        "bundle_path": str(target),
        "bundle_sha256": _sha256(target),
        "bundle_bytes": target.stat().st_size,
        "file_count": file_count,
        "evidence_status": SCREEN_ONLY_STATUS,
    }
