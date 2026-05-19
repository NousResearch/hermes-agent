from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCREEN_ONLY_STATUS = "SCREEN-ONLY · NOT EXECUTABLE"
PUBLISHABILITY = "internal-only until quote-verified and counsel-approved"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_evidence_manifest(
    *,
    run_id: str,
    as_of_cst: str,
    artifacts: dict[str, str | Path],
) -> dict[str, Any]:
    present = []
    missing = []
    for label, raw_path in artifacts.items():
        path = Path(raw_path)
        if not path.exists():
            missing.append(label)
            continue
        present.append(
            {
                "label": label,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "evidence_status": SCREEN_ONLY_STATUS,
            }
        )
    present.sort(key=lambda item: item["label"])
    missing.sort()
    return {
        "run_id": run_id,
        "as_of_cst": as_of_cst,
        "evidence_status": SCREEN_ONLY_STATUS,
        "publishability": PUBLISHABILITY,
        "artifact_count": len(present),
        "missing_artifacts": missing,
        "artifacts": present,
    }


def _markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# BTC Vol Desk Evidence Manifest",
        "",
        f"Run: `{manifest.get('run_id', '')}`",
        f"As of: `{manifest.get('as_of_cst', '')}`",
        f"Evidence status: **{manifest.get('evidence_status', SCREEN_ONLY_STATUS)}**",
        f"Publishability: **{manifest.get('publishability', PUBLISHABILITY)}**",
        "",
        "## Artifacts",
        "",
        "| Label | Bytes | sha256 | Path |",
        "|---|---:|---|---|",
    ]
    for artifact in manifest.get("artifacts", []):
        lines.append(
            f"| {artifact.get('label', '')} | {artifact.get('bytes', 0)} | `{artifact.get('sha256', '')}` | `{artifact.get('path', '')}` |"
        )
    missing = manifest.get("missing_artifacts") or []
    if missing:
        lines.extend(["", "## Missing Artifacts", ""])
        lines.extend(f"- {label}" for label in missing)
    lines.extend(
        [
            "",
            "## Control Note",
            "",
            "This manifest is an internal evidence-control artifact. It does not validate executable economics, quote firmness, legal approval, or investor publishability.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_evidence_manifest(
    json_path: str | Path,
    markdown_path: str | Path,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_target.write_text(_markdown(manifest), encoding="utf-8")
    return {"json": json_target, "markdown": markdown_target}
