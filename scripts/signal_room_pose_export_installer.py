#!/usr/bin/env python3
"""Install approved Signal Room pose PNG exports into a scaffold."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


POSE_MAP = {
    "neutral_read": "neutral_read.png",
    "bill_shock": "bill_shock.png",
    "look_to_machine": "look_to_machine.png",
    "skeptical_point": "skeptical_point.png",
    "slight_lean": "lean_weight_shift.png",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _missing_candidate_frames(candidate_dir: Path) -> list[str]:
    return [filename for filename in POSE_MAP.values() if not (candidate_dir / filename).exists()]


def _rewrite_pose_refs(scaffold_dir: Path) -> None:
    index_path = scaffold_dir / "index.html"
    html = index_path.read_text()
    for pose_name in POSE_MAP:
        html = html.replace(f"assets/poses/{pose_name}.svg", f"assets/poses/{pose_name}.png")
    index_path.write_text(html)


def install_pose_export(candidate_dir: Path, scaffold_dir: Path, *, out: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    for filename in _missing_candidate_frames(candidate_dir):
        errors.append(f"missing required candidate frame: {filename}")
    manifest_path = candidate_dir / "rig_pass_manifest.json"
    if not manifest_path.exists():
        errors.append("missing rig_pass_manifest.json")
        source_manifest: dict[str, Any] = {}
    else:
        source_manifest = read_json(manifest_path)

    if errors:
        result = {
            "passed": False,
            "errors": errors,
            "candidate_dir": str(candidate_dir),
            "scaffold_dir": str(scaffold_dir),
            "installed_pose_count": 0,
        }
        if out:
            _write_json(out, result)
        return result

    poses_dir = scaffold_dir / "assets" / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    for pose_name, source_filename in POSE_MAP.items():
        shutil.copy2(candidate_dir / source_filename, poses_dir / f"{pose_name}.png")

    _rewrite_pose_refs(scaffold_dir)

    candidate_name = str(source_manifest.get("candidate_name") or candidate_dir.name)
    installed_source = {
        **source_manifest,
        "installed_from": str(candidate_dir),
        "installed_pose_map": POSE_MAP,
    }
    _write_json(scaffold_dir / "source_rig_manifest.json", installed_source)

    scaffold_manifest_path = scaffold_dir / "manifest.json"
    scaffold_manifest = read_json(scaffold_manifest_path) if scaffold_manifest_path.exists() else {}
    scaffold_manifest["approved_pose_export"] = {
        "candidate_name": candidate_name,
        "source_dir": str(candidate_dir),
        "pose_count": len(POSE_MAP),
        "source_manifest": "source_rig_manifest.json",
    }
    _write_json(scaffold_manifest_path, scaffold_manifest)

    result = {
        "passed": True,
        "errors": [],
        "candidate_dir": str(candidate_dir),
        "scaffold_dir": str(scaffold_dir),
        "candidate_name": candidate_name,
        "installed_pose_count": len(POSE_MAP),
        "installed_poses": sorted(f"{pose}.png" for pose in POSE_MAP),
    }
    if out:
        _write_json(out, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--scaffold", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = install_pose_export(args.candidate, args.scaffold, out=args.out)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
