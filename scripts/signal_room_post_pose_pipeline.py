#!/usr/bin/env python3
"""Run the Signal Room review refresh after approved pose PNGs land."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable


HYPERFRAMES_VERSION = "0.6.38"


Step = dict[str, Any]
Runner = Callable[[Step], int | None]


def _step(name: str, command: list[str], *, cwd: Path | None = None, allow_failure: bool = False) -> Step:
    return {
        "name": name,
        "command": command,
        "cwd": str(cwd) if cwd else None,
        "allow_failure": allow_failure,
    }


def build_pipeline_plan(package_dir: Path, candidate_dir: Path) -> list[Step]:
    scaffold = package_dir / "fee_machine_v2_scaffold"
    draft = package_dir / "fee_machine_v2_draft_pose_refresh.mp4"
    draft_with_audio = package_dir / "fee_machine_v2_draft_pose_refresh_with_audio.mp4"
    proof_frames = package_dir / "proof_frames"
    frame_plan = scaffold / "retention_frame_plan.json"
    cue_sheet = scaffold / "audio_cue_sheet.json"
    audio_assets = package_dir / "audio_assets"
    contact_sheet = package_dir / f"{candidate_dir.name}_contact_sheet.svg"

    return [
        _step(
            "rig_acting_gate",
            [
                "python",
                "scripts/signal_room_rig_acting_gate.py",
                str(candidate_dir),
                "--out",
                str(package_dir / "rig_acting_scorecard.json"),
            ],
        ),
        _step(
            "acting_contact_sheet",
            [
                "python",
                "scripts/signal_room_contact_sheet.py",
                str(candidate_dir),
                "--out",
                str(contact_sheet),
            ],
        ),
        _step(
            "install_pose_export",
            [
                "python",
                "scripts/signal_room_pose_export_installer.py",
                "--candidate",
                str(candidate_dir),
                "--scaffold",
                str(scaffold),
                "--out",
                str(package_dir / "pose_export_install_scorecard.json"),
            ],
        ),
        _step("hyperframes_lint", ["npx", "--yes", f"hyperframes@{HYPERFRAMES_VERSION}", "lint"], cwd=scaffold),
        _step("hyperframes_inspect", ["npx", "--yes", f"hyperframes@{HYPERFRAMES_VERSION}", "inspect", "--json"], cwd=scaffold),
        _step(
            "hyperframes_render",
            [
                "npx",
                "--yes",
                f"hyperframes@{HYPERFRAMES_VERSION}",
                "render",
                "--quality",
                "draft",
                "--output",
                str(draft),
            ],
            cwd=scaffold,
        ),
        _step(
            "render_asset_gate",
            [
                "python",
                "scripts/signal_room_render_asset_gate.py",
                str(draft),
                "--out",
                str(package_dir / "render_asset_scorecard.json"),
            ],
        ),
        _step(
            "motion_smoothness_gate",
            [
                "python",
                "scripts/signal_room_motion_smoothness_gate.py",
                str(draft),
                "--frame-diff-report",
                str(package_dir / "motion_frame_diff_report.json"),
                "--choreography",
                str(scaffold / "scene_choreography.json"),
                "--motion-primitives",
                str(scaffold / "motion_primitives.json"),
                "--out",
                str(package_dir / "motion_smoothness_scorecard.json"),
            ],
        ),
        _step(
            "sample_proof_frames",
            [
                "python",
                "scripts/signal_room_render_frame_sampler.py",
                "--plan",
                str(frame_plan),
                "--render",
                str(draft),
                "--out",
                str(proof_frames),
                "--force",
            ],
        ),
        _step(
            "retention_frame_gate",
            [
                "python",
                "scripts/signal_room_retention_frame_gate.py",
                "--plan",
                str(frame_plan),
                "--frames",
                str(proof_frames),
                "--choreography",
                str(scaffold / "scene_choreography.json"),
                "--out",
                str(package_dir / "retention_frame_scorecard.json"),
            ],
        ),
        _step(
            "proof_contact_sheet",
            [
                "python",
                "scripts/signal_room_proof_contact_sheet.py",
                "--plan",
                str(frame_plan),
                "--frames",
                str(proof_frames),
                "--out",
                str(package_dir / "proof_retention_contact_sheet.svg"),
            ],
        ),
        _step(
            "audio_mux",
            [
                "python",
                "scripts/signal_room_audio_mix_muxer.py",
                "--cue-sheet",
                str(cue_sheet),
                "--assets",
                str(audio_assets),
                "--render",
                str(draft),
                "--out",
                str(draft_with_audio),
                "--scorecard-out",
                str(package_dir / "audio_mux_scorecard.json"),
            ],
        ),
        _step(
            "motion_quality_plan",
            [
                "python",
                "scripts/signal_room_motion_quality_plan.py",
                str(package_dir),
                "--json-out",
                str(package_dir / "motion_quality_plan.json"),
                "--markdown-out",
                str(package_dir / "MOTION_QUALITY_PASS.md"),
            ],
        ),
        _step(
            "review_package_index",
            [
                "python",
                "scripts/signal_room_review_package_index.py",
                str(package_dir),
                "--out",
                str(package_dir / "review_package_index.json"),
                "--markdown-out",
                str(package_dir / "REVIEW_STATUS.md"),
            ],
            allow_failure=True,
        ),
        _step(
            "handoff_bundle",
            [
                "python",
                "scripts/signal_room_review_handoff_bundle.py",
                str(package_dir),
                "--out",
                str(package_dir / "signal_room_fee_machine_v2_review_handoff.tar.gz"),
                "--manifest-out",
                str(package_dir / "handoff_manifest.json"),
            ],
        ),
    ]


def default_runner(step: Step) -> int:
    completed = subprocess.run(step["command"], cwd=step.get("cwd") or None)
    return completed.returncode


def run_pipeline(
    package_dir: Path,
    candidate_dir: Path,
    *,
    dry_run: bool = False,
    runner: Runner = default_runner,
) -> dict[str, Any]:
    plan = build_pipeline_plan(package_dir, candidate_dir)
    if dry_run:
        return {
            "passed": True,
            "dry_run": True,
            "step_count": len(plan),
            "steps": plan,
            "completed_steps": [],
        }

    completed_steps: list[str] = []
    for step in plan:
        returncode = runner(step)
        if returncode not in (0, None) and not step.get("allow_failure"):
            return {
                "passed": False,
                "dry_run": False,
                "failed_step": step["name"],
                "returncode": returncode,
                "completed_steps": completed_steps,
                "steps": plan,
            }
        completed_steps.append(step["name"])

    return {
        "passed": True,
        "dry_run": False,
        "step_count": len(plan),
        "completed_steps": completed_steps,
        "steps": plan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = run_pipeline(args.package_dir, args.candidate, dry_run=args.dry_run)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
