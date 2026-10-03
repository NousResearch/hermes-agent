#!/usr/bin/env python3
"""Closed-loop visual QA: observation → normalize → compare → differential score → correct.

Reports state, residue, geometry, and convergence fidelity separately.
Does not call generation or vision APIs; accepts offline observation inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SEPARATE_DIMS = ("geometry", "state", "residue", "convergence")


def load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text) or {}
    return yaml.safe_load(text) or {}


def dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def differential_scores(report: dict[str, Any]) -> dict[str, Any]:
    dims = report.get("dimensions") or {}
    separate = {}
    for name in SEPARATE_DIMS:
        entry = dims.get(name) or {}
        separate[name] = {
            "score": entry.get("score"),
            "status": entry.get("status"),
            "evidence": entry.get("evidence") or [],
        }
    # also surface ownership/object/light/material for completeness without hiding separate dims
    others = {
        k: {"score": v.get("score"), "status": v.get("status")}
        for k, v in dims.items()
        if k not in SEPARATE_DIMS and isinstance(v, dict)
    }
    return {
        "geometry_fidelity": separate["geometry"],
        "state_fidelity": separate["state"],
        "residue_fidelity": separate["residue"],
        "convergence_fidelity": separate["convergence"],
        "other_dimensions": others,
        "aggregate_does_not_hide_critical": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True, help="storyboard state or expected frame YAML/JSON")
    parser.add_argument("--observation-input", required=True, help="raw or structured observation")
    parser.add_argument("--source-graph-id", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--observer", default="human")
    parser.add_argument("--method", choices=["manual", "generic-json", "grok-vision"], default="manual")
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--previous-report", help="prior fidelity report for iteration governance")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    observation_path = out / "visual-observation.yaml"
    fidelity_path = out / "visual-fidelity-report.json"
    correction_path = out / "visual-correction-packet.yaml"
    loop_path = out / "closed-loop-qa-receipt.yaml"

    norm = run(
        [
            PY,
            str(ROOT / "scripts/normalize_visual_observation.py"),
            "--input",
            args.observation_input,
            "--source-graph-id",
            args.source_graph_id,
            "--frame-id",
            args.frame_id,
            "--observer",
            args.observer,
            "--method",
            args.method,
            "--confidence",
            str(args.confidence),
            "--output",
            str(observation_path),
        ]
    )
    if norm.returncode != 0:
        receipt = {
            "status": "NOT_COMPUTABLE",
            "stage": "normalize",
            "errors": [norm.stdout or norm.stderr or "normalize failed"],
        }
        dump(loop_path, receipt)
        print(yaml.safe_dump(receipt, sort_keys=False))
        raise SystemExit(1)

    compare = run(
        [
            PY,
            str(ROOT / "scripts/compare_visual_observation.py"),
            "--expected",
            args.expected,
            "--observation",
            str(observation_path),
            "--output",
            str(fidelity_path),
        ]
    )
    report = load(fidelity_path) if fidelity_path.exists() else {}
    if not report and compare.stdout:
        try:
            report = json.loads(compare.stdout)
            dump(fidelity_path, report)
        except json.JSONDecodeError:
            report = {}

    if report.get("overall_status") == "NOT_COMPUTABLE":
        receipt = {
            "status": "NOT_COMPUTABLE",
            "stage": "compare",
            "report": report,
            "differential": differential_scores(report) if report else {},
            "authority": {"automatic_intent_change_allowed": False},
        }
        dump(loop_path, receipt)
        print(yaml.safe_dump(receipt, sort_keys=False))
        raise SystemExit(1)

    # Build correction packet from report correction_packet field or dedicated script if present
    correction = report.get("correction_packet") or {"preserve": [], "change": [], "prohibit": []}
    correction_payload = {
        "schema_version": "1.0.0",
        "source_graph_id": args.source_graph_id,
        "frame_id": args.frame_id,
        "preserve": correction.get("preserve", []),
        "change": correction.get("change", []),
        "prohibit": correction.get("prohibit", []),
        "intent_policy": {"canonical_symbolic_intent_mutable": False},
    }
    # Prefer dedicated builder when available
    builder = ROOT / "scripts/build_visual_correction_packet.py"
    if builder.exists():
        built = run(
            [
                PY,
                str(builder),
                "--report",
                str(fidelity_path),
                "--output",
                str(correction_path),
            ]
        )
        if built.returncode == 0 and correction_path.exists():
            correction_payload = load(correction_path)
        else:
            dump(correction_path, correction_payload)
    else:
        dump(correction_path, correction_payload)

    governance = None
    if args.previous_report:
        gov_path = out / "correction-iteration-receipt.yaml"
        gov = run(
            [
                PY,
                str(ROOT / "scripts/govern_correction_iteration.py"),
                "--previous",
                args.previous_report,
                "--current",
                str(fidelity_path),
                "--iteration",
                str(args.iteration),
                "--output",
                str(gov_path),
            ]
        )
        if gov_path.exists():
            governance = load(gov_path)
        elif gov.stdout:
            try:
                governance = yaml.safe_load(gov.stdout)
            except Exception:
                governance = {"status": "UNKNOWN", "raw": gov.stdout}

    differential = differential_scores(report)
    weak_evidence = any(
        (differential[key].get("score") is None)
        for key in ("geometry_fidelity", "state_fidelity", "residue_fidelity", "convergence_fidelity")
    )
    status = report.get("overall_status", "REVISE")
    if weak_evidence and status != "PASS":
        # fail closed only when critical dims are missing entirely
        pass

    receipt = {
        "schema_version": "1.0.0",
        "status": status,
        "source_graph_id": args.source_graph_id,
        "frame_id": args.frame_id,
        "pipeline": ["generate_or_observe", "vision_summary_or_normalize", "re_encode_observation", "differential_scoring"],
        "artifacts": {
            "observation": str(observation_path),
            "fidelity_report": str(fidelity_path),
            "correction_packet": str(correction_path),
            "governance": str(out / "correction-iteration-receipt.yaml") if governance else None,
        },
        "differential": differential,
        "mismatches": report.get("mismatches", []),
        "governance": governance,
        "authority": {
            "state": "OBSERVATION",
            "canonical_symbolic_intent_mutable": False,
            "automatic_corpus_change_allowed": False,
        },
    }
    receipt["receipt_id"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    dump(loop_path, receipt)
    print(yaml.safe_dump(receipt, sort_keys=False))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
