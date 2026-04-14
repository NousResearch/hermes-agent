#!/usr/bin/env python3
"""Build and gate a Hermes-native release evidence ladder bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway.evidence_ladder import build_release_evidence_bundle, evaluate_release_evidence_ladder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and gate a Hermes-native release evidence ladder bundle")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--activation-mode", choices=["shadow", "canary", "progressive", "broad_activation"], default="shadow")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = build_release_evidence_bundle(
        release_id=args.release_id,
        activation_mode=args.activation_mode,
        repo_root=args.repo_root,
    )
    payload = {"bundle": bundle}
    if not args.skip_gate:
        payload["decision"] = evaluate_release_evidence_ladder(bundle=bundle, repo_root=args.repo_root)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Release: {bundle['release_id']}")
        print(f"Activation mode: {bundle['activation_mode']}")
        if payload.get("decision"):
            print(f"Verdict: {payload['decision']['verdict']}")
            print("Gates:")
            for row in payload["decision"]["gate_results"]:
                print(f"- {row['gate_id']}: {row['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
