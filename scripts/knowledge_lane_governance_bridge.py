#!/usr/bin/env python3
"""Export Hermes knowledge-lane items into governed Marson promotion artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.knowledge_bridge import export_lane_item_to_governance_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge Hermes knowledge lanes into governed promotion artifacts")
    parser.add_argument("--id", required=True)
    parser.add_argument("--lane", choices=["draft", "promoted"], default="draft")
    parser.add_argument("--target-surface", choices=["doctrine", "memory", "playbook"], required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--merge-mode", choices=["append", "patch", "replace_section"], default="append")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = export_lane_item_to_governance_package(
        lane_item_id=args.id,
        lane=args.lane,
        repo_root=args.repo_root,
        target_surface=args.target_surface,
        target_path=args.target_path,
        merge_mode=args.merge_mode,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
