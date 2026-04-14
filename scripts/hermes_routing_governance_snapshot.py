#!/usr/bin/env python3
"""Render a Hermes-native session-topology routing governance snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.routing_policy_bridge import build_routing_governance_snapshot
from hermes_constants import get_hermes_home


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Hermes-native session-topology routing governance snapshot")
    parser.add_argument("--config", type=Path, default=get_hermes_home() / "config.yaml")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--json", action="store_true")
    return parser


def _load_config_routes(config_path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model = payload.get("model") or {}
    primary = {
        "provider": str(model.get("provider") or "").strip(),
        "model": str(model.get("default") or "").strip(),
    }
    fallbacks = []
    for item in payload.get("fallback_providers") or []:
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider") or "").strip()
        model_name = str(item.get("model") or "").strip()
        if provider and model_name:
            fallbacks.append({"provider": provider, "model": model_name})
    return primary, fallbacks


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    primary, fallbacks = _load_config_routes(args.config)
    payload = build_routing_governance_snapshot(
        primary_route=primary,
        fallback_routes=fallbacks,
        repo_root=args.repo_root,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Policy: {payload['policy']['policy_id']}")
        print(f"Routes: {len(payload['available_routes'])}")
        print(f"Tasks with selected route: {payload['parity_validation']['tasks_with_selected_route']}")
        if payload['parity_validation']['tasks_without_any_policy_candidate']:
            print("Tasks without policy candidate:")
            for item in payload['parity_validation']['tasks_without_any_policy_candidate']:
                print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
