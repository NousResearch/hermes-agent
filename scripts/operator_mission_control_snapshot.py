#!/usr/bin/env python3
"""Hermes-native operator mission surface snapshot."""

from __future__ import annotations

import argparse
import json

from gateway.operator_surfaces import build_operator_mission_surface


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the Hermes operator mission-control snapshot")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_operator_mission_surface()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Headline: {payload['headline']}")
        print(f"Gateway: {payload['gateway']['state']}")
        print(f"Active agents: {payload['gateway']['active_agents']}")
        if payload.get("recommended_actions"):
            print("Recommended actions:")
            for item in payload["recommended_actions"]:
                print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
