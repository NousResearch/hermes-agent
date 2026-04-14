#!/usr/bin/env python3
"""Hermes-native operator triage surface snapshot."""

from __future__ import annotations

import argparse
import json

from gateway.operator_surfaces import build_operator_triage_surface


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the Hermes operator triage snapshot")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_operator_triage_surface()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Severity: {payload['severity']}")
        print(f"Summary: {payload['summary']}")
        for issue in payload.get("issues", []):
            print(f"- [{issue['severity']}] {issue['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
