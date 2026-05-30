#!/usr/bin/env python3.11
"""Validate that a Hermes Lab process is not pointed at stable state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.dev_control.lab_environment import cli_report, validate_lab_environment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Hermes Lab isolation settings.")
    parser.add_argument("--hermes-home", default=None)
    parser.add_argument("--gateway-port", default=None)
    parser.add_argument("--repo-root", action="append", default=[])
    args = parser.parse_args()
    result = validate_lab_environment(
        hermes_home=args.hermes_home,
        gateway_port=args.gateway_port,
        repo_roots=args.repo_root,
    )
    print(cli_report(result))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
