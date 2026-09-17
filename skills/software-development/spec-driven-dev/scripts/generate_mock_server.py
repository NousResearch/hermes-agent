#!/usr/bin/env python3
"""
Wrapper for generating a mock server from a schema-first API contract.

Shells out to Stoplight Prism (https://github.com/stoplightio/prism) --
does not reimplement mock-server generation. Requires `npx` and a
network-reachable npm registry the first time (npx installs prism-cli
on demand unless it's already globally installed).

This script does NOT attempt to install Node/npm -- if the environment's
npm/node versions are mismatched or prism fails to launch, it reports
the real error and exits non-zero rather than masking the failure.

Usage:
  python3 generate_mock_server.py <contract.yaml|contract.json> [--port 4010]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_prism_available() -> bool:
    return shutil.which("npx") is not None


def run_mock_server(contract: Path, port: int) -> int:
    if not contract.exists():
        print(f"ERROR: contract file not found: {contract}", file=sys.stderr)
        return 1

    if not check_prism_available():
        print(
            "ERROR: npx not found. Prism (https://github.com/stoplightio/"
            "prism) is an npm package; install Node.js/npm first, or "
            "install prism-cli globally (`npm install -g @stoplight/"
            "prism-cli`) and re-run.",
            file=sys.stderr,
        )
        return 1

    cmd = [
        "npx",
        "--yes",
        "@stoplight/prism-cli",
        "mock",
        str(contract),
        "--port",
        str(port),
    ]
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    try:
        result = subprocess.run(cmd)
    except FileNotFoundError as exc:
        print(f"ERROR: failed to launch prism: {exc}", file=sys.stderr)
        return 1

    if result.returncode != 0:
        print(
            "Prism exited non-zero. Common cause: a Node/npm version "
            "mismatch in this environment (npx reports the real error "
            "above) -- this script does not attempt to fix Node/npm "
            "itself.",
            file=sys.stderr,
        )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--port", type=int, default=4010)
    args = parser.parse_args()
    sys.exit(run_mock_server(args.contract, args.port))


if __name__ == "__main__":
    main()
