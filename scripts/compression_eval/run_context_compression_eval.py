"""Run and validate the external hermes-compression-eval harness."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from report_contract import validate_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--hermes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.hermes_root.is_dir() or not args.harness.is_dir() or not args.command:
        raise SystemExit("--harness, --hermes-root, and a harness command are required")
    result = subprocess.run(args.command, cwd=args.harness, env={**os.environ, "HERMES_AGENT_ROOT": str(args.hermes_root)}, text=True, capture_output=True, check=False)
    if result.returncode:
        raise SystemExit(f"compression harness failed with exit {result.returncode}")
    report_path = args.harness / "results" / "latest" / "report.json"
    if not report_path.exists():
        raise SystemExit(f"compression report missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    errors = validate_report(report)
    if errors:
        raise SystemExit("invalid compression report: " + ", ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
