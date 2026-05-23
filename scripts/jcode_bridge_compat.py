#!/usr/bin/env python3
"""Stdlib-only compatibility gate for the Hermes <-> jcode bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.jcode_bridge.tools import handle_jcode_contract_check  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jcode-bin", help="Optional path to a jcode executable.")
    parser.add_argument("--cwd", help="Optional working directory for live jcode checks.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also run lightweight live checks against a local jcode binary.",
    )
    parser.add_argument(
        "--live-run",
        action="store_true",
        help="With --live, also run a single jcode prompt and validate JSON output.",
    )
    parser.add_argument(
        "--live-run-message",
        default="Reply with exactly OK.",
        help="Prompt to use for --live-run.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Timeout for live jcode checks.",
    )
    ns = parser.parse_args(argv)

    report = json.loads(handle_jcode_contract_check({
        "jcode_bin": ns.jcode_bin,
        "cwd": ns.cwd,
        "live": ns.live,
        "live_run": ns.live_run,
        "live_run_message": ns.live_run_message,
        "timeout_seconds": ns.timeout_seconds,
    }))
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
