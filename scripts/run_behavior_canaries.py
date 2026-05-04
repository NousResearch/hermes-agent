#!/usr/bin/env python
"""Run Hermes behavior-regression canaries against the shipped base identity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.behavior_canaries import summarize_behavior_canaries
from hermes_cli.default_soul import DEFAULT_SOUL_MD


def main() -> int:
    summary = summarize_behavior_canaries(DEFAULT_SOUL_MD)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
