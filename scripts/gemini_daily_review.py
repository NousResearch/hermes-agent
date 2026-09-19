#!/usr/bin/env python3
"""Run the previous local day's Gemini routing review.

Successful and already-complete ticks intentionally emit no stdout. Quality and
pipeline failures are delivered by the configured exact Slack sender; uncaught
operational failures go to stderr and return nonzero for Hermes cron health.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.gemini_daily_review import main as run_configured_review


def main() -> int:
    try:
        run_configured_review()
    except Exception as exc:
        print(f"Gemini daily review failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
