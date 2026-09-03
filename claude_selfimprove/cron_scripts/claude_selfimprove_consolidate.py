#!/usr/bin/env python3
"""claude_selfimprove_consolidate.py — weekly threshold + conflict + write pass.

Hermes cron runs this once a week with ``--no-agent``. It evaluates every
pending candidate the nightly scan has accumulated evidence for: excludes
repo-specific ones, checks each against its evidence/confidence threshold,
checks the ones that clear the bar for conflicts with what is already
applied, and writes through — only for what survives every gate. See
``claude_selfimprove.pipeline.consolidate`` for the full sequence.

Notifications for applied changes, blocked conflicts, and failures go
through the shared self-improvement queue (``claude_selfimprove.notify``),
delivered by the existing ``self_improvement_watch.sh`` watcher — this
script does not send Slack messages directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claude_selfimprove import pipeline  # noqa: E402


def main() -> int:
    result = pipeline.consolidate()

    if result.skipped_busy:
        print("claude-selfimprove consolidate: skipped, another run holds the lock")
        return 0

    if result.error:
        print(f"claude-selfimprove consolidate: FAILED: {result.error}", file=sys.stderr)
        return 1

    print(
        f"claude-selfimprove consolidate: {result.evaluated} pending evaluated, "
        f"{len(result.applied)} applied, {len(result.conflict_blocked)} conflict-blocked, "
        f"{len(result.excluded_repo_scope)} excluded (repo-specific), "
        f"{result.not_yet_eligible} not yet eligible, {len(result.failed)} failed"
    )
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
