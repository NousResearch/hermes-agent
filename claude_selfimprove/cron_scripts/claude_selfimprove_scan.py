#!/usr/bin/env python3
"""claude_selfimprove_scan.py — nightly incremental transcript scan.

Hermes cron runs this once a night with ``--no-agent``. It reads only the
transcript bytes that have arrived since the last run (see
``claude_selfimprove.scanner``), flags candidate lessons, and classifies
them — it never writes to any Claude Code target file. That happens in the
separate weekly consolidation pass (``claude_selfimprove_consolidate.py``),
once evidence has had time to accumulate across several nights.

Deployed alongside a same-directory copy of the ``claude_selfimprove``
package (see the pipeline's install step) — this script only adds that
directory to ``sys.path`` and calls into it, matching the
``t3_update_run.py`` / ``t3_update_common.py`` pattern already used for the
sibling nightly/update watchers in this profile.

Exit code communicates success/failure to the cron scheduler; stdout is one
short summary line for ``hermes cron output``, which stays cheap regardless
of the job's ``deliver`` setting since Slack visibility for this job comes
from the shared self-improvement notification queue instead (see
``claude_selfimprove.notify``), not from cron's own delivery of stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claude_selfimprove import pipeline  # noqa: E402


def main() -> int:
    result = pipeline.scan()

    if result.skipped_busy:
        print("claude-selfimprove scan: skipped, another run holds the lock")
        return 0

    if result.error:
        print(f"claude-selfimprove scan: FAILED: {result.error}", file=sys.stderr)
        return 1

    print(
        f"claude-selfimprove scan: {result.turns_scanned} new turns, "
        f"{result.files_touched} files tracked, {result.candidates_flagged} flagged, "
        f"{result.candidates_classified} classified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
