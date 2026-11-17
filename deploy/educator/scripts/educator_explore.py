"""explore pre-run wake gate (educator profile, 3x per working day).

The explore job always runs when its schedule fires — budget is enforced by
the scheduler itself before this script starts (``cost_budget`` preflight:
the job is discretionary, so over budget it is skipped and the alert comes
from the gateway). This script only adds the continuity facts the agent
needs and that it cannot read itself:

- ``notebook_not_updated``: the previous explore run did not rewrite
  ``workspace/notebook.md`` (compared with the run's start in the cron
  executions ledger);
- ``previous_run_failed``;
- ``notebook_bytes``: so the agent compacts before the ~12 KB soft cap.

No network calls, no LLM.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import educator_wake as w  # noqa: E402  (sibling module)

JOB = "explore"
SCRIPT_NAME = "educator_explore.py"


def main(now: Optional[datetime] = None) -> dict:
    now = now or w.utcnow()
    state = w.load_state(JOB) or {"version": 1}
    previous = w.previous_woken_run(w.find_job_id(SCRIPT_NAME), state)
    _mtime, size = w.notebook_info()
    payload = {
        "wakeAgent": True,
        "job": JOB,
        "now": w.iso(now),
        "notebook_not_updated": w.notebook_missed(previous),
        "notebook_bytes": size,
        "previous_run_failed": bool(previous and previous.get("status") == "failed"),
    }
    state["last_wake"] = {"at": w.iso(now)}
    w.save_state(JOB, state)
    return w.emit(payload)


if __name__ == "__main__":
    main()
