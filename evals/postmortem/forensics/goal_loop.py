"""Lane 4: /goal loop behaviour — nudges, judge verdicts, parked time. OBSERVED from the root session's
messages and the persisted goal state.

    python -m evals.postmortem.forensics.goal_loop --db state_copy.db

Reproduces: how many ``[Continuing toward your standing goal]`` nudges fired and how soon after the
previous assistant turn; how many of those turns had just said they were waiting; the goal's final
persisted wait barrier (what it was parked on, since when); and the count of async-batch and
background-process notifications that re-entered the root.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from evals.postmortem.forensics.common import Run

_WAITING = re.compile(r"\b(waiting|wait(s)? on|nothing (else |new )?to dispatch|no action needed|standing by|until .* (finish|return|complete)|in flight|still running)\b", re.I)


def main(argv=None) -> int:
    run = Run.from_args(argv, (__doc__ or "").split("\n\n")[0])
    root = run.root
    msgs = run.messages(root, "role, content, timestamp")
    nudges: List[Dict[str, Any]] = []
    batch_notices = bgproc_notices = 0
    last_assistant = None
    for m in msgs:
        c = m.get("content") or ""
        if m["role"] == "assistant":
            last_assistant = m
        elif m["role"] == "user":
            if c.startswith("[Continuing toward your standing goal]"):
                gap = float(m["timestamp"]) - float(last_assistant["timestamp"]) if last_assistant else None
                nudges.append({"ts": m["timestamp"], "gap_s": round(gap, 1) if gap is not None else None,
                               "prev_turn_said_waiting": bool(last_assistant and _WAITING.search(last_assistant.get("content") or ""))})
            elif c.startswith("[ASYNC DELEGATION"):
                batch_notices += 1
            elif c.startswith("[IMPORTANT: Background process"):
                bgproc_notices += 1
    goal_state: Dict[str, Any] = {}
    try:
        row = run._conn.execute("SELECT value FROM state_meta WHERE key=?", (f"goal:{root}",)).fetchone()
        if row:
            goal_state = json.loads(row[0])
    except Exception:
        pass
    parked = {k: goal_state.get(k) for k in ("status", "waiting_on_pid", "waiting_on_session", "waiting_until", "waiting_since", "waiting_reason", "last_verdict", "turns_used")}
    if goal_state.get("waiting_since"):
        # The barrier is never auto-cleared unless a turn re-evaluates it, so its age at session end is
        # the time the loop sat parked (the state may have been cleared by the user afterwards).
        end = float(run.sessions[root].get("ended_at") or 0)
        parked["parked_minutes_until_session_end"] = round((end - float(goal_state["waiting_since"])) / 60, 1) if end else None
    report = {"observed": {
        "nudges": len(nudges), "nudges_within_180s_of_a_waiting_turn": sum(1 for n in nudges if n["prev_turn_said_waiting"] and (n["gap_s"] or 1e9) < 180),
        "nudge_detail": nudges, "async_batch_notices": batch_notices, "background_process_notices": bgproc_notices,
        "final_goal_state": parked,
    }}
    path = run.write("goal_loop.json", report)
    o = report["observed"]
    print(f"[goal] nudges {o['nudges']}, of which {o['nudges_within_180s_of_a_waiting_turn']} fired <180 s after a turn that said it was waiting; "
          f"batch notices {o['async_batch_notices']}, bg-process notices {o['background_process_notices']}")
    print(f"[goal] final goal state: {o['final_goal_state']}")
    print(f"[goal] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
