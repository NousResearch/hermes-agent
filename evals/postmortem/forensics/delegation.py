"""Lane 2: delegation behaviour — nested-delegate timeouts, orphaned children, polling cost, batch-join
delivery delay, and summary truncation. All OBSERVED from ``messages`` in the run tree.

    python -m evals.postmortem.forensics.delegation --db state_copy.db

Reproduces: how many ``delegate_task`` tool results were deadline timeouts and in how many sessions;
how many nested calls ever returned a result; what those sessions spent after their first timeout
(lifetime cost, with the caveat that it includes real work); total ``sleep N`` seconds; child results
withheld by batch joins (concurrent child-hours, NOT critical path); and how many parent-side summaries
carried the truncation footer.
"""
from __future__ import annotations

import collections
import json
import re
import statistics
from typing import Any, Dict, List

from evals.postmortem.forensics.common import Run

_TIMEOUT = re.compile(r"timed out after ([\d.]+)s")
_SLEEP = re.compile(r"\bsleep\s+(\d+)")
_TRUNC = re.compile(r"\[SUMMARY TRUNCATED\]|middle omitted|trimmed to protect the parent")


def main(argv=None) -> int:
    run = Run.from_args(argv, (__doc__ or "").split("\n\n")[0])
    parent_of = {s: run.sessions[s].get("parent_session_id") for s in run.in_run}
    children_of: Dict[str, List[str]] = collections.defaultdict(list)
    for s, p in parent_of.items():
        if p:
            children_of[p].append(s)
    orchestrators = [s for s in run.in_run if children_of.get(s)]

    timeouts_by_sess: Dict[str, int] = collections.Counter()
    delegate_results = delegate_ok = 0
    sleep_seconds_after_timeout = 0
    first_timeout_ts: Dict[str, float] = {}
    truncated_summaries = total_summaries = 0
    for sid in orchestrators:
        for m in run.messages(sid, "role, tool_name, content, tool_calls, timestamp"):
            if m["role"] == "tool" and (m.get("tool_name") or "") == "delegate_task":
                c = m.get("content") or ""
                delegate_results += 1
                if _TIMEOUT.search(c) and "delegate_task" in c:
                    timeouts_by_sess[sid] += 1
                    first_timeout_ts.setdefault(sid, float(m.get("timestamp") or 0))
                elif '"status"' in c and ("completed" in c or "summary" in c):
                    delegate_ok += 1
                    total_summaries += 1
                    if _TRUNC.search(c):
                        truncated_summaries += 1
            if m["role"] == "user" and (m.get("content") or "").startswith("[ASYNC DELEGATION"):
                # background batches re-enter as a user-role completion block, one entry per child
                c = m["content"]
                entries = c.count("--- ✓ TASK") + c.count("--- ✗ TASK") + c.count("--- ⚠ TASK") or (1 if "COMPLETE" in c else 0)
                total_summaries += entries
                truncated_summaries += c.count("[SUMMARY TRUNCATED]")
            if m["role"] == "assistant" and sid in first_timeout_ts and float(m.get("timestamp") or 0) >= first_timeout_ts[sid]:
                for mt in _SLEEP.finditer(m.get("tool_calls") or ""):
                    sleep_seconds_after_timeout += int(mt.group(1))

    timeout_sessions = list(timeouts_by_sess)
    nested_timeout_sessions = [s for s in timeout_sessions if run.depth[s] >= 1]
    lifetime_cost = sum(run.cost(s) for s in timeout_sessions)
    lifetime_cache_write = sum(float(run.sessions[s].get("cache_write_tokens") or 0) for s in timeout_sessions) * run.price_per_token["cache_write_tokens"]

    # batch-join delivery delay: children of one parent dispatched within 60 s of each other = one batch
    withheld: Dict[int, List[float]] = collections.defaultdict(list)
    for p, kids in children_of.items():
        kids = sorted(kids, key=lambda k: float(run.sessions[k].get("started_at") or 0))
        batch: List[str] = []
        def flush(batch: List[str]) -> None:
            if len(batch) < 2:
                return
            ends = [float(run.sessions[k].get("ended_at") or 0) for k in batch]
            last = max(ends)
            withheld[run.depth.get(str(p), 0)].extend(last - e for e in ends if e)
        for k in kids:
            if batch and float(run.sessions[k].get("started_at") or 0) - float(run.sessions[batch[-1]].get("started_at") or 0) > 60:
                flush(batch); batch = []
            batch.append(k)
        flush(batch)

    report: Dict[str, Any] = {
        "observed": {
            "orchestrators": len(orchestrators),
            "delegate_task_results": delegate_results,
            "delegate_task_timeouts": sum(timeouts_by_sess.values()),
            "sessions_with_timeouts": len(timeout_sessions),
            "nested_sessions_with_timeouts": len(nested_timeout_sessions),
            "delegate_task_results_that_carried_a_result": delegate_ok,
            "sleep_hours_after_first_timeout": round(sleep_seconds_after_timeout / 3600, 1),
            "timeout_sessions_lifetime_cost_usd": round(lifetime_cost, 2),
            "timeout_sessions_lifetime_cache_write_usd": round(lifetime_cache_write, 2),
            "lifetime_cost_caveat": "whole-session spend; includes legitimate work after the timeout, and its cache writes overlap the tokens lane's excess proxy",
            "summaries_truncated": truncated_summaries, "summaries_total": total_summaries,
            "batch_join_withheld_child_hours_by_parent_depth": {d: round(sum(v) / 3600, 1) for d, v in sorted(withheld.items())},
            "batch_join_withheld_minutes_median_by_parent_depth": {d: round(statistics.median(v) / 60, 1) for d, v in sorted(withheld.items()) if v},
            "withheld_caveat": "concurrent child-hours held back from the parent; not critical-path time",
        }
    }
    path = run.write("delegation.json", report)
    o = report["observed"]
    print(f"[delegation] {o['orchestrators']} orchestrators; delegate_task results {o['delegate_task_results']:,}: timeouts {o['delegate_task_timeouts']} in {o['sessions_with_timeouts']} sessions "
          f"({o['nested_sessions_with_timeouts']} nested); carried a result: {o['delegate_task_results_that_carried_a_result']}")
    print(f"[delegation] after first timeout: sleep {o['sleep_hours_after_first_timeout']} h; those sessions' lifetime ${o['timeout_sessions_lifetime_cost_usd']:,} (cache writes ${o['timeout_sessions_lifetime_cache_write_usd']:,}; includes real work)")
    print(f"[delegation] summaries truncated {o['summaries_truncated']}/{o['summaries_total']}; batch-join withheld child-hours by parent depth {o['batch_join_withheld_child_hours_by_parent_depth']}")
    print(f"[delegation] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
