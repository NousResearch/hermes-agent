#!/usr/bin/env python3
"""harvest.py — pure fold: one cycle's task-plan results → ledger records.

Turns per-task verdicts into evidence for the next clarify/replan round:
  - verdict=failed      → kind=dead-end  "Tried <method>: failed — <evidence>"
  - verdict=worked      → kind=fact      "Done <method>: <evidence>"
  - verdict=needs_split → kind=fact      "SPLIT HINT: task '<method>' is too coarse —
                          split into: <split items>" (fp "split <method>" — the method
                          is NOT dead, the task was too coarse; the driver forces a
                          partial replan off this verdict)
  - verdict=skipped     → nothing (its failed dependency already produced the dead-end)
  - EITHER verdict's `learnings` (optional, from run_task) → one extra kind=fact record
    PER learning, fp'd on the learning's own text — folded regardless of whether the
    task worked or failed, since a failure can still surface something worth
    remembering beyond its dead-end evidence text.

Dead-end fp keys on the METHOD LABEL, not the failure evidence: a method dying twice
with a freshly-worded reason is the flap this guard exists to catch. Worked facts
fingerprint in a distinct namespace ("ok <method>") so a method that failed in c0 and
worked in c2 records BOTH transitions. Learnings fp on their own text (a free-form
"what/why/what-happened/why" mini post-mortem, not a method label) — a distinct
namespace from both, so no collision is possible in practice.

The `exhausted` / `needs_decision` plan dispositions are folded by the driver directly
(relentless.py fold_one) — this module only folds task results.

Stdlib only; no file IO, no env, no LLM.
"""

import hashlib
import re


def fp(text):
    """Anti-flap fingerprint: case/whitespace/punctuation-insensitive identity hash."""
    t = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def harvest_tasks(plan, results, cycle):
    """One cycle's plan + per-task results → ledger records (source=harvest).

    plan: the validated plan dict (success criteria travel in each record's meta);
    results: [{id, method, verdict, evidence, learnings?}] as produced by the driver's
    run_task (plus its dependency-skip synthesis) — `learnings` is optional and, when
    present, is a list of free-form strings independent of `verdict`.
    """
    criteria = {t["id"]: t.get("success_criterion", "") for t in plan.get("tasks") or []}
    records = []
    for r in results:
        meta = {"task": r["id"], "criterion": criteria.get(r["id"], "")}
        if r["verdict"] == "failed":
            records.append({"cycle": cycle, "source": "harvest", "kind": "dead-end",
                            "text": f"Tried {r['method']}: failed — {r['evidence']}",
                            "fp": fp(r["method"]), "meta": meta})
        elif r["verdict"] == "worked":
            records.append({"cycle": cycle, "source": "harvest", "kind": "fact",
                            "text": f"Done {r['method']}: {r['evidence']}",
                            "fp": fp("ok " + r["method"]), "meta": meta})
        elif r["verdict"] == "needs_split":
            # A FACT, not a dead-end: the method isn't dead, the task was too coarse.
            # fp on "split <method>" (own namespace) — declaring the same split twice
            # is the flap this dedup catches; the driver forces a partial replan off
            # this verdict, and the split hint reaches the replan prompt via this text.
            split = "; ".join(r.get("split") or []) or r["evidence"]
            records.append({"cycle": cycle, "source": "harvest", "kind": "fact",
                            "text": f"SPLIT HINT: task '{r['method']}' is too coarse — "
                                    f"split into: {split}",
                            "fp": fp("split " + r["method"]), "meta": meta})
        # skipped: no record — the failed dependency already recorded the dead-end
        for learning in r.get("learnings") or []:
            records.append({"cycle": cycle, "source": "harvest", "kind": "fact",
                            "text": learning, "fp": fp(learning),
                            "meta": {**meta, "learning_from": r["id"]}})
    return records
