"""Live judge A/B on the run's real "waiting" response shape. Usage: python judge_ab.py <repo_root> [n]"""
import os, sys
root = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
sys.path.insert(0, root)
os.environ.setdefault("HERMES_HOME", os.path.expanduser("~/.hermes"))  # LIVE: real auxiliary judge calls (cents)
from hermes_cli import goals
goal = ("Simplify the hermes-agent codebase by >=30% LOC with zero behavior change, decomposing every god file, "
        "through parallel subagent waves; integrate, run the full test suite, and open one PR.")
response = ("Round 3 status: 4 worker batches are still running (deleg_9f1a2b, deleg_77c0de, deleg_a1b2c3, deleg_0e9f88; "
            "8 workers on adapters/gateway god files, median 38 min in). Round-2 integration merged clean; full suite "
            "green on the integration branch. Nothing else is dispatchable until these return: their summaries decide "
            "which files round 4 takes. Waiting on the batch-complete notifications; no action needed from me now.")
kw = {"active_delegations": 4} if "active_delegations" in goals.judge_goal.__code__.co_varnames else {}
verdicts = []
for _ in range(n):
    v, reason, pf, directive, tf = goals.judge_goal(goal, response, **kw)
    verdicts.append((v, directive.get("seconds") if directive else None))
    print(v, directive, "|", reason[:90])
print("SUMMARY", verdicts)
