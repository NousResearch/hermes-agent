"""Test 12 — don't redo an already-completed task (resume edge).

Resumability completeness: if a prior run already reached SUCCESS, a re-invocation must
recognize the completed plan-tree and report the existing result — NOT redo the work.

We seed a SUCCESS-terminated plan-tree + a finished result file (old mtime), then invoke
the skill. RECEIPTS: the result file is unchanged (same content + mtime → not redone),
and the agent's reply acknowledges the task was already complete (distinguishing a
correct "already done" from a no-op).

Run:  python3 run.py test_12_resume_completed
"""
import time

import pytest

from helpers import PLANS, _dex, read_file, run_planner, setup_sandbox, write_container_file

SLUG = "test-resume-done"
BASE = "/tmp/test-resume-done"
RESULT = f"{BASE}/result.txt"
DONE_VALUE = "ALREADY-DONE-VALUE-42"
SEED = """# Plan-Tree: test-resume-done   STATE: SUCCESS

INTENT: write the answer to result.txt
SUCCESS: result.txt exists with the answer  [x]

NODES   (markers: ○ open/untried · ▶ active · ✝ dead · ✓ done)
- S1  compute + write the answer   ✓ result.txt written with the answer; intent met
FRONTIER: (empty)
"""

PROMPT = (
    "Use the method-explorer skill: skill_view to load it, then follow it INCLUDING the "
    "'Resuming an interrupted run' guidance. REAL run.\n"
    f"A PRIOR run already COMPLETED this task — the plan-tree at {PLANS}/{SLUG}/plan-tree.md "
    f"shows TERMINAL SUCCESS and the answer is already at {RESULT}.\n"
    f"INTENT: ensure {RESULT} contains the answer.\n"
    "Per the Resuming guidance: if the existing plan-tree is already SUCCESS, REPORT that "
    "result and STOP — do NOT redo the work or overwrite the result."
)


def _reset():
    _dex(f"mkdir -p {BASE}; printf '{DONE_VALUE}' > {RESULT}; "
         f"touch -d '2020-01-01 00:00:00' {RESULT}; chown -R 10000:10000 {BASE}")
    write_container_file(f"{PLANS}/{SLUG}/plan-tree.md", SEED)
    _dex(f"rm -f {PLANS}/{SLUG}/journal.jsonl")


@pytest.mark.agent
def test_does_not_redo_completed_task():
    setup_sandbox(SLUG)
    proc, m0, m1 = None, None, None
    for _ in range(3):
        _reset()
        m0 = _dex(f"stat -c %Y {RESULT}").stdout.strip()
        proc = run_planner(PROMPT)
        if (proc.stdout or "").strip():
            m1 = _dex(f"stat -c %Y {RESULT}").stdout.strip()
            break
        time.sleep(8)

    out = (proc.stdout or "").strip()
    assert out, "persistent no-op (empty stdout)"
    # Receipt 1: the completed result was NOT redone (content + mtime unchanged).
    assert read_file(RESULT) == DONE_VALUE, "the completed result was overwritten (redone)"
    assert m1 == m0, "result.txt mtime changed → the finished task was redone"
    # Receipt 2: the agent ran and acknowledged completion (not a silent no-op).
    low = out.lower()
    assert any(w in low for w in ("already", "complete", "done", "success", "terminal",
                                  "not redo", "no further")), \
        f"agent did not acknowledge the task was already complete: ...{out[-200:]!r}"
