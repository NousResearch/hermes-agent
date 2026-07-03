"""Test 10 — resume an interrupted run from a partial plan-tree.

The dominant runtime flake (backend timeout) can kill a run mid-task. The skill's
'Resuming an interrupted run' guidance says: on invocation, read an existing,
non-terminated plan-tree and continue from its Dead-set/Frontier — don't restart.

We seed a partial plan-tree (alfa already tombstoned, frontier = charlie, NOT
terminated), then invoke the skill. RESUME receipt: it must NOT re-attempt alfa (a
restart would try the alfa happy-path first) and must reach success via charlie.

Run:  python3 run.py test_10_resume
"""
import pytest

from helpers import (PLANS, deploy_scenario, is_succ, run_until_journal,
                     setup_sandbox, write_container_file)
from scenario_builder import build_prompt, build_scenario, canonical_resume_methods

SLUG = "test-resume"
METHODS = canonical_resume_methods()  # alfa (seeded ✝) + charlie → delta (shared)
SEED = """# Plan-Tree: test-resume   STATE: active

INTENT: obtain the data (a valid result)
HARD (inviolable): real data only
SOFT (relaxable, ranked): (none)

NODES   (markers: ○ open/untried · ▶ active · ✝ dead · ✓ done)
- S1  alfa (primary fetch)   ✝ tombstoned: primary source down (non-transient); LOCAL — method dead
FRONTIER: charlie (local cache)
"""


@pytest.mark.agent
def test_resumes_from_partial_plan_tree():
    setup_sandbox(SLUG)
    scen = build_scenario("obtain the data", METHODS, notes="resume regression")
    cont = deploy_scenario("resume-scen.json", scen)
    write_container_file(f"{PLANS}/{SLUG}/plan-tree.md", SEED)  # seed the partial (active) tree

    prompt = build_prompt(
        "obtain the data", METHODS, SLUG,
        meanings={"alfa": "primary", "charlie": "cache", "delta": "verify"},
        extra=("A PRIOR run on this task was INTERRUPTED — a partial plan-tree already "
               f"exists at {PLANS}/{SLUG}/plan-tree.md. Follow the skill's 'Resuming an "
               "interrupted run' guidance: READ it and RESUME from its Dead-set + Frontier. "
               "Do NOT restart from scratch and do NOT re-attempt any method already in the "
               "Dead-set."))
    # preserve_tree=True: a no-op retry keeps the seeded plan-tree (a wipe would restart).
    rows, _ = run_until_journal(prompt, SLUG, scenario=cont, preserve_tree=True)

    assert rows, "persistent no-op (empty journal after retries)"
    # RESUME receipt: the pre-tombstoned method alfa was NOT re-attempted.
    assert not any("alfa" in str(r.get("chosen", "")).lower() for r in rows), (
        "did NOT resume — it re-attempted the pre-tombstoned method 'alfa' (it restarted "
        "from scratch instead of honoring the seeded Dead-set)"
    )
    # And it still reached success via the frontier (charlie → delta).
    assert any(is_succ(r.get("verdict")) for r in rows), "did not reach success on resume"
