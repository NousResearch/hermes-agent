"""Test 16 — driver resumes a seeded interrupted run to a terminal STATE (real model).

End-to-end proof that scripts/drive.py drives the real skill: we seed an `active` partial
plan-tree (alfa ✝ tombstoned; FRONTIER: charlie) — exactly the state a cap/no-op cut-off
leaves — then run the REAL `drive()` wired to the test harness (run_planner / read_file /
_dex mv). The driver must re-invoke `hermes -z`, the skill must RESUME from the on-disk
plan-tree (not restart), and the loop must stop at a terminal STATE.

Receipts: driver status SUCCESS; final plan-tree STATE == SUCCESS; the UNION of journal
archives never chose `alfa` (resumed from the dead-set, didn't restart); plan-tree.md
still present at the end (the driver never deletes it).

Non-deterministic (real model) -> run via `run.py --reps 3` and require a pass-rate.

Run:  python3 run.py test_16_driver_resume_integration --reps 3
"""
import os
import sys

import pytest

import helpers
from helpers import (PLANS, SCEN_DIR, _dex, file_exists, is_succ, parse_journal_text,
                     read_file, setup_sandbox, write_container_file)
from scenario_builder import build_prompt, build_scenario, canonical_resume_methods

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from drive import drive, parse_state  # noqa: E402

SLUG = "test-driver-resume"
PT = f"{PLANS}/{SLUG}/plan-tree.md"

METHODS = canonical_resume_methods()  # alfa (seeded ✝) + charlie→delta (shared)
MEANINGS = {"alfa": "primary source", "charlie": "cache", "delta": "verify"}

# A partial, NON-terminal plan-tree: alfa already dead, charlie open on the frontier.
SEED = """# Plan-Tree: test-driver-resume   STATE: active

INTENT: produce a data file with a top-level key "ok" (network preferred)
HARD (inviolable): valid JSON containing key "ok"
SOFT (relaxable, ranked): 1) prefer the network source

NODES   (markers: ○ open/untried · ▶ active · ✝ dead · ✓ done)
- S1  alfa (primary fetch)   ✝ tombstoned: primary source down (non-transient); LOCAL — method dead
FRONTIER: charlie (local cache)
"""


def _deploy_scenario():
    scen = build_scenario("produce data with key ok (network preferred)", METHODS,
                          notes="driver resume integration")
    return helpers.deploy_scenario("driver-resume-scen.json", scen)


@pytest.mark.agent
def test_driver_resumes_seeded_run_to_success():
    setup_sandbox(SLUG)
    write_container_file(PT, SEED)                       # seed the active plan-tree
    _dex(f"rm -f {PLANS}/{SLUG}/journal.jsonl {PLANS}/{SLUG}/journal.tick*.jsonl")
    cont = _deploy_scenario()

    prompt = build_prompt("produce a data file with a top-level key \"ok\" (network preferred)",
                          METHODS, SLUG, meanings=MEANINGS, mode="sim")

    res = drive(
        prompt, SLUG,
        invoke=lambda p, t: helpers.run_planner(p, scenario=cont, timeout=t),
        read_plan_tree=lambda s: read_file(f"{PLANS}/{s}/plan-tree.md"),
        archive_journal=lambda s, n: _dex(
            f"mv {PLANS}/{s}/journal.jsonl {PLANS}/{s}/journal.tick{n}.jsonl 2>/dev/null || true"),
        plan_path=PT, max_ticks=6, bump_guard=False,
    )

    # Receipt 1: the driver reached a terminal SKILL state, specifically SUCCESS.
    assert res.terminal, f"driver stopped on a backstop, not a terminal state: {res.status} ({res.detail})"
    assert res.status == "SUCCESS", f"expected SUCCESS, got {res.status} ({res.detail})"

    # Receipt 2: the on-disk plan-tree confirms it, and was never deleted.
    assert file_exists(PT), "driver deleted the plan-tree (it must only ever read it)"
    final = read_file(PT) or ""
    assert parse_state(final) == "SUCCESS", f"final plan-tree STATE != SUCCESS:\n{final[:400]}"

    # Receipt 3: across the UNION of all per-tick journals, the seeded ✝ method was never
    # re-chosen -> it resumed from the dead-set instead of restarting.
    raw = _dex(f"cat {PLANS}/{SLUG}/journal.tick*.jsonl {PLANS}/{SLUG}/journal.jsonl 2>/dev/null").stdout
    rows = parse_journal_text(raw or "")
    assert rows, "no journal written across any tick"
    assert not any("alfa" in str(r.get("chosen", "")).lower() for r in rows), (
        "RESUME FAILED: the driver/skill re-attempted the seeded ✝ method 'alfa' "
        "(restarted from scratch instead of honoring the on-disk dead-set)"
    )
    assert any(is_succ(r.get("verdict")) for r in rows), "no success verdict in the journal union"

    # Informational: how many ticks + whether the skill appended or overwrote the journal.
    n_arch = int(_dex(f"ls {PLANS}/{SLUG}/journal.tick*.jsonl 2>/dev/null | wc -l").stdout.strip() or 0)
    print(f"  [driver] status={res.status} productive_ticks={res.productive_ticks} "
          f"invokes={res.invokes} archived_journals={n_arch}")
