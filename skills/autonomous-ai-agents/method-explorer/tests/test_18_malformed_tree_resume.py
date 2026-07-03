"""Test 18 — self-repair a torn (headerless) plan-tree on resume.

An interruption can tear the plan-tree mid-write, leaving a fragment with NO parseable
STATE header. The skill's 'Artifact self-repair' rule: harvest the ✝ dead-set from the
readable fragment + the journal's fail verdicts, re-derive the frontier via P3, write a
fresh templated tree, and resume — WITHOUT re-attempting the harvested-dead method and
without restarting from scratch.

Bare resume path (the driver path is test_16's job). Uses run_planner directly with a
manual no-op retry loop: run_until_journal's empty-journal no-op detection is defeated
by a seeded journal (rows are never empty), so seed rows are counted and sliced off.

Run:  python3 run.py test_18_malformed_tree_resume
"""
import json
import time

import pytest

from helpers import (PLANS, _dex, deploy_scenario, is_succ, load_journal,
                     read_file, run_planner, setup_sandbox, write_container_file)
from scenario_builder import build_prompt, build_scenario, canonical_resume_methods

SLUG = "test-torn-tree"
METHODS = canonical_resume_methods()  # alfa (dead) + charlie -> delta (succeed)

# Torn-write fixture: a NODES fragment, readable ✝ alfa line, NO "# Plan-Tree:"/STATE
# header, truncated mid-word — exactly what a killed write_file can leave behind.
TORN_TREE = """NODES   (markers: ○ open/untried · ▶ active · ✝ dead · ✓ done)
- S1  alfa (primary fetch)   ✝ tombstoned: primary source down (non-transient for this task); dead
FRONT"""

SEED_ROW = {"node": "S1", "q": "fetch the data from the primary source?",
            "chosen": "alfa", "expected": "data returned",
            "verdict": "fail", "evidence": "primary source down", "next": "backtrack->cache"}


def _has_state_header(tree):
    import re
    return bool(re.search(r"STATE:\s*[A-Za-z][A-Za-z-]*", tree))  # drive.parse_state semantics


@pytest.mark.agent
def test_repairs_torn_tree_and_resumes_without_redoing_dead_method():
    setup_sandbox(SLUG)
    scen = build_scenario("obtain the data", METHODS, notes="torn-tree self-repair regression")
    cont = deploy_scenario("torn-tree-scen.json", scen)
    write_container_file(f"{PLANS}/{SLUG}/plan-tree.md", TORN_TREE)
    write_container_file(f"{PLANS}/{SLUG}/journal.jsonl", json.dumps(SEED_ROW) + "\n")
    seed_count = 1

    prompt = build_prompt(
        "obtain the data", METHODS, SLUG,
        meanings={"alfa": "primary", "charlie": "cache", "delta": "verify"},
        extra=(f"A PRIOR run was INTERRUPTED MID-WRITE: {PLANS}/{SLUG}/plan-tree.md exists "
               "but is TORN (no STATE header). Follow the skill's 'Artifact self-repair' "
               "rule for a malformed plan-tree: harvest the dead-set from the fragment + "
               "journal, re-derive the frontier, write a fresh templated tree, and resume. "
               "Do NOT restart from scratch and do NOT re-attempt a harvested-dead method."))

    rows, tree = [], ""
    for attempt in range(3):  # manual no-op retry: seeded rows defeat run_until_journal's
        run_planner(prompt, scenario=cont)            # empty-journal detection
        rows = load_journal(SLUG)
        tree = read_file(f"{PLANS}/{SLUG}/plan-tree.md")
        if len(rows) > seed_count or _has_state_header(tree):
            break
        time.sleep(8)
    new_rows = rows[seed_count:]
    assert new_rows, "persistent no-op: no new journal rows after 3 attempts"

    # (a) The harvested ✝ survived the repair: alfa was NOT re-attempted.
    assert not any("alfa" in str(r.get("chosen", "")).lower() for r in new_rows), (
        "re-attempted the harvested-dead method 'alfa' — the torn tree's ✝ line (and the "
        "journal's fail verdict) were discarded instead of harvested"
    )
    # (b) The tree was rebuilt with a parseable STATE header (drive.py can resume it).
    assert _has_state_header(tree), (
        f"plan-tree still headerless after the run — no self-repair happened; head: "
        f"{tree.splitlines()[0][:80] if tree.strip() else '(empty)'!r}"
    )
    # (c) It resumed to success via the frontier (charlie -> delta).
    assert any(is_succ(r.get("verdict")) for r in new_rows), \
        "did not reach success after the repair"

    # The repair note is the stochastic slot: log, don't assert.
    noted = any("repair" in (str(r.get("next", "")) + str(r.get("evidence", ""))).lower()
                for r in new_rows)
    print(f"  [info] explicit repair note in journal: {noted}")
