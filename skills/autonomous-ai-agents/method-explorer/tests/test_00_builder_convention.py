"""Gate 1 / Test 00 — the builder's labeling convention holds against the real model.

scenario_builder.py introduced a NEW convention the prior natural-word scenarios never
used: phonetic tags (alfa/bravo/…) + a prompt instruction telling the planner to label
its methods with those exact tags. This is the ONE load-bearing unvalidated assumption
in the whole construction plan — every Tier-2 generated scenario and Tier-3 trap is
built by the builder, so this must hold *first*.

This test builds the backtrack scenario via the builder, deploys it, runs once, and
asserts the tags resolved AND backtrack→success happened. Logic: with default:tombstone,
if the model ignored the tag labels, NO progress rule could match and everything would
tombstone → exhaustion. So *reaching success* is itself proof the tags resolved; the
explicit tag check is belt-and-suspenders.

Run:  python3 run.py test_00_builder_convention
"""
import json
import os
import subprocess

import pytest

from helpers import (PLANS, SCEN_DIR, is_fail, is_succ, run_until_journal,
                     setup_sandbox)
from scenario_builder import (build_scenario, canonical_backtrack_methods,
                              expected_terminal, method_list_prompt)

SLUG = "gate1-builder"
SCEN_NAME = "gate1-builder-backtrack.json"
HOST_SCEN = os.path.join(os.path.dirname(__file__), "..", "assets", "scenarios", SCEN_NAME)
CONT_SCEN = f"{SCEN_DIR}/{SCEN_NAME}"

METHODS = canonical_backtrack_methods()  # shared 4-method backtrack scenario


@pytest.mark.agent
def test_builder_generated_scenario_drives_the_skill():
    # Build + deploy a builder-generated scenario.
    scen = build_scenario(
        'produce a data file containing valid JSON with a top-level key "ok"',
        METHODS,
        notes="GATE1: builder-generated backtrack; validates the phonetic-tag convention",
    )
    assert expected_terminal(METHODS) == "success"
    with open(HOST_SCEN, "w") as f:
        json.dump(scen, f, indent=2)
    subprocess.run(["docker", "cp", HOST_SCEN, f"hermes:{CONT_SCEN}"], check=True)
    subprocess.run(["docker", "exec", "hermes", "sh", "-c", f"chmod a+r {CONT_SCEN}"],
                   check=True)

    setup_sandbox(SLUG)
    prompt = (
        "Use the method-explorer skill: skill_view to load it, then follow it INCLUDING "
        "Simulation Mode and the Decision Records discipline. SIMULATION MODE IS ACTIVE "
        "(read $HERMES_SIM_SCENARIO); do NOT perform real actions.\n"
        'INTENT: produce a data file containing valid JSON with a top-level key "ok".\n'
        "Methods, in preference order: alfa = fetch from the primary network source; "
        "bravo = fetch from the mirror network source; charlie = read the local cache; "
        "delta = verify the file is valid JSON with key ok.\n"
        + method_list_prompt(METHODS) + "\n"
        f"Write the plan-tree to {PLANS}/{SLUG}/plan-tree.md as a COMPACT MARKER MAP "
        "(STATE header; NODES with ○/▶/✝/✓ markers + one-line receipts; FRONTIER) and "
        f"append one lean single-line JSON record per cycle (node, q, chosen, expected, "
        f"verdict, evidence, next) to {PLANS}/{SLUG}/journal.jsonl."
    )
    rows, _ = run_until_journal(prompt, SLUG, scenario=CONT_SCEN)
    assert rows, "no journal after retries — persistent no-op (see stdout tail above)"
    verdicts = [r.get("verdict") for r in rows]
    blob = " ".join(
        str(r.get("chosen", "")) + " " + str(r.get("next", "")) for r in rows
    ).lower()

    # (1) Convention: the model actually used the phonetic recovery tags.
    assert any(t in blob for t in ("charlie", "delta")), (
        "BUILDER CONVENTION FAILED: the model did not use the recovery tags "
        "(charlie/delta) in its actions, so the simulation could not resolve the path"
    )
    # (2) Outcome: reaching success requires the progress/success tags to have matched.
    assert any(is_succ(v) for v in verdicts), (
        "did not reach success — the builder-generated backtrack path was not driven correctly"
    )
    # (3) Backtrack: a failure preceded the success.
    fs = next((i for i, v in enumerate(verdicts) if is_succ(v)), len(verdicts))
    assert any(is_fail(v) for v in verdicts[:fs]), \
        "success not preceded by a tombstone (no backtrack happened)"
