"""Test 05 — upstream-jump at K=5 (don't grind a dead sub-tree).

Sim run: five DISTINCT sibling methods (attempt-a..attempt-e) under one parent
decision all tombstone. Per the skill's upstream-jump heuristic (K=5), after the
5th sibling tombstone the planner must STOP trying siblings, climb to the parent,
and RELAX the single ranked soft constraint — taking the external alternative path
instead of grinding a 6th sibling.

Asserts on receipts: >=5 fail/tombstone verdicts in the journal, then a NON-sibling
move evidenced by a later record's next/chosen/evidence (relax / upstream jump /
alternative) OR by the plan-tree showing the soft constraint relaxed — never on the
agent's prose.

Run:  pytest -m agent -k upstream_jump -v   |   python3 run.py test_05_upstream_jump_k5
"""
import pytest

from helpers import (
    PLANS,
    SCEN_DIR,
    is_fail,
    load_journal,
    read_file,
    run_planner,
    setup_sandbox,
    sim_prompt,
)

SLUG = "test-upstream-jump-k5"
TREE = f"{PLANS}/{SLUG}/plan-tree.md"

# Semantic markers of a non-sibling move (upstream jump / soft-constraint relax).
JUMP = ("relax", "upstream", "alternative", "soft constraint", "soft-constraint", "climb")


@pytest.mark.agent
def test_upstream_jump_after_five_sibling_tombstones():
    setup_sandbox(SLUG)

    prompt = sim_prompt(
        intent=(
            "secure a working build of the release artifact under the parent decision "
            "'use the in-house native build pipeline'"
        ),
        slug=SLUG,
        extra=(
            "SINGLE RANKED SOFT CONSTRAINT (relaxable): 'prefer the in-house native "
            "pipeline over an external alternative build service'. No other soft "
            "constraints.\n"
            "Try siblings FIRST: attempt the five in-house native build configs labeled "
            "'attempt-a', 'attempt-b', 'attempt-c', 'attempt-d', 'attempt-e' (all "
            "siblings/methods under the one parent decision) — each will tombstone.\n"
            "Per the skill's upstream-jump heuristic, after K=5 sibling tombstones do NOT "
            "try a 6th sibling: climb upstream to the parent decision and RELAX the soft "
            "constraint to take the external alternative path. Label that move with "
            "'relax'/'upstream'/'alternative'; reserve those words for it alone."
        ),
    )
    run_planner(prompt, scenario=f"{SCEN_DIR}/k5-siblings.json")

    rows = load_journal(SLUG)
    assert rows, "no journal written — cannot confirm the loop ran"

    verdicts = [r.get("verdict") for r in rows]
    fails = [i for i, v in enumerate(verdicts) if is_fail(v)]
    assert len(fails) >= 5, (
        f"expected >=5 sibling tombstones before the upstream jump, saw {len(fails)}"
    )

    # After the 5th tombstone the loop must make a NON-sibling move: a relax /
    # upstream-jump recorded in a later record's next/chosen/evidence, OR the
    # plan-tree showing the soft constraint relaxed. Tolerant + semantic.
    after = rows[fails[4]:]
    blob = " ".join(
        str(r.get(f, "")) for r in after for f in ("next", "chosen", "evidence")
    ).lower()
    jumped_in_journal = any(k in blob for k in JUMP)

    tree = (read_file(TREE) or "").lower()
    relaxed_in_tree = ("relax" in tree and ("soft" in tree or "constraint" in tree)) \
        or "upstream" in tree

    assert jumped_in_journal or relaxed_in_tree, (
        "no upstream-jump / soft-constraint relaxation after 5 sibling tombstones — "
        "the loop ground the dead sub-tree instead of climbing"
    )
